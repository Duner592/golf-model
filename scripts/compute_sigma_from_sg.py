#!/usr/bin/env python3
# scripts/compute_sigma_from_sg.py
#
# Estimate player-specific round-to-round sigma (volatility) from strokes-gained per round.
# Inputs (expected, best-effort):
#   - data/processed/{tour}/event_{event_id}_player_form.parquet  (should contain per-round sg_total with dates)
# Fallbacks:
#   - If per-round file is missing or lacks required cols, produce a constant sigma=2.8 for all players in field.
# Output:
#   - data/processed/{tour}/event_{event_id}_player_sigma.parquet with columns [dg_id(or player_id), sigma]
#
# You can later replace the loader to point to your exact per-round SG endpoint output.

import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


TOUR = "pga"
DEFAULT_SIGMA = 2.8
BOUNDS = (1.8, 3.8)  # clip sigma to this range
LOOKBACK = 100  # last N rounds
MIN_ROUNDS = 20  # minimum to estimate; else shrink heavily
PRIOR_K = 60.0  # strength of prior (pseudo rounds)
HL = 50.0  # recency half-life (rounds)


def _kish_eff_n(weights: np.ndarray) -> float:
    s1 = weights.sum()
    s2 = (weights**2).sum() + 1e-12
    return float(s1 * s1 / s2)


def _weighted_stats(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    w = w / (w.sum() + 1e-12)
    mu = float((w * x).sum())
    var = float((w * (x - mu) ** 2).sum())
    return mu, np.sqrt(var)


def compute_sigma(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    # Sort and keep only last LOOKBACK rounds per player
    df = df.sort_values([id_col, "date"]).copy()
    parts = []
    for pid, g in df.groupby(id_col, sort=False):
        x = g["sg_total"].to_numpy(dtype=float)
        if x.size == 0:
            parts.append((pid, DEFAULT_SIGMA, 0))
            continue
        # keep last LOOKBACK
        x = x[-LOOKBACK:]
        n = len(x)
        # recency weights (oldest to newest)
        idx = np.arange(n)
        w = 0.5 ** ((idx.max() - idx) / HL)
        w = w / (w.sum() + 1e-12)
        # baseline (weighted mean), residuals, sample sigma
        mu, s = _weighted_stats(x, w)
        eff_n = _kish_eff_n(w)
        parts.append((pid, float(s), float(eff_n)))

    sig = pd.DataFrame(parts, columns=[id_col, "sample_sigma", "eff_n"])

    # pooled median sigma among those with enough rounds, else default
    pool = sig.loc[sig["eff_n"] >= MIN_ROUNDS, "sample_sigma"]
    s_pool = float(pool.median()) if not pool.empty else DEFAULT_SIGMA
    s_pool = float(np.clip(s_pool, *BOUNDS))

    # EB shrinkage
    var_i = (sig["eff_n"] / (sig["eff_n"] + PRIOR_K)) * (sig["sample_sigma"] ** 2) + (
        PRIOR_K / (sig["eff_n"] + PRIOR_K)
    ) * (s_pool**2)
    sigma = np.sqrt(var_i).clip(*BOUNDS)
    out = sig[[id_col]].copy()
    out["sigma"] = sigma
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No event meta; run parse_field_updates.py first.")
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    event_id = str(meta["event_id"])

    # Try to load per-round SG file (you should wire your player_form fetch to produce this)
    # Expected columns minimally: [dg_id or player_id], date, sg_total
    candidates = [
        processed / f"event_{event_id}_player_form.parquet",
        processed / f"event_{event_id}_player_form.csv",
    ]
    df_rounds = None
    for p in candidates:
        if p.exists():
            df_rounds = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            break

    # Determine id column
    id_col = None
    if df_rounds is not None:
        for cand in ["dg_id", "player_id", "id"]:
            if cand in df_rounds.columns:
                id_col = cand
                break

    # If we can compute sigma from rounds:
    if df_rounds is not None and id_col and "sg_total" in df_rounds.columns:
        # Ensure date
        if "date" in df_rounds.columns:
            df_rounds["date"] = pd.to_datetime(df_rounds["date"], errors="coerce")
        else:
            # If no date, fabricate monotonically increasing to allow weighting (not ideal)
            df_rounds = df_rounds.sort_values([id_col]).copy()
            df_rounds["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
                np.arange(len(df_rounds)), unit="D"
            )

        sigma_df = compute_sigma(
            df_rounds[[id_col, "date", "sg_total"]].dropna(subset=["sg_total"]),
            id_col=id_col,
        )

    else:
        # Fallback: constant sigma for players in field
        # Load field table to get ids
        field_candidates = [
            processed / f"event_{event_id}_field_teetimes.parquet",
            processed / f"event_{event_id}_field.parquet",
            processed / f"event_{event_id}_field_teetimes.csv",
            processed / f"event_{event_id}_field.csv",
        ]
        field = None
        for p in field_candidates:
            if p.exists():
                field = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                break
        if field is None:
            raise FileNotFoundError("Field table not found for sigma fallback.")

        for cand in ["dg_id", "player_id", "id"]:
            if cand in (field.columns):
                id_col = cand
                break
        if id_col is None:
            raise ValueError("No id column in field table for sigma fallback.")

        sigma_df = field[[id_col]].drop_duplicates().copy()
        sigma_df["sigma"] = DEFAULT_SIGMA

    out_path = processed / f"event_{event_id}_player_sigma.parquet"
    sigma_df.to_parquet(out_path, index=False)
    print(f"Saved sigma: {out_path} (rows={len(sigma_df)})")


if __name__ == "__main__":
    main()
