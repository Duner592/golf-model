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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOUR = "pga"
DEFAULT_SIGMA = 2.8
BOUNDS = (1.8, 3.8)
LOOKBACK = 100
MIN_ROUNDS = 20
PRIOR_K = 60.0
HL = 50.0


def weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = w / (w.sum() + 1e-12)
    mu = float((w * x).sum())
    var = float((w * (x - mu) ** 2).sum())
    return mu, np.sqrt(var)


def kish_eff_n(w: np.ndarray) -> float:
    s1 = w.sum()
    s2 = (w**2).sum() + 1e-12
    return float(s1 * s1 / s2)


def compute_sigma_df(rounds: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = rounds.sort_values([id_col, "date"]).copy()
    parts = []
    for pid, g in df.groupby(id_col, sort=False):
        x = g["sg_total"].astype(float).to_numpy()
        if x.size == 0:
            parts.append((pid, DEFAULT_SIGMA, 0.0))
            continue
        x = x[-LOOKBACK:]
        n = len(x)
        idx = np.arange(n)
        w = 0.5 ** ((idx.max() - idx) / HL)
        w = w / (w.sum() + 1e-12)
        _, s = weighted_stats(x, w)
        eff_n = kish_eff_n(w)
        parts.append((pid, float(s), float(eff_n)))
    sig = pd.DataFrame(parts, columns=[id_col, "sample_sigma", "eff_n"])
    pool = sig.loc[sig["eff_n"] >= MIN_ROUNDS, "sample_sigma"]
    s_pool = float(pool.median()) if not pool.empty else DEFAULT_SIGMA
    s_pool = float(np.clip(s_pool, *BOUNDS))
    var_i = (sig["eff_n"] / (sig["eff_n"] + PRIOR_K)) * (sig["sample_sigma"] ** 2) + (PRIOR_K / (sig["eff_n"] + PRIOR_K)) * (s_pool**2)
    sigma = np.sqrt(var_i).clip(*BOUNDS)
    out = sig[[id_col]].copy()
    out["sigma"] = sigma
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute per-player sigma (volatility) for the pinned/current event.")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR

    # Determine event_id
    if args.event_id:
        event_id = str(args.event_id)
    else:
        meta = json.loads(sorted(processed.glob("event_*_meta.json"))[-1].read_text(encoding="utf-8"))
        event_id = str(meta["event_id"])

    # Load SG rounds (your pipeline should have created this; adjust name if needed)
    # Fallback: if not present, create a constant sigma for the event field players
    rounds_pq = processed / f"event_{event_id}_player_form.parquet"
    if rounds_pq.exists():
        rounds = pd.read_parquet(rounds_pq)
        id_col = "player_id" if "player_id" in rounds.columns else ("dg_id" if "dg_id" in rounds.columns else None)
        if id_col is None:
            raise ValueError("player_form parquet missing id col")
        if "date" not in rounds.columns or "sg_total" not in rounds.columns:
            raise ValueError("player_form parquet needs [date, sg_total]")
        rounds["date"] = pd.to_datetime(rounds["date"], errors="coerce")
        sigma_df = compute_sigma_df(
            rounds[[id_col, "date", "sg_total"]].dropna(subset=["date", "sg_total"]),
            id_col=id_col,
        )
    else:
        # Fallback: constant sigma for players in field
        field = None
        for name in [
            f"event_{event_id}_field_teetimes.parquet",
            f"event_{event_id}_field.parquet",
            f"event_{event_id}_field_teetimes.csv",
            f"event_{event_id}_field.csv",
        ]:
            p = processed / name
            if p.exists():
                field = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                break
        if field is None:
            raise FileNotFoundError("Field table not found for sigma fallback.")
        id_col = "player_id" if "player_id" in field.columns else ("dg_id" if "dg_id" in field.columns else None)
        if id_col is None:
            raise ValueError("No id column in field table for sigma fallback.")
        sigma_df = field[[id_col]].drop_duplicates().copy()
        sigma_df["sigma"] = DEFAULT_SIGMA

    out_path = processed / f"event_{event_id}_player_sigma.parquet"
    sigma_df.to_parquet(out_path, index=False)
    print(f"Saved sigma: {out_path} (rows={len(sigma_df)})")


if __name__ == "__main__":
    main()
