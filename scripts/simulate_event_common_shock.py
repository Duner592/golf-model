#!/usr/bin/env python3
# scripts/simulate_event_common_shock.py
#
# Enhanced simulator:
# - Uses per-player sigma from features_full (added by merge_sigma_into_features.py)
# - Adds common shocks:
#     * round-level shared shock (all players each round)
#     * wave-level shared shock (AM/PM for R1/R2 if waves available)
# - 36-hole cut (top 65 and ties approximation)
# Output:
#   - data/preds/{tour}/event_{event_id}_preds_common_shock.parquet / .csv

import json
from pathlib import Path
import numpy as np
import pandas as pd

TOUR = "pga"
CUT_TOP = 65
N_SIMS = 30000
SEED = 42


def load_meta() -> dict:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No event meta found.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def load_features(event_id: str) -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    feats = pd.read_parquet(
        root / "data" / "features" / TOUR / f"event_{event_id}_features_full.parquet"
    )
    # IDs and names
    if "dg_id" not in feats.columns and "player_id" in feats.columns:
        feats = feats.rename(columns={"player_id": "dg_id"})
    if "player_name" not in feats.columns:
        feats["player_name"] = feats["dg_id"].astype(str)

    # Skill column selection (adjust if you want a specific one)
    for cand in ["sr_sg_total", "sr_dg_rating", "dr_dg_rating", "dr_sg_total"]:
        if cand in feats.columns:
            skill_col = cand
            break
    else:
        skill_col = None

    # Weather matrices
    wave_cols = [f"weather_r{r}_delta_wave" for r in [1, 2, 3, 4]]
    neu_cols = [f"weather_r{r}_delta_neutral" for r in [1, 2, 3, 4]]
    if all(c in feats.columns for c in wave_cols):
        weather = feats[wave_cols].astype(float).fillna(0.0).to_numpy()
    elif all(c in feats.columns for c in neu_cols):
        weather = feats[neu_cols].astype(float).fillna(0.0).to_numpy()
    else:
        weather = np.zeros((len(feats), 4))

    # Mu base from skill: convert SG (+ is good) to strokes offset (lower better)
    if skill_col:
        mu_base = -feats[skill_col].astype(float).fillna(0.0).to_numpy()
    else:
        mu_base = np.zeros(len(feats), dtype=float)

    # Sigma
    sigma = feats["sigma"].astype(float).fillna(2.8).clip(1.8, 3.8).to_numpy()

    # Waves for R1/R2 (optional)
    r1_wave = (
        feats["r1_wave"].fillna("NA").to_numpy()
        if "r1_wave" in feats.columns
        else np.array(["NA"] * len(feats))
    )
    r2_wave = (
        feats["r2_wave"].fillna("NA").to_numpy()
        if "r2_wave" in feats.columns
        else np.array(["NA"] * len(feats))
    )

    ids = feats["dg_id"].astype(str).to_numpy()
    names = feats["player_name"].astype(str).to_numpy()
    return feats, ids, names, mu_base, sigma, weather, r1_wave, r2_wave


def simulate(
    ids,
    names,
    mu_base,
    sigma,
    weather,
    r1_wave,
    r2_wave,
    n_sims=N_SIMS,
    cut_top=CUT_TOP,
    seed=SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(ids)
    wins = np.zeros(n, dtype=np.int64)
    top10 = np.zeros(n, dtype=np.int64)
    made_cut = np.zeros(n, dtype=np.int64)
    scores = np.empty((n, 4), dtype=float)

    for _ in range(n_sims):
        # Round-level common shocks (e.g., forecast miss)
        # Scale ~0.15–0.30 strokes; tune via backtesting
        round_shock = rng.normal(0.0, 0.20, size=4)

        for r in range(4):
            mu_r = mu_base + weather[:, r] + round_shock[r]

            if r in (0, 1):  # R1=0, R2=1 wave-level shared shocks
                # Wave shocks (AM vs PM) ~0.10–0.20
                shock_am = rng.normal(0.0, 0.12)
                shock_pm = rng.normal(0.0, 0.12)
                wave_adj = np.zeros(n, dtype=float)
                if r == 0:
                    wave_adj[r1_wave == "AM"] += shock_am
                    wave_adj[r1_wave == "PM"] += shock_pm
                else:
                    wave_adj[r2_wave == "AM"] += shock_am
                    wave_adj[r2_wave == "PM"] += shock_pm
                mu_r = mu_r + wave_adj

            scores[:, r] = rng.normal(loc=mu_r, scale=sigma)

        # Apply 36-hole cut
        totals_r2 = scores[:, :2].sum(axis=1)
        cut_idx = min(max(cut_top - 1, 0), n - 1)
        cut_line = np.partition(totals_r2, cut_idx)[cut_idx]
        alive = totals_r2 <= cut_line
        made_cut += alive.astype(np.int64)

        totals = scores.sum(axis=1)
        totals_masked = totals.copy()
        totals_masked[~alive] = np.inf
        order = np.argsort(totals_masked)
        wins[order[0]] += 1
        top10[order[: min(10, n)]] += 1

    return pd.DataFrame(
        {
            "dg_id": ids,
            "player_name": names,
            "p_win": wins / n_sims,
            "p_top10": top10 / n_sims,
            "p_mc": made_cut / n_sims,
        }
    )


def main():
    root = Path(__file__).resolve().parent.parent
    preds_dir = root / "data" / "preds" / TOUR
    preds_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    event_id = str(meta["event_id"])
    feats, ids, names, mu_base, sigma, weather, r1_wave, r2_wave = load_features(
        event_id
    )

    preds = simulate(
        ids,
        names,
        mu_base,
        sigma,
        weather,
        r1_wave,
        r2_wave,
        n_sims=N_SIMS,
        cut_top=CUT_TOP,
        seed=SEED,
    )

    out_parquet = preds_dir / f"event_{event_id}_preds_common_shock.parquet"
    out_csv = preds_dir / f"event_{event_id}_preds_common_shock.csv"
    preds.to_parquet(out_parquet, index=False)
    preds.to_csv(out_csv, index=False)
    print("Saved predictions:")
    print("-", out_parquet)
    print("-", out_csv)
    print(preds.sort_values("p_win", ascending=False).head(10))


if __name__ == "__main__":
    main()
