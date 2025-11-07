#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOUR = "pga"


def load_meta() -> dict:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No event meta found. Run parse_field_updates.py first.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def load_features_full(event_id: str) -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    feats_dir = root / "data" / "features" / TOUR
    p = feats_dir / f"event_{event_id}_features_full.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing features_full: {p}. Run merge_player_data_into_features.py first.")
    df = pd.read_parquet(p)
    # Ensure an ID
    if "dg_id" not in df.columns and "player_id" in df.columns:
        df = df.rename(columns={"player_id": "dg_id"})
    if "dg_id" not in df.columns:
        raise ValueError("Features must contain 'dg_id' or 'player_id'.")
    # Ensure player_name exists for readable outputs
    if "player_name" not in df.columns:
        if "name" in df.columns:
            df = df.rename(columns={"name": "player_name"})
        else:
            df["player_name"] = df["dg_id"].astype(str)
    return df


def infer_skill_columns(df: pd.DataFrame) -> dict:
    """
    Pick columns for baseline skill and volatility if available.
    Preference order:
      - Skill (sr_*) from skill_ratings
      - Rankings (dr_*) as fallback
    """
    col_skill_candidates = [
        "sr_sg_total",
        "sr_rating",
        "sr_dg_rating",
        "dr_dg_rating",
        "dr_rating",
        "dr_sg_total",
    ]
    skill_col = next((c for c in col_skill_candidates if c in df.columns), None)

    # Per-player volatility: fallback to constant if not provided
    vol_candidates = ["sr_volatility", "dr_volatility", "volatility"]
    vol_col = next((c for c in vol_candidates if c in df.columns), None)

    return {"skill": skill_col, "vol": vol_col}


def build_mu_sigma(df: pd.DataFrame, cols: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - mu_base: baseline strokes per round (lower is better)
      - sigma: player round-to-round standard deviation
      - weather_mat: Nx4 weather delta matrix (wave-aware if present, else neutral)
    Assumptions:
      - skill is in "strokes gained per round"; we invert sign to model strokes to par.
      - If skill is a rating on different scale, you can rescale below.
    """
    n = len(df)
    # Baseline skill
    if cols["skill"] and cols["skill"] in df.columns:
        sg = df[cols["skill"]].astype(float).fillna(0.0).to_numpy()
        # Convert SG per round (positive better) to strokes to par adjustment (negative better)
        mu_base = -sg  # model mean round score offset; larger SG => lower scores
    else:
        mu_base = np.zeros(n, dtype=float)

    # Weather deltas per round (prefer wave-aware if present; fallback to neutral)
    weather_cols_wave = [f"weather_r{r}_delta_wave" for r in [1, 2, 3, 4]]
    weather_cols_neu = [f"weather_r{r}_delta_neutral" for r in [1, 2, 3, 4]]
    if all(c in df.columns for c in weather_cols_wave):
        weather_mat = df[weather_cols_wave].astype(float).fillna(0.0).to_numpy()
    elif all(c in df.columns for c in weather_cols_neu):
        weather_mat = df[weather_cols_neu].astype(float).fillna(0.0).to_numpy()
    else:
        weather_mat = np.zeros((n, 4), dtype=float)

    # Volatility
    if cols["vol"] and cols["vol"] in df.columns:
        sigma = df[cols["vol"]].astype(float).fillna(2.8).to_numpy()
        # Clip to reasonable range
        sigma = np.clip(sigma, 1.8, 3.8)
    else:
        sigma = np.full(n, 2.8, dtype=float)

    return mu_base, sigma, weather_mat


def simulate_event(df: pd.DataFrame, n_sims: int = 20000, seed: int = 42, cut_top: int = 65) -> pd.DataFrame:
    """
    Simple Monte Carlo:
      score_ir ~ Normal(mu_base_i + weather_r, sigma_i)
      After R2: Top 65 and ties advance (approx).
    """
    rng = np.random.default_rng(seed)
    players = df["player_name"].to_numpy()
    ids = df["dg_id"].astype(str).to_numpy()

    cols = infer_skill_columns(df)
    mu_base, sigma, weather = build_mu_sigma(df, cols)

    n = len(df)
    wins = np.zeros(n, dtype=np.int64)
    top10 = np.zeros(n, dtype=np.int64)
    made_cut = np.zeros(n, dtype=np.int64)

    # Pre-allocate per round
    scores = np.empty((n, 4), dtype=float)

    for _ in range(n_sims):
        # Simulate rounds
        for r in range(4):
            mu_r = mu_base + weather[:, r]
            scores[:, r] = rng.normal(loc=mu_r, scale=sigma)

        # Cut after R2 (top 65 and ties)
        totals_r2 = scores[:, :2].sum(axis=1)
        # compute cut line by order statistic
        cut_idx = min(max(cut_top - 1, 0), n - 1)
        cut_line = np.partition(totals_r2, cut_idx)[cut_idx]
        alive = totals_r2 <= cut_line
        made_cut += alive.astype(np.int64)

        # Apply "missed cut" penalty by setting large totals
        totals = scores.sum(axis=1)
        totals_masked = totals.copy()
        totals_masked[~alive] = np.inf

        order = np.argsort(totals_masked)
        wins[order[0]] += 1
        top10[order[: min(10, n)]] += 1

    out = pd.DataFrame(
        {
            "dg_id": ids,
            "player_name": players,
            "p_win": wins / n_sims,
            "p_top10": top10 / n_sims,
            "p_mc": made_cut / n_sims,
        }
    )
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    preds_dir = root / "data" / "preds" / TOUR
    preds_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    event_id = str(meta["event_id"])
    df = load_features_full(event_id)

    preds = simulate_event(df, n_sims=20000, seed=42, cut_top=65)

    out_parquet = preds_dir / f"event_{event_id}_preds_baseline.parquet"
    out_csv = preds_dir / f"event_{event_id}_preds_baseline.csv"
    preds.to_parquet(out_parquet, index=False)
    preds.to_csv(out_csv, index=False)
    print("Saved predictions:")
    print("-", out_parquet)
    print("-", out_csv)
    # Show top 10 by win probability
    print(preds.sort_values("p_win", ascending=False).head(10))


if __name__ == "__main__":
    main()
