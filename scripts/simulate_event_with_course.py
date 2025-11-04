#!/usr/bin/env python3
# scripts/simulate_event_with_course.py
#
# Course-aware Monte Carlo simulator (config-driven):
# - Reads configs/simulator.yaml for all tuning constants
# - Uses skill (auto-detected), weather (neutral or wave-aware), sigma, DIY course fit, and venue history
# - Adds common round and wave shocks
#
# Outputs:
#   data/preds/{tour}/event_{event_id}_preds_with_course.(parquet|csv)
#
# CLI overrides (optional):
#   --event_id 457         Force event_id
#   --n_sims 50000         Override sim count (else use config)
#   --seed 123             Override RNG seed
#
#!/usr/bin/env python3
# scripts/simulate_event_with_course.py

from __future__ import annotations

# stdlib/third-party
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yaml

# ensure src import works when running directly
import _bootstrap  # noqa: F401

from src.utils_event import (
    resolve_event_id,
)

TOUR = "pga"


# -----------------------
# Config loading
# -----------------------

DEFAULTS = {
    "seed": 42,
    "n_sims": 30000,
    "cut_top": 65,
    "weights": {"beta_course_hist": -0.30, "beta_course_fit": -0.10},
    "shocks": {"round_sd": 0.20, "wave_sd": 0.12},
    "sigma": {"default": 2.8, "min": 1.8, "max": 3.8},
}


def load_sim_config(root: Path) -> dict:
    p = root / "configs" / "simulator.yaml"
    cfg = {}
    if p.exists():
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    # merge with defaults (shallow for top-level and known sections)
    merged = DEFAULTS.copy()
    for k in ["seed", "n_sims", "cut_top"]:
        if k in cfg:
            merged[k] = cfg[k]
    for sect in ["weights", "shocks", "sigma"]:
        merged[sect] = {**DEFAULTS[sect], **(cfg.get(sect, {}) or {})}
    # clamp sigma sanity
    s = merged["sigma"]
    s["min"] = float(s.get("min", DEFAULTS["sigma"]["min"]))
    s["max"] = float(max(s["min"], s.get("max", DEFAULTS["sigma"]["max"])))
    s["default"] = float(
        np.clip(
            float(s.get("default", DEFAULTS["sigma"]["default"])), s["min"], s["max"]
        )
    )
    # shocks sanity
    sh = merged["shocks"]
    sh["round_sd"] = float(max(0.0, sh.get("round_sd", DEFAULTS["shocks"]["round_sd"])))
    sh["wave_sd"] = float(max(0.0, sh.get("wave_sd", DEFAULTS["shocks"]["wave_sd"])))
    return merged


# -----------------------
# Feature loading
# -----------------------


def load_features_table(tour: str, event_id: str) -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    feats_dir = root / "data" / "features" / tour
    p_course = feats_dir / f"event_{event_id}_features_course.parquet"
    p_full = feats_dir / f"event_{event_id}_features_full.parquet"

    if p_course.exists():
        df = pd.read_parquet(p_course)
    elif p_full.exists():
        df = pd.read_parquet(p_full)
    else:
        raise FileNotFoundError(
            "Missing features parquet; run merge_player_data_into_features.py and course-feature merges first."
        )
    # IDs / Names
    if "dg_id" not in df.columns and "player_id" in df.columns:
        df = df.rename(columns={"player_id": "dg_id"})
    if "dg_id" not in df.columns:
        raise ValueError(
            "Features must contain 'dg_id' (or 'player_id' to be renamed)."
        )
    if "player_name" not in df.columns:
        df["player_name"] = df["dg_id"].astype(str)
    return df


def pick_skill_column(df: pd.DataFrame) -> str | None:
    for c in [
        "sr_sg_total",
        "sr_dg_rating",
        "dr_dg_rating",
        "dr_sg_total",
        "dg_rating",
        "sg_total",
    ]:
        if c in df.columns:
            return c
    return None


def get_weather_matrix(df: pd.DataFrame) -> np.ndarray:
    wave_cols = [f"weather_r{r}_delta_wave" for r in [1, 2, 3, 4]]
    neu_cols = [f"weather_r{r}_delta_neutral" for r in [1, 2, 3, 4]]
    if all(c in df.columns for c in wave_cols):
        return df[wave_cols].astype(float).fillna(0.0).to_numpy()
    if all(c in df.columns for c in neu_cols):
        return df[neu_cols].astype(float).fillna(0.0).to_numpy()
    return np.zeros((len(df), 4), dtype=float)


def get_sigma_vector(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    s_cfg = cfg["sigma"]
    if "sigma" in df.columns:
        s = df["sigma"].astype(float).fillna(s_cfg["default"]).to_numpy()
        return np.clip(s, s_cfg["min"], s_cfg["max"])
    return np.full(len(df), s_cfg["default"], dtype=float)


def get_waves(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    r1 = (
        df["r1_wave"].fillna("NA").to_numpy()
        if "r1_wave" in df.columns
        else np.array(["NA"] * len(df))
    )
    r2 = (
        df["r2_wave"].fillna("NA").to_numpy()
        if "r2_wave" in df.columns
        else np.array(["NA"] * len(df))
    )
    return r1, r2


def build_mu_base(df: pd.DataFrame, skill_col: str | None, cfg: dict) -> np.ndarray:
    w_cfg = cfg["weights"]
    if skill_col and skill_col in df.columns:
        skill = df[skill_col].astype(float).fillna(0.0).to_numpy()
        mu = -skill
    else:
        mu = np.zeros(len(df), dtype=float)

    if "sg_course_mean_shrunk" in df.columns:
        ch = df["sg_course_mean_shrunk"].astype(float).fillna(0.0).to_numpy()
        mu = mu + w_cfg["beta_course_hist"] * ch

    if "course_fit_score" in df.columns:
        cf = df["course_fit_score"].astype(float).fillna(0.0).to_numpy()
        mu = mu + w_cfg["beta_course_fit"] * cf

    return mu


# -----------------------
# Simulator
# -----------------------


def simulate(
    ids: np.ndarray,
    names: np.ndarray,
    mu_base: np.ndarray,
    sigma: np.ndarray,
    weather: np.ndarray,
    r1_wave: np.ndarray,
    r2_wave: np.ndarray,
    cfg: dict,
) -> pd.DataFrame:

    rng = np.random.default_rng(cfg["seed"])
    n = len(ids)
    wins = np.zeros(n, dtype=np.int64)
    top10 = np.zeros(n, dtype=np.int64)
    made_cut = np.zeros(n, dtype=np.int64)
    scores = np.empty((n, 4), dtype=float)

    round_sd = float(cfg["shocks"]["round_sd"])
    wave_sd = float(cfg["shocks"]["wave_sd"])
    n_sims = int(cfg["n_sims"])
    cut_top = int(cfg["cut_top"])

    for _ in range(n_sims):
        round_shock = rng.normal(0.0, round_sd, size=4)

        for r in range(4):
            mu_r = mu_base + weather[:, r] + round_shock[r]

            if r in (0, 1):
                shock_am = rng.normal(0.0, wave_sd)
                shock_pm = rng.normal(0.0, wave_sd)
                wave_adj = np.zeros(n, dtype=float)
                if r == 0:
                    wave_adj[r1_wave == "AM"] += shock_am
                    wave_adj[r1_wave == "PM"] += shock_pm
                else:
                    wave_adj[r2_wave == "AM"] += shock_am
                    wave_adj[r2_wave == "PM"] += shock_pm
                mu_r = mu_r + wave_adj

            scores[:, r] = rng.normal(loc=mu_r, scale=sigma)

        # 36-hole cut: top 65 and ties approximation
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


# -----------------------
# Main
# -----------------------


def main():
    ap = argparse.ArgumentParser(
        description="Simulate event with course features (config-driven)."
    )
    ap.add_argument("--event_id", default=None, help="Override event_id")
    ap.add_argument(
        "--n_sims", type=int, default=None, help="Override sim count (else use config)"
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="Override RNG seed (else use config)"
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    preds_dir = root / "data" / "preds" / TOUR
    preds_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_sim_config(root)
    if args.n_sims is not None:
        cfg["n_sims"] = int(args.n_sims)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    event_id = resolve_event_id(args.event_id, TOUR)
    df = load_features_table(TOUR, event_id)

    ids = df["dg_id"].astype(str).to_numpy()
    names = df["player_name"].astype(str).to_numpy()
    skill_col = pick_skill_column(df)
    mu_base = build_mu_base(df, skill_col, cfg)
    weather = get_weather_matrix(df)
    sigma = get_sigma_vector(df, cfg)
    r1_wave, r2_wave = get_waves(df)

    preds = simulate(ids, names, mu_base, sigma, weather, r1_wave, r2_wave, cfg)

    out_parquet = preds_dir / f"event_{event_id}_preds_with_course.parquet"
    out_csv = preds_dir / f"event_{event_id}_preds_with_course.csv"
    preds.to_parquet(out_parquet, index=False)
    preds.to_csv(out_csv, index=False)
    print("Saved predictions with course features:")
    print("-", out_parquet)
    print("-", out_csv)
    print(preds.sort_values("p_win", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
