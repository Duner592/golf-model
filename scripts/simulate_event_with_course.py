#!/usr/bin/env python3
# scripts/simulate_event_with_course.py
#
# Course-aware Monte Carlo simulator (hardened):
# - Skill (auto-detected)
# - Weather (neutral or wave-aware)
# - Course history (sg_course_mean_shrunk)
# - DIY course fit (course_fit_score)
# - Per-player sigma
# - Common round + wave shocks
# - Optional starting strokes (e.g., TOUR Championship)
# - No-cut handling; small-field shock/sigma adjustments; optional course beta softening
#
# Outputs:
#   data/preds/{tour}/event_{event_id}_preds_with_course.(parquet|csv)
#
# CLI:
#   --event_id <id>   Force event_id (pinned/backtest)
#
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -----------------------
# Globals / Defaults
# -----------------------
TOUR = "pga"

# Defaults (tweak in code, or externalize to YAML if desired)
N_SIMS = 30000
CUT_TOP = 65
SEED = 42

# Course terms (can be softened per-event)
BETA_COURSE_HIST = -0.30
BETA_COURSE_FIT = -0.10

# Shocks
ROUND_SHOCK_SD = 0.20
WAVE_SHOCK_SD = 0.12

# Sigma
SIGMA_DEFAULT = 2.8
SIGMA_MIN = 1.8
SIGMA_MAX = 3.8

# Signature events to soften course effects a touch (ids will vary by your schedule mapping)
SOFTEN_COURSE_FOR = {"7", "9", "10", "12", "34"}


# -----------------------
# Utilities
# -----------------------
def resolve_event_id(cli_event_id: str | None) -> str:
    if cli_event_id:
        return str(cli_event_id)
    processed = Path("data/processed") / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("Missing meta; run parse_field_updates.py first.")
    return str(json.loads(metas[-1].read_text(encoding="utf-8"))["event_id"])


def load_features_table(event_id: str) -> pd.DataFrame:
    feats_dir = Path("data/features") / TOUR
    p_course = feats_dir / f"event_{event_id}_features_course.parquet"
    p_full = feats_dir / f"event_{event_id}_features_full.parquet"

    if p_course.exists():
        df = pd.read_parquet(p_course)
    elif p_full.exists():
        df = pd.read_parquet(p_full)
    else:
        raise FileNotFoundError("Missing features parquet; run merge_player_data_into_features.py and course-feature merges first.")

    # IDs / Names
    if "dg_id" not in df.columns and "player_id" in df.columns:
        df = df.rename(columns={"player_id": "dg_id"})
    if "dg_id" not in df.columns:
        raise ValueError("Features must contain 'dg_id' (or 'player_id' to be renamed).")
    if "player_name" not in df.columns:
        df["player_name"] = df["dg_id"].astype(str)
    return df


def pick_skill_column(df: pd.DataFrame) -> str | None:
    for c in (
        "sr_sg_total",
        "sr_dg_rating",
        "dr_dg_rating",
        "dr_sg_total",
        "dg_rating",
        "sg_total",
    ):
        if c in df.columns:
            return c
    return None


def get_weather_matrix(df: pd.DataFrame) -> np.ndarray:
    wave_cols = [f"weather_r{r}_delta_wave" for r in (1, 2, 3, 4)]
    neu_cols = [f"weather_r{r}_delta_neutral" for r in (1, 2, 3, 4)]
    if all(c in df.columns for c in wave_cols):
        return df[wave_cols].astype(float).fillna(0.0).to_numpy()
    if all(c in df.columns for c in neu_cols):
        return df[neu_cols].astype(float).fillna(0.0).to_numpy()
    return np.zeros((len(df), 4), dtype=float)


def get_sigma_vector(df: pd.DataFrame, sigma_default: float) -> np.ndarray:
    if "sigma" in df.columns:
        s = df["sigma"].astype(float).fillna(sigma_default).to_numpy()
        return np.clip(s, SIGMA_MIN, SIGMA_MAX)
    return np.full(len(df), sigma_default, dtype=float)


def get_waves(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    r1 = df["r1_wave"].fillna("NA").to_numpy() if "r1_wave" in df.columns else np.array(["NA"] * len(df))
    r2 = df["r2_wave"].fillna("NA").to_numpy() if "r2_wave" in df.columns else np.array(["NA"] * len(df))
    return r1, r2


def build_mu_base(
    df: pd.DataFrame,
    skill_col: str | None,
    betas: dict,
) -> np.ndarray:
    mu = np.zeros(len(df), dtype=float)

    # Skill
    if skill_col and skill_col in df.columns:
        skill = df[skill_col].astype(float).fillna(0.0).to_numpy()
        mu = -skill  # lower is better

    # Course history
    if "sg_course_mean_shrunk" in df.columns:
        ch = df["sg_course_mean_shrunk"].astype(float).fillna(0.0).to_numpy()
        mu = mu + betas["hist"] * ch

    # Course fit
    if "course_fit_score" in df.columns:
        cf = df["course_fit_score"].astype(float).fillna(0.0).to_numpy()
        mu = mu + betas["fit"] * cf

    return mu


def load_event_rules(root: Path) -> dict:
    """Optional per-event rules from configs/event_rules.yaml; fallback dict otherwise."""
    p = root / "configs" / "event_rules.yaml"
    if p.exists():
        try:
            rules = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            # ensure keys are strings
            return {str(k): (v or {}) for k, v in rules.items()}
        except Exception:
            pass
    return {}


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
    n_sims: int,
    cut_top: int | None,
    seed: int,
    round_sd: float,
    wave_sd: float,
    use_starting_strokes: bool = False,
    starting_strokes: np.ndarray | None = None,
    no_cut: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n = len(ids)
    wins = np.zeros(n, dtype=np.int64)
    top10 = np.zeros(n, dtype=np.int64)
    made_cut = np.zeros(n, dtype=np.int64)
    scores = np.empty((n, 4), dtype=float)

    if starting_strokes is None:
        starting_strokes = np.zeros(n, dtype=float)

    for _ in range(int(n_sims)):
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

        # Cut or no-cut
        if no_cut or cut_top is None:
            alive = np.ones(n, dtype=bool)
        else:
            totals_r2 = scores[:, :2].sum(axis=1)
            cut_idx = min(max(int(cut_top) - 1, 0), n - 1)
            cut_line = np.partition(totals_r2, cut_idx)[cut_idx]
            alive = totals_r2 <= cut_line
        made_cut += alive.astype(np.int64)

        # Final totals (apply starting strokes if enabled)
        totals = scores.sum(axis=1)
        if use_starting_strokes:
            totals = totals + starting_strokes

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
    ap = argparse.ArgumentParser(description="Course-aware tournament simulator")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned/event to simulate")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent

    # Load event rules
    rules_map = load_event_rules(root)

    # Resolve event_id and load features
    event_id = resolve_event_id(args.event_id)
    df = load_features_table(event_id)

    # Detect per-event rules
    rules = rules_map.get(str(event_id), {})
    use_starting_strokes = bool(rules.get("use_starting_strokes", False))
    no_cut = bool(rules.get("no_cut", False))
    soften_course = bool(rules.get("soften_course", False))

    # Build inputs
    ids = df["dg_id"].astype(str).to_numpy()
    names = df["player_name"].astype(str).to_numpy()
    skill_col = pick_skill_column(df)

    # Soften course effects for selected events if requested
    if soften_course or str(event_id) in SOFTEN_COURSE_FOR:
        betas = {"hist": -0.25, "fit": -0.08}
    else:
        betas = {"hist": BETA_COURSE_HIST, "fit": BETA_COURSE_FIT}
    mu_base = build_mu_base(df, skill_col, betas)

    weather = get_weather_matrix(df)

    # Sigma (with optional small-field tweak)
    field_size = len(df)
    adj_round_sd = ROUND_SHOCK_SD
    adj_sigma_default = SIGMA_DEFAULT
    if field_size <= 80:
        adj_round_sd = max(ROUND_SHOCK_SD, 0.26)
        # Only bump sigma if many players lack individual sigma
        if "sigma" not in df.columns or df["sigma"].isna().mean() > 0.25:
            adj_sigma_default = max(SIGMA_DEFAULT, 3.0)
    sigma_vec = get_sigma_vector(df, adj_sigma_default)

    r1_wave, r2_wave = get_waves(df)

    # Starting strokes (e.g., TOUR Championship)
    start_strokes = np.zeros(field_size, dtype=float)
    if use_starting_strokes:
        if "starting_strokes" in df.columns:
            start_strokes = df["starting_strokes"].astype(float).fillna(0.0).to_numpy()
        else:
            # Optional: overlay CSV if you generated one for TOUR Championship
            overlay = Path("data/processed") / TOUR / f"event_{event_id}_starting_strokes.csv"
            if overlay.exists():
                ss = pd.read_csv(overlay)
                key = "dg_id" if "dg_id" in df.columns and "dg_id" in ss.columns else "player_name"
                merged = df[[key]].merge(ss[[key, "starting_strokes"]], on=key, how="left")
                start_strokes = merged["starting_strokes"].fillna(0.0).astype(float).to_numpy()
            else:
                print("[warn] use_starting_strokes enabled, but 'starting_strokes' column not found. Defaulting to 0s.")

    # Cut rules
    cut_top = None if no_cut else int(CUT_TOP)

    # Simulate
    preds = simulate(
        ids=ids,
        names=names,
        mu_base=mu_base,
        sigma=sigma_vec,
        weather=weather,
        r1_wave=r1_wave,
        r2_wave=r2_wave,
        n_sims=int(N_SIMS),
        cut_top=cut_top,
        seed=int(SEED),
        round_sd=float(adj_round_sd),
        wave_sd=float(WAVE_SHOCK_SD),
        use_starting_strokes=use_starting_strokes,
        starting_strokes=start_strokes,
        no_cut=no_cut,
    )

    # Save
    preds_dir = root / "data" / "preds" / TOUR
    preds_dir.mkdir(parents=True, exist_ok=True)
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
