#!/usr/bin/env python3
# scripts/build_course_fit_from_history.py
#
# DIY course-fit using historical rounds + player skills, now INCLUDING driving accuracy and driving distance.
#
# What it does:
# 1) Load the combined historical rounds parquet for this tournament (wide by round).
# 2) Reshape to long: one row per (player, round) with sg_total, driving_acc, driving_dist.
# 3) Load current event skill_ratings and map category skills (4-cat OTT/APP/ARG/PUTT or 2-cat T2G+PUTT).
# 4) Fit a Ridge regression: sg_total ~ [category skills] + driving_acc + driving_dist (standardized).
#    - Coefficients are converted to positive weights that sum to 1 across all predictors used.
# 5) Compute per-player course_fit_score for the current field:
#    - For categories: use their skill_ratings (as-is).
#    - For driving_acc/dist: use their venue means from history if available; otherwise overall means; if missing, field mean.
#    - Standardize driving metrics using the same (venue) mean/std used in the model, then apply driving weights.
#
# Outputs:
#   - data/processed/{tour}/event_{event_id}_course_fit_weights.json
#       { "weights": {pred: weight, ...}, "cats_used": [...], "driving_norm": {"da_field_mean", "da_field_std", "dd_field_mean", "dd_field_std"} }
#   - data/processed/{tour}/event_{event_id}_course_fit_diy.parquet
#       [dg_id/player_id, player_name?, cats..., course_fit_score, da_inputs, dd_inputs]
#
# Requirements:
#   - scikit-learn (pip install scikit-learn)
#
# Notes:
# - The historical rounds parquet must have wide columns like round_1.sg_total, round_1.driving_acc, round_1.driving_dist, etc.
# - The skill_ratings parquet must include either OTT/APP/ARG/PUTT or T2G + PUTT (prefixed names are auto-mapped).
# - This is a first-pass heuristic; tune and calibrate with backtesting.

from __future__ import annotations

# stdlib/third-party
from pathlib import Path
import json
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ensure src import works when running directly
import _bootstrap  # noqa: F401


TOUR = "pga"

CATS_4 = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
CATS_2 = ["sg_t2g", "sg_putt"]

# Allowed aliases in skill_ratings for category columns
ALIASES_SKILL = {
    "sg_ott": ["sg_ott", "sr_sg_ott", "ott", "sg_off_tee"],
    "sg_app": ["sg_app", "sr_sg_app", "app", "sg_approach"],
    "sg_arg": ["sg_arg", "sr_sg_arg", "arg", "sg_around_green"],
    "sg_putt": ["sg_putt", "sr_sg_putt", "putt", "sg_putting"],
    "sg_t2g": ["sg_t2g", "sr_sg_t2g", "t2g", "sg_tee_to_green"],
}


def normalize_name(s: str) -> str:
    s0 = (s or "").lower()
    s0 = re.sub(r"[^a-z0-9]+", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0.replace(" ", "_")


def load_event_meta() -> dict:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(
            "No event meta found. Run parse_field_updates.py first."
        )
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def find_hist_parquet(event_name: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    safe = normalize_name(event_name)
    return (
        root
        / "data"
        / "raw"
        / "historical"
        / TOUR
        / f"tournament_{safe}_rounds_combined.parquet"
    )


def wide_rounds_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    From columns like round_1.sg_total, round_1.driving_acc, round_1.driving_dist, ...
    to long rows with columns: [player_id, year, round, sg_total, driving_acc, driving_dist]
    """
    id_col = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(
            "No player id column found in historical rounds (expected dg_id/player_id)."
        )

    pat_sg = re.compile(r"^round_(\d+)\.sg_total$")
    pat_acc = re.compile(r"^round_(\d+)\.driving_acc$")
    pat_dist = re.compile(r"^round_(\d+)\.driving_dist$")

    sg_cols, acc_cols, dist_cols = {}, {}, {}
    for c in df.columns:
        m = pat_sg.match(c)
        if m:
            sg_cols[int(m.group(1))] = c
        m = pat_acc.match(c)
        if m:
            acc_cols[int(m.group(1))] = c
        m = pat_dist.match(c)
        if m:
            dist_cols[int(m.group(1))] = c

    if not sg_cols:
        raise ValueError(
            "No round_N.sg_total columns found in historical rounds parquet."
        )

    rounds = set(sg_cols.keys()) | set(acc_cols.keys()) | set(dist_cols.keys())
    recs = []
    for _, row in df.iterrows():
        pid = row[id_col]
        year = row.get("year", None)
        for r in rounds:
            recs.append(
                {
                    "player_id": pid,
                    "year": year,
                    "round": r,
                    "sg_total": row.get(sg_cols.get(r), np.nan),
                    "driving_acc": row.get(acc_cols.get(r), np.nan),
                    "driving_dist": row.get(dist_cols.get(r), np.nan),
                }
            )
    long_df = pd.DataFrame.from_records(recs)
    # keep rows with sg_total present
    long_df = long_df.dropna(subset=["sg_total"])
    return long_df


def map_skill_columns(
    skills: pd.DataFrame, prefer_four=True
) -> tuple[pd.DataFrame, str, list]:
    """
    Map skill_ratings columns to canonical category columns.
    Returns:
      df_small: id + cats + optional player_name
      id_col: join key
      cats: list of cats used (CATS_4 or CATS_2)
    """
    id_col = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in skills.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError("No id column in skill_ratings (expected dg_id/player_id).")

    def find_first(df, keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    # Try 4-cat
    cats_found = {}
    if prefer_four:
        ok = True
        for c in CATS_4:
            col = find_first(skills, ALIASES_SKILL[c])
            if col:
                cats_found[c] = col
            else:
                ok = False
                break
        if ok:
            keep = [id_col] + list(cats_found.values())
            if "player_name" in skills.columns:
                keep.append("player_name")
            df_small = skills[keep].copy()
            df_small = df_small.rename(columns=cats_found)
            return df_small, id_col, CATS_4

    # Fallback 2-cat
    t2g = find_first(skills, ALIASES_SKILL["sg_t2g"])
    putt = find_first(skills, ALIASES_SKILL["sg_putt"])
    if not (t2g and putt):
        raise ValueError(
            "Could not find skill columns for 4-cat (OTT/APP/ARG/PUTT) or 2-cat (T2G+PUTT)."
        )
    keep = [id_col, t2g, putt]
    if "player_name" in skills.columns:
        keep.append("player_name")
    df_small = skills[keep].copy().rename(columns={t2g: "sg_t2g", putt: "sg_putt"})
    return df_small, id_col, CATS_2


def fit_course_weights(
    df_long: pd.DataFrame, skills_small: pd.DataFrame, id_col: str, cats: list
) -> tuple[pd.Series, dict]:
    """
    Join historical long df (sg_total per round) to player skills, then fit Ridge regression:
      sg_total ~ [cats] + driving_acc + driving_dist
    Returns:
      weights: normalized positive weights for predictors used
      driving_norm: dict with venue field mean/std for driving metrics (for later scoring standardization)
    """
    # Prepare driving standardization on venue (historical) data
    da_mu = (
        float(df_long["driving_acc"].mean())
        if "driving_acc" in df_long.columns
        else None
    )
    da_sd = (
        float(df_long["driving_acc"].std())
        if "driving_acc" in df_long.columns
        else None
    )
    dd_mu = (
        float(df_long["driving_dist"].mean())
        if "driving_dist" in df_long.columns
        else None
    )
    dd_sd = (
        float(df_long["driving_dist"].std())
        if "driving_dist" in df_long.columns
        else None
    )

    # Join with skills
    skills = skills_small.rename(columns={id_col: "player_id"})
    dfm = df_long.merge(skills, on="player_id", how="inner")

    # Build predictors X
    preds = []
    for c in cats:
        if c in dfm.columns:
            preds.append(c)
        else:
            raise ValueError(f"Missing skill column after join: {c}")

    # Add driving predictors (standardized with venue stats)
    def z(x, mu, sd):
        if mu is None or sd is None or sd == 0 or pd.isna(sd):
            return np.zeros_like(x, dtype=float)
        return (x - mu) / sd

    if "driving_acc" in dfm.columns and dfm["driving_acc"].notna().any():
        dfm["_da_z"] = z(dfm["driving_acc"].astype(float), da_mu, da_sd)
        preds.append("_da_z")
    if "driving_dist" in dfm.columns and dfm["driving_dist"].notna().any():
        dfm["_dd_z"] = z(dfm["driving_dist"].astype(float), dd_mu, dd_sd)
        preds.append("_dd_z")

    if not preds:
        raise ValueError("No predictors available to fit course weights.")

    X = dfm[preds].astype(float).to_numpy()
    y = dfm["sg_total"].astype(float).to_numpy()

    # Guard: minimum sample requirement
    if len(dfm) < max(80, 10 * len(preds)):
        # fallback to equal weights
        w = np.ones(len(preds)) / len(preds)
        weights = pd.Series(w, index=preds)
    else:
        # Standardize + Ridge (light regularization)
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            Ridge(alpha=0.5, fit_intercept=True, random_state=42),
        )
        model.fit(X, y)
        ridge = model.named_steps["ridge"]
        coefs = np.abs(ridge.coef_)
        if coefs.sum() == 0:
            w = np.ones(len(preds)) / len(preds)
        else:
            w = coefs / coefs.sum()
        weights = pd.Series(w, index=preds).round(6)

    driving_norm = {
        "da_field_mean": da_mu,
        "da_field_std": da_sd,
        "dd_field_mean": dd_mu,
        "dd_field_std": dd_sd,
    }
    return weights, driving_norm


def compute_player_driving_inputs_for_scoring(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player venue means for driving_acc and driving_dist (and overall as fallback).
    Returns a DataFrame with columns:
      player_id, da_venue_mean, dd_venue_mean, da_overall_mean, dd_overall_mean
    """
    # Venue-level (same as df_long since df_long is venue history only)
    venue = df_long.groupby("player_id", as_index=False).agg(
        da_venue_mean=("driving_acc", "mean"),
        dd_venue_mean=("driving_dist", "mean"),
    )

    # Overall (identical to venue here since df_long is venue-only; kept for API symmetry)
    overall = df_long.groupby("player_id", as_index=False).agg(
        da_overall_mean=("driving_acc", "mean"),
        dd_overall_mean=("driving_dist", "mean"),
    )

    drv = venue.merge(overall, on="player_id", how="outer")
    return drv


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR

    # Meta / event
    meta = load_event_meta()
    event_id = str(meta["event_id"])
    event_name = meta.get("event_name") or "current_event"

    # 1) Historical rounds combined parquet (venue history)
    hist_path = find_hist_parquet(event_name)
    if not hist_path.exists():
        raise FileNotFoundError(
            f"Historical rounds combined parquet not found:\n  {hist_path}\n"
            "Run scripts/fetch_historical_rounds.py first (ensure name matching)."
        )
    df_hist = pd.read_parquet(hist_path)
    df_long = wide_rounds_to_long(df_hist)

    # 2) Load skill_ratings for current event players
    sr_path = processed / f"event_{event_id}_skill_ratings.parquet"
    if not sr_path.exists():
        raise FileNotFoundError(
            "Missing skill_ratings parquet. Run fetch_player_data.py first."
        )
    skills = pd.read_parquet(sr_path)

    skills_small, id_col, cats = map_skill_columns(skills, prefer_four=True)

    # 3) Fit weights via Ridge regression on (cats + driving_z)
    weights, driving_norm = fit_course_weights(
        df_long, skills_small, id_col=id_col, cats=cats
    )
    print("Raw predictor weights:", weights.to_dict())

    # 4) Compute per-player driving inputs (means) for scoring
    drv_inputs = compute_player_driving_inputs_for_scoring(
        df_long
    )  # player_id + da_venue_mean/dd_venue_mean/overall

    # 5) Prepare scoring table for current event players (align IDs)
    score_df = skills_small.copy()
    score_df = score_df.rename(columns={id_col: "player_id"})
    score_df = score_df.merge(drv_inputs, on="player_id", how="left")

    # Fill driving means: prefer venue_mean, fallback overall_mean, fallback field mean
    da_field_mean = driving_norm["da_field_mean"]
    dd_field_mean = driving_norm["dd_field_mean"]

    # Compute field fallback if needed
    if "da_venue_mean" in score_df.columns and score_df["da_venue_mean"].notna().any():
        da_fallback = float(score_df["da_venue_mean"].mean(skipna=True))
    else:
        da_fallback = da_field_mean if da_field_mean is not None else 0.0

    if "dd_venue_mean" in score_df.columns and score_df["dd_venue_mean"].notna().any():
        dd_fallback = float(score_df["dd_venue_mean"].mean(skipna=True))
    else:
        dd_fallback = dd_field_mean if dd_field_mean is not None else 0.0

    # Compose final driving inputs
    def choose_mean(row, which):
        v = row.get(f"{which}_venue_mean", np.nan)
        if pd.notna(v):
            return float(v)
        v = row.get(f"{which}_overall_mean", np.nan)
        if pd.notna(v):
            return float(v)
        return da_fallback if which == "da" else dd_fallback

    score_df["da_input"] = score_df.apply(lambda r: choose_mean(r, "da"), axis=1)
    score_df["dd_input"] = score_df.apply(lambda r: choose_mean(r, "dd"), axis=1)

    # Standardize driving inputs with venue field stats used in model
    def z(val, mu, sd):
        if mu is None or sd is None or sd == 0 or pd.isna(sd):
            return 0.0
        return float((val - mu) / sd)

    score_df["da_z"] = score_df["da_input"].apply(
        lambda x: z(x, driving_norm["da_field_mean"], driving_norm["da_field_std"])
    )
    score_df["dd_z"] = score_df["dd_input"].apply(
        lambda x: z(x, driving_norm["dd_field_mean"], driving_norm["dd_field_std"])
    )

    # 6) Convert predictor weights to category + driving dictionary for readability
    # weights index contains cats and possibly _da_z/_dd_z
    weight_dict = {}
    for c in cats:
        if c in weights.index:
            weight_dict[c] = float(weights[c])
        else:
            weight_dict[c] = 0.0
    weight_dict["da_z"] = float(weights.get("_da_z", 0.0))
    weight_dict["dd_z"] = float(weights.get("_dd_z", 0.0))

    # 7) Compute per-player course_fit_score
    score_df["course_fit_score"] = 0.0
    for c in cats:
        if c in score_df.columns:
            score_df["course_fit_score"] += score_df[c].astype(float) * weight_dict[c]
    score_df["course_fit_score"] += score_df["da_z"].astype(float) * weight_dict["da_z"]
    score_df["course_fit_score"] += score_df["dd_z"].astype(float) * weight_dict["dd_z"]

    # 8) Save outputs
    processed.mkdir(parents=True, exist_ok=True)
    weights_payload = {
        "weights": weight_dict,
        "cats_used": cats,
        "driving_norm": driving_norm,
    }
    (processed / f"event_{event_id}_course_fit_weights.json").write_text(
        json.dumps(weights_payload, indent=2), encoding="utf-8"
    )

    # Restore original id column name for downstream merge
    out_df = score_df.rename(columns={"player_id": id_col})
    keep_cols = [
        id_col,
        "course_fit_score",
        "da_input",
        "dd_input",
        "da_z",
        "dd_z",
    ] + cats
    if "player_name" in out_df.columns:
        keep_cols.insert(1, "player_name")
    out_df = out_df[[c for c in keep_cols if c in out_df.columns]].drop_duplicates(
        subset=[id_col]
    )
    out_df.to_parquet(
        processed / f"event_{event_id}_course_fit_diy.parquet", index=False
    )

    print("Saved weights:", processed / f"event_{event_id}_course_fit_weights.json")
    print(
        "Saved per-player DIY course_fit (with driving):",
        processed / f"event_{event_id}_course_fit_diy.parquet",
    )


if __name__ == "__main__":
    main()
