#!/usr/bin/env python3
# scripts/build_course_fit_from_history.py
#
# DIY course-fit using historical rounds + player skills, including driving accuracy/distance.
# Robust to missing history: skips gracefully and writes empty/default weights payload.
#
# Outputs (under data/processed/{tour}/):
#   - event_{event_id}_course_fit_weights.json
#   - event_{event_id}_course_fit_diy.parquet
#
# Notes:
# - Historical combined parquet must be at:
#     data/raw/historical/{tour}/tournament_{normalized_event_name}_rounds_combined.parquet
#   where normalized_event_name is lowercased, non-alnum -> space, multiple spaces -> single underscore.
# - Player skills parquet:
#     data/processed/{tour}/event_{event_id}_skill_ratings.parquet
#   Must contain either 4-cat (sg_ott, sg_app, sg_arg, sg_putt) or 2-cat (sg_t2g, sg_putt) columns.
# - Driving features are learned from the historical rounds (driving_acc/driving_dist) and standardized
#   using historical venue mean/std. If history is missing, driving weights default to 0.
#
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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


# ------------------------- helpers ------------------------- #
def normalize_name(s: str) -> str:
    s0 = (s or "").lower()
    s0 = re.sub(r"[^a-z0-9]+", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0.replace(" ", "_")


def resolve_event_id_arg() -> str | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_id", type=str, default=None)
    args, _ = ap.parse_known_args()
    return args.event_id


def load_event_meta(override_event_id: str | None = None) -> dict:
    """
    If override_event_id provided, use the latest meta matching that event_id if present;
    else fall back to the latest meta file.
    """
    processed = Path("data") / "processed" / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(
            "No event meta found. Run parse_field_updates.py first."
        )
    if override_event_id:
        for p in reversed(metas):
            meta = json.loads(p.read_text(encoding="utf-8"))
            if str(meta.get("event_id")) == str(override_event_id):
                return meta
    # fallback: latest
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


def _choose_id_col(df: pd.DataFrame) -> str | None:
    for c in ["dg_id", "player_id", "id"]:
        if c in df.columns:
            return c
    return None


def wide_rounds_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract long rows with sg_total (and optional driving_acc/driving_dist) from 'round_N.*' wide columns.
    Output columns: ['player_id', 'year', 'round', 'sg_total', 'driving_acc', 'driving_dist']
    """
    id_col = _choose_id_col(df)
    if id_col is None:
        raise ValueError(
            "No player id column found in historical rounds (expected dg_id/player_id)."
        )

    # Patterns to detect columns
    pat_sg = re.compile(r"^round_(\d+)\.(sg_total|sg)$", re.IGNORECASE)
    pat_da = re.compile(r"^round_(\d+)\.driving_acc$", re.IGNORECASE)
    pat_dd = re.compile(r"^round_(\d+)\.driving_dist$", re.IGNORECASE)

    sg_cols, da_cols, dd_cols = {}, {}, {}
    for c in df.columns:
        m = pat_sg.match(c)
        if m:
            sg_cols[int(m.group(1))] = c
            continue
        m = pat_da.match(c)
        if m:
            da_cols[int(m.group(1))] = c
            continue
        m = pat_dd.match(c)
        if m:
            dd_cols[int(m.group(1))] = c

    if not sg_cols:
        # No usable sg_total in history
        return pd.DataFrame(
            columns=[
                "player_id",
                "year",
                "round",
                "sg_total",
                "driving_acc",
                "driving_dist",
            ]
        )

    rounds = sorted(
        set(list(sg_cols.keys()) + list(da_cols.keys()) + list(dd_cols.keys()))
    )
    records = []
    for _, row in df.iterrows():
        pid = row[id_col]
        year = row.get("year", None)
        for r in rounds:
            sg_val = row.get(sg_cols.get(r), np.nan) if r in sg_cols else np.nan
            if pd.isna(sg_val):
                continue
            rec = {
                "player_id": str(pid),
                "year": year,
                "round": int(r),
                "sg_total": float(sg_val),
                "driving_acc": (
                    float(row.get(da_cols.get(r), np.nan)) if r in da_cols else np.nan
                ),
                "driving_dist": (
                    float(row.get(dd_cols.get(r), np.nan)) if r in dd_cols else np.nan
                ),
            }
            records.append(rec)
    return pd.DataFrame.from_records(records)


def map_skill_columns(
    skills: pd.DataFrame, prefer_four=True
) -> tuple[pd.DataFrame, str, list]:
    """
    Map skill_ratings columns to canonical category columns (4-cat or 2-cat).
    Returns: (df_small, id_col, cats_used)
    """
    id_col = _choose_id_col(skills)
    if id_col is None:
        raise ValueError("No id column in skill_ratings (expected dg_id/player_id).")

    def find_first(df: pd.DataFrame, keys: list[str]) -> str | None:
        for k in keys:
            if k in df.columns:
                return k
        return None

    # Try 4-cat
    if prefer_four:
        cats_found = {}
        for c in CATS_4:
            col = find_first(skills, ALIASES_SKILL[c])
            if not col:
                cats_found = {}
                break
            cats_found[c] = col
        if cats_found:
            keep = [id_col] + list(cats_found.values())
            if "player_name" in skills.columns:
                keep.append("player_name")
            df_small = skills[keep].copy().rename(columns=cats_found)
            return df_small, id_col, CATS_4

    # Fallback 2-cat
    t2g = find_first(skills, ALIASES_SKILL["sg_t2g"])
    putt = find_first(skills, ALIASES_SKILL["sg_putt"])
    if not (t2g and putt):
        raise ValueError(
            "Could not find skill columns for either 4-cat (OTT/APP/ARG/PUTT) or 2-cat (T2G+PUTT)."
        )
    keep = [id_col, t2g, putt]
    if "player_name" in skills.columns:
        keep.append("player_name")
    df_small = skills[keep].copy().rename(columns={t2g: "sg_t2g", putt: "sg_putt"})
    return df_small, id_col, CATS_2


def fit_course_weights(
    df_long: pd.DataFrame, skills_small: pd.DataFrame, id_col: str, cats: list
) -> tuple[pd.Series | None, dict]:
    """
    Join historical long df (sg_total per round) to player skills, then fit Ridge regression:
      sg_total ~ [cats] + driving_acc + driving_dist (driving standardized using venue mean/std).
    Returns: (weights: pd.Series or None, driving_norm: dict)
    """
    if df_long.empty:
        # No history → return no weights
        driving_norm = {
            "da_field_mean": None,
            "da_field_std": None,
            "dd_field_mean": None,
            "dd_field_std": None,
        }
        return None, driving_norm

    # Driving standardization stats (venue field)
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
    driving_norm = {
        "da_field_mean": da_mu,
        "da_field_std": da_sd,
        "dd_field_mean": dd_mu,
        "dd_field_std": dd_sd,
    }

    # Join
    skills = skills_small.rename(columns={id_col: "player_id"})
    skills["player_id"] = skills["player_id"].astype(str)
    dfm = df_long.merge(skills, on="player_id", how="inner")
    if dfm.empty:
        return None, driving_norm

    # Predictors
    preds = []
    for c in cats:
        if c not in dfm.columns:
            raise ValueError(f"Missing skill column after join: {c}")
        preds.append(c)

    # Standardize driving and append
    def z(x: pd.Series, mu, sd):
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
        # No usable predictors
        return None, driving_norm

    X = dfm[preds].astype(float).to_numpy()
    y = dfm["sg_total"].astype(float).to_numpy()

    # If too few samples, fallback to equal weights
    if len(dfm) < max(80, 10 * len(preds)):
        w = np.ones(len(preds)) / len(preds)
        weights = pd.Series(w, index=preds)
        return weights.round(6), driving_norm

    # Ridge (light regularization)
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
    return weights, driving_norm


def compute_player_driving_inputs_for_scoring(
    df_long: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    """
    Compute per-player driving means for venue (historical df_long) to use at scoring time.
    Returns (drv_df: [key, da_venue_mean, dd_venue_mean, da_overall_mean, dd_overall_mean], join_key_name).
    """
    key = None
    for cand in ["player_id", "dg_id", "player_name"]:
        if cand in df_long.columns:
            key = cand
            break
    if key is None:
        key = "player_id"

    dfl = df_long.copy()
    if key not in dfl.columns:
        # create key from available id
        dfl[key] = df_long.get("player_id", "").astype(str)

    venue = dfl.groupby(key, as_index=False).agg(
        da_venue_mean=(
            ("driving_acc", "mean")
            if "driving_acc" in dfl.columns
            else ("round", "size")
        ),
        dd_venue_mean=(
            ("driving_dist", "mean")
            if "driving_dist" in dfl.columns
            else ("round", "size")
        ),
    )
    if "driving_acc" not in dfl.columns:
        venue["da_venue_mean"] = pd.NA
    if "driving_dist" not in dfl.columns:
        venue["dd_venue_mean"] = pd.NA

    overall = dfl.groupby(key, as_index=False).agg(
        da_overall_mean=(
            ("driving_acc", "mean")
            if "driving_acc" in dfl.columns
            else ("round", "size")
        ),
        dd_overall_mean=(
            ("driving_dist", "mean")
            if "driving_dist" in dfl.columns
            else ("round", "size")
        ),
    )
    if "driving_acc" not in dfl.columns:
        overall["da_overall_mean"] = pd.NA
    if "driving_dist" not in dfl.columns:
        overall["dd_overall_mean"] = pd.NA

    drv = venue.merge(overall, on=key, how="outer")
    return drv, key


# ------------------------- main ------------------------- #
def main():
    # Resolve event
    eid_override = resolve_event_id_arg()
    meta = load_event_meta(eid_override)
    event_id = str(meta["event_id"])
    event_name = meta.get("event_name") or "current_event"

    processed = Path("data") / "processed" / TOUR
    processed.mkdir(parents=True, exist_ok=True)

    # Load historical combined parquet (venue history)
    hist_path = find_hist_parquet(event_name)
    weights = None
    driving_norm = {
        "da_field_mean": None,
        "da_field_std": None,
        "dd_field_mean": None,
        "dd_field_std": None,
    }

    try:
        if not hist_path.exists():
            raise FileNotFoundError(
                f"Historical rounds combined parquet not found:\n  {hist_path}\n"
                "Run scripts/fetch_historical_rounds.py (ensure name matching)."
            )
        df_hist = pd.read_parquet(hist_path)
        df_long = wide_rounds_to_long(df_hist)

        # If no sg_total rows, skip gracefully
        if df_long.empty:
            print(
                "[warn] No long rows created from historical rounds parquet. Skipping course-fit."
            )
        else:
            # Load skills for current event players
            sr_path = processed / f"event_{event_id}_skill_ratings.parquet"
            if not sr_path.exists():
                raise FileNotFoundError(
                    "Missing skill_ratings parquet. Run fetch_player_data.py first."
                )
            skills = pd.read_parquet(sr_path)
            skills_small, id_col, cats = map_skill_columns(skills, prefer_four=True)

            # Fit weights (may return None)
            weights, driving_norm = fit_course_weights(
                df_long, skills_small, id_col=id_col, cats=cats
            )

            # Prepare scoring table and compute per-player driving inputs
            drv_inputs, drv_key = compute_player_driving_inputs_for_scoring(df_long)

            score_df = skills_small.copy().rename(columns={id_col: "player_id"})
            drv_aligned = drv_inputs.copy()
            if drv_key != "player_id":
                drv_aligned = drv_aligned.rename(columns={drv_key: "player_id"})
            score_df["player_id"] = score_df["player_id"].astype(str)
            drv_aligned["player_id"] = drv_aligned["player_id"].astype(str)
            score_df = score_df.merge(drv_aligned, on="player_id", how="left")

            # Fallback driving means
            da_field_mean = driving_norm.get("da_field_mean")
            dd_field_mean = driving_norm.get("dd_field_mean")

            if (
                "da_venue_mean" in score_df.columns
                and score_df["da_venue_mean"].notna().any()
            ):
                da_fallback = float(score_df["da_venue_mean"].mean(skipna=True))
            else:
                da_fallback = da_field_mean if da_field_mean is not None else 0.0

            if (
                "dd_venue_mean" in score_df.columns
                and score_df["dd_venue_mean"].notna().any()
            ):
                dd_fallback = float(score_df["dd_venue_mean"].mean(skipna=True))
            else:
                dd_fallback = dd_field_mean if dd_field_mean is not None else 0.0

            def choose_mean(row, which):
                v = row.get(f"{which}_venue_mean", np.nan)
                if pd.notna(v):
                    return float(v)
                v = row.get(f"{which}_overall_mean", np.nan)
                if pd.notna(v):
                    return float(v)
                return da_fallback if which == "da" else dd_fallback

            score_df["da_input"] = score_df.apply(
                lambda r: choose_mean(r, "da"), axis=1
            )
            score_df["dd_input"] = score_df.apply(
                lambda r: choose_mean(r, "dd"), axis=1
            )

            # Standardize with venue stats used in model
            def z(val, mu, sd):
                if mu is None or sd is None or sd == 0 or pd.isna(sd):
                    return 0.0
                return float((val - mu) / sd)

            score_df["da_z"] = score_df["da_input"].apply(
                lambda x: z(
                    x, driving_norm["da_field_mean"], driving_norm["da_field_std"]
                )
            )
            score_df["dd_z"] = score_df["dd_input"].apply(
                lambda x: z(
                    x, driving_norm["dd_field_mean"], driving_norm["dd_field_std"]
                )
            )

            # Convert weights to a readable dict and compute course_fit_score
            # weights may be None if no history usable
            weight_dict = {}
            cats_used = cats
            if weights is not None:
                # Build readable dict: cats + driving
                for c in cats_used:
                    weight_dict[c] = float(weights.get(c, 0.0))
                weight_dict["da_z"] = float(weights.get("_da_z", 0.0))
                weight_dict["dd_z"] = float(weights.get("_dd_z", 0.0))
            else:
                # No weights: zero everything
                for c in cats_used:
                    weight_dict[c] = 0.0
                weight_dict["da_z"] = 0.0
                weight_dict["dd_z"] = 0.0

            score_df["course_fit_score"] = 0.0
            for c in cats_used:
                if c in score_df.columns:
                    score_df["course_fit_score"] += (
                        score_df[c].astype(float) * weight_dict[c]
                    )
            score_df["course_fit_score"] += (
                score_df["da_z"].astype(float) * weight_dict["da_z"]
            )
            score_df["course_fit_score"] += (
                score_df["dd_z"].astype(float) * weight_dict["dd_z"]
            )

            # Save outputs
            weights_payload = {
                "weights": weight_dict,
                "cats_used": cats_used,
                "driving_norm": driving_norm,
            }
            (processed / f"event_{event_id}_course_fit_weights.json").write_text(
                json.dumps(weights_payload, indent=2), encoding="utf-8"
            )

            # Restore original id column name for downstream
            out_df = score_df.rename(columns={"player_id": id_col})
            keep_cols = [
                id_col,
                "course_fit_score",
                "da_input",
                "dd_input",
                "da_z",
                "dd_z",
            ] + cats_used
            if "player_name" in out_df.columns:
                keep_cols.insert(1, "player_name")
            out_df = out_df[
                [c for c in keep_cols if c in out_df.columns]
            ].drop_duplicates(subset=[id_col])
            out_df.to_parquet(
                processed / f"event_{event_id}_course_fit_diy.parquet", index=False
            )

            print(
                "Saved weights:",
                processed / f"event_{event_id}_course_fit_weights.json",
            )
            print(
                "Saved per-player DIY course_fit (with driving):",
                processed / f"event_{event_id}_course_fit_diy.parquet",
            )

    except Exception as ex:
        # Robust fallback: write empty/default weights payload to avoid crashes upstream
        print("[warn] DIY course-fit unavailable or failed:", ex)
        empty_payload = {"weights": {}, "cats_used": [], "driving_norm": driving_norm}
        (processed / f"event_{event_id}_course_fit_weights.json").write_text(
            json.dumps(empty_payload, indent=2), encoding="utf-8"
        )
        # still write an empty diy parquet with minimal columns for downstream merges
        stub = pd.DataFrame(
            columns=[
                "dg_id",
                "player_id",
                "player_name",
                "course_fit_score",
                "da_input",
                "dd_input",
                "da_z",
                "dd_z",
            ]
        )
        stub.to_parquet(
            processed / f"event_{event_id}_course_fit_diy.parquet", index=False
        )
        print("[warn] Wrote empty weights and stub DIY parquet.")


if __name__ == "__main__":
    main()
