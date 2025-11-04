#!/usr/bin/env python3
# scripts/merge_player_data_into_features.py
#
# Refactored to merge the new preds endpoints:
# - event_{event_id}_dg_rankings.parquet
# - event_{event_id}_skill_ratings.parquet
# into your weather-driven features:
# - data/features/{tour}/event_{event_id}_features_weather.parquet
#
# Output:
# - data/features/{tour}/event_{event_id}_features_full.parquet

from pathlib import Path
import json
import pandas as pd

TOUR_DEFAULT = "pga"


def load_meta(tour: str):
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(
            "No event meta found. Run parse_field_updates.py first."
        )
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def main():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    tour = TOUR_DEFAULT
    meta = load_meta(tour)
    event_id = str(meta["event_id"])

    features_dir = root / "data" / "features" / tour
    processed_dir = root / "data" / "processed" / tour
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load base features (from weather step)
    feat_path = features_dir / f"event_{event_id}_features_weather.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing weather features: {feat_path}. Run build_features_from_weather.py first."
        )
    feats = pd.read_parquet(feat_path)

    # Decide ID column
    id_col = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in feats.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError("Features table missing dg_id/player_id for merge.")

    # Load rankings + skills
    r_path = processed_dir / f"event_{event_id}_dg_rankings.parquet"
    s_path = processed_dir / f"event_{event_id}_skill_ratings.parquet"
    if not r_path.exists() and not s_path.exists():
        raise FileNotFoundError("No player data found. Run fetch_player_data.py first.")

    rankings = pd.read_parquet(r_path) if r_path.exists() else pd.DataFrame()
    skills = pd.read_parquet(s_path) if s_path.exists() else pd.DataFrame()

    # Ensure consistent ID column
    if id_col not in rankings.columns and not rankings.empty:
        for alt in ["dg_id", "player_id", "id"]:
            if alt in rankings.columns:
                rankings = rankings.rename(columns={alt: id_col})
                break
    if id_col not in skills.columns and not skills.empty:
        for alt in ["dg_id", "player_id", "id"]:
            if alt in skills.columns:
                skills = skills.rename(columns={alt: id_col})
                break

    # Prefix columns to avoid collisions and keep provenance
    def prefix_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return df
        keep = df.columns.tolist()
        # never prefix the id col
        new_cols = {c: (c if c == id_col else f"{prefix}{c}") for c in keep}
        return df.rename(columns=new_cols)

    rankings_p = prefix_df(rankings, "dr_")
    skills_p = prefix_df(skills, "sr_")

    # Merge progressively
    out = (
        feats.merge(rankings_p, on=id_col, how="left")
        if not rankings_p.empty
        else feats.copy()
    )
    out = out.merge(skills_p, on=id_col, how="left") if not skills_p.empty else out

    # Optionally fill a few common numeric columns if present
    for col in [c for c in out.columns if c.startswith(("dr_", "sr_"))]:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())

    out_path = features_dir / f"event_{event_id}_features_full.parquet"
    out.to_parquet(out_path, index=False)
    print("Saved merged features:", out_path)

    # Quick preview of some likely useful columns
    preview_cols = [
        c
        for c in [
            id_col,
            "player_name",
            "dr_rank",
            "dr_rating",
            "dr_dg_rating",
            "sr_sg_total",
            "sr_sg_t2g",
            "sr_sg_ott",
            "sr_sg_app",
            "sr_sg_putt",
            "weather_r1_delta_wave",
            "weather_r2_delta_wave",
        ]
        if c in out.columns
    ]
    if preview_cols:
        print(out[preview_cols].head(12))


if __name__ == "__main__":
    main()
