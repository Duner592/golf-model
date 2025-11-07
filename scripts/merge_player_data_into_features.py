#!/usr/bin/env python3
# scripts/merge_player_data_into_features.py
from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd

TOUR = "pga"


def latest_meta(processed: Path) -> dict:
    metas = sorted(processed.glob("event_*_meta.json"), key=lambda p: p.stat().st_mtime)
    if not metas:
        raise FileNotFoundError("No meta files.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_id", type=str, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR
    features.mkdir(parents=True, exist_ok=True)

    meta = latest_meta(processed)
    event_id = str(args.event_id) if args.event_id else str(meta["event_id"])

    feat_path = features / f"event_{event_id}_features_weather.parquet"
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
        raise ValueError("Features missing dg_id/player_id for merge.")

    # Load player files for this event
    r_path = processed / f"event_{event_id}_dg_rankings.parquet"
    s_path = processed / f"event_{event_id}_skill_ratings.parquet"
    rankings = pd.read_parquet(r_path) if r_path.exists() else pd.DataFrame()
    skills = pd.read_parquet(s_path) if s_path.exists() else pd.DataFrame()

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

    def prefix_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return df
        new_cols = {c: (c if c == id_col else f"{prefix}{c}") for c in df.columns}
        return df.rename(columns=new_cols)

    rankings_p = prefix_df(rankings, "dr_")
    skills_p = prefix_df(skills, "sr_")

    out = (
        feats.merge(rankings_p, on=id_col, how="left")
        if not rankings_p.empty
        else feats.copy()
    )
    out = out.merge(skills_p, on=id_col, how="left") if not skills_p.empty else out

    out.to_parquet(features / f"event_{event_id}_features_full.parquet", index=False)
    print(
        "Saved merged features:", features / f"event_{event_id}_features_full.parquet"
    )


if __name__ == "__main__":
    main()
