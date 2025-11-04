#!/usr/bin/env python3
# scripts/merge_course_history_into_features.py
#
# Merges venue course history stats into features table.
# Adds: rounds_course, sg_course_mean, sg_course_mean_shrunk
#
# Inputs:
#   data/processed/{tour}/event_{event_id}_course_history_stats.parquet
#   data/features/{tour}/event_{event_id}_features_course.parquet (preferred) or features_full.parquet
# Output:
#   updates features_course.parquet (or features_full if snapshot absent)

from pathlib import Path
import json
import pandas as pd

TOUR = "pga"


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR

    meta = json.loads(
        sorted(processed.glob("event_*_meta.json"))[-1].read_text(encoding="utf-8")
    )
    event_id = str(meta["event_id"])

    stats_path = processed / f"event_{event_id}_course_history_stats.parquet"
    if not stats_path.exists():
        raise FileNotFoundError(
            "Missing course history stats; run build_course_history_from_hist.py first."
        )
    stats = pd.read_parquet(stats_path)

    # Pick features file
    feat_course = features / f"event_{event_id}_features_course.parquet"
    feat_full = features / f"event_{event_id}_features_full.parquet"
    target = feat_course if feat_course.exists() else feat_full
    if not target.exists():
        raise FileNotFoundError(
            "Missing features; run merge_player_data_into_features.py first (and optionally course merges)."
        )

    df = pd.read_parquet(target)

    # Align join key
    key = None
    for cand in ["player_id", "dg_id", "id"]:
        if cand in df.columns and cand in stats.columns:
            key = cand
            break
    if key is None:
        # try to rename stats
        for fkey in ["player_id", "dg_id", "id"]:
            if fkey in df.columns:
                for skey in ["player_id", "dg_id", "id"]:
                    if skey in stats.columns:
                        stats = stats.rename(columns={skey: fkey})
                        key = fkey
                        break
            if key:
                break
    if key is None:
        raise ValueError(
            "Could not align id columns between features and course history stats."
        )

    # Merge and fill sensible defaults
    keep = [key, "rounds_course", "sg_course_mean", "sg_course_mean_shrunk"]
    out = df.merge(stats[keep], on=key, how="left")
    out["rounds_course"] = out["rounds_course"].fillna(0).astype(int)
    out["sg_course_mean"] = out["sg_course_mean"].fillna(0.0)
    out["sg_course_mean_shrunk"] = out["sg_course_mean_shrunk"].fillna(0.0)

    out.to_parquet(target, index=False)
    print(f"Updated features with course history: {target}")
    # Quick confirmation
    cols = [
        c
        for c in ["rounds_course", "sg_course_mean", "sg_course_mean_shrunk"]
        if c in out.columns
    ]
    print("Added columns present:", cols)


if __name__ == "__main__":
    main()
