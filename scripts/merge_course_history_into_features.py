#!/usr/bin/env python3
# scripts/merge_course_history_into_features.py
#
# Merge course history stats into features_full using a dtype-safe join.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TOUR = "pga"


def resolve_event_id(cli_event_id: str | None) -> str:
    if cli_event_id:
        return str(cli_event_id)
    processed = Path("data/processed") / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta found. Run parse_field_updates.py first.")
    return str(json.loads(metas[-1].read_text(encoding="utf-8"))["event_id"])


def main():
    ap = argparse.ArgumentParser(description="Merge course history stats into features (dtype-safe).")
    ap.add_argument("--event_id", type=str, default=None)
    args = ap.parse_args()

    event_id = resolve_event_id(args.event_id)
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR

    feats_path = features / f"event_{event_id}_features_full.parquet"
    if not feats_path.exists():
        raise FileNotFoundError("Missing features_full; run merge_player_data_into_features.py first.")
    feats = pd.read_parquet(feats_path)

    stats_path = processed / f"event_{event_id}_course_history_stats.parquet"
    if not stats_path.exists():
        print("[warn] Missing course history stats parquet. Skipping merge.")
        return
    stats = pd.read_parquet(stats_path)

    # Align ID column
    key = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in feats.columns and cand in stats.columns:
            key = cand
            break
    if key is None:
        for feats_key in ["dg_id", "player_id", "id"]:
            if feats_key in feats.columns:
                for stats_key in ["dg_id", "player_id", "id"]:
                    if stats_key in stats.columns:
                        stats = stats.rename(columns={stats_key: feats_key})
                        key = feats_key
                        break
            if key:
                break
    if key is None:
        raise ValueError("Could not align ID columns between features_full and course history stats.")

    # Ensure consistent dtypes for merge (fix for object vs float64)
    feats[key] = feats[key].astype(str)
    stats[key] = stats[key].astype(str)

    # Select columns to merge
    stats_cols_available = [c for c in ["rounds_course", "sg_course_mean", "sg_course_mean_shrunk"] if c in stats.columns]
    keep = [key] + stats_cols_available

    stats_small = stats[keep].drop_duplicates(subset=[key]).copy()
    out = feats.merge(stats_small, on=key, how="left")

    out.to_parquet(feats_path, index=False)
    print("Merged course history stats into features:", feats_path)


if __name__ == "__main__":
    main()
