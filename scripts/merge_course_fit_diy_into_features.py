#!/usr/bin/env python3
# scripts/merge_course_fit_diy_into_features.py
#
# Merge DIY course fit (with driving features) into features_full using a dtype-safe join.
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
    ap = argparse.ArgumentParser(description="Merge DIY course fit into features (dtype-safe).")
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

    diy_path = processed / f"event_{event_id}_course_fit_diy.parquet"
    if not diy_path.exists():
        print("[warn] Missing DIY course fit parquet. Skipping merge.")
        return
    diy = pd.read_parquet(diy_path)

    # Align ID column
    key = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in feats.columns and cand in diy.columns:
            key = cand
            break
    if key is None:
        for feats_key in ["dg_id", "player_id", "id"]:
            if feats_key in feats.columns:
                for diy_key in ["dg_id", "player_id", "id"]:
                    if diy_key in diy.columns:
                        diy = diy.rename(columns={diy_key: feats_key})
                        key = feats_key
                        break
            if key:
                break
    if key is None:
        raise ValueError("Could not align ID columns between features_full and DIY course fit.")

    # Dtype-safe merge
    feats[key] = feats[key].astype(str)
    diy[key] = diy[key].astype(str)

    # Select columns to merge
    diy_cols_available = [c for c in ["course_fit_score", "da_input", "dd_input", "da_z", "dd_z"] if c in diy.columns]
    extra_skill_cols = [c for c in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g"] if c in diy.columns]
    merge_cols = [key] + list(dict.fromkeys(diy_cols_available + extra_skill_cols))

    diy_small = diy[merge_cols].drop_duplicates(subset=[key]).copy()
    out = feats.merge(diy_small, on=key, how="left", suffixes=("", "_diy"))

    # Fill sensible defaults
    if "course_fit_score" in out.columns:
        out["course_fit_score"] = pd.to_numeric(out["course_fit_score"], errors="coerce")
        out["course_fit_score"] = out["course_fit_score"].fillna(out["course_fit_score"].median())
    for zcol in ["da_z", "dd_z"]:
        if zcol in out.columns:
            out[zcol] = pd.to_numeric(out[zcol], errors="coerce").fillna(0.0)

    out.to_parquet(feats_path, index=False)
    print("Merged DIY course fit (and driving) into features:", feats_path)


if __name__ == "__main__":
    main()
