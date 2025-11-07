#!/usr/bin/env python3
# scripts/merge_sigma_into_features.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TOUR = "pga"
DEFAULT_SIGMA = 2.8
BOUNDS = (1.8, 3.8)


def resolve_event_id(processed: Path, override: str | None) -> str:
    if override:
        return str(override)
    meta = json.loads(sorted(processed.glob("event_*_meta.json"))[-1].read_text(encoding="utf-8"))
    return str(meta["event_id"])


def main():
    ap = argparse.ArgumentParser(description="Merge computed sigma into features_full for the event.")
    ap.add_argument("--event_id", type=str, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR
    features.mkdir(parents=True, exist_ok=True)

    event_id = resolve_event_id(processed, args.event_id)

    feats_path = features / f"event_{event_id}_features_full.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(f"Missing features_full for event {event_id}: {feats_path}")
    feats = pd.read_parquet(feats_path)

    sigma_path = processed / f"event_{event_id}_player_sigma.parquet"
    if sigma_path.exists():
        sigma = pd.read_parquet(sigma_path)
    else:
        print(f"[warn] No sigma file for event {event_id}: {sigma_path}. Will use default for all.")
        sigma = pd.DataFrame()

    # Determine join key
    key = None
    for cand in ["player_id", "dg_id", "id"]:
        if cand in feats.columns and (sigma.empty or cand in sigma.columns):
            key = cand
            break
    if key is None:
        # try to coerce sigma id to feats id
        if not sigma.empty:
            for cand in ["player_id", "dg_id", "id"]:
                if cand in feats.columns:
                    for alt in ["player_id", "dg_id", "id"]:
                        if alt in sigma.columns:
                            sigma = sigma.rename(columns={alt: cand})
                            key = cand
                            break
                    if key:
                        break
    if key is None:
        # no merge possible; just add a default sigma column
        feats["sigma"] = DEFAULT_SIGMA
        feats.to_parquet(feats_path, index=False)
        print(f"[warn] Could not align IDs; wrote default sigma to {feats_path}")
        return

    out = feats.copy()
    if not sigma.empty:
        # keep only id + sigma
        keep = [c for c in sigma.columns if c == key or c == "sigma"]
        sigma_small = sigma[keep].drop_duplicates(subset=[key])
        out = out.merge(sigma_small, on=key, how="left")

    # Ensure sigma exists
    if "sigma" not in out.columns:
        out["sigma"] = DEFAULT_SIGMA

    # Fill and clip
    out["sigma"] = out["sigma"].fillna(DEFAULT_SIGMA).clip(BOUNDS[0], BOUNDS[1])

    out.to_parquet(feats_path, index=False)
    print(f"Merged sigma into features and saved: {feats_path}")


if __name__ == "__main__":
    main()
