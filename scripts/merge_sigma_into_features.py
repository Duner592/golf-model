#!/usr/bin/env python3
# scripts/merge_sigma_into_features.py
#
# Merge computed sigma into your features_full table.
# Input:
#   - data/features/{tour}/event_{event_id}_features_full.parquet
#   - data/processed/{tour}/event_{event_id}_player_sigma.parquet
# Output:
#   - data/features/{tour}/event_{event_id}_features_full.parquet (updated in-place)
#   - data/features/{tour}/event_{event_id}_features_full_with_sigma.parquet (backup snapshot)

from pathlib import Path
import json
import pandas as pd

TOUR = "pga"
DEFAULT_SIGMA = 2.8
BOUNDS = (1.8, 3.8)


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR
    features.mkdir(parents=True, exist_ok=True)

    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta; run parse_field_updates.py first.")
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    event_id = str(meta["event_id"])

    feats_path = features / f"event_{event_id}_features_full.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(
            "Missing features_full; run merge_player_data_into_features.py first."
        )
    feats = pd.read_parquet(feats_path)

    sigma_path = processed / f"event_{event_id}_player_sigma.parquet"
    if not sigma_path.exists():
        raise FileNotFoundError(
            "Missing sigma file; run compute_sigma_from_sg.py first."
        )
    sigma = pd.read_parquet(sigma_path)

    # Determine join key
    key = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in feats.columns and cand in sigma.columns:
            key = cand
            break
    if key is None:
        # Try to rename in sigma or feats
        for cand in ["dg_id", "player_id", "id"]:
            if cand in feats.columns:
                if (
                    "dg_id" in sigma.columns
                    or "player_id" in sigma.columns
                    or "id" in sigma.columns
                ):
                    for alt in ["dg_id", "player_id", "id"]:
                        if alt in sigma.columns:
                            sigma = sigma.rename(columns={alt: cand})
                            key = cand
                            break
                if key:
                    break
        if key is None:
            raise ValueError("Could not align ID columns between features and sigma.")

    out_prev = features / f"event_{event_id}_features_full_with_sigma.parquet"
    feats.to_parquet(out_prev, index=False)

    out = feats.merge(sigma[[key, "sigma"]], on=key, how="left")
    out["sigma"] = out["sigma"].fillna(DEFAULT_SIGMA).clip(*BOUNDS)

    out.to_parquet(feats_path, index=False)
    print(f"Merged sigma into features and saved: {feats_path}")
    print(f"Backup of pre-merge features saved: {out_prev}")


if __name__ == "__main__":
    main()
