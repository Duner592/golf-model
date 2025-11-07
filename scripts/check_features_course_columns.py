#!/usr/bin/env python3
# scripts/check_features_course_columns.py
import json
from pathlib import Path

import pandas as pd

TOUR = "pga"


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR

    meta = json.loads(sorted(processed.glob("event_*_meta.json"))[-1].read_text(encoding="utf-8"))
    event_id = str(meta["event_id"])

    # Prefer course snapshot, else full
    for name in [
        f"event_{event_id}_features_course.parquet",
        f"event_{event_id}_features_full.parquet",
    ]:
        p = features / name
        if p.exists():
            df = pd.read_parquet(p)
            print("File:", p.name)
            cols = set(df.columns)
            needed = {
                "course_fit_score",
                "sg_course_mean_shrunk",
                "da_z",
                "dd_z",
                "sigma",
            }
            print("Has columns:", {c: (c in cols) for c in needed})
            print("Sample rows:", len(df), "Sample cols:", list(df.columns)[:20])
            break
    else:
        print("No features parquet found.")


if __name__ == "__main__":
    main()
