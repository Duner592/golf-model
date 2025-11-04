#!/usr/bin/env python3
# scripts/verify_leaderboard.py
from __future__ import annotations

# stdlib/third-party
from pathlib import Path
import json
import pandas as pd

# ensure src import works when running directly
import _bootstrap  # noqa: F401


TOUR = "pga"


def load_latest_meta():
    p = sorted((Path("data/processed") / TOUR).glob("event_*_meta.json"))[-1]
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    meta = load_latest_meta()
    event_id = str(meta["event_id"])
    preds_dir = Path("data/preds") / TOUR

    # Pick the timestamped leaderboard, fallback to non-timestamp if needed
    candidates = sorted(preds_dir.glob(f"event_{event_id}_*_leaderboard.csv")) or [
        preds_dir / f"event_{event_id}_leaderboard.csv"
    ]

    lf = candidates[-1]
    df = pd.read_csv(lf)
    print("Leaderboard:", lf)
    print("Rows:", len(df))
    print(df.head(10).to_string(index=False))

    # If you want to verify p_win sums, load raw preds (with p_win)
    p_pred = None
    for name in [
        f"event_{event_id}_preds_with_course.parquet",
        f"event_{event_id}_preds_common_shock.parquet",
        f"event_{event_id}_preds_baseline.parquet",
    ]:
        p = Path("data/preds") / TOUR / name
        if p.exists():
            p_pred = pd.read_parquet(p)
            break

    if p_pred is not None:
        print("\nRaw preds summary (p_win):")
        print(" field_size:", len(p_pred))
        print(" p_win_sum :", round(float(p_pred["p_win"].sum()), 6))
        print(" p_win_max :", round(float(p_pred["p_win"].max()), 6))
        print(" p_win_median:", round(float(p_pred["p_win"].median()), 6))
    else:
        print("\nRaw preds file not found for p_win sum check (optional).")


if __name__ == "__main__":
    main()
