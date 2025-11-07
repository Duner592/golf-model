#!/usr/bin/env python3
# scripts/plot_quick.py
#
# Quick plots for the current event:
# - p_win histogram
# - cumulative p_win (sorted desc)
#
# Usage:
#   python scripts/plot_quick.py                  # auto-resolve latest event
#   python scripts/plot_quick.py --event_id 457   # force specific event
#   python scripts/plot_quick.py --tour pga       # choose tour
#
# Notes:
# - Fixes: avoid next(sorted(...)) TypeError by selecting the last element of the list.

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def latest_meta(processed_dir: Path) -> dict:
    metas = sorted(processed_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta files under {processed_dir}")
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    return meta


def load_preds(preds_dir: Path, event_id: str) -> pd.DataFrame:
    candidates = [
        preds_dir / f"event_{event_id}_preds_with_course.parquet",
        preds_dir / f"event_{event_id}_preds_common_shock.parquet",
        preds_dir / f"event_{event_id}_preds_baseline.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"No preds parquet found for event_id={event_id} in {preds_dir}")


def main():
    ap = argparse.ArgumentParser(description="Quick plots for current event predictions.")
    ap.add_argument("--tour", default="pga", help="Tour key (default: pga)")
    ap.add_argument("--event_id", default=None, help="Force event_id")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / args.tour
    preds_dir = root / "data" / "preds" / args.tour

    # Resolve event_id
    if args.event_id:
        event_id = str(args.event_id)
        event_name = f"event_{event_id}"
    else:
        meta = latest_meta(processed_dir)
        event_id = str(meta["event_id"])
        event_name = meta.get("event_name", f"event_{event_id}")

    # Load predictions
    df = load_preds(preds_dir, event_id).sort_values("p_win", ascending=False)

    # p_win histogram
    plt.figure(figsize=(7, 4))
    df["p_win"].hist(bins=30, edgecolor="black")
    plt.title(f"{event_name} — p_win histogram")
    plt.xlabel("p_win")
    plt.ylabel("count")
    plt.tight_layout()

    # cumulative p_win
    plt.figure(figsize=(7, 4))
    df["p_win"].reset_index(drop=True).sort_values(ascending=False).cumsum().plot()
    plt.title(f"{event_name} — cumulative p_win (sorted desc)")
    plt.xlabel("rank (desc by p_win)")
    plt.ylabel("cumulative p_win")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
