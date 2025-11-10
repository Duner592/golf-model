#!/usr/bin/env python3
# scripts/build_course_history_from_hist.py
#
# Build course history stats from historical combined parquet.
# Outputs: event_{event_id}_course_history_stats.parquet
# And emits web assets.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TOUR = "pga"


def normalize_name(s: str) -> str:
    s0 = (s or "").lower()
    s0 = re.sub(r"[^a-z0-9]+", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0.replace(" ", "_")


def load_event_meta(root: Path, tour: str, event_id: str | None = None) -> dict:
    processed = root / "data" / "processed" / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No event meta found. Run parse_field_updates.py first.")
    if event_id:
        for p in reversed(metas):
            meta = json.loads(p.read_text(encoding="utf-8"))
            if str(meta.get("event_id")) == str(event_id):
                return meta
    # fallback: latest
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def find_hist_parquet(event_name: str, tour: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    safe = normalize_name(event_name)
    return root / "data" / "raw" / "historical" / tour / f"tournament_{safe}_rounds_combined.parquet"


def build_course_history_stats(df_hist: pd.DataFrame) -> pd.DataFrame:
    if df_hist.empty or "sg_total" not in df_hist.columns:
        return pd.DataFrame()

    # Aggregate per player: mean sg_total, count rounds, etc.
    stats = df_hist.groupby("player_id", as_index=False).agg(
        sg_course_mean_shrunk=("sg_total", "mean"),
        rounds_course=("sg_total", "count"),
    )
    # You can add more stats here, e.g., std, min/max sg
    return stats


def main():
    ap = argparse.ArgumentParser(description="Build course history stats from historical rounds.")
    ap.add_argument("--event_id", type=str, default=None, help="Override event_id")
    ap.add_argument("--tour", type=str, default="pga", help="Tour to process")
    args = ap.parse_args()

    TOUR = args.tour
    root = Path(__file__).resolve().parent.parent

    meta = load_event_meta(root, TOUR, args.event_id)
    event_id = str(meta["event_id"])
    event_name = meta.get("event_name") or "current_event"

    processed = root / "data" / "processed" / TOUR
    processed.mkdir(parents=True, exist_ok=True)

    hist_path = find_hist_parquet(event_name, TOUR)
    if not hist_path.exists():
        print(f"[warn] Historical rounds parquet not found: {hist_path}. Skipping course history.")
        return

    df_hist = pd.read_parquet(hist_path)
    stats_df = build_course_history_stats(df_hist)

    if stats_df.empty:
        print("[warn] No course history stats generated.")
        return

    out_path = processed / f"event_{event_id}_course_history_stats.parquet"
    stats_df.to_parquet(out_path, index=False)
    print(f"Saved course history stats: {out_path}")

    # Emit web assets (optional)
    web_dir = root / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    # Example: save a summary JSON for web
    summary = {
        "event_id": event_id,
        "event_name": event_name,
        "stats_rows": len(stats_df),
        "mean_sg_course": float(stats_df["sg_course_mean_shrunk"].mean()) if "sg_course_mean_shrunk" in stats_df.columns else None,
    }
    (web_dir / "course_history_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote web assets under web/:")


if __name__ == "__main__":
    main()
