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


def first_nonempty(values: pd.Series):
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    return cleaned.iloc[0] if not cleaned.empty else None


def choose_name_col(df: pd.DataFrame) -> str | None:
    for col in ["player_name", "name", "Player", "player"]:
        if col in df.columns:
            return col
    return None


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
    if df_hist.empty:
        return pd.DataFrame()

    id_col = "player_id" if "player_id" in df_hist.columns else ("dg_id" if "dg_id" in df_hist.columns else None)
    if id_col is None:
        return pd.DataFrame()
    df_hist = df_hist.copy().rename(columns={id_col: "player_id"})
    df_hist["player_id"] = df_hist["player_id"].astype(str)

    if "sg_total" in df_hist.columns:
        df_long = df_hist.copy()
        df_long["sg_total"] = pd.to_numeric(df_long["sg_total"], errors="coerce")
        agg = {
            "sg_course_mean_shrunk": ("sg_total", "mean"),
            "rounds_course": ("sg_total", "count"),
        }
        name_col = choose_name_col(df_long)
        if name_col:
            agg["player_name"] = (name_col, first_nonempty)
        stats = df_long.dropna(subset=["sg_total"]).groupby("player_id", as_index=False).agg(**agg)
        return stats

    # Find all round-specific sg_total columns
    sg_cols = [col for col in df_hist.columns if col.endswith(".sg_total")]
    if not sg_cols:
        return pd.DataFrame()  # No SG data available

    # Compute total SG per row (sum across rounds)
    df_hist = df_hist.copy()
    df_hist["total_sg"] = df_hist[sg_cols].sum(axis=1, skipna=True)

    # Aggregate per player: mean total_sg, count rounds (using number of non-null SG entries)
    agg = {
        "sg_course_mean_shrunk": ("total_sg", "mean"),
        "rounds_course": ("total_sg", lambda x: x.notna().sum()),  # Count non-null total_sg as rounds played
    }
    name_col = choose_name_col(df_hist)
    if name_col:
        agg["player_name"] = (name_col, first_nonempty)
    stats = df_hist.groupby("player_id", as_index=False).agg(**agg)
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
        "rows": len(stats_df),
        "mean_sg_course": float(stats_df["sg_course_mean_shrunk"].mean()) if "sg_course_mean_shrunk" in stats_df.columns else None,
        "top_rounds": stats_df.sort_values("rounds_course", ascending=False).head(10).to_dict(orient="records")
        if "rounds_course" in stats_df
        else [],
        "top_sg_course": stats_df.sort_values("sg_course_mean_shrunk", ascending=False).head(10).to_dict(orient="records")
        if "sg_course_mean_shrunk" in stats_df
        else [],
    }
    (web_dir / "course_history_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote web assets under web/:")


if __name__ == "__main__":
    main()
