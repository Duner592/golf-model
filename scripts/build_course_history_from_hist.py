#!/usr/bin/env python3
# scripts/build_course_history_from_hist.py
#
# Computes per-player venue-level course history using your combined historical parquet:
# - rounds_course: count of rounds at this venue
# - sg_course_mean: mean sg_total at this venue
# - sg_course_mean_shrunk: EB shrinkage of sg_course_mean toward 0 with prior n0 (default 16)
#
# Input:
#   data/raw/historical/{tour}/tournament_{normalized_event_name}_rounds_combined.parquet
# Output:
#   data/processed/{tour}/event_{event_id}_course_history_stats.parquet

from __future__ import annotations

# stdlib/third-party
from pathlib import Path
import json
import pandas as pd
import re

# ensure src import works when running directly
import _bootstrap  # noqa: F401


TOUR = "pga"
N0 = 16.0  # prior pseudo-rounds for shrinkage


def normalize_name(s: str) -> str:
    s0 = (s or "").lower()
    s0 = re.sub(r"[^a-z0-9]+", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0.replace(" ", "_")


def load_event_meta(root: Path) -> dict:
    processed = root / "data" / "processed" / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No event meta; run parse_field_updates.py first.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def wide_rounds_to_long_sg(df: pd.DataFrame) -> pd.DataFrame:
    # Extract round_N.sg_total into long form
    id_col = None
    for cand in ["player_id", "dg_id", "id"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(
            "No player id column found in historical rounds (player_id/dg_id)."
        )

    pat_sg = re.compile(r"^round_(\d+)\.sg_total$")
    sg_cols = {}
    for c in df.columns:
        m = pat_sg.match(c)
        if m:
            sg_cols[int(m.group(1))] = c
    if not sg_cols:
        raise ValueError("No round_N.sg_total columns found in historical parquet.")

    recs = []
    for _, row in df.iterrows():
        pid = row[id_col]
        year = row.get("year", None)
        for r, col in sg_cols.items():
            val = row.get(col, None)
            if pd.notna(val):
                recs.append(
                    {"player_id": pid, "year": year, "round": r, "sg_total": float(val)}
                )

    long_df = pd.DataFrame.from_records(recs)
    if long_df.empty:
        raise ValueError("No non-null sg_total values after reshaping.")
    return long_df


def eb_shrink(
    mean: pd.Series, n: pd.Series, mu0: float = 0.0, n0: float = N0
) -> pd.Series:
    # Posterior mean of normal-normal with known variance (heuristic): (n*mean + n0*mu0)/(n+n0)
    return (n * mean + n0 * mu0) / (n + n0)


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR

    meta = load_event_meta(root)
    event_id = str(meta["event_id"])
    event_name = meta.get("event_name") or "current_event"
    safe_name = normalize_name(event_name)

    hist_path = (
        root
        / "data"
        / "raw"
        / "historical"
        / TOUR
        / f"tournament_{safe_name}_rounds_combined.parquet"
    )
    if not hist_path.exists():
        raise FileNotFoundError(
            f"Historical combined parquet not found: {hist_path}. Run fetch_historical_rounds.py first."
        )

    df_hist = pd.read_parquet(hist_path)
    long_sg = wide_rounds_to_long_sg(df_hist)

    # Aggregate per player at this venue
    agg = long_sg.groupby("player_id", as_index=False).agg(
        rounds_course=("sg_total", "count"),
        sg_course_mean=("sg_total", "mean"),
    )
    agg["sg_course_mean_shrunk"] = eb_shrink(
        agg["sg_course_mean"], agg["rounds_course"], mu0=0.0, n0=N0
    )

    out_path = processed / f"event_{event_id}_course_history_stats.parquet"
    agg.to_parquet(out_path, index=False)
    print("Saved course history stats:", out_path)
    print(agg.head(10))


if __name__ == "__main__":
    main()
