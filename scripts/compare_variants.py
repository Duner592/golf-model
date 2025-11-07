#!/usr/bin/env python3
# scripts/compare_variants.py
#
# Compare prediction variants for the current event (or a specified event_id):
#   - with_course vs common_shock
#   - with_course vs baseline
#   - common_shock vs baseline
#
# Usage:
#   python scripts/compare_variants.py
#   python scripts/compare_variants.py --event_id 457 --topK 15

import argparse
from pathlib import Path

# Ensure src is importable when running scripts directly
import _bootstrap  # noqa: F401
import pandas as pd

from src.utils_event import resolve_event_id

TOUR_DEFAULT = "pga"


def load_preds(preds_dir: Path, event_id: str, stem: str) -> pd.DataFrame | None:
    p = preds_dir / f"event_{event_id}_{stem}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def choose_join_key(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    for k in ["dg_id", "player_id", "player_name"]:
        if k in a.columns and k in b.columns:
            return k
    # try renames to align
    if "dg_id" in a.columns and "player_id" in b.columns:
        b.rename(columns={"player_id": "dg_id"}, inplace=True)
        return "dg_id"
    if "player_id" in a.columns and "dg_id" in b.columns:
        b.rename(columns={"dg_id": "player_id"}, inplace=True)
        return "player_id"
    return None


def align_variants(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    a = a.copy()
    b = b.copy()
    key = choose_join_key(a, b)
    if not key:
        raise ValueError("Could not find a common join key (dg_id/player_id/player_name).")
    a[key] = a[key].astype(str)
    b[key] = b[key].astype(str)

    cols_a = [c for c in [key, "player_name", "p_win"] if c in a.columns]
    a_small = a[cols_a].copy()
    if "player_name" not in a_small.columns and "player_name" in b.columns:
        a_small = a_small.merge(b[[key, "player_name"]].drop_duplicates(), on=key, how="left")

    cols_b = [c for c in [key, "p_win"] if c in b.columns]
    b_small = b[cols_b].copy()

    m = a_small.merge(b_small, on=key, suffixes=("_a", "_b"))
    return m, key


def rank_agreement(df_merged: pd.DataFrame) -> float:
    # Spearman rank without scipy
    ra = df_merged["p_win_a"].rank(ascending=False, method="first")
    rb = df_merged["p_win_b"].rank(ascending=False, method="first")
    return float(ra.corr(rb, method="spearman"))


def compare_pair(a: pd.DataFrame | None, b: pd.DataFrame | None, label: str, topK: int):
    if a is None or b is None:
        print(f"[skip] Missing preds for {label}")
        return
    merged, key = align_variants(a, b)
    rho = rank_agreement(merged)
    print(f"\n{label}: Spearman rho={rho:.3f}  (n={len(merged)})")

    merged = merged.sort_values("p_win_a", ascending=False)
    merged["delta"] = merged["p_win_a"] - merged["p_win_b"]
    show_cols = [c for c in [key, "player_name", "p_win_a", "p_win_b", "delta"] if c in merged.columns]
    print(f"Top-{topK} deltas (p_win_a - p_win_b):")
    print(merged[show_cols].head(topK).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Compare prediction variants for the current event.")
    ap.add_argument("--tour", default=TOUR_DEFAULT, help="Tour key (default: pga)")
    ap.add_argument("--event_id", type=str, default=None, help="Force event_id")
    ap.add_argument("--topK", type=int, default=10, help="Rows to show in delta tables (default 10)")
    args = ap.parse_args()

    preds_dir = Path("data/preds") / args.tour
    event_id = resolve_event_id(args.event_id)

    with_course = load_preds(preds_dir, event_id, "preds_with_course")
    common_shock = load_preds(preds_dir, event_id, "preds_common_shock")
    baseline = load_preds(preds_dir, event_id, "preds_baseline")

    compare_pair(with_course, common_shock, "with_course vs common_shock", args.topK)
    compare_pair(with_course, baseline, "with_course vs baseline", args.topK)
    compare_pair(common_shock, baseline, "common_shock vs baseline", args.topK)


if __name__ == "__main__":
    main()
