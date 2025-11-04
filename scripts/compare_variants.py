#!/usr/bin/env python3
# scripts/compare_variants.py
#
# Compare prediction variants for the current event (or a specified event_id):
#   - with_course vs common_shock
#   - with_course vs baseline
#   - common_shock vs baseline
#
# Outputs:
#   - Rank agreement (Spearman rho) between variants
#   - Top-K delta table (p_win differences) for each comparison
#
# Usage:
#   python scripts/compare_variants.py
#   python scripts/compare_variants.py --event_id 457 --topK 15
#
# Notes:
# - Avoids next(sorted(...)) TypeError by selecting the last element from a list.
# - Joins robustly on dg_id -> player_id -> player_name and coerces types to str.

from pathlib import Path
import argparse
import json
import pandas as pd

TOUR_DEFAULT = "pga"


def latest_meta_path(processed_dir: Path) -> Path:
    metas = sorted(processed_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta files found under {processed_dir}")
    return metas[-1]  # last in sorted list


def resolve_event_id(processed_dir: Path, override: str | None) -> str:
    if override:
        return str(override)
    meta_p = latest_meta_path(processed_dir)
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    eid = meta.get("event_id")
    if eid is None:
        raise ValueError(f"event_id missing in {meta_p}")
    return str(eid)


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


def align_variants(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    # pick key and coerce to string
    a = a.copy()
    b = b.copy()
    key = choose_join_key(a, b)
    if not key:
        raise ValueError(
            "Could not find a common join key (dg_id/player_id/player_name)."
        )

    a[key] = a[key].astype(str)
    b[key] = b[key].astype(str)

    cols = [c for c in [key, "player_name", "p_win"] if c in a.columns]
    a_small = a[cols].copy()
    if "player_name" not in a_small.columns and "player_name" in b.columns:
        a_small = a_small.merge(
            b[[key, "player_name"]].drop_duplicates(), on=key, how="left"
        )

    cols_b = [c for c in [key, "p_win"] if c in b.columns]
    b_small = b[cols_b].copy()

    m = a_small.merge(b_small, on=key, suffixes=("_a", "_b"))
    return m, key


def rank_agreement(df_merged: pd.DataFrame) -> float:
    # Spearman rank (no scipy dependency)
    ra = df_merged["p_win_a"].rank(ascending=False, method="first")
    rb = df_merged["p_win_b"].rank(ascending=False, method="first")
    rho = ra.corr(rb, method="spearman")
    return float(rho)


def compare_pair(a: pd.DataFrame, b: pd.DataFrame, label: str, topK: int):
    if a is None or b is None:
        print(f"[skip] Missing preds for {label}")
        return
    merged, key = align_variants(a, b)
    rho = rank_agreement(merged)
    print(f"\n{label}: Spearman rho={rho:.3f}  (n={len(merged)})")

    # Top-K deltas by variant A rank
    merged = merged.sort_values("p_win_a", ascending=False)
    show_cols = [c for c in [key, "player_name"] if c in merged.columns] + [
        "p_win_a",
        "p_win_b",
    ]
    merged["delta"] = merged["p_win_a"] - merged["p_win_b"]
    print(f"Top-{topK} deltas (p_win_a - p_win_b):")
    print(merged[show_cols + ["delta"]].head(topK).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Compare prediction variants for the current event."
    )
    ap.add_argument("--tour", default=TOUR_DEFAULT, help="Tour key (default: pga)")
    ap.add_argument("--event_id", type=str, default=None, help="Force event_id")
    ap.add_argument(
        "--topK", type=int, default=10, help="Rows to show in delta tables (default 10)"
    )
    args = ap.parse_args()

    processed_dir = Path("data/processed") / args.tour
    preds_dir = Path("data/preds") / args.tour
    event_id = resolve_event_id(processed_dir, args.event_id)

    with_course = load_preds(preds_dir, event_id, "preds_with_course")
    common_shock = load_preds(preds_dir, event_id, "preds_common_shock")
    baseline = load_preds(preds_dir, event_id, "preds_baseline")

    compare_pair(with_course, common_shock, "with_course vs common_shock", args.topK)
    compare_pair(with_course, baseline, "with_course vs baseline", args.topK)
    compare_pair(common_shock, baseline, "common_shock vs baseline", args.topK)


if __name__ == "__main__":
    main()
