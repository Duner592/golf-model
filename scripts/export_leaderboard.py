#!/usr/bin/env python3
# scripts/export_leaderboard.py
#
# Export a clean, shareable leaderboard with QoL features:
# - Player names; hide IDs
# - Timestamp + event name slug in filenames
# - Full + Top-N variants (default --topN 20)
# - Optional HTML (--html)
# - Auto-include tee times if present
# - Summary footer (field size, max/median/sum of p_win)
#
# Outputs:
#   data/preds/{tour}/event_{id}_{slug}_{date}_leaderboard.csv
#   data/preds/{tour}/event_{id}_{slug}_{date}_leaderboard_top{N}.csv
#   data/preds/{tour}/event_{id}_{slug}_{date}_summary.json

from __future__ import annotations

# stdlib/third-party
from pathlib import Path
import json
import argparse
import pandas as pd
import re
from datetime import datetime

# ensure src import works when running directly
import _bootstrap  # noqa: F401

from src.utils_event import (
    resolve_event_id,
)

TOUR_DEFAULT = "pga"

ENRICH_CANDIDATES = [
    "player_name",
    "r1_teetime",
    "r2_teetime",
    "course_fit_score",
]


def slugify(s: str) -> str:
    s0 = (s or "").lower()
    s0 = re.sub(r"[^a-z0-9]+", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0.replace(" ", "_")


def load_latest_meta(tour: str, root: Path) -> dict:
    processed = root / "data" / "processed" / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(
            "No event meta found. Run parse_field_updates.py first."
        )
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def pick_preds_file(tour: str, event_id: str, root: Path) -> Path:
    preds_dir = root / "data" / "preds" / tour
    candidates = [
        preds_dir / f"event_{event_id}_preds_with_course.parquet",
        preds_dir / f"event_{event_id}_preds_common_shock.parquet",
        preds_dir / f"event_{event_id}_preds_baseline.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No predictions found in {preds_dir}")


def load_features_snapshot(tour: str, event_id: str, root: Path) -> pd.DataFrame | None:
    feats_dir = root / "data" / "features" / tour
    p_course = feats_dir / f"event_{event_id}_features_course.parquet"
    p_full = feats_dir / f"event_{event_id}_features_full.parquet"
    if p_course.exists():
        return pd.read_parquet(p_course)
    if p_full.exists():
        return pd.read_parquet(p_full)
    return None


def _pick_best_join_key(df_left: pd.DataFrame, df_right: pd.DataFrame):
    L, R = df_left.copy(), df_right.copy()

    def try_key(lkey, rkey, rename_right_to=None):
        dl, dr = L.copy(), R.copy()
        if rename_right_to and rkey in dr.columns:
            dr = dr.rename(columns={rkey: rename_right_to})
            rkey = rename_right_to
        if lkey not in dl.columns or rkey not in dr.columns:
            return (None, -1.0, None, None)
        dl[lkey] = dl[lkey].astype(str)
        dr[rkey] = dr[rkey].astype(str)
        coverage = len(set(dl[lkey]) & set(dr[rkey])) / max(1, len(set(dl[lkey])))
        return (lkey, coverage, dl, dr)

    tests = [
        try_key("dg_id", "dg_id", None),
        try_key("dg_id", "player_id", "dg_id"),
        try_key("player_id", "player_id", None),
        try_key("player_id", "dg_id", "player_id"),
    ]
    best = max(tests, key=lambda x: x[1])
    if best[0] is None or best[1] <= 0:
        return None, L, R
    return best[0], best[2], best[3]


def build_display_table(
    preds: pd.DataFrame, feats: pd.DataFrame | None
) -> pd.DataFrame:
    df = preds.copy()

    # Ensure player_name
    if "player_name" not in df.columns:
        if "name" in df.columns:
            df = df.rename(columns={"name": "player_name"})
        else:
            id_guess = (
                "dg_id"
                if "dg_id" in df.columns
                else ("player_id" if "player_id" in df.columns else None)
            )
            df["player_name"] = df.get(id_guess, df.index).astype(str)

    # Probabilities for display
    for col, rnd in [("p_win", 2), ("p_top10", 1), ("p_mc", 1)]:
        if col in df.columns:
            df[f"{col}_%"] = (df[col] * 100).round(rnd)

    # Enrichment
    if feats is not None and len(feats):
        key, df_aligned, feats_aligned = _pick_best_join_key(df, feats)
        if key:
            keep = [c for c in ENRICH_CANDIDATES if c in feats_aligned.columns]
            if keep:
                feats_small = feats_aligned[[key] + keep].drop_duplicates(subset=[key])
                merged = df_aligned.merge(
                    feats_small, on=key, how="left", suffixes=("_preds", "_feat")
                )
                # keep preds player_name
                if (
                    "player_name_preds" in merged.columns
                    or "player_name_feat" in merged.columns
                ):
                    merged["player_name"] = merged.get(
                        "player_name_preds", merged.get("player_name", None)
                    )
                    if merged["player_name"].isna().any():
                        merged["player_name"] = merged["player_name"].fillna(
                            merged.get("player_name_feat")
                        )
                    merged = merged.drop(
                        columns=[
                            c
                            for c in ["player_name_preds", "player_name_feat"]
                            if c in merged.columns
                        ]
                    )
                df = merged

    # Rank by p_win
    if "p_win" in df.columns:
        df = df.sort_values("p_win", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # Final columns
    cols = ["rank", "player_name", "p_win_%", "p_top10_%", "p_mc_%"]
    for tcol in ["r1_teetime", "r2_teetime"]:
        if tcol in df.columns and df[tcol].notna().any():
            cols.append(tcol)
    if "course_fit_score" in df.columns:
        cols.append("course_fit_score")

    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def compute_summary(preds_raw: pd.DataFrame) -> dict:
    res = {}
    n = len(preds_raw)
    res["field_size"] = int(n)
    if "p_win" in preds_raw.columns:
        p = preds_raw["p_win"].astype(float)
        res["p_win_max"] = float(p.max()) if n else 0.0
        res["p_win_median"] = float(p.median()) if n else 0.0
        res["p_win_sum"] = float(p.sum()) if n else 0.0
    else:
        res["p_win_max"] = res["p_win_median"] = res["p_win_sum"] = None
    return res


def save_outputs(
    leaderboard: pd.DataFrame,
    preds_raw: pd.DataFrame,
    preds_dir: Path,
    event_id: str,
    event_name: str,
    topN: int | None,
    html: bool,
) -> None:
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    slug = slugify(event_name)
    base = f"event_{event_id}_{slug}_{date_str}_leaderboard"

    out_full = preds_dir / f"{base}.csv"
    leaderboard.to_csv(out_full, index=False)
    print("Saved:", out_full)

    N = topN if topN and topN > 0 else 20
    lb_top = leaderboard.head(N).copy()
    out_top = preds_dir / f"{base}_top{N}.csv"
    lb_top.to_csv(out_top, index=False)
    print("Saved:", out_top)

    if html:
        out_full_html = preds_dir / f"{base}.html"
        out_top_html = preds_dir / f"{base}_top{N}.html"
        leaderboard.to_html(out_full_html, index=False)
        lb_top.to_html(out_top_html, index=False)
        print("Saved:", out_full_html)
        print("Saved:", out_top_html)

    summary = compute_summary(preds_raw)
    summary_path = preds_dir / f"{base}_summary.json"
    summary_payload = {
        "event_id": event_id,
        "event_name": event_name,
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ"),
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print("Saved summary:", summary_path)

    print("\nSummary footer:")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Export leaderboard CSV/HTML with QoL features."
    )
    parser.add_argument("--tour", type=str, default=TOUR_DEFAULT)
    parser.add_argument("--topN", type=int, default=20)
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--event_id", type=str, default=None)  # optional override
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    preds_dir = root / "data" / "preds" / args.tour
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Resolve meta and preds
    # (keep meta for event_name/slug)
    meta = load_latest_meta(args.tour, root)
    event_id = resolve_event_id(args.event_id, args.tour)
    event_name = meta.get("event_name") or "event"

    preds_path = pick_preds_file(args.tour, event_id, root)
    preds_raw = pd.read_parquet(preds_path)
    feats = load_features_snapshot(args.tour, event_id, root)

    leaderboard = build_display_table(preds_raw, feats)
    save_outputs(
        leaderboard, preds_raw, preds_dir, event_id, event_name, args.topN, args.html
    )

    print("\nTop rows:")
    print(leaderboard.head(max(10, args.topN or 10)).to_string(index=False))


if __name__ == "__main__":
    main()
