#!/usr/bin/env python3
# scripts/build_field_and_results_from_hist.py
#
# Build a field table and a simple results file (winner_flag) from historical rounds JSON
# for the pinned or specified event. Designed for backtesting.
#
# Inputs:
#   - data/raw/historical/{tour}/event_{event_id}_{year}_rounds.json
#     (Use --year to select the correct file)
# Meta handling:
#   - event_id is taken from --event_id if provided, else from latest meta by mtime.
#
# Outputs:
#   - data/processed/{tour}/event_{event_id}_field.csv
#   - data/processed/{tour}/event_{event_id}_results.csv

from __future__ import annotations
import re
import json
import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

TOUR_DEFAULT = "pga"


def latest_meta_by_mtime(processed_dir: Path) -> Path | None:
    metas = list(processed_dir.glob("event_*_meta.json"))
    if not metas:
        return None
    return max(metas, key=lambda p: p.stat().st_mtime)


def load_event_id_from_meta(processed_dir: Path) -> str:
    p = latest_meta_by_mtime(processed_dir)
    if not p:
        raise FileNotFoundError(f"No meta found under {processed_dir}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    if "event_id" not in meta:
        raise ValueError(f"Meta missing event_id: {p}")
    return str(meta["event_id"])


def pick_hist_json(root: Path, tour: str, event_id: str, year: Optional[str]) -> Path:
    hist_dir = root / "data" / "raw" / "historical" / tour
    if year:
        p = hist_dir / f"event_{event_id}_{year}_rounds.json"
        if not p.exists():
            raise FileNotFoundError(f"Historical JSON not found: {p}. Fetch it first.")
        return p
    # fallback: pick latest file for this event_id
    candidates = sorted(hist_dir.glob(f"event_{event_id}_*_rounds.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No historical JSON for event {event_id} in {hist_dir}"
        )
    return candidates[-1]


def extract_rows(data: Any) -> List[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(
        "Unrecognized historical JSON structure: expected list or dict containing a list."
    )


def compute_winner(df: pd.DataFrame) -> Tuple[str, str]:
    # fin_text winner
    if "fin_text" in df.columns:
        mask = df["fin_text"].astype(str).str.upper().isin(["1", "W", "WIN"])
        if mask.any():
            winner_name = df.loc[mask, "player_name"].iloc[0]
            return str(winner_name), "fin_text"
    # lowest total across round_N.score
    score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
    if score_cols:
        df["__total"] = (
            df[score_cols]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=1)
        )
        tmp = df.dropna(subset=["__total"])
        if not tmp.empty:
            idx = tmp["__total"].idxmin()
            winner_name = tmp.loc[idx, "player_name"]
            return str(winner_name), "lowest_total"
    raise ValueError("Could not determine winner (no fin_text and no round_* scores).")


def build_field(df: pd.DataFrame) -> pd.DataFrame:
    if "player_name" not in df.columns:
        raise ValueError("Historical JSON must include 'player_name'.")
    out = df[["player_name"]].drop_duplicates().copy()
    # attach a player_id if present
    if "dg_id" in df.columns:
        ids = (
            df[["player_name", "dg_id"]]
            .drop_duplicates()
            .rename(columns={"dg_id": "player_id"})
        )
        out = out.merge(ids, on="player_name", how="left")
    elif "player_id" in df.columns:
        ids = df[["player_name", "player_id"]].drop_duplicates()
        out = out.merge(ids, on="player_name", how="left")
    return out.drop_duplicates(subset=["player_name"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description="Build field/results from historical rounds JSON."
    )
    ap.add_argument("--tour", default=TOUR_DEFAULT)
    ap.add_argument(
        "--year", default=None, help="YYYY (required to select the correct rounds file)"
    )
    ap.add_argument(
        "--event_id",
        default=None,
        help="Override event_id (else use latest meta by mtime)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / args.tour
    processed.mkdir(parents=True, exist_ok=True)

    # Resolve event_id robustly
    event_id = (
        str(args.event_id) if args.event_id else load_event_id_from_meta(processed)
    )

    # Find historical JSON
    hist_path = pick_hist_json(root, args.tour, event_id, args.year)
    data = json.loads(hist_path.read_text(encoding="utf-8"))
    rows = extract_rows(data)
    df = pd.json_normalize(rows)

    # Basic normalize
    if "player_name" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "player_name"})

    # Build and save field
    field_df = build_field(df)
    field_out = processed / f"event_{event_id}_field.csv"
    field_df.to_csv(field_out, index=False, encoding="utf-8")
    print(f"Saved field: {field_out}  rows={len(field_df)}")

    # Winner
    winner_name, method = compute_winner(df)
    results_df = field_df[["player_name"]].copy()
    results_df["winner_flag"] = (
        results_df["player_name"].astype(str) == winner_name
    ).astype(int)
    results_out = processed / f"event_{event_id}_results.csv"
    results_df.to_csv(results_out, index=False, encoding="utf-8")
    print(f"Saved results: {results_out}  Winner: {winner_name}  (method: {method})")

    print(
        "\nNext: python scripts/run_weekly_all.py --skip-html; then python scripts/evaluate_preds.py"
    )


if __name__ == "__main__":
    main()
