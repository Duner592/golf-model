#!/usr/bin/env python3
# scripts/build_field_and_results_from_hist.py
#
# Build a field table and a simple results file (winner_flag) from your saved historical
# rounds JSON for a pinned event. This is useful for backtesting completed events.
#
# Inputs (one of):
#   - data/raw/historical/{tour}/event_{event_id}_{year}_rounds.json     (preferred via --year)
#   - If --year not provided, the script will auto-pick the latest matching file:
#       data/raw/historical/{tour}/event_{event_id}_*_rounds.json
#
# Outputs:
#   - data/processed/{tour}/event_{event_id}_field.csv      (columns: player_name, [player_id if available])
#   - data/processed/{tour}/event_{event_id}_results.csv    (columns: player_name, winner_flag)
#
# Winner selection strategy:
#   1) If a "fin_text" is present with "W" or "1", pick that player.
#   2) Else, if round_*\.score columns exist, compute the lowest 72-hole total.
#   3) Else, fail with a helpful message (provide a manual results CSV or adjust logic).
#
# Notes:
#   - This script relies on event meta saved by your pin or parse steps:
#       data/processed/{tour}/event_{event_id}_meta.json
#   - It does not require tee times or live leaderboard endpoints.

import re
import json
import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

TOUR_DEFAULT = "pga"


def load_latest_meta(root: Path, tour: str) -> dict:
    processed = root / "data" / "processed" / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(
            f"No meta found under {processed}. Run pin_event_from_schedule.py or parse_field_updates.py."
        )
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def pick_hist_json(root: Path, tour: str, event_id: str, year: Optional[str]) -> Path:
    hist_dir = root / "data" / "raw" / "historical" / tour
    if year:
        p = hist_dir / f"event_{event_id}_{year}_rounds.json"
        if not p.exists():
            raise FileNotFoundError(f"Historical JSON not found: {p}. Fetch it first.")
        return p
    # Auto-pick the latest by filename
    candidates = sorted(hist_dir.glob(f"event_{event_id}_*_rounds.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No historical JSON found matching {hist_dir}/event_{event_id}_*_rounds.json"
        )
    return candidates[-1]


def extract_rows(data: Any) -> List[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # try to find first list in dict
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(
        "Unrecognized historical JSON structure: expected list or dict containing a list."
    )


def compute_winner(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Returns: (winner_name, method_used)
    """
    # 1) fin_text winner
    if "fin_text" in df.columns:
        mask = df["fin_text"].astype(str).str.upper().isin(["1", "W", "WIN"])
        if mask.any():
            winner_name = df.loc[mask, "player_name"].iloc[0]
            return str(winner_name), "fin_text"

    # 2) lowest total score using round_*\.score
    score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
    if score_cols:
        # Some rows may be missing a round score; sum numeric-only
        df["__total_score"] = (
            df[score_cols]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=1)
        )
        # Drop rows with all NaNs in score columns just in case
        tmp = df.dropna(subset=["__total_score"])
        if not tmp.empty:
            idx = tmp["__total_score"].idxmin()
            winner_name = tmp.loc[idx, "player_name"]
            return str(winner_name), "lowest_total"

    # 3) no luck
    raise ValueError(
        "Could not determine winner (no fin_text winner and no round_* scores to total)."
    )


def build_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer dg_id as player_id; else keep player_name only.
    """
    if "player_name" not in df.columns:
        raise ValueError("Historical JSON must include 'player_name' to build a field.")

    # Attempt to standardize a player_id column
    out = df[["player_name"]].drop_duplicates().copy()
    if "dg_id" in df.columns:
        ids = df[["player_name", "dg_id"]].drop_duplicates()
        ids = ids.rename(columns={"dg_id": "player_id"})
        out = out.merge(ids, on="player_name", how="left")
    elif "player_id" in df.columns:
        ids = df[["player_name", "player_id"]].drop_duplicates()
        out = out.merge(ids, on="player_name", how="left")
    # else leave only player_name

    return out.drop_duplicates(subset=["player_name"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description="Build field and results (winner) from historical rounds JSON for the pinned event."
    )
    ap.add_argument("--tour", default=TOUR_DEFAULT, help="Tour key (default: pga)")
    ap.add_argument(
        "--year",
        default=None,
        help="Year of the historical JSON (YYYY). If omitted, picks latest available.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / args.tour
    processed.mkdir(parents=True, exist_ok=True)

    # 0) Load meta for current (pinned) event
    meta = load_latest_meta(root, args.tour)
    event_id = str(meta["event_id"])

    # 1) Pick historical JSON file
    hist_path = pick_hist_json(root, args.tour, event_id, args.year)
    data = json.loads(hist_path.read_text(encoding="utf-8"))
    rows = extract_rows(data)
    df = pd.json_normalize(rows)

    # 2) Build and save field
    field_df = build_field(df)
    field_out = processed / f"event_{event_id}_field.csv"
    field_df.to_csv(field_out, index=False, encoding="utf-8")
    print(f"Saved field: {field_out}  rows={len(field_df)}")

    # 3) Compute and save results (winner_flag)
    winner_name, method = compute_winner(df)
    results_df = field_df[["player_name"]].copy()
    results_df["winner_flag"] = (
        results_df["player_name"].astype(str) == winner_name
    ).astype(int)
    results_out = processed / f"event_{event_id}_results.csv"
    results_df.to_csv(results_out, index=False, encoding="utf-8")
    print(f"Saved results: {results_out}  Winner: {winner_name}  (method: {method})")

    # 4) Small hint for next steps
    print("\nNext:")
    print("  - Run: python scripts/evaluate_preds.py")
    print("  - Or re-run your simulation and export leaderboard for this pinned event.")


if __name__ == "__main__":
    main()
