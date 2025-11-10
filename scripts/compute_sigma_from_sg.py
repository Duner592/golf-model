#!/usr/bin/env python3
# scripts/compute_sigma_from_sg.py
#
# Fallback: constant sigma per player for this event (hardened to --event_id).
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SIGMA = 2.8
BOUNDS = (1.8, 3.8)


def resolve_event_id(cli_event_id: str | None, tour: str) -> str:
    if cli_event_id:
        return str(cli_event_id)
    processed = Path("data/processed") / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta found.")
    return str(json.loads(metas[-1].read_text(encoding="utf-8"))["event_id"])


def main():
    ap = argparse.ArgumentParser(description="Compute per-player sigma (volatility) for the pinned/current event.")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id")
    ap.add_argument("--tour", type=str, default="pga", help="Tour to process")
    args = ap.parse_args()

    TOUR = args.tour
    processed = Path("data/processed") / TOUR
    event_id = resolve_event_id(args.event_id, TOUR)

    # Load field to get ids
    field = None
    for name in [
        f"event_{event_id}_field_teetimes.parquet",
        f"event_{event_id}_field.parquet",
        f"event_{event_id}_field_teetimes.csv",
        f"event_{event_id}_field.csv",
    ]:
        p = processed / name
        if p.exists():
            field = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            break
    if field is None:
        raise FileNotFoundError("Field table not found for sigma fallback.")

    id_col = "player_id" if "player_id" in field.columns else ("dg_id" if "dg_id" in field.columns else None)
    if id_col is None:
        raise ValueError("No id column in field table for sigma fallback.")
    sigma_df = field[[id_col]].drop_duplicates().copy()
    sigma_df["sigma"] = DEFAULT_SIGMA
    sigma_df["sigma"] = sigma_df["sigma"].clip(*BOUNDS)

    out_path = processed / f"event_{event_id}_player_sigma.parquet"
    sigma_df.to_parquet(out_path, index=False)
    print(f"Saved sigma: {out_path} (rows={len(sigma_df)})")


if __name__ == "__main__":
    main()
