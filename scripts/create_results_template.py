#!/usr/bin/env python3
# scripts/create_results_template.py
#
# Build a results template for the current event (winner_flag=0).
# Edit this file after the event finishes and set winner_flag=1 for the winner.

import json
from pathlib import Path

import pandas as pd

TOUR = "pga"


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR

    # Load event meta to get id
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta found. Run parse_field_updates.py first.")
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    event_id = str(meta["event_id"])

    # Pick field table (tee-time enriched preferred)
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
        raise FileNotFoundError("Field table not found. Run parse_field_updates.py first.")

    # Build template with player_name (fallbacks)
    df = field.copy()
    if "player_name" not in df.columns:
        if "name" in df.columns:
            df = df.rename(columns={"name": "player_name"})
        else:
            # as last resort, try an ID (not ideal)
            id_col = "dg_id" if "dg_id" in df.columns else ("player_id" if "player_id" in df.columns else None)
            if id_col is None:
                raise ValueError("Cannot find player_name or an id to derive names.")
            df["player_name"] = df[id_col].astype(str)

    out = df[["player_name"]].drop_duplicates().copy()
    out["winner_flag"] = 0
    out_path = processed / f"event_{event_id}_results.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved results template:", out_path)
    print("Next: open the CSV after the event completes and set winner_flag=1 for the winner.")


if __name__ == "__main__":
    main()
