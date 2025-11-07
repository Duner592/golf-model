#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


def load_event_meta(tour: str = "pga") -> dict:
    meta_dir = Path(__file__).resolve().parent.parent / "data" / "meta" / tour
    metas = sorted(meta_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No event meta found.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def main():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    tour = "pga"
    meta = load_event_meta(tour)
    event_id = meta["event_id"]

    processed_dir = root / "data" / "processed" / tour
    tee_times_raw = root / "data" / "raw" / tour / f"event_{event_id}_tee-times.json"

    # Field table
    field_parquet = processed_dir / f"event_{event_id}_field.parquet"
    field_csv = processed_dir / f"event_{event_id}_field.csv"
    if field_parquet.exists():
        df_field = pd.read_parquet(field_parquet)
    elif field_csv.exists():
        df_field = pd.read_csv(field_csv)
    else:
        raise FileNotFoundError("Field table not found. Run parse_field_updates.py first.")

    # If tee-times file not present, skip merge
    if not tee_times_raw.exists():
        print("No tee-times file yet. Skipping merge; keep neutral pipeline.")
        # Save a pass-through copy if you want:
        (processed_dir / f"event_{event_id}_field_with_teetimes.csv").write_text(df_field.to_csv(index=False), encoding="utf-8")
        return

    tee = json.loads(tee_times_raw.read_text(encoding="utf-8"))

    # Normalize tee-times into a DataFrame. Adjust keys per actual schema.
    # Expecting list of dict items with player_id, round, tee_time_local, wave, etc.
    if isinstance(tee, dict) and "tee_times" in tee:
        tt_df = pd.json_normalize(tee["tee_times"])
    elif isinstance(tee, list):
        tt_df = pd.json_normalize(tee)
    else:
        raise ValueError("Unexpected tee-times schema; inspect the JSON structure.")

    # Simplify to R1/R2 wave and time if present
    for r in [1, 2]:
        # Example pivot; adjust column names to actual keys from the API
        mask = tt_df.get("round", pd.Series([None] * len(tt_df))) == r
        sub = tt_df[mask]
        if "player_id" in sub.columns and "wave" in sub.columns:
            waves = sub[["player_id", "wave"]].drop_duplicates()
            df_field = df_field.merge(waves, on="player_id", how="left", suffixes=("", f"_r{r}"))
            if f"wave_r{r}" not in df_field.columns:
                df_field = df_field.rename(columns={"wave": f"wave_r{r}"})
        if "player_id" in sub.columns and "tee_time_local" in sub.columns:
            times = sub[["player_id", "tee_time_local"]].drop_duplicates()
            df_field = df_field.merge(times, on="player_id", how="left", suffixes=("", f"_r{r}"))
            if f"tee_time_r{r}" not in df_field.columns:
                df_field = df_field.rename(columns={"tee_time_local": f"tee_time_r{r}"})

    out_path = processed_dir / f"event_{event_id}_field_with_teetimes.parquet"
    df_field.to_parquet(out_path, index=False)
    print(f"Saved merged field + tee times: {out_path}")


if __name__ == "__main__":
    main()
