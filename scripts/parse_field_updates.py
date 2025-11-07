#!/usr/bin/env python3
# scripts/parse_field_updates.py
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

TOUR = "pga"

_time_pat = re.compile(r"^(\d{1,2}):(\d{2})")


def infer_wave_from_time(tstr: str | None) -> str | None:
    if not tstr or not isinstance(tstr, str):
        return None
    m = _time_pat.match(tstr.strip())
    if not m:
        return None
    hour = int(m.group(1))
    return "AM" if hour < 12 else "PM"


def load_field_updates(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing file: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def normalize_field(data: dict) -> pd.DataFrame:
    field = data.get("field", [])
    if not isinstance(field, list):
        raise ValueError("Expected 'field' to be a list")
    df = pd.json_normalize(field, sep=".")

    # Ensure a player identifier
    if "player_id" not in df.columns:
        for alt in ["dg_id", "id", "player.player_id", "pga_number"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "player_id"})
                break
    if "player_id" not in df.columns:
        raise ValueError("Could not find a suitable player_id column")

    df["event_id"] = data.get("event_id")
    return df


def add_teetimes_and_waves(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Keep common columns if present
    keep_cols = [
        c
        for c in [
            "player_id",
            "player_name",
            "dg_id",
            "pga_number",
            "country",
            "flag",
            "start_hole",
            "early_late",
            "r1_teetime",
            "r2_teetime",
            "r3_teetime",
            "r4_teetime",
        ]
        if c in out.columns
    ]
    out = out[keep_cols + ["event_id"]]

    # Infer waves for R1/R2 from time strings (if present)
    for r in (1, 2):
        col = f"r{r}_teetime"
        wave_col = f"r{r}_wave"
        if col in out.columns:
            out[wave_col] = out[col].apply(infer_wave_from_time)
        else:
            out[wave_col] = None
    return out


def save_outputs(data: dict, df_field: pd.DataFrame, df_tt: pd.DataFrame, tour: str):
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / tour
    processed_dir.mkdir(parents=True, exist_ok=True)

    event_id = data.get("event_id")
    event_name = data.get("event_name")
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

    # Save base field table
    base_field = processed_dir / f"event_{event_id}_field"
    df_field.to_csv(base_field.with_suffix(".csv"), index=False)
    try:
        df_field.to_parquet(base_field.with_suffix(".parquet"), index=False)
    except Exception:
        pass

    # Save tee-time–enriched table
    base_tt = processed_dir / f"event_{event_id}_field_teetimes"
    df_tt.to_csv(base_tt.with_suffix(".csv"), index=False)
    try:
        df_tt.to_parquet(base_tt.with_suffix(".parquet"), index=False)
    except Exception:
        pass

    # Save/refresh meta
    meta = {
        "event_id": event_id,
        "event_name": event_name,
        "tour": tour,
        "current_round": data.get("current_round"),
        "last_updated": data.get("last_updated"),
        "saved_at_utc": ts,
        "n_players": int(len(df_field)),
        "has_r1_teetimes": (bool(df_tt["r1_teetime"].notna().any()) if "r1_teetime" in df_tt.columns else False),
        "has_r2_teetimes": (bool(df_tt["r2_teetime"].notna().any()) if "r2_teetime" in df_tt.columns else False),
    }
    meta_path = processed_dir / f"event_{event_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved base field: {base_field.with_suffix('.csv')}")
    print(f"Saved tee-times field: {base_tt.with_suffix('.csv')}")
    print(f"Saved/updated meta: {meta_path}")


def main():
    # Read the file produced by fetch_field_updates.py
    script_dir = Path(__file__).resolve().parent
    field_updates_json = script_dir / "field-updates.json"
    data = load_field_updates(field_updates_json)

    df_field = normalize_field(data)
    df_tt = add_teetimes_and_waves(df_field)

    # Sample output for sanity
    if "r1_teetime" in df_tt.columns:
        sample = df_tt[df_tt["r1_teetime"].notna()].head(10)
        if not sample.empty:
            print("Sample R1 tee times:")
            cols = [c for c in ["player_id", "player_name", "r1_teetime", "r1_wave"] if c in sample.columns]
            print(sample[cols].to_string(index=False))

    save_outputs(data, df_field, df_tt, tour=TOUR)


if __name__ == "__main__":
    main()
