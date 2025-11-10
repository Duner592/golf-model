#!/usr/bin/env python3
# scripts/build_features_from_weather.py
#
# Robust features builder with explicit event selection:
# Priority:
#   1) --event_id CLI
#   2) scripts/field-updates.json event_id
#   3) processed meta that already has weather summaries
#   4) latest processed meta
#
# Then:
# - Load field (tee-times if exists, else base)
# - Load neutral/wave weather (build neutral from wave if missing)
# - Emit features_weather parquet

from __future__ import annotations

import argparse

# stdlib/third-party
from pathlib import Path

# ensure src import works when running directly
import _bootstrap  # noqa: F401
import pandas as pd

from src.utils_event import (
    load_field_table,
    load_weather_neutral,
    resolve_event_id,
    try_load_weather_wave,
)

TOUR = "pga"


def build_neutral_from_wave(df_wave: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in [1, 2, 3, 4]:
        sub = df_wave[df_wave["round"] == r]
        if sub.empty:
            rows.append({"round": r, "delta_strokes": 0.0})
            continue
        waves = sub.set_index("wave")["delta_strokes"].to_dict() if "wave" in sub.columns else {}
        if {"AM", "PM"}.issubset(set(waves)):
            mean_delta = float(pd.Series([waves["AM"], waves["PM"]]).mean())
        elif "ALL" in waves:
            mean_delta = float(waves["ALL"])
        else:
            mean_delta = float(sub["delta_strokes"].mean())
        rows.append({"round": r, "delta_strokes": mean_delta})
    return pd.DataFrame(rows)


def attach_weather_features(df_field: pd.DataFrame, df_neutral: pd.DataFrame, df_wave: pd.DataFrame | None) -> pd.DataFrame:
    feats = df_field.copy()

    # Neutral map
    neutral_map = {int(r): float(d) for r, d in zip(df_neutral["round"], df_neutral["delta_strokes"], strict=False)}
    for r in [1, 2, 3, 4]:
        feats[f"weather_r{r}_delta_neutral"] = neutral_map.get(r, 0.0)

    # Wave-aware (fallback to neutral)
    if df_wave is not None and "wave" in df_wave.columns:
        wave_map = {}
        for r in [1, 2, 3, 4]:
            sub = df_wave[df_wave["round"] == r]
            wave_map[r] = {str(w): float(d) for w, d in zip(sub["wave"], sub["delta_strokes"], strict=False)} if not sub.empty else {}

        for r in [1, 2]:
            wave_col = f"r{r}_wave" if f"r{r}_wave" in feats.columns else None
            if wave_col:
                feats[f"weather_r{r}_delta_wave"] = feats[wave_col].map(wave_map.get(r, {})).fillna(feats[f"weather_r{r}_delta_neutral"])
            else:
                feats[f"weather_r{r}_delta_wave"] = feats[f"weather_r{r}_delta_neutral"]

        for r in [3, 4]:
            all_delta = wave_map.get(r, {}).get("ALL", None)
            feats[f"weather_r{r}_delta_wave"] = all_delta if all_delta is not None else feats[f"weather_r{r}_delta_neutral"]
    else:
        for r in [1, 2, 3, 4]:
            feats[f"weather_r{r}_delta_wave"] = feats[f"weather_r{r}_delta_neutral"]

    return feats


def main():
    ap = argparse.ArgumentParser(description="Build weather-based features for the current event.")
    ap.add_argument("--event_id", type=str, default=None, help="Override event_id")
    args = ap.parse_args()

    eid = resolve_event_id(args.event_id, TOUR)
    print("Resolved event_id:", eid)

    df_field = load_field_table(eid, TOUR)

    # Weather (neutral with fallback from wave)
    try:
        df_neutral = load_weather_neutral(eid, TOUR)
        print("Loaded neutral summary.")
    except FileNotFoundError:
        df_wave = try_load_weather_wave(eid, TOUR)
        if df_wave is None:
            raise FileNotFoundError("Missing both neutral and wave weather summaries. Run summarize_weather_from_schedule.py first.") from None
        print("Neutral summary missing; building from wave...")
        df_neutral = build_neutral_from_wave(df_wave)
        out_neu = Path(__file__).resolve().parent.parent / "data" / "processed" / TOUR / f"event_{eid}_weather_round_neutral.parquet"
        df_neutral.to_parquet(out_neu, index=False)
        print("Built and saved neutral:", out_neu)
    df_wave = try_load_weather_wave(eid, TOUR)

    feats = attach_weather_features(df_field, df_neutral, df_wave)
    out_path = Path(__file__).resolve().parent.parent / "data" / "features" / TOUR / f"event_{eid}_features_weather.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    feats.to_parquet(out_path, index=False)
    print(f"Saved features: {out_path}  rows={len(feats)}  cols={len(feats.columns)}")


if __name__ == "__main__":
    main()
