#!/usr/bin/env python3
# scripts/summarize_weather_from_schedule.py
#
# Summarize hourly weather (mph and %) into:
#   - event_{eid}_weather_round_neutral.parquet  (round, wind_mph, gust_mph, temp_c, precip_pct, delta_strokes)
#   - event_{eid}_weather_round_wave.parquet     (round, wave, wind_mph, gust_mph, temp_c, precip_pct, delta_strokes)
#
# Notes:
# - Assumes fetch_weather_from_schedule.py requested windspeed_unit=mph.
# - delta_strokes is computed from mph: max(0, wind_mph - BASE_MPH) * SLOPE
# - BASE_MPH and SLOPE can be tuned if needed.

import json
from pathlib import Path
import pandas as pd

TOUR = "pga"
BASE_MPH = 8.0  # wind baseline
SLOPE = 0.12  # strokes per mph above baseline


def load_meta(event_id: str | None = None) -> tuple[dict, Path]:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    metas = sorted(processed.glob("event_*_weather_meta.json"))
    if not metas:
        # fallback to event meta if no weather_meta yet
        metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(
            "No meta found; run fetch_weather_from_schedule.py first."
        )
    if event_id:
        p = processed / f"event_{event_id}_weather_meta.json"
        if not p.exists():
            p = processed / f"event_{event_id}_meta.json"
        meta = json.loads(p.read_text(encoding="utf-8"))
        return meta, processed
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    return meta, processed


def load_hourly(processed_dir: Path, event_id: str) -> pd.DataFrame:
    p = processed_dir / f"event_{event_id}_weather_hourly.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    # mph expected because we requested windspeed_unit=mph
    return pd.DataFrame(
        {
            "time_local": pd.to_datetime(raw["hourly"]["time"]),
            "wind_mph": raw["hourly"]["wind_speed_10m"],  # mph
            "gust_mph": raw["hourly"]["wind_gusts_10m"],  # mph
            "temp_c": raw["hourly"]["temperature_2m"],
            "precip_pct": raw["hourly"]["precipitation_probability"],  # 0..100
        }
    )


def summarize_day(df_day: pd.DataFrame) -> dict:
    if df_day.empty:
        return {
            "wind_mph": None,
            "gust_mph": None,
            "temp_c": None,
            "precip_pct": None,
            "delta_strokes": 0.0,
        }
    w = float(df_day["wind_mph"].mean())
    gust = float(df_day["gust_mph"].mean())
    t = float(df_day["temp_c"].mean())
    p = float(df_day["precip_pct"].mean())
    delta = max(0.0, w - BASE_MPH) * SLOPE
    return {
        "wind_mph": round(w, 2),
        "gust_mph": round(gust, 2),
        "temp_c": round(t, 1),
        "precip_pct": round(p, 0),
        "delta_strokes": round(delta, 2),
    }


def main():
    meta, processed_dir = load_meta()
    event_id = str(meta["event_id"])
    df_hourly = load_hourly(processed_dir, event_id)

    # Neutral summary per round using schedule dates
    rows = []
    for r, key in [(1, "r1_date"), (2, "r2_date"), (3, "r3_date"), (4, "r4_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d]
        s = summarize_day(day)
        s["round"] = r
        rows.append(s)
    df_neutral = pd.DataFrame(rows)
    df_neutral.to_parquet(
        processed_dir / f"event_{event_id}_weather_round_neutral.parquet", index=False
    )
    print("Saved:", processed_dir / f"event_{event_id}_weather_round_neutral.parquet")

    # Wave-aware split: AM (<12) vs PM (>=12) for R1/R2; R3/R4 as ALL
    rows = []
    for r, key in [(1, "r1_date"), (2, "r2_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d].copy()
        if day.empty:
            rows += [
                {"round": r, "wave": "AM", **summarize_day(day)},
                {"round": r, "wave": "PM", **summarize_day(day)},
            ]
            continue
        day["hour"] = day["time_local"].dt.hour
        am = day[day["hour"] < 12]
        pm = day[day["hour"] >= 12]
        rows.append({"round": r, "wave": "AM", **summarize_day(am)})
        rows.append({"round": r, "wave": "PM", **summarize_day(pm)})
    for r, key in [(3, "r3_date"), (4, "r4_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d]
        rows.append({"round": r, "wave": "ALL", **summarize_day(day)})

    df_wave = pd.DataFrame(rows)
    df_wave.to_parquet(
        processed_dir / f"event_{event_id}_weather_round_wave.parquet", index=False
    )
    print("Saved:", processed_dir / f"event_{event_id}_weather_round_wave.parquet")


if __name__ == "__main__":
    main()
