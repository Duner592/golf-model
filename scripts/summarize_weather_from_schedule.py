#!/usr/bin/env python3
# scripts/summarize_weather_from_schedule.py
#
# Summarize hourly weather (mph and %) into:
#   - event_{eid}_weather_round_neutral.parquet  (round, wind_mph, gust_mph, temp_c, precip_pct, delta_strokes)
#   - event_{eid}_weather_round_wave.parquet     (round, wave, wind_mph, gust_mph, temp_c, precip_pct, delta_strokes)
#
# Notes:
# - Assumes fetch_weather_from_schedule.py requested windspeed_unit=mph (forecast) or converted archive.
# - If --event_id is provided, summarizes only that pinned event_id. Otherwise, falls back to the latest.
# - delta_strokes is computed from mph: max(0, wind_mph - BASE_MPH) * SLOPE

import argparse
import json
from pathlib import Path

import pandas as pd

TOUR = "pga"
BASE_MPH = 8.0  # wind baseline
SLOPE = 0.12  # strokes per mph above baseline


def load_meta(processed_dir: Path, event_id: str | None) -> dict:
    """
    Prefer weather_meta for explicit event, else event meta; or fall back to latest.
    """
    if event_id:
        for name in (
            f"event_{event_id}_weather_meta.json",
            f"event_{event_id}_meta.json",
        ):
            p = processed_dir / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No weather/meta found for event_id={event_id}")
    # no event_id: pick latest
    metas = sorted(processed_dir.glob("event_*_weather_meta.json"))
    if not metas:
        metas = sorted(processed_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta found; run fetch_weather_from_schedule.py first.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def load_hourly(processed_dir: Path, event_id: str) -> pd.DataFrame:
    p = processed_dir / f"event_{event_id}_weather_hourly.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing hourly weather JSON: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    # mph expected (forecast), but archive conversion code in fetch also populates precipitation_probability
    return pd.DataFrame(
        {
            "time_local": pd.to_datetime(raw["hourly"]["time"]),
            "wind_mph": raw["hourly"]["wind_speed_10m"],  # mph
            "gust_mph": raw["hourly"]["wind_gusts_10m"],  # mph
            "temp_c": raw["hourly"]["temperature_2m"],
            "precip_pct": raw["hourly"].get("precipitation_probability", [0] * len(raw["hourly"]["time"])),
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
    g = float(df_day["gust_mph"].mean())
    t = float(df_day["temp_c"].mean())
    p = float(pd.to_numeric(df_day["precip_pct"], errors="coerce").fillna(0).mean())
    delta = max(0.0, w - BASE_MPH) * SLOPE
    return {
        "wind_mph": round(w, 2),
        "gust_mph": round(g, 2),
        "temp_c": round(t, 1),
        "precip_pct": round(p, 0),
        "delta_strokes": round(delta, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize weather into round/day and wave splits (mph/%).")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id to summarize")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR

    meta = load_meta(processed_dir, args.event_id)
    eid = str(meta["event_id"])
    df_hourly = load_hourly(processed_dir, eid)

    # Neutral summary per round using schedule/pinned dates
    neutral_rows = []
    for r, key in [(1, "r1_date"), (2, "r2_date"), (3, "r3_date"), (4, "r4_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d]
        s = summarize_day(day)
        s["round"] = r
        neutral_rows.append(s)
    df_neutral = pd.DataFrame(neutral_rows)
    out_neu = processed_dir / f"event_{eid}_weather_round_neutral.parquet"
    df_neutral.to_parquet(out_neu, index=False)
    print("Saved:", out_neu)

    # Wave-aware split: AM (<12) vs PM (>=12) for R1/R2; R3/R4 as ALL
    wave_rows = []
    for r, key in [(1, "r1_date"), (2, "r2_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d].copy()
        if day.empty:
            wave_rows += [
                {"round": r, "wave": "AM", **summarize_day(day)},
                {"round": r, "wave": "PM", **summarize_day(day)},
            ]
            continue
        day["hour"] = day["time_local"].dt.hour
        am = day[day["hour"] < 12]
        pm = day[day["hour"] >= 12]
        wave_rows.append({"round": r, "wave": "AM", **summarize_day(am)})
        wave_rows.append({"round": r, "wave": "PM", **summarize_day(pm)})
    for r, key in [(3, "r3_date"), (4, "r4_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d]
        wave_rows.append({"round": r, "wave": "ALL", **summarize_day(day)})
    df_wave = pd.DataFrame(wave_rows)
    out_wv = processed_dir / f"event_{eid}_weather_round_wave.parquet"
    df_wave.to_parquet(out_wv, index=False)
    print("Saved:", out_wv)


if __name__ == "__main__":
    main()
