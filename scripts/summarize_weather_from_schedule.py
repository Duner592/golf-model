#!/usr/bin/env python3
# scripts/summarize_weather_from_schedule.py
#
# Summarize hourly weather (mph and %) into:
#   - event_{eid}_weather_round_neutral.parquet
#   - event_{eid}_weather_round_wave.parquet
#
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TOUR = "pga"
BASE_MPH = 8.0
SLOPE = 0.12  # strokes per mph above baseline


def resolve_event_id(cli_event_id: str | None) -> str:
    if cli_event_id:
        return str(cli_event_id)
    processed = Path("data/processed") / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta found; run fetch_weather_from_schedule.py first.")
    return str(json.loads(metas[-1].read_text(encoding="utf-8"))["event_id"])


def load_hourly(processed_dir: Path, event_id: str) -> pd.DataFrame:
    p = processed_dir / f"event_{event_id}_weather_hourly.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing hourly weather JSON: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(
        {
            "time_local": pd.to_datetime(raw["hourly"]["time"]),
            "wind_mph": raw["hourly"].get("wind_speed_10m") or raw["hourly"].get("wind_mph"),
            "gust_mph": raw["hourly"].get("wind_gusts_10m") or raw["hourly"].get("gust_mph"),
            "temp_c": raw["hourly"].get("temperature_2m") or raw["hourly"].get("temp_c"),
            "precip_pct": raw["hourly"].get("precipitation_probability") or raw["hourly"].get("precip_pct"),
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
        "delta_strokes": round(delta, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize weather into round/day and wave splits (mph/%).")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id to summarize")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR

    eid = resolve_event_id(args.event_id)
    df_hourly = load_hourly(processed_dir, eid)

    # Round dates from event meta
    meta = json.loads((processed_dir / f"event_{eid}_meta.json").read_text(encoding="utf-8"))
    rows_neutral = []
    for r, key in [(1, "r1_date"), (2, "r2_date"), (3, "r3_date"), (4, "r4_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d]
        rows_neutral.append({"round": r, **summarize_day(day)})
    df_neutral = pd.DataFrame(rows_neutral)
    out_neu = processed_dir / f"event_{eid}_weather_round_neutral.parquet"
    df_neutral.to_parquet(out_neu, index=False)
    print("Saved:", out_neu)

    # Wave-aware (R1/R2 split AM/PM)
    wave_rows = []
    for r, key in [(1, "r1_date"), (2, "r2_date")]:
        d = meta[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d].copy()
        if day.empty:
            wave_rows += [{"round": r, "wave": "AM", **summarize_day(day)}, {"round": r, "wave": "PM", **summarize_day(day)}]
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
