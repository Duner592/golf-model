#!/usr/bin/env python3
# scripts/summarize_weather_from_schedule.py
#
# Summarize hourly weather (mph and %) into:
#   - event_{eid}_weather_round_neutral.parquet  (round, wind_mph, gust_mph, temp_c, precip_pct, delta_strokes, weathercode)
#   - event_{eid}_weather_round_wave.parquet     (round, wave, wind_mph, gust_mph, temp_c, precip_pct, delta_strokes, weathercode)
#
# Hardened:
# - Accepts --event_id
# - If r1_date..r4_date missing in weather_meta, derive the first four unique local dates from hourly time
#
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_MPH = 8.0  # wind baseline
SLOPE = 0.12  # strokes per mph above baseline


def resolve_event_id(cli_event_id: str | None, tour: str) -> str:
    if cli_event_id:
        return str(cli_event_id)
    processed = Path("data/processed") / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta found; run fetch_weather_from_schedule.py first.")
    return str(json.loads(metas[-1].read_text(encoding="utf-8"))["event_id"])


def load_hourly(processed_dir: Path, event_id: str) -> pd.DataFrame:
    p = processed_dir / f"event_{event_id}_weather_hourly.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing hourly weather JSON: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    # mph and %, plus weathercode
    return pd.DataFrame(
        {
            "time_local": pd.to_datetime(raw["hourly"]["time"]),
            "wind_mph": raw["hourly"].get("wind_speed_10m") or raw["hourly"].get("wind_mph"),
            "gust_mph": raw["hourly"].get("wind_gusts_10m") or raw["hourly"].get("gust_mph"),
            "temp_c": raw["hourly"].get("temperature_2m") or raw["hourly"].get("temp_c"),
            "precip_pct": raw["hourly"].get("precipitation_probability") or raw["hourly"].get("precip_pct"),
            "weathercode": raw["hourly"].get("weathercode"),
        }
    )


def try_load_weather_meta(processed_dir: Path, event_id: str) -> dict | None:
    p = processed_dir / f"event_{event_id}_weather_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def derive_round_dates_from_hourly(df_hourly: pd.DataFrame) -> dict:
    """
    Return {'r1_date','r2_date','r3_date','r4_date'} from the first four
    unique local calendar dates present in the hourly dataframe.
    """
    dates = df_hourly["time_local"].dt.strftime("%Y-%m-%d").dropna().drop_duplicates().sort_values().tolist()
    if len(dates) < 4:
        # If fewer than 4 unique days, pad with last known date(s)
        while len(dates) < 4 and dates:
            dates.append(dates[-1])
    elif len(dates) > 4:
        dates = dates[:4]
    if not dates or len(dates) < 4:
        raise ValueError("Cannot derive 4 round dates from hourly weather; check the hourly time range.")
    return {"r1_date": dates[0], "r2_date": dates[1], "r3_date": dates[2], "r4_date": dates[3]}


def summarize_day(df_day: pd.DataFrame) -> dict:
    if df_day.empty:
        return {
            "wind_mph": None,
            "gust_mph": None,
            "temp_c": None,
            "precip_pct": None,
            "delta_strokes": 0.0,
            "weathercode": None,
        }
    w = float(pd.to_numeric(df_day["wind_mph"], errors="coerce").dropna().mean()) if "wind_mph" in df_day else None
    g = float(pd.to_numeric(df_day["gust_mph"], errors="coerce").dropna().mean()) if "gust_mph" in df_day else None
    t = float(pd.to_numeric(df_day["temp_c"], errors="coerce").dropna().mean()) if "temp_c" in df_day else None
    p = float(pd.to_numeric(df_day["precip_pct"], errors="coerce").fillna(0).mean()) if "precip_pct" in df_day else None
    weathercode = None
    if "weathercode" in df_day.columns:
        mode_series = df_day["weathercode"].mode()
        weathercode = int(mode_series.iloc[0]) if not mode_series.empty else None
    w = w if w is not None else 0.0
    delta = max(0.0, w - BASE_MPH) * SLOPE
    return {
        "wind_mph": round(w, 2) if w is not None else None,
        "gust_mph": round(g, 2) if g is not None else None,
        "temp_c": round(t, 1) if t is not None else None,
        "precip_pct": round(p, 0) if p is not None else None,
        "delta_strokes": round(delta, 3),
        "weathercode": weathercode,
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize weather into round/day and wave splits (mph/%).")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id to summarize")
    ap.add_argument("--tour", type=str, default="pga", help="Tour to process")
    args = ap.parse_args()

    TOUR = args.tour
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR

    eid = resolve_event_id(args.event_id, TOUR)
    df_hourly = load_hourly(processed_dir, eid)

    # Round dates: prefer weather_meta; if missing, derive from hourly
    meta_w = try_load_weather_meta(processed_dir, eid) or {}
    if not all(k in meta_w for k in ("r1_date", "r2_date", "r3_date", "r4_date")):
        print("[warn] weather_meta missing r1_date..r4_date; deriving dates from hourly.")
        dates_map = derive_round_dates_from_hourly(df_hourly)
    else:
        dates_map = {k: meta_w[k] for k in ("r1_date", "r2_date", "r3_date", "r4_date")}

    # Neutral summary per round using dates
    neutral_rows = []
    for r, key in [(1, "r1_date"), (2, "r2_date"), (3, "r3_date"), (4, "r4_date")]:
        d = dates_map[key]
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
        d = dates_map[key]
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
        d = dates_map[key]
        day = df_hourly[df_hourly["time_local"].dt.strftime("%Y-%m-%d") == d]
        wave_rows.append({"round": r, "wave": "ALL", **summarize_day(day)})
    df_wave = pd.DataFrame(wave_rows)
    out_wv = processed_dir / f"event_{eid}_weather_round_wave.parquet"
    df_wave.to_parquet(out_wv, index=False)
    print("Saved:", out_wv)


if __name__ == "__main__":
    main()
