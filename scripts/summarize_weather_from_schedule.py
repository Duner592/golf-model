#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

TOUR = "pga"


def load_meta(event_id: str | None = None) -> tuple[dict, Path]:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    metas = sorted(processed.glob("event_*_weather_meta.json"))
    if not metas:
        raise FileNotFoundError(
            "No weather_meta found. Run fetch_weather_from_schedule.py first."
        )
    if event_id:
        m = processed / f"event_{event_id}_weather_meta.json"
        return json.loads(m.read_text(encoding="utf-8")), processed
    return json.loads(metas[-1].read_text(encoding="utf-8")), processed


def load_hourly(processed_dir: Path, event_id: str) -> pd.DataFrame:
    raw = json.loads(
        (processed_dir / f"event_{event_id}_weather_hourly.json").read_text(
            encoding="utf-8"
        )
    )
    return pd.DataFrame(
        {
            "time_local": pd.to_datetime(raw["hourly"]["time"]),
            "wind_mps": raw["hourly"]["wind_speed_10m"],
            "gust_mps": raw["hourly"]["wind_gusts_10m"],
            "temp_c": raw["hourly"]["temperature_2m"],
            "precip_prob": raw["hourly"]["precipitation_probability"],
        }
    )


def summarize_day(df_day: pd.DataFrame) -> dict:
    if df_day.empty:
        return {
            "wind_mps": None,
            "gust_mps": None,
            "temp_c": None,
            "precip_prob": None,
            "delta_strokes": 0.0,
        }
    wind = df_day["wind_mps"].mean()
    mph = wind * 2.237
    delta = max(0.0, mph - 8.0) * 0.12
    return {
        "wind_mps": round(wind, 3),
        "gust_mps": round(df_day["gust_mps"].mean(), 3),
        "temp_c": round(df_day["temp_c"].mean(), 1),
        "precip_prob": round(df_day["precip_prob"].mean(), 1),
        "delta_strokes": round(delta, 3),
    }


def main():
    meta, processed_dir = load_meta()
    event_id = str(meta["event_id"])
    df_hourly = load_hourly(processed_dir, event_id)

    # Neutral per round using schedule dates
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

    # Simple AM/PM split for R1/R2 (R3/R4 = ALL)
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
