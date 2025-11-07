#!/usr/bin/env python3
# scripts/fetch_weather_from_schedule.py
#
# Fetch hourly weather for an event and write:
#   data/processed/{tour}/event_{event_id}_weather_hourly.json
#   data/processed/{tour}/event_{event_id}_weather_meta.json
#
# Behavior:
# - Default: use schedule to pick the next event (existing logic).
# - Override: --event_id will fetch lat/lon/start from processed meta for that event_id
#             (pinned/backtest mode) and save files with that event_id.

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Helpers
# -------------------------


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def extract_events(s: dict | list) -> list[dict]:
    """
    Try common schedule shapes:
      - list of events
      - dict with 'events' or 'schedule' or 'data' keys
    """
    if isinstance(s, list):
        return s
    if isinstance(s, dict):
        for key in ("events", "schedule", "data"):
            v = s.get(key)
            if isinstance(v, list):
                return v
    return []


def deep_get(d: dict, *keys) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def pick_event_from_schedule(events: list[dict], known_event_id: str | int | None = None) -> dict | None:
    # Match known event_id first
    if known_event_id is not None:
        for e in events:
            if str(e.get("event_id")) == str(known_event_id):
                return e
    # Otherwise pick the next upcoming by start date, else the most recent past
    candidates = []
    today = datetime.utcnow().date()
    for e in events:
        start = e.get("start") or e.get("start_date") or deep_get(e, "dates", "start")
        sd = parse_date(str(start)) if start else None
        if sd is not None:
            candidates.append((sd, e))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        for sd, e in candidates:
            if sd >= today:
                return e
        return candidates[-1][1]
    return events[0] if events else None


def to_round_dates(start_str: str) -> tuple[str, str, dict[int, str]]:
    start = parse_date(start_str)
    if start is None:
        raise ValueError(f"Cannot parse start date: {start_str}")
    r1 = start
    r2 = start + timedelta(days=1)
    r3 = start + timedelta(days=2)
    r4 = start + timedelta(days=3)
    return (
        r1.isoformat(),
        r4.isoformat(),
        {1: r1.isoformat(), 2: r2.isoformat(), 3: r3.isoformat(), 4: r4.isoformat()},
    )


def fetch_open_meteo_hourly(lat: float, lon: float, start_date: str, end_date: str, tz: str = "auto") -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation_probability,weathercode",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": tz,
        "windspeed_unit": "mph",  # return mph directly
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_open_meteo_archive(lat: float, lon: float, start_date: str, end_date: str, tz: str = "auto") -> dict:
    """
    Historical weather via ERA5 archive. Note:
    - Units are m/s (no windspeed_unit param here).
    - No precipitation_probability; we derive a 0/100% proxy from 'precipitation' > 0.
    """
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": tz,
        "hourly": "wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Derive precipitation_probability (0 or 100) from precipitation > 0
    if "hourly" in data and "precipitation" in data["hourly"]:
        pp = []
        for p in data["hourly"]["precipitation"]:
            try:
                pp.append(100.0 if float(p) > 0 else 0.0)
            except Exception:
                pp.append(0.0)
        data["hourly"]["precipitation_probability"] = pp
    return data


def fetch_hourly_auto(lat: float, lon: float, start_date: str, end_date: str, tz: str = "auto") -> dict:
    """
    Choose forecast vs archive automatically:
    - If end_date is in the past (< today), use archive (ERA5).
    - Else use forecast (as your current code does).
    """
    today = datetime.utcnow().date()
    ed = parse_date(end_date)
    if ed and ed < today:
        # Archive path (m/s units, no windspeed_unit)
        return fetch_open_meteo_archive(lat, lon, start_date, end_date, tz=tz)
    # Forecast path (keeps your existing behavior)
    return fetch_open_meteo_hourly(lat, lon, start_date, end_date, tz=tz)


def load_meta_for_event(processed_dir: Path, event_id: str) -> dict:
    # Prefer weather_meta (if exists), else use meta
    wm = processed_dir / f"event_{event_id}_weather_meta.json"
    if wm.exists():
        return json.loads(wm.read_text(encoding="utf-8"))
    m = processed_dir / f"event_{event_id}_meta.json"
    if not m.exists():
        raise FileNotFoundError(f"Missing meta for pinned event_id={event_id}: {m}")
    return json.loads(m.read_text(encoding="utf-8"))


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser(description="Fetch hourly weather for event (schedule or pinned via --event_id).")
    parser.add_argument(
        "--event_id",
        type=str,
        default=None,
        help="Pinned event id; if set, use processed meta for lat/lon/dates",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    # Load config and env
    cfg = load_yaml(root / "configs" / "datagolf.yaml")
    base_url = cfg["base_url"].rstrip("/")
    key_param = cfg["auth"]["key_param"]
    env_var = cfg["auth"]["env_var"]
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {env_var}")
    tour = cfg.get("defaults", {}).get("tour", "pga")
    sched_path = cfg["endpoints"]["schedule"]["path"]

    out_dir = root / "data" / "processed" / tour
    out_dir.mkdir(parents=True, exist_ok=True)

    # If pinned (--event_id), bypass schedule, read meta, and fetch for that event id.
    if args.event_id:
        pinned_meta = load_meta_for_event(out_dir, args.event_id)
        lat = pinned_meta.get("lat")
        lon = pinned_meta.get("lon")
        start = pinned_meta.get("r1_date")
        if not (lat and lon and start):
            raise ValueError(f"Pinned meta for event_id={args.event_id} missing lat/lon/start")
        start_date, end_date, round_dates = to_round_dates(str(start))
        weather = fetch_hourly_auto(float(lat), float(lon), start_date, end_date, tz="auto")
        (out_dir / f"event_{args.event_id}_weather_hourly.json").write_text(json.dumps(weather, indent=2, ensure_ascii=False), encoding="utf-8")
        pinned_weather_meta = {
            "event_id": int(args.event_id),
            "event_name": pinned_meta.get("event_name"),
            "lat": float(lat),
            "lon": float(lon),
            "r1_date": round_dates[1],
            "r2_date": round_dates[2],
            "r3_date": round_dates[3],
            "r4_date": round_dates[4],
            "saved_at_utc": now_utc().strftime("%Y-%m-%dT%H%M%SZ"),
            "source": "pinned + open-meteo",
        }
        (out_dir / f"event_{args.event_id}_weather_meta.json").write_text(
            json.dumps(pinned_weather_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[pinned] Saved hourly weather: {out_dir / f'event_{args.event_id}_weather_hourly.json'}")
        print(f"[pinned] Saved weather meta: {out_dir / f'event_{args.event_id}_weather_meta.json'}")
        return

    # Default behavior (existing): use schedule to pick an event
    # Optionally, try to use latest meta to match event_id
    known_event_id = None
    meta_dirs = [
        root / "data" / "meta" / tour,
        out_dir,
    ]
    for mdir in meta_dirs:
        metas = sorted(mdir.glob("event_*_meta.json"))
        if metas:
            meta = json.loads(metas[-1].read_text(encoding="utf-8"))
            known_event_id = meta.get("event_id")
            break

    # Fetch schedule
    requests_cache.install_cache("dg_cache", expire_after=600)
    session = requests.Session()
    sched_url = f"{base_url}/{sched_path.lstrip('/')}"
    params = {key_param: api_key, "tour": tour}
    resp = session.get(sched_url, params=params, timeout=20)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("HTTP error:", e)
        print("Status:", getattr(e.response, "status_code", "unknown"))
        print("Body:", getattr(e.response, "text", "")[:1000])
        raise

    try:
        sched_data = resp.json()
    except Exception:
        print("Non-JSON schedule response (first 1000 chars):")
        print(resp.text[:1000])
        raise

    events = extract_events(sched_data)
    if not events:
        print("Schedule payload top-level type:", type(sched_data))
        if isinstance(sched_data, dict):
            print("Top-level keys:", list(sched_data.keys()))
            for k in list(sched_data.keys())[:5]:
                v = sched_data.get(k)
                print(f"- {k}: type={type(v)} preview={str(v)[:120]}")
        else:
            print("Schedule payload preview:", str(sched_data)[:500])
        raise RuntimeError("No events found in schedule payload.")

    ev = pick_event_from_schedule(events, known_event_id)
    if not ev:
        print(
            "Could not pick an event. First event preview:",
            json.dumps(events[0], indent=2)[:1000],
        )
        raise RuntimeError("Could not select an event from schedule.")

    # Extract location and start date (try multiple key names)
    lat = ev.get("latitude") or ev.get("lat") or ev.get("course_lat") or deep_get(ev, "course", "lat")
    lon = ev.get("longitude") or ev.get("lon") or ev.get("course_lon") or deep_get(ev, "course", "lon")
    start = ev.get("start") or ev.get("start_date") or deep_get(ev, "dates", "start")
    if lat is None or lon is None or not start:
        print("Event preview:\n", json.dumps(ev, indent=2)[:1000])
        raise ValueError(f"Schedule record missing lat/lon/start. Got lat={lat}, lon={lon}, start={start}")

    event_id = ev.get("event_id") or known_event_id
    event_name = ev.get("event_name") or ev.get("name")

    # Compute R1..R4 date range and fetch weather
    start_date, end_date, round_dates = to_round_dates(str(start))
    weather = fetch_open_meteo_hourly(float(lat), float(lon), start_date, end_date, tz="auto")

    weather_path = out_dir / f"event_{event_id}_weather_hourly.json"
    weather_path.write_text(json.dumps(weather, indent=2, ensure_ascii=False), encoding="utf-8")

    weather_meta = {
        "event_id": event_id,
        "event_name": event_name,
        "lat": float(lat),
        "lon": float(lon),
        "r1_date": round_dates[1],
        "r2_date": round_dates[2],
        "r3_date": round_dates[3],
        "r4_date": round_dates[4],
        "saved_at_utc": now_utc().strftime("%Y-%m-%dT%H%M%SZ"),
        "source": "schedule + open-meteo",
    }
    (out_dir / f"event_{event_id}_weather_meta.json").write_text(json.dumps(weather_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved hourly weather: {weather_path}")
    print(f"Saved weather meta: {out_dir / f'event_{event_id}_weather_meta.json'}")


if __name__ == "__main__":
    main()
