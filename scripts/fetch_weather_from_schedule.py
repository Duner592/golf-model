#!/usr/bin/env python3
# scripts/fetch_weather_from_schedule.py
import os
import json
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def parse_date(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def extract_events(s: Union[dict, list]) -> List[dict]:
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


def pick_event_from_schedule(
    events: List[dict], known_event_id: Optional[Union[str, int]] = None
) -> Optional[dict]:
    # Match known event_id first
    if known_event_id is not None:
        for e in events:
            if str(e.get("event_id")) == str(known_event_id):
                return e

    # Otherwise pick the next upcoming by start date, else first in list
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
        # If all are in the past, return the last one (most recent)
        return candidates[-1][1]

    # No parseable dates; return first event if available
    return events[0] if events else None


def to_round_dates(start_str: str) -> Tuple[str, str, Dict[int, str]]:
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


def fetch_open_meteo_hourly(
    lat: float, lon: float, start_date: str, end_date: str, tz: str = "auto"
) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation_probability",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": tz,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# -------------------------
# Main
# -------------------------


def main():
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

    # Optionally, try to use latest meta to match event_id
    known_event_id = None
    meta_dirs = [
        root / "data" / "meta" / tour,
        root / "data" / "processed" / tour,
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
    # If your schedule endpoint supports season/weeks filters, you can add them here:
    # params.update({"season": 2025, "weeks_ahead": 6})
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
        # Extra diagnostics to help adjust the parser/params
        print("Schedule payload top-level type:", type(sched_data))
        if isinstance(sched_data, dict):
            print("Top-level keys:", list(sched_data.keys()))
            # Print a short preview of values
            for k in list(sched_data.keys())[:5]:
                v = sched_data.get(k)
                print(f"- {k}: type={type(v)} preview={str(v)[:120]}")
        else:
            print("Schedule payload preview:", str(sched_data)[:500])
        raise RuntimeError("No events found in schedule payload.")

    ev = pick_event_from_schedule(events, known_event_id)
    if not ev:
        # Show first event to help adjust matching logic
        print(
            "Could not pick an event. First event preview:",
            json.dumps(events[0], indent=2)[:1000],
        )
        raise RuntimeError("Could not select an event from schedule.")

    # Extract location and start date (try multiple key names)
    lat = (
        ev.get("latitude")
        or ev.get("lat")
        or ev.get("course_lat")
        or deep_get(ev, "course", "lat")
    )
    lon = (
        ev.get("longitude")
        or ev.get("lon")
        or ev.get("course_lon")
        or deep_get(ev, "course", "lon")
    )
    start = ev.get("start") or ev.get("start_date") or deep_get(ev, "dates", "start")

    if lat is None or lon is None or not start:
        print("Event preview:\n", json.dumps(ev, indent=2)[:1000])
        raise ValueError(
            f"Schedule record missing lat/lon/start. Got lat={lat}, lon={lon}, start={start}"
        )

    event_id = ev.get("event_id") or known_event_id
    event_name = ev.get("event_name") or ev.get("name")

    # Compute R1..R4 date range and fetch weather
    start_date, end_date, round_dates = to_round_dates(str(start))
    weather = fetch_open_meteo_hourly(
        float(lat), float(lon), start_date, end_date, tz="auto"
    )

    out_dir = root / "data" / "processed" / tour
    out_dir.mkdir(parents=True, exist_ok=True)

    weather_path = out_dir / f"event_{event_id}_weather_hourly.json"
    weather_path.write_text(
        json.dumps(weather, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
    (out_dir / f"event_{event_id}_weather_meta.json").write_text(
        json.dumps(weather_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved hourly weather: {weather_path}")
    print(f"Saved weather meta: {out_dir / f'event_{event_id}_weather_meta.json'}")


if __name__ == "__main__":
    main()
