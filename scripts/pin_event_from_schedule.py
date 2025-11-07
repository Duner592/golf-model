#!/usr/bin/env python3
# scripts/pin_event_from_schedule.py
# Pin a past (or future) event (by name + year) as the active meta:
# - Queries schedule for the given year
# - Fuzzy-matches the event name
# - Writes data/processed/{tour}/event_{event_id}_meta.json with lat/lon and R1–R4 dates

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()
TOUR = "pga"


def slugify(s: str) -> str:
    s = (s or "").lower()
    # FIX: pass the input string to re.sub
    s0 = re.sub(r"[^a-z0-9]+", " ", s)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def schedule_events_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("events", "schedule", "data"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
    return []


def main():
    ap = argparse.ArgumentParser(description="Pin a past/future event (by name + year) as the active meta.")
    ap.add_argument(
        "--name",
        required=True,
        help='Event name (or substring), e.g. "Bank of Utah Championship"',
    )
    ap.add_argument("--year", required=True, type=int, help="Year (e.g., 2024)")
    ap.add_argument("--tour", default=TOUR, help="Tour key (default: pga)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(root / "configs" / "datagolf.yaml")

    base = cfg["base_url"].rstrip("/")
    keyp = cfg["auth"]["key_param"]
    envv = cfg["auth"]["env_var"]
    api_key = os.getenv(envv)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {envv}")

    tour = args.tour
    sched_path = cfg["endpoints"]["schedule"]["path"]

    # Fetch the schedule for the given year
    requests_cache.install_cache("dg_cache", expire_after=600)
    url = f"{base}/{sched_path.lstrip('/')}"
    resp = requests.get(url, params={keyp: api_key, "tour": tour, "year": str(args.year)}, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("HTTP error:", e)
        print("Status:", getattr(e.response, "status_code", "unknown"))
        print("Body:", getattr(e.response, "text", "")[:800])
        raise

    events = schedule_events_from_payload(resp.json())
    if not events:
        raise RuntimeError(f"No schedule events returned for year={args.year} tour={tour}")

    target_slug = slugify(args.name)
    matches = []
    for ev in events:
        nm = ev.get("event_name") or ev.get("name") or ""
        if target_slug in slugify(nm):
            matches.append(ev)

    if not matches:
        sample = [e.get("event_name") for e in events[:10]]
        raise RuntimeError(f'No match for "{args.name}" in {args.year}. Sample schedule names: {sample}')

    # If multiple matches, pick the first; you can refine selection if needed
    ev = matches[0]
    event_id = ev.get("event_id")
    event_name = ev.get("event_name") or args.name

    # Try common keys for lat/lon and start date
    lat = ev.get("latitude") or ev.get("lat") or ev.get("course_lat")
    lon = ev.get("longitude") or ev.get("lon") or ev.get("course_lon")
    start = ev.get("start") or ev.get("start_date")

    if not event_id:
        raise ValueError("schedule record missing event_id")
    if lat is None or lon is None:
        raise ValueError("schedule record missing latitude/longitude")
    if not start:
        raise ValueError("schedule record missing start/start_date")

    try:
        d0 = datetime.strptime(str(start), "%Y-%m-%d").date()
    except Exception:
        raise ValueError(f"Unparseable start date in schedule record: {start}") from None

    dates = {
        "r1_date": d0.isoformat(),
        "r2_date": (d0 + timedelta(days=1)).isoformat(),
        "r3_date": (d0 + timedelta(days=2)).isoformat(),
        "r4_date": (d0 + timedelta(days=3)).isoformat(),
    }

    meta = {
        "event_id": event_id,
        "event_name": event_name,
        "lat": float(lat),
        "lon": float(lon),
        **dates,
        "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ"),
        "source": f"schedule pin ({args.year})",
    }

    processed = root / "data" / "processed" / tour
    processed.mkdir(parents=True, exist_ok=True)
    out = processed / f"event_{event_id}_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Pinned event meta:", out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
