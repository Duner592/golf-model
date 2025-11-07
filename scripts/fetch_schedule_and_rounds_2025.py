#!/usr/bin/env python3
# scripts/fetch_schedule_and_rounds_2025.py
import argparse
import json
import os
from pathlib import Path

import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()
TOUR = "pga"


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Fetch 2025 schedule and per-event round JSONs.")
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(root / "configs" / "datagolf.yaml")

    base_url = cfg["base_url"].rstrip("/")
    keyp = cfg["auth"]["key_param"]
    api_key = os.getenv(cfg["auth"]["env_var"])
    if not api_key:
        raise RuntimeError("Missing API key")
    tour = cfg["defaults"]["tour"]
    sched_path = cfg["endpoints"]["schedule"]["path"]
    hist_path = cfg["endpoints"]["historical_rounds"]["path"]  # make sure this exists in your YAML

    requests_cache.install_cache("dg_cache", expire_after=900)
    # 1) schedule for 2025
    url = f"{base_url}/{sched_path.lstrip('/')}"
    r = requests.get(url, params={keyp: api_key, "tour": tour, "year": str(args.year)}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    events = []
    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict):
        for k in ("events", "schedule", "data"):
            v = payload.get(k)
            if isinstance(v, list):
                events = v
                break
    if not events:
        raise RuntimeError("No events in schedule")

    out_dir = root / "data" / "raw" / "historical" / tour
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Download each event's 2025 round JSON
    for e in events:
        eid = e.get("event_id")
        ename = e.get("event_name") or e.get("name")
        if not eid:
            continue
        print(f"[schedule] {eid} • {ename}")
        url_h = f"{base_url}/{hist_path.lstrip('/')}"
        params = {
            keyp: api_key,
            "tour": tour,
            "event_id": str(eid),
            "year": str(args.year),
            "file_format": "json",
        }
        try:
            rh = requests.get(url_h, params=params, timeout=60)
            if rh.status_code in (400, 404):
                print(f"  -> skip: HTTP {rh.status_code}")
                continue
            rh.raise_for_status()
            data = rh.json()
            out = out_dir / f"event_{eid}_{args.year}_rounds.json"
            out.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  -> saved: {out}")
        except requests.HTTPError as ex:
            print(f"  -> error: {ex}")

    # Save a simple event list snapshot for convenience
    (root / "data" / "raw" / f"schedule_{tour}_{args.year}.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
