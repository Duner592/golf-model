#!/usr/bin/env python3
# scripts/fetch_historical_rounds_single.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Fetch historical rounds for pinned event (single year).")
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--event_id", type=str, default=None, help="Event ID to fetch. Defaults to latest processed meta.")
    ap.add_argument("--tour", type=str, default=None, help="Tour to process. Defaults to datagolf.yaml.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(root / "configs" / "datagolf.yaml")
    base = cfg["base_url"].rstrip("/")
    keyp = cfg["auth"]["key_param"]
    envv = cfg["auth"]["env_var"]
    api_key = os.getenv(envv)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {envv}")

    tour = args.tour or cfg["defaults"]["tour"]
    endpoint = cfg["endpoints"]["historical_rounds"]["path"]
    event_id = args.event_id
    if event_id is None:
        proc = root / "data" / "processed" / tour
        metas = sorted(proc.glob("event_*_meta.json"))
        if not metas:
            raise FileNotFoundError("No pinned meta found. Pass --event_id for explicit historical fetches.")
        meta = json.loads(metas[-1].read_text(encoding="utf-8"))
        event_id = str(meta["event_id"])

    url = f"{base}/{endpoint.lstrip('/')}"
    params = {
        keyp: api_key,
        "tour": tour,
        "event_id": event_id,
        "year": str(args.year),
        "file_format": "json",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    out_dir = root / "data" / "raw" / "historical" / tour
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"event_{event_id}_{args.year}_rounds.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
