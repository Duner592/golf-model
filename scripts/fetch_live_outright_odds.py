#!/usr/bin/env python3
"""Fetch current DataGolf outright odds for the published Odds & Value board.

Only UK bookmaker prices required by the site are retained. The DataGolf key is
read from the configured environment variable and is never written to output.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

from request_safety import raise_for_status_safely


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "datagolf.yaml"
OUTPUT_PATH = ROOT / "web" / "odds" / "current.json"
BOOKS = ("bet365", "skybet")


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fetch_tour(config: dict, tour: str, api_key: str) -> dict:
    endpoint = config["endpoints"]["live_outright_odds"]["path"]
    url = f"{config['base_url'].rstrip('/')}/{endpoint}"
    params = {
        config["auth"]["key_param"]: api_key,
        "tour": tour,
        "market": "win",
        "odds_format": "fraction",
    }
    response = requests.get(url, params=params, timeout=30)
    raise_for_status_safely(response)
    payload = response.json()
    odds = payload.get("odds") if isinstance(payload, dict) else None
    if not isinstance(odds, list):
        raise ValueError(f"DataGolf odds response for {tour} has no odds list")

    rows = []
    for row in odds:
        if not isinstance(row, dict):
            continue
        prices = {book: str(row[book]).strip() for book in BOOKS if str(row.get(book) or "").strip()}
        if not prices:
            continue
        rows.append({
            "player_name": row.get("player_name", ""),
            "dg_id": row.get("dg_id"),
            "prices": prices,
        })
    return {
        "tour": tour,
        "event_name": payload.get("event_name", "Unknown event"),
        "market": payload.get("market", "win"),
        "last_updated": payload.get("last_updated", ""),
        "books": list(BOOKS),
        "odds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live Bet365 and Sky Bet outright odds from DataGolf.")
    parser.add_argument("--tour", choices=("pga", "euro", "both"), default="both")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config()
    api_key = os.getenv(config["auth"]["env_var"])
    if not api_key:
        raise RuntimeError(f"Missing API key in environment: {config['auth']['env_var']}")

    tours = ("pga", "euro") if args.tour == "both" else (args.tour,)
    events = [fetch_tour(config, tour, api_key) for tour in tours]
    output = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "DataGolf betting-tools/outrights",
        "events": events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.relative_to(ROOT)} for {len(events)} tour(s).")


if __name__ == "__main__":
    main()
