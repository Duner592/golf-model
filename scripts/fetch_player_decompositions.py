#!/usr/bin/env python3
"""Publish current-event DataGolf skill decompositions for Player Drilldown.

The values are reference data from DataGolf, deliberately kept separate from
the project's own win/top-10/make-cut model outputs.
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
OUTPUT_PATH = ROOT / "web" / "player_decompositions" / "current.json"
PLAYER_FIELDS = (
    "dg_id",
    "player_name",
    "baseline_pred",
    "final_pred",
    "total_fit_adjustment",
    "total_course_history_adjustment",
    "course_experience_adjustment",
    "driving_distance_adjustment",
    "driving_accuracy_adjustment",
    "cf_approach_comp",
    "cf_short_comp",
    "other_fit_adjustment",
    "strokes_gained_category_adjustment",
    "age_adjustment",
    "country_adjustment",
    "timing_adjustment",
    "sample_size",
)
SKILL_FIELDS = (
    "driving_acc",
    "driving_dist",
    "sg_ott",
    "sg_app",
    "sg_arg",
    "sg_putt",
    "sg_total",
)


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fetch_skill_ratings(config: dict, api_key: str) -> dict[str, dict]:
    endpoint = config["endpoints"]["skill_ratings"]["path"]
    response = requests.get(
        f"{config['base_url'].rstrip('/')}/{endpoint}",
        params={config["auth"]["key_param"]: api_key, "display": "value", "file_format": "json"},
        timeout=30,
    )
    raise_for_status_safely(response)
    payload = response.json()
    players = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(players, list):
        raise ValueError("DataGolf skill ratings response has no players list")
    return {
        str(row["dg_id"]): {field: row.get(field) for field in SKILL_FIELDS if field in row}
        for row in players
        if isinstance(row, dict) and row.get("dg_id")
    }


def fetch_tour(config: dict, tour: str, api_key: str, skills_by_id: dict[str, dict]) -> dict:
    endpoint = config["endpoints"]["player_decompositions"]["path"]
    response = requests.get(
        f"{config['base_url'].rstrip('/')}/{endpoint}",
        params={config["auth"]["key_param"]: api_key, "tour": tour, "file_format": "json"},
        timeout=30,
    )
    raise_for_status_safely(response)
    payload = response.json()
    players = payload.get("players") if isinstance(payload, dict) else None
    if not isinstance(players, list):
        raise ValueError(f"DataGolf decomposition response for {tour} has no players list")
    return {
        "tour": tour,
        "event_name": payload.get("event_name", "Unknown event"),
        "course_name": payload.get("course_name", ""),
        "last_updated": payload.get("last_updated", ""),
        "players": [
            {
                **{field: row.get(field) for field in PLAYER_FIELDS if field in row},
                **({"skill_ratings": skills_by_id[str(row.get("dg_id"))]} if str(row.get("dg_id")) in skills_by_id else {}),
            }
            for row in players
            if isinstance(row, dict) and row.get("player_name")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch current DataGolf player decompositions.")
    parser.add_argument("--tour", choices=("pga", "euro", "both"), default="both")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config()
    api_key = os.getenv(config["auth"]["env_var"])
    if not api_key:
        raise RuntimeError(f"Missing API key in environment: {config['auth']['env_var']}")

    existing = {}
    if args.output.exists():
        try:
            existing = {item.get("tour"): item for item in json.loads(args.output.read_text(encoding="utf-8")).get("events", []) if isinstance(item, dict) and item.get("tour")}
        except (json.JSONDecodeError, AttributeError):
            existing = {}
    tours = ("pga", "euro") if args.tour == "both" else (args.tour,)
    skills_by_id = fetch_skill_ratings(config, api_key)
    for tour in tours:
        existing[tour] = fetch_tour(config, tour, api_key, skills_by_id)
    output = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "DataGolf player-decompositions (reference data; not this site's model)",
        "events": [existing[tour] for tour in ("pga", "euro") if tour in existing],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.relative_to(ROOT)} for {len(tours)} tour(s).")


if __name__ == "__main__":
    main()
