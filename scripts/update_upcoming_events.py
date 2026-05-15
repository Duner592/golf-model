#!/usr/bin/env python3
"""Refresh upcoming-events.json from DataGolf schedules.

The output intentionally matches the existing repo shape:

{
  "schedule": [...],
  "season": 2026,
  "tour": "all",
  "upcoming_only": "no"
}

By default this pulls the current UTC year/season for PGA and DPWT/Euro. Tours
already present in the file but not refreshed are preserved, so existing KFT/LIV
schedule entries are not silently dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOURS = ("pga", "euro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh upcoming-events.json from DataGolf schedules.")
    parser.add_argument("--season", action="append", type=int, help="Season/year to fetch. Defaults to current UTC year.")
    parser.add_argument("--tour", action="append", help="Tour to fetch. Repeat for multiple tours. Defaults to pga/euro/kft/liv.")
    parser.add_argument("--out", default="upcoming-events.json", help="Output path relative to the repo root.")
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    path = ROOT / "configs" / "datagolf.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def events_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict):
        for key in ("schedule", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [event for event in value if isinstance(event, dict)]
    return []


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def is_empty_value(value: Any, *, coordinate: bool = False) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned in {"", "tbd", "tbc", "unknown", "none", "null", "nan"}
    if coordinate and isinstance(value, (int, float)):
        return float(value) == 0.0
    return False


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return text[:10] if len(text) >= 10 else text


def normalize_winner(value: Any) -> str:
    if is_empty_value(value):
        return "TBD"
    return str(value)


def normalize_status(event: dict[str, Any], winner: str) -> str:
    raw = first_non_empty(event.get("status"), event.get("event_status"), event.get("state"))
    if raw is not None:
        cleaned = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned in {"complete", "completed", "final", "finished"}:
            return "completed"
        if cleaned in {"in_progress", "active", "live"}:
            return "in_progress"
        if cleaned in {"upcoming", "scheduled"}:
            return "upcoming"
        return cleaned
    return "completed" if winner != "TBD" else "upcoming"


def existing_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    events = payload.get("schedule") if isinstance(payload, dict) else None
    return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []


def event_keys(event: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    tour = str(event.get("tour") or "").lower()
    event_id = str(event.get("event_id") or "")
    name = str(event.get("event_name") or "").lower()
    start = str(event.get("start_date") or "")
    return [
        (tour, event_id, start, name),
        (tour, event_id, "", name),
        (tour, "", start, name),
    ]


def existing_lookup(events: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in events:
        for key in event_keys(event):
            lookup.setdefault(key, event)
    return lookup


def fallback_value(event: dict[str, Any], existing: dict[str, Any] | None, key: str, *, coordinate: bool = False) -> Any:
    value = event.get(key)
    if is_empty_value(value, coordinate=coordinate) and existing:
        prior = existing.get(key)
        if not is_empty_value(prior, coordinate=coordinate):
            return prior
    return value


def normalize_event(raw: dict[str, Any], tour: str, existing: dict[str, Any] | None) -> dict[str, Any]:
    event_name = first_non_empty(raw.get("event_name"), raw.get("name"), raw.get("tournament_name"), raw.get("event"))
    start_date = normalize_date(first_non_empty(raw.get("start_date"), raw.get("start"), raw.get("date")))
    winner = normalize_winner(first_non_empty(raw.get("winner"), raw.get("champion")))
    event = {
        "country": first_non_empty(raw.get("country"), raw.get("course_country"), raw.get("country_code"), "TBD"),
        "course": first_non_empty(raw.get("course"), raw.get("course_name"), raw.get("host_course"), "TBD"),
        "course_key": first_non_empty(raw.get("course_key"), raw.get("course_id"), raw.get("course_num"), raw.get("course"), "TBD"),
        "event_id": str(first_non_empty(raw.get("event_id"), raw.get("eventId"), raw.get("id"), "TBD")),
        "event_name": str(event_name or "TBD"),
        "latitude": coerce_float(first_non_empty(raw.get("latitude"), raw.get("lat"), raw.get("course_lat"), 0.0)) or 0.0,
        "location": first_non_empty(raw.get("location"), raw.get("city"), raw.get("course_location"), "TBD"),
        "longitude": coerce_float(first_non_empty(raw.get("longitude"), raw.get("lon"), raw.get("lng"), raw.get("course_lon"), 0.0)) or 0.0,
        "start_date": start_date or "TBD",
        "status": normalize_status(raw, winner),
        "tour": tour,
        "winner": winner,
    }

    for key in ("country", "course", "course_key", "location", "winner"):
        event[key] = fallback_value(event, existing, key)
    for key in ("latitude", "longitude"):
        event[key] = fallback_value(event, existing, key, coordinate=True)
    if event["winner"] != "TBD":
        event["status"] = "completed"
    elif existing and existing.get("winner") and not is_empty_value(existing.get("winner")):
        event["winner"] = existing["winner"]
        event["status"] = existing.get("status") or "completed"
    return event


def fetch_schedule(base_url: str, path: str, key_param: str, api_key: str, tour: str, season: int) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    params = {key_param: api_key, "tour": tour, "year": str(season)}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    events = events_from_payload(response.json())
    if events:
        return events

    # Some DataGolf schedule examples use "season"; keep a fallback so this
    # script remains compatible if the endpoint rejects/ignores "year".
    params = {key_param: api_key, "tour": tour, "season": str(season)}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return events_from_payload(response.json())


def sort_key(event: dict[str, Any]) -> tuple[str, str, int | str, str]:
    event_id = str(event.get("event_id", ""))
    try:
        event_id_key: int | str = int(event_id)
    except ValueError:
        event_id_key = event_id
    return str(event.get("start_date", "")), str(event.get("tour", "")), event_id_key, str(event.get("event_name", ""))


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    cfg = load_config()
    key_param = cfg["auth"]["key_param"]
    api_key = os.getenv(cfg["auth"]["env_var"])
    if not api_key:
        print(f"Missing API key in env var: {cfg['auth']['env_var']}", file=sys.stderr)
        sys.exit(1)

    seasons = args.season or [datetime.utcnow().year]
    tours = [tour.lower() for tour in (args.tour or DEFAULT_TOURS)]
    out_path = ROOT / args.out
    prior_events = existing_events(out_path)
    previous_lookup = existing_lookup(prior_events)

    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    failures: list[str] = []
    for season in seasons:
        for tour in tours:
            try:
                raw_events = fetch_schedule(
                    cfg["base_url"],
                    cfg["endpoints"]["schedule"]["path"],
                    key_param,
                    api_key,
                    tour,
                    season,
                )
            except Exception as exc:
                failures.append(f"{tour} {season}: {exc}")
                continue

            print(f"Fetched {len(raw_events)} events for {tour} {season}")
            for raw in raw_events:
                probe = {
                    "tour": tour,
                    "event_id": str(first_non_empty(raw.get("event_id"), raw.get("eventId"), raw.get("id"), "TBD")),
                    "event_name": str(first_non_empty(raw.get("event_name"), raw.get("name"), raw.get("tournament_name"), raw.get("event"), "")),
                    "start_date": normalize_date(first_non_empty(raw.get("start_date"), raw.get("start"), raw.get("date"))) or "",
                }
                existing = next((previous_lookup.get(key) for key in event_keys(probe) if previous_lookup.get(key)), None)
                event = normalize_event(raw, tour, existing)
                merged[event_keys(event)[0]] = event

    if not merged:
        for failure in failures:
            print(f"Fetch failed: {failure}", file=sys.stderr)
        raise RuntimeError("No schedule events fetched; leaving upcoming-events.json unchanged.")

    if failures:
        for failure in failures:
            print(f"Fetch failed: {failure}", file=sys.stderr)
        raise RuntimeError("One or more schedule fetches failed; leaving upcoming-events.json unchanged.")

    refreshed_tours = set(tours)
    for prior in prior_events:
        prior_tour = str(prior.get("tour") or "").lower()
        if prior_tour and prior_tour not in refreshed_tours:
            merged[event_keys(prior)[0]] = prior

    schedule = sorted(merged.values(), key=sort_key)
    payload: dict[str, Any] = {
        "schedule": schedule,
        "season": seasons[0] if len(seasons) == 1 else seasons,
        "tour": "all",
        "upcoming_only": "no",
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(schedule)} events to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
