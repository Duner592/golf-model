#!/usr/bin/env python3
# scripts/update_archived_event.py
#
# Update an archived event's tournament_summary.json for a completed event.
# Usage: python scripts/update_archived_event.py --event_id <event_id>
#
# This script:
# - Checks if the event status is "completed" in upcoming-events.json; skips if "upcoming" unless --force is used.
# - Finds the archived tournament_summary.json for the given event_id, or skips if no archive exists.
# - Updates status to "completed".
# - Fetches the winner from DataGolf API, or falls back to upcoming-events.json if API fails.
# - Updates only the "winner" field (does not modify "previous_winners").
# - Overwrites the file with updated data.

import argparse
import json
import os
from pathlib import Path

import re

import requests


# ---------- helper functions ----------
def load_upcoming_events(root: Path) -> dict:
    """
    Load upcoming-events.json.

    Args:
        root (Path): Project root.

    Returns:
        dict: Upcoming events data.
    """
    upcoming_file = root / "upcoming-events.json"
    if not upcoming_file.exists():
        raise FileNotFoundError("upcoming-events.json not found")
    with open(upcoming_file, encoding="utf-8") as f:
        return json.load(f)


def get_event_details(event_id: str, upcoming_data: dict) -> dict | None:
    """
    Get event details from upcoming-events.json.

    Args:
        event_id (str): Event ID.
        upcoming_data (dict): Upcoming events data.

    Returns:
        dict or None: Event details.
    """
    for event in upcoming_data.get("schedule", []):
        if str(event.get("event_id")) == str(event_id):
            return event
    return None


def _schedule_events_from_payload(payload):
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict):
        for key in ("schedule", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [event for event in value if isinstance(event, dict)]
    return []


def fetch_winner_from_api(event_id: str, year: str, tour: str | None) -> str | None:
    """
    Fetch winner from DataGolf API.

    Args:
        event_id (str): Event ID.
        year (str): Year.

    Returns:
        str or None: Winner name.
    """
    try:
        if not tour:
            return None

        api_key = os.getenv("DATAGOLF_API_KEY")
        params_base = {"tour": tour}
        if api_key:
            params_base["key"] = api_key
        for year_key in ("year", "season"):
            params = dict(params_base)
            params[year_key] = year
            response = requests.get("https://feeds.datagolf.com/get-schedule", params=params, timeout=20)
            if response.status_code != 200:
                print(f"Warn: API response status {response.status_code} for {event_id} using {year_key}")
                continue
            for sched_event in _schedule_events_from_payload(response.json()):
                if str(sched_event.get("event_id")) == str(event_id):
                    winner = sched_event.get("winner")
                    print(f"DEBUG: API returned winner for {event_id}: {winner}")
                    return winner
        print(f"Warn: no winner found from API for {event_id}")
    except Exception as e:
        print(f"Warn: Failed to fetch winner from API: {e}")
    return None


def normalize_slug(name: str | None) -> str:
    """Create a filesystem-friendly slug for the event name."""
    if not isinstance(name, str):
        return ""
    txt = name.lower().strip()
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt.replace(" ", "_")


def find_archive_summary_path(root: Path, event_details: dict, year: str) -> Path:
    """Resolve the archived summary path, preferring the committed archive index."""
    event_id = str(event_details.get("event_id"))
    tour = str(event_details.get("tour", "")).lower()
    index_path = root / "web" / "archive" / "index.json"

    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index_data = []
        if isinstance(index_data, list):
            for entry in index_data:
                if not isinstance(entry, dict):
                    continue
                same_event = str(entry.get("event_id")) == event_id
                same_tour = str(entry.get("tour", "")).lower() == tour
                same_year = str(entry.get("year")) == str(year)
                slug = entry.get("slug")
                if same_event and same_tour and same_year and slug:
                    return root / "web" / "archive" / str(year) / str(slug) / "tournament_summary.json"

    slug = event_details.get("slug") or normalize_slug(event_details.get("event_name"))
    return root / "web" / "archive" / year / str(slug) / "tournament_summary.json"


def update_tournament_summary(ts_path: Path, winner: str | None) -> None:
    """
    Update the tournament_summary.json file.

    Args:
        ts_path (Path): Path to tournament_summary.json.
        winner (str or None): Winner name.
    """
    if not ts_path.exists():
        raise FileNotFoundError(f"Archived tournament_summary.json not found at {ts_path}")

    with open(ts_path, encoding="utf-8") as f:
        data = json.load(f)

    # Update status
    data["status"] = "completed"

    # Update winner
    data["winner"] = winner
    print(f"DEBUG: Setting winner to {winner}")

    # Write back
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {ts_path}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Update archived tournament_summary.json for a completed event.")
    ap.add_argument("--event_id", type=str, required=True, help="Event ID to update")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore local upcoming/in-progress status. Used by scheduled previous-week archive updates once a winner is available.",
    )
    args = ap.parse_args()

    event_id = args.event_id
    root = Path(__file__).resolve().parent.parent

    # Load upcoming events
    upcoming_data = load_upcoming_events(root)

    # Get event details
    event_details = get_event_details(event_id, upcoming_data)
    if not event_details:
        print(f"Error: Event ID {event_id} not found in upcoming-events.json")
        return

    local_status = str(event_details.get("status", "")).lower()
    if local_status in {"upcoming", "in_progress", "in-progress"} and not args.force:
        print(f"Skipping: Event {event_id} is still upcoming or in progress.")
        return

    event_name = event_details.get("event_name")
    start_date = event_details.get("start_date")
    if not start_date:
        print(f"Error: No start_date for event {event_id}")
        return

    # Extract year
    year = start_date.split("-")[0]

    # Build archive path
    archive_path = find_archive_summary_path(root, event_details, year)
    if not archive_path.exists():
        print(
            "Skipping: archived tournament_summary.json not found for "
            f"event {event_id} ({event_name}) at {archive_path}"
        )
        return

    # Fetch winner from API, fallback to upcoming-events.json
    tour = event_details.get("tour")
    winner = fetch_winner_from_api(event_id, year, tour)
    if winner is None:
        winner = event_details.get("winner")
        print(f"DEBUG: Using fallback winner from upcoming-events.json: {winner}")
    if args.force and local_status in {"upcoming", "in_progress", "in-progress"} and winner is None:
        print(f"Skipping: Event {event_id} is locally {local_status!r} and no winner is available yet.")
        return

    # Update the file
    update_tournament_summary(archive_path, winner)

    print(f"Successfully updated archived event {event_id} ({event_name}) for {year}")


if __name__ == "__main__":
    main()
