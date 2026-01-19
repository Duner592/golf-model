#!/usr/bin/env python3
# scripts/update_archived_event.py
#
# Update an archived event's tournament_summary.json for a completed event.
# Usage: python scripts/update_archived_event.py --event_id <event_id>
#
# This script:
# - Checks if the event status is "completed" in upcoming-events.json; skips if "upcoming".
# - Finds the archived tournament_summary.json for the given event_id.
# - Updates status to "completed".
# - Fetches the winner from DataGolf API, or falls back to upcoming-events.json if API fails.
# - Updates only the "winner" field (does not modify "previous_winners").
# - Overwrites the file with updated data.

import argparse
import json
from pathlib import Path

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


def fetch_winner_from_api(event_id: str, year: str) -> str | None:
    """
    Fetch winner from DataGolf API.

    Args:
        event_id (str): Event ID.
        year (str): Year.

    Returns:
        str or None: Winner name.
    """
    try:
        response = requests.get(f"https://datagolf.ca/api/get-schedule?season={year}")
        if response.status_code == 200:
            schedule_data = response.json()
            for sched_event in schedule_data.get("schedule", []):
                if str(sched_event.get("event_id")) == str(event_id):
                    winner = sched_event.get("winner")
                    print(f"DEBUG: API returned winner for {event_id}: {winner}")
                    return winner
        print(f"Warn: API response status {response.status_code} or no winner found for {event_id}")
    except Exception as e:
        print(f"Warn: Failed to fetch winner from API: {e}")
    return None


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

    # Check status
    if event_details.get("status") == "upcoming":
        print(f"Skipping: Event {event_id} is still upcoming or in progress.")
        return

    event_name = event_details.get("event_name")
    start_date = event_details.get("start_date")
    if not start_date:
        print(f"Error: No start_date for event {event_id}")
        return

    # Extract year and slug
    year = start_date.split("-")[0]
    slug = event_name.lower().replace(" ", "_")  # Simple slugify

    # Build archive path
    archive_path = root / "web" / "archive" / year / slug / "tournament_summary.json"

    # Fetch winner from API, fallback to upcoming-events.json
    winner = fetch_winner_from_api(event_id, year)
    if winner is None:
        winner = event_details.get("winner")
        print(f"DEBUG: Using fallback winner from upcoming-events.json: {winner}")

    # Update the file
    update_tournament_summary(archive_path, winner)

    print(f"Successfully updated archived event {event_id} ({event_name}) for {year}")


if __name__ == "__main__":
    main()
