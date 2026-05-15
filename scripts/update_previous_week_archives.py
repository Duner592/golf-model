#!/usr/bin/env python3
"""Update archive summaries for events that started in the previous week.

The default previous-week window is Monday through Sunday before the reference
date. This is intended for Monday archive-update runs after the prior week's
events have completed.

Usage:
    python scripts/update_previous_week_archives.py --dry-run
    python scripts/update_previous_week_archives.py
    python scripts/update_previous_week_archives.py --date 2026-05-18 --tour pga
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOURS = ("pga", "euro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update archived events from the previous Monday-Sunday window.")
    parser.add_argument("--date", help="Reference date in YYYY-MM-DD format. Defaults to today's date.")
    parser.add_argument("--tour", action="append", choices=DEFAULT_TOURS, help="Tour to include. Repeat for multiple tours.")
    parser.add_argument("--dry-run", action="store_true", help="Print the matching event IDs and commands without updating files.")
    parser.add_argument("--completed-only", action="store_true", help="Only include events marked completed in upcoming-events.json.")
    return parser.parse_args()


def parse_reference_date(raw: str | None) -> date:
    if not raw:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()


def previous_week_window(reference_date: date) -> tuple[date, date]:
    current_monday = reference_date - timedelta(days=reference_date.weekday())
    previous_monday = current_monday - timedelta(days=7)
    previous_sunday = current_monday - timedelta(days=1)
    return previous_monday, previous_sunday


def load_schedule() -> list[dict[str, Any]]:
    path = ROOT / "upcoming-events.json"
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    schedule = data.get("schedule")
    if not isinstance(schedule, list):
        raise ValueError(f"Expected upcoming-events.json schedule to be a list: {path}")
    return [event for event in schedule if isinstance(event, dict)]


def event_start_date(event: dict[str, Any]) -> date | None:
    raw = event.get("start_date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).date()
    except ValueError:
        return None


def matching_events(reference_date: date, tours: set[str], completed_only: bool) -> list[dict[str, Any]]:
    start, end = previous_week_window(reference_date)
    matches: list[dict[str, Any]] = []
    for event in load_schedule():
        tour = str(event.get("tour", "")).lower()
        if tour not in tours:
            continue
        started = event_start_date(event)
        if started is None or not start <= started <= end:
            continue
        if completed_only and str(event.get("status", "")).lower() != "completed":
            continue
        matches.append(event)

    def sort_key(event: dict[str, Any]) -> tuple[date, str, int | str]:
        started = event_start_date(event) or date.min
        event_id = str(event.get("event_id", ""))
        try:
            event_id_key: int | str = int(event_id)
        except ValueError:
            event_id_key = event_id
        return started, str(event.get("tour", "")), event_id_key

    matches.sort(key=sort_key)
    return matches


def print_event(event: dict[str, Any]) -> None:
    event_id = event.get("event_id")
    tour = event.get("tour")
    start_date = event.get("start_date")
    name = event.get("event_name")
    status = event.get("status")
    print(f"{tour} {start_date} event_id={event_id} status={status} - {name}")


def main() -> None:
    args = parse_args()
    reference_date = parse_reference_date(args.date)
    tours = {tour.lower() for tour in (args.tour or DEFAULT_TOURS)}
    start, end = previous_week_window(reference_date)

    events = matching_events(reference_date, tours, args.completed_only)
    print(f"Previous-week window: {start.isoformat()} to {end.isoformat()}")
    print(f"Tours: {', '.join(sorted(tours))}")

    if not events:
        print("No matching previous-week events found.")
        return

    for event in events:
        print_event(event)
        event_id = str(event.get("event_id"))
        cmd = [sys.executable, str(ROOT / "scripts" / "update_archived_event.py"), "--event_id", event_id, "--force"]
        print("Command:", f"python scripts/update_archived_event.py --event_id {event_id} --force")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
