#!/usr/bin/env python3
"""Fail before a Pages deploy would publish stale live model assets.

Most live model files under web/{tour}/ are generated in GitHub Actions and are
not committed back to master. Non-model workflows that deploy the whole web/
tree must therefore verify that the checked-out assets already match the active
week's events; otherwise they can overwrite a good model deployment with stale
checked-in files.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOURS = ("pga", "euro")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def current_window() -> tuple[datetime.date, datetime.date]:
    today = datetime.now().date()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    if today.weekday() == 6:
        end = start + timedelta(days=13)
    return start, end


def current_event_ids(tour: str) -> list[str]:
    upcoming_path = ROOT / "upcoming-events.json"
    if not upcoming_path.exists():
        return []

    start, end = current_window()
    payload = read_json(upcoming_path)
    events = payload.get("schedule", []) if isinstance(payload, dict) else []
    ids: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("tour", "")).lower() != tour:
            continue
        event_id = str(event.get("event_id", "")).strip()
        start_date = event.get("start_date")
        if not event_id or event_id.upper() == "TBD" or not start_date:
            continue
        try:
            event_date = datetime.fromisoformat(str(start_date)).date()
        except ValueError:
            continue
        if start <= event_date <= end and event_id not in ids:
            ids.append(event_id)
    return ids


def published_event_map(tour: str) -> dict[str, bool]:
    meta_path = ROOT / "web" / tour / "meta.json"
    if not meta_path.exists():
        return {}

    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        return {}

    active_events = meta.get("active_events")
    if isinstance(active_events, list) and active_events:
        out: dict[str, bool] = {}
        for event in active_events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id", "")).strip()
            if event_id and event_id != "0":
                out[event_id] = bool(event.get("has_predictions", False))
        return out

    event_id = str(meta.get("event_id", "")).strip()
    if event_id and event_id != "0":
        leaderboard_path = ROOT / "web" / tour / "leaderboard.json"
        has_predictions = False
        if leaderboard_path.exists():
            try:
                leaderboard = read_json(leaderboard_path)
                has_predictions = isinstance(leaderboard, list) and bool(leaderboard)
            except Exception:
                has_predictions = False
        return {event_id: has_predictions}
    return {}


def main() -> int:
    failures: list[str] = []
    for tour in TOURS:
        expected_ids = current_event_ids(tour)
        if not expected_ids:
            continue

        published = published_event_map(tour)
        missing = [event_id for event_id in expected_ids if not published.get(event_id)]
        if missing:
            failures.append(
                f"{tour}: expected active model event(s) {', '.join(expected_ids)} "
                f"but checked-out web assets contain {', '.join(published) or 'none'}"
            )

    if failures:
        print("::error::Refusing to deploy stale Pages model assets.")
        for failure in failures:
            print(f"::error::{failure}")
        print("::error::Run the scheduled model workflow successfully, or rebuild all active model assets before deploying web/.")
        return 1

    print("Current checked-out model assets match active scheduled events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
