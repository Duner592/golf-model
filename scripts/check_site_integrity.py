#!/usr/bin/env python3
"""Check generated site assets for stale or inconsistent model state.

This script is intentionally read-only. It is useful locally before deploys and
in CI after web assets have been rebuilt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOURS = ("pga", "euro")
UTC = timezone.utc


@dataclass
class Finding:
    severity: str
    code: str
    message: str


class IntegrityCheck:
    def __init__(self, *, strict_status_age_hours: int, archive_lookback_days: int) -> None:
        self.strict_status_age_hours = strict_status_age_hours
        self.archive_lookback_days = archive_lookback_days
        self.findings: list[Finding] = []

    def error(self, code: str, message: str) -> None:
        self.findings.append(Finding("error", code, message))

    def warn(self, code: str, message: str) -> None:
        self.findings.append(Finding("warning", code, message))

    def read_json(self, rel_path: str) -> Any | None:
        path = ROOT / rel_path
        if not path.exists():
            self.error("missing-file", f"{rel_path} is missing")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.error("invalid-json", f"{rel_path} is not valid JSON: {exc}")
            return None

    def check(self) -> list[Finding]:
        upcoming = self.read_json("upcoming-events.json")
        status = self.read_json("web/status.json")
        archive_index = self.read_json("web/archive/index.json")

        schedule = upcoming.get("schedule", []) if isinstance(upcoming, dict) else []
        archive_entries = archive_index if isinstance(archive_index, list) else []
        if archive_index is not None and not isinstance(archive_index, list):
            self.error("archive-index-shape", "web/archive/index.json must be a list")

        self.check_status(status)
        self.check_tours(schedule, status)
        self.check_archive(schedule, archive_entries)
        return self.findings

    def check_status(self, status: Any | None) -> None:
        if not isinstance(status, dict):
            return
        updated_raw = status.get("updated_at_utc")
        updated_at = parse_datetime(updated_raw)
        if not updated_at:
            self.warn("status-updated-missing", "web/status.json has no parseable updated_at_utc")
            return

        age = datetime.now(UTC) - updated_at
        if age > timedelta(hours=self.strict_status_age_hours):
            self.warn(
                "status-stale",
                f"web/status.json is {format_timedelta(age)} old; latest update was {updated_at.isoformat()}",
            )

    def check_tours(self, schedule: list[Any], status: Any | None) -> None:
        active_by_tour = {tour: current_events(schedule, tour) for tour in TOURS}
        status_runs = status.get("model_runs", {}) if isinstance(status, dict) else {}

        for tour in TOURS:
            meta = self.read_json(f"web/{tour}/meta.json")
            if not isinstance(meta, dict):
                continue

            active_events = active_by_tour[tour]
            published = published_events(meta)
            status_run = status_runs.get(tour, {}) if isinstance(status_runs, dict) else {}
            status_ids = {str(eid) for eid in status_run.get("event_ids", [])} if isinstance(status_run, dict) else set()
            if isinstance(status_run, dict) and status_run.get("event_id"):
                status_ids.add(str(status_run["event_id"]))

            if active_events:
                active_ids = {event.event_id for event in active_events}
                published_ids = set(published)
                missing = active_ids - published_ids
                if missing:
                    self.error(
                        "active-event-mismatch",
                        f"{tour}: DataGolf schedule current-week event(s) {sorted(active_ids)} "
                        f"but web/{tour}/meta.json publishes {sorted(published_ids) or 'none'}",
                    )

                no_prediction = [eid for eid, has_predictions in published.items() if eid in active_ids and not has_predictions]
                if no_prediction:
                    self.error(
                        "active-event-no-predictions",
                        f"{tour}: active event(s) {no_prediction} are published without predictions",
                    )

                if status_ids and status_ids != active_ids:
                    self.warn(
                        "status-model-run-mismatch",
                        f"{tour}: web/status.json model run event(s) {sorted(status_ids)} "
                        f"do not match current-week schedule {sorted(active_ids)}",
                    )
            elif published:
                self.warn(
                    "published-without-current-event",
                    f"{tour}: web/{tour}/meta.json publishes {sorted(published)} but no current-week event was found",
                )

    def check_archive(self, schedule: list[Any], archive_entries: list[Any]) -> None:
        archive_by_id: dict[str, dict[str, Any]] = {}
        for entry in archive_entries:
            if not isinstance(entry, dict):
                self.error("archive-entry-shape", "web/archive/index.json contains a non-object entry")
                continue
            event_id = str(entry.get("event_id", "")).strip()
            if not event_id:
                self.error("archive-entry-id-missing", f"Archive entry has no event_id: {entry}")
                continue
            archive_by_id[event_id] = entry
            self.check_archive_entry_files(entry)

        for event in schedule:
            if not isinstance(event, dict):
                continue
            tour = str(event.get("tour", "")).lower()
            if tour not in TOURS:
                continue
            event_id = str(event.get("event_id", "")).strip()
            if not event_id or event_id.upper() == "TBD":
                continue
            status = str(event.get("status", "")).lower()
            start = parse_date(event.get("start_date"))
            if status == "completed":
                if start and start > date.today():
                    self.warn(
                        "completed-event-future-start",
                        f"{tour} {event_id} is marked completed but starts {start.isoformat()}",
                    )
                if event_id not in archive_by_id and self.should_expect_archive(start):
                    self.warn(
                        "completed-event-missing-archive",
                        f"{tour} {event_id} {event.get('event_name', '')!r} is completed but missing from web/archive/index.json",
                    )
            elif start and start + timedelta(days=4) < date.today() and status == "upcoming":
                self.warn(
                    "past-event-still-upcoming",
                    f"{tour} {event_id} {event.get('event_name', '')!r} started {start.isoformat()} but is still marked upcoming",
                )

    def check_archive_entry_files(self, entry: dict[str, Any]) -> None:
        year = str(entry.get("year", "")).strip()
        slug = str(entry.get("slug", "")).strip()
        event_id = str(entry.get("event_id", "")).strip()
        if not year or not slug:
            self.error("archive-entry-path-missing", f"Archive event {event_id} has no year/slug")
            return

        event_dir = ROOT / "web" / "archive" / year / slug
        if not event_dir.exists():
            self.error("archive-dir-missing", f"Archive event {event_id} points to missing {event_dir.relative_to(ROOT)}")
            return

        leaderboard_json = event_dir / "leaderboard.json"
        leaderboard_csv = event_dir / "leaderboard.csv"
        if not leaderboard_json.exists() and not leaderboard_csv.exists():
            self.error("archive-leaderboard-missing", f"Archive event {event_id} has no leaderboard JSON or CSV")

        start = parse_date(entry.get("start_date"))
        if start and start + timedelta(days=4) < date.today() and not (event_dir / "results.json").exists():
            self.warn(
                "archive-results-missing",
                f"Archive event {event_id} {entry.get('event_name', '')!r} started {start.isoformat()} but has no results.json",
            )

    def should_expect_archive(self, start: date | None) -> bool:
        if not start:
            return True
        if self.archive_lookback_days <= 0:
            return True
        return start >= date.today() - timedelta(days=self.archive_lookback_days)


@dataclass(frozen=True)
class ScheduleEvent:
    event_id: str
    event_name: str
    start_date: date


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw[:-1] + "+00:00")
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def current_window(today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    if today.weekday() == 6:
        end = start + timedelta(days=13)
    return start, end


def current_events(schedule: list[Any], tour: str) -> list[ScheduleEvent]:
    start, end = current_window()
    events: list[ScheduleEvent] = []
    for event in schedule:
        if not isinstance(event, dict):
            continue
        if str(event.get("tour", "")).lower() != tour:
            continue
        event_id = str(event.get("event_id", "")).strip()
        event_date = parse_date(event.get("start_date"))
        if not event_id or event_id.upper() == "TBD" or not event_date:
            continue
        if start <= event_date <= end:
            events.append(ScheduleEvent(event_id, str(event.get("event_name", "")), event_date))
    return events


def published_events(meta: dict[str, Any]) -> dict[str, bool]:
    active_events = meta.get("active_events")
    if isinstance(active_events, list) and active_events:
        published: dict[str, bool] = {}
        for event in active_events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id", "")).strip()
            if event_id and event_id != "0":
                published[event_id] = bool(event.get("has_predictions", False))
        return published

    event_id = str(meta.get("event_id", "")).strip()
    if event_id and event_id != "0":
        return {event_id: bool(meta.get("resources", {}).get("leaderboard"))}
    return {}


def format_timedelta(delta: timedelta) -> str:
    total_hours = int(delta.total_seconds() // 3600)
    days, hours = divmod(total_hours, 24)
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check generated web assets for site integrity issues.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when warnings are present as well as errors.",
    )
    parser.add_argument(
        "--status-age-hours",
        type=int,
        default=24,
        help="Warn when web/status.json is older than this many hours.",
    )
    parser.add_argument(
        "--archive-lookback-days",
        type=int,
        default=45,
        help="Warn about completed schedule events missing archives only within this many days; use 0 for all.",
    )
    args = parser.parse_args()

    checker = IntegrityCheck(
        strict_status_age_hours=args.status_age_hours,
        archive_lookback_days=args.archive_lookback_days,
    )
    findings = checker.check()
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]

    if not findings:
        print("Site integrity check passed.")
        return 0

    print(f"Site integrity check found {len(errors)} error(s), {len(warnings)} warning(s).")
    for finding in findings:
        print(f"[{finding.severity}] {finding.code}: {finding.message}")

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
