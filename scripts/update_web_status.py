#!/usr/bin/env python3
"""Update the public web/status.json health snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = ROOT / "web" / "status.json"


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_now() -> str:
    return format_utc(datetime.now(timezone.utc))


def file_mtime_utc(path: Path) -> str:
    return format_utc(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))


def git_relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def git_file_is_dirty(path: Path) -> bool:
    try:
        rel_path = git_relative_path(path)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", rel_path],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def git_last_commit_utc(path: Path) -> str | None:
    try:
        rel_path = git_relative_path(path)
    except ValueError:
        return None
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", rel_path],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    if not value:
        return None
    try:
        return format_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {"model_runs": {}, "schedule": {}, "betting_data": {}}
    try:
        data = read_json(STATUS_PATH)
    except Exception:
        return {"model_runs": {}, "schedule": {}, "betting_data": {}}
    if not isinstance(data, dict):
        return {"model_runs": {}, "schedule": {}, "betting_data": {}}
    data.setdefault("model_runs", {})
    data.setdefault("schedule", {})
    data.setdefault("betting_data", {})
    return data


def write_status(status: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at_utc"] = utc_now()
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Updated {STATUS_PATH.relative_to(ROOT)}")


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(event.get("event_id", "")),
        "event_name": event.get("event_name") or "Unknown Event",
        "start_date": event.get("start_date") or event.get("r1_date"),
        "generated_utc": event.get("generated_utc"),
        "has_predictions": bool(event.get("has_predictions", False)),
    }


def load_tour_meta(tour: str) -> dict[str, Any]:
    meta_path = ROOT / "web" / tour / "meta.json"
    if not meta_path.exists():
        return {}
    data = read_json(meta_path)
    return data if isinstance(data, dict) else {}


def update_model_run(status: dict[str, Any], tour: str, workflow: str | None) -> None:
    tour_key = tour.lower()
    meta = load_tour_meta(tour_key)
    active_events = meta.get("active_events")
    events = [compact_event(event) for event in active_events if isinstance(event, dict)] if isinstance(active_events, list) else []
    if not events and meta:
        events = [compact_event(meta)]

    event_ids = [event["event_id"] for event in events if event.get("event_id") and event.get("event_id") != "0"]
    status["model_runs"][tour_key] = {
        "last_run_utc": utc_now(),
        "tour": tour_key,
        "event_id": str(meta.get("event_id", "")) if meta else "",
        "event_name": meta.get("event_name") if meta else None,
        "event_ids": event_ids,
        "events": events,
        "artifact_generated_utc": meta.get("generated_utc") if meta else None,
        "workflow": workflow,
    }


def sync_model_from_assets(status: dict[str, Any], workflow: str | None) -> None:
    for tour in ("pga", "euro"):
        tour_key = tour.lower()
        existing = status.get("model_runs", {}).get(tour_key, {})
        meta = load_tour_meta(tour_key)
        if not meta:
            continue
        active_events = meta.get("active_events")
        events = [compact_event(event) for event in active_events if isinstance(event, dict)] if isinstance(active_events, list) else []
        if not events:
            events = [compact_event(meta)]
        event_ids = [event["event_id"] for event in events if event.get("event_id") and event.get("event_id") != "0"]
        status["model_runs"][tour_key] = {
            "last_run_utc": existing.get("last_run_utc"),
            "tour": tour_key,
            "event_id": str(meta.get("event_id", "")),
            "event_name": meta.get("event_name"),
            "event_ids": event_ids,
            "events": events,
            "artifact_generated_utc": meta.get("generated_utc"),
            "workflow": existing.get("workflow") or workflow,
        }


def update_schedule(status: dict[str, Any], workflow: str | None) -> None:
    schedule_path = ROOT / "upcoming-events.json"
    payload = read_json(schedule_path) if schedule_path.exists() else {}
    events = payload.get("schedule") if isinstance(payload, dict) else []
    status["schedule"] = {
        "last_refreshed_utc": utc_now(),
        "source": "DataGolf",
        "season": payload.get("season") if isinstance(payload, dict) else None,
        "tour": payload.get("tour") if isinstance(payload, dict) else None,
        "event_count": len(events) if isinstance(events, list) else None,
        "workflow": workflow,
    }


def update_betting_data(status: dict[str, Any]) -> None:
    csv_path = ROOT / "web" / "spreadsheet_data.csv"
    if not csv_path.exists():
        status["betting_data"] = {"source": "web/spreadsheet_data.csv", "last_modified": None}
        return
    betting_data = status.get("betting_data")
    existing = betting_data.get("last_modified") if isinstance(betting_data, dict) else None
    if git_file_is_dirty(csv_path):
        last_modified = file_mtime_utc(csv_path)
        timestamp_source = "filesystem_mtime"
    else:
        committed_modified = git_last_commit_utc(csv_path)
        last_modified = committed_modified or existing or file_mtime_utc(csv_path)
        timestamp_source = "git_commit" if committed_modified else "existing_or_filesystem_mtime"
    status["betting_data"] = {
        "source": "web/spreadsheet_data.csv",
        "last_modified": last_modified,
        "timestamp_source": timestamp_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update web/status.json")
    parser.add_argument("--model-run", action="store_true", help="Record a completed model/web asset run.")
    parser.add_argument("--schedule-refreshed", action="store_true", help="Record a completed upcoming-events refresh.")
    parser.add_argument("--sync-assets", action="store_true", help="Refresh model event metadata from current web/{tour}/meta.json assets.")
    parser.add_argument("--tour", action="append", choices=["pga", "euro"], help="Tour for --model-run. Repeat for multiple tours.")
    parser.add_argument("--workflow", help="Workflow or command name that updated the status.")
    args = parser.parse_args()

    if args.model_run and not args.tour:
        parser.error("--model-run requires at least one --tour")

    status = read_status()
    if args.sync_assets:
        sync_model_from_assets(status, args.workflow)
    if args.model_run:
        for tour in args.tour or []:
            update_model_run(status, tour, args.workflow)
    if args.schedule_refreshed:
        update_schedule(status, args.workflow)
    update_betting_data(status)

    write_status(status)


if __name__ == "__main__":
    main()
