#!/usr/bin/env python3
"""Update the public web/status.json health snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = ROOT / "web" / "status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {"model_runs": {}, "schedule": {}}
    try:
        data = read_json(STATUS_PATH)
    except Exception:
        return {"model_runs": {}, "schedule": {}}
    if not isinstance(data, dict):
        return {"model_runs": {}, "schedule": {}}
    data.setdefault("model_runs", {})
    data.setdefault("schedule", {})
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

    write_status(status)


if __name__ == "__main__":
    main()
