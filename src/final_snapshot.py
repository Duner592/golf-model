"""Capture the latest fresh evening prediction before the event's local start date."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.player_integrity import validate_unique_players
from src.provenance import build_snapshot_provenance


def capture_window(start_date: str, timezone_name: str):
    """Use local midnight as a conservative cutoff, never guess a first tee time."""
    zone = ZoneInfo(timezone_name)
    cutoff = datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time(), zone)
    opening = datetime.combine(cutoff.date() - timedelta(days=1), time(18), zone)
    return opening.astimezone(timezone.utc), cutoff.astimezone(timezone.utc)


def prediction_time(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def capture_final_snapshot(root: Path, *, event: dict, source_dir: Path, leaderboard_csv: Path | None,
                           generated_utc: str | None, now: datetime | None = None) -> dict | None:
    now = now or datetime.now(timezone.utc)
    weather_path = source_dir / "weather_meta.json"
    weather = json.loads(weather_path.read_text()) if weather_path.exists() else {}
    try:
        opening, cutoff = capture_window(event["start_date"], weather.get("timezone", ""))
        generated = prediction_time(str(generated_utc))
        if generated.tzinfo is None:
            return None
    except (KeyError, ValueError, TypeError, ZoneInfoNotFoundError):
        return None
    if event.get("reconstruction") or not opening <= generated <= now < cutoff or now - generated > timedelta(hours=4):
        return None
    if str(weather.get("event_id")) != str(event["event_id"]) or weather.get("r1_date") != event["start_date"]:
        return None
    if not leaderboard_csv or not leaderboard_csv.exists():
        return None
    rows = json.loads((source_dir / "leaderboard.json").read_text())
    if not isinstance(rows, list) or not rows:
        return None
    validate_unique_players(rows, context="Final pre-event snapshot")
    tournament = json.loads((source_dir / "tournament_summary.json").read_text())
    if str(tournament.get("event_id")) != str(event["event_id"]) or tournament.get("status") in ("completed", "finished", "in_progress", "in-progress"):
        return None
    target = root / "web/archive" / str(event["year"]) / event["slug"] / "final"
    metadata_path = target / "snapshot.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if datetime.fromisoformat(previous["prediction_generated_utc"].replace("Z", "+00:00")) >= generated:
            return previous
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("leaderboard.json", "summary.json", "tournament_summary.json", "weather_meta.json", "field_teetimes.csv"):
        source = source_dir / filename
        if source.exists():
            shutil.copyfile(source, target / filename)
    shutil.copyfile(leaderboard_csv, target / "leaderboard.csv")
    iso = lambda value: value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    metadata = {
        "snapshot_type": "final", "event_id": str(event["event_id"]), "tour": event["tour"], "year": str(event["year"]),
        "start_date": event["start_date"], "timezone": weather["timezone"],
        "snapshot_created_utc": iso(now), "prediction_generated_utc": iso(generated),
        "capture_window_open_utc": iso(opening), "cutoff_utc": iso(cutoff),
        "cutoff_basis": "midnight before local event start date",
        "provenance": build_snapshot_provenance(root, target),
    }
    metadata["provenance"]["artifact_sha256"].pop("snapshot.json", None)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
