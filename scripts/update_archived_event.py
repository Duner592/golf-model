#!/usr/bin/env python3
# scripts/update_archived_event.py
#
# Update an archived event's tournament_summary.json for a completed event.
# Usage: python scripts/update_archived_event.py --event_id <event_id>
#
# This script:
# - Checks if the event status is "completed" in upcoming-events.json; skips if "upcoming" unless --force is used.
# - Finds the archived tournament_summary.json for the given event_id.
# - If the archive is missing, creates it from the saved initial snapshot or checked-in event assets.
# - Updates status to "completed".
# - Fetches the winner from DataGolf API, or falls back to upcoming-events.json if API fails.
# - Updates only the "winner" field (does not modify "previous_winners").
# - Overwrites the file with updated data.

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import re

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.player_integrity import _name, validate_unique_players
from src.provenance import sha256_file


load_dotenv()


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


def get_event_details(event_id: str, upcoming_data: dict, tour: str | None = None) -> dict | None:
    """
    Get event details from upcoming-events.json.

    Args:
        event_id (str): Event ID.
        upcoming_data (dict): Upcoming events data.

    Returns:
        dict or None: Event details.
    """
    matches = [
        event
        for event in upcoming_data.get("schedule", [])
        if str(event.get("event_id")) == str(event_id)
    ]
    if tour:
        normalized_tour = str(tour).lower()
        matches = [event for event in matches if str(event.get("tour", "")).lower() == normalized_tour]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None

    candidates = ", ".join(str(event.get("tour", "unknown")) for event in matches)
    raise ValueError(
        f"Event ID {event_id} is ambiguous across tours ({candidates}). Pass --tour explicitly."
    )


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


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def archive_time(value: str | None = None) -> str:
    if value:
        for fmt in (None, "%d-%b-%Y %H:%M:%S"):
            try:
                if fmt:
                    parsed = datetime.strptime(value, fmt)
                else:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                continue
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def archive_sort_value(entry: dict) -> str:
    for key in ("start_date", "r1_date", "archived_at"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_web_resource(root: Path, resource: str | None) -> Path | None:
    if not resource:
        return None
    resource_path = Path(resource)
    if resource_path.is_absolute():
        return resource_path if resource_path.exists() else None
    candidate = root / "web" / resource_path
    return candidate if candidate.exists() else None


def source_candidates(root: Path, event_details: dict, year: str) -> list[tuple[Path, str]]:
    event_id = str(event_details.get("event_id"))
    tour = str(event_details.get("tour", "")).lower()
    return [
        (root / "web" / tour / "initial" / year / f"event_{event_id}", "initial"),
        (root / "web" / tour / "events" / f"event_{event_id}", "event_assets"),
    ]


def copy_optional_csv(root: Path, source_dir: Path, archive_dir: Path) -> bool:
    direct_csv = source_dir / "leaderboard.csv"
    if direct_csv.exists():
        shutil.copy(direct_csv, archive_dir / "leaderboard.csv")
        return True

    meta_path = source_dir / "meta.json"
    if not meta_path.exists():
        return False
    meta = read_json(meta_path)
    resources = meta.get("resources") if isinstance(meta.get("resources"), dict) else {}
    for key in ("initial_downloads_csv", "downloads_csv"):
        csv_path = resolve_web_resource(root, resources.get(key))
        if csv_path:
            shutil.copy(csv_path, archive_dir / "leaderboard.csv")
            return True
    return False


def materialize_missing_archive(root: Path, event_details: dict, year: str, archive_path: Path) -> bool:
    event_id = str(event_details.get("event_id"))
    event_name = event_details.get("event_name")
    tour = str(event_details.get("tour", "")).lower()
    event_slug = archive_path.parent.name

    for source_dir, source_type in source_candidates(root, event_details, year):
        if not (source_dir / "leaderboard.json").exists() or not (source_dir / "tournament_summary.json").exists():
            continue

        validate_unique_players(json.loads((source_dir / "leaderboard.json").read_text()), context=f"Archive source {tour}/{year}/{event_id}")

        archive_dir = archive_path.parent
        archive_dir.mkdir(parents=True, exist_ok=True)
        for file_name in (
            "model_page.html",
            "leaderboard.json",
            "meta.json",
            "tournament_summary.json",
            "summary.json",
            "weather_round_neutral.json",
            "weather_round_wave.json",
            "weather_meta.json",
            "course_fit_weights.json",
            "course_history_summary.json",
            "snapshot.json",
            "field_teetimes.csv",
        ):
            src = source_dir / file_name
            if src.exists():
                shutil.copy(src, archive_dir / file_name)

        csv_available = copy_optional_csv(root, source_dir, archive_dir)
        source_meta = read_json(source_dir / "meta.json") if (source_dir / "meta.json").exists() else {}
        initial_snapshot = source_meta.get("initial_snapshot") if isinstance(source_meta.get("initial_snapshot"), dict) else {}
        snapshot_created = source_meta.get("snapshot_created_utc") or initial_snapshot.get("snapshot_created_utc")
        generated_utc = source_meta.get("generated_utc")

        index_file = root / "web" / "archive" / "index.json"
        try:
            index_data = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else []
        except json.JSONDecodeError as exc:
            raise ValueError(f"Archive index is invalid JSON: {index_file}") from exc
        if not isinstance(index_data, list):
            raise ValueError(f"Archive index must be a list: {index_file}")

        index_data = [
            entry
            for entry in index_data
            if not (
                isinstance(entry, dict)
                and str(entry.get("event_id")) == event_id
                and str(entry.get("tour", "")).lower() == tour
                and str(entry.get("year")) == str(year)
            )
        ]
        index_entry = {
            "event_id": event_id,
            "event_name": event_name,
            "tour": tour,
            "slug": event_slug,
            "year": year,
            "start_date": event_details.get("start_date"),
            "csv_available": csv_available,
            "archived_at": archive_time(snapshot_created or generated_utc),
            "prediction_snapshot": "initial" if source_type == "initial" else "event_assets",
            "initial_snapshot_created_utc": snapshot_created,
        }
        if (archive_dir / "model_page.html").exists():
            index_entry["model_page"] = f"archive/{year}/{event_slug}/model_page.html"
        index_data.append(index_entry)
        index_data.sort(key=archive_sort_value, reverse=True)
        write_json(index_file, index_data)

        print(f"Created missing archive for {event_id} ({event_name}) from {source_dir}")
        return True

    print(
        "Skipping: archived tournament_summary.json not found and no source snapshot/assets were available for "
        f"event {event_id} ({event_name})"
    )
    return False


def find_archive_summary_path(root: Path, event_details: dict, year: str) -> Path:
    """Resolve the archived summary path, preferring the committed archive index."""
    event_id = str(event_details.get("event_id"))
    tour = str(event_details.get("tour", "")).lower()
    index_path = root / "web" / "archive" / "index.json"

    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Archive index is invalid JSON: {index_path}") from exc
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
def deduplicate_archive(root: Path, *, event_id: str, tour: str, year: str) -> int:
    """Explicit repair only: retain first saved row, never average or use results.

    Initial snapshots remain untouched. A machine-readable audit records the
    removed rows and pre/post hashes without adding a website badge.
    """
    entries = json.loads((root / "web/archive/index.json").read_text())
    matches = [e for e in entries if str(e.get("event_id")) == event_id and e.get("tour") == tour and str(e.get("year")) == year]
    if len(matches) != 1:
        raise ValueError("Repair requires one exact tour/year/event archive")
    entry = matches[0]
    directory = root / "web/archive" / year / entry["slug"]
    original = root / "web" / tour / "initial" / year / f"event_{event_id}"
    paths = [directory / "leaderboard.json", directory / "leaderboard.csv"]
    for path in paths:
        if sha256_file(path) is None:
            raise ValueError(f"Missing repair input: {path}")
    rows = json.loads(paths[0].read_text())
    with paths[1].open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        csv_rows = list(reader)
    if len(rows) != len(csv_rows):
        raise ValueError("JSON and CSV row counts differ; refusing ambiguous repair")
    for json_row, csv_row in zip(rows, csv_rows):
        if _name(json_row.get("player_name")) != _name(csv_row.get("player_name")):
            raise ValueError("JSON and CSV player order differs")
        for key in ("p_win_%", "p_top10_%", "p_mc_%"):
            if float(json_row[key]) != float(csv_row[key]):
                raise ValueError(f"JSON and CSV probabilities differ: {key}")
    seen, kept, removed = set(), [], []
    for i, row in enumerate(rows):
        key = _name(row.get("player_name"))
        if not key:
            raise ValueError("Missing player name")
        if key in seen:
            removed.append(row)
        else:
            seen.add(key)
            kept.append(i)
    if not removed:
        return 0
    for path in paths:
        if sha256_file(path) != sha256_file(original / path.name):
            raise ValueError(f"Original snapshot does not preserve {path.name}; refusing to overwrite")
    clean = [{**rows[i], "rank": rank} for rank, i in enumerate(kept, 1)]
    validate_unique_players(clean, context=f"Repaired archive {tour}/{year}/{event_id}")
    audit = {
        "operation": "remove_duplicate_players_keep_first_saved_row",
        "event_id": event_id, "tour": tour, "year": year,
        "original_snapshot": str(original.relative_to(root)),
        "original_sha256": {path.name: sha256_file(path) for path in paths},
        "removed_rows": removed,
        "probabilities_recalculated": False,
        "limitation": "Removing rows does not undo any duplicate-field effects on the original simulation.",
    }
    write_json(paths[0], clean)
    with paths[1].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({**csv_rows[i], "rank": rank} for rank, i in enumerate(kept, 1))
    # Keep archived field and derived field-size metadata consistent too.
    field = directory / "field_teetimes.csv"
    if field.exists():
        with field.open(newline="") as handle:
            reader = csv.DictReader(handle)
            field_columns, field_rows = reader.fieldnames, list(reader)
        unique, seen_ids = [], set()
        for row in field_rows:
            key = row.get("player_id") or row.get("dg_id") or _name(row.get("player_name"))
            if key not in seen_ids:
                unique.append(row)
                seen_ids.add(key)
        if len(unique) != len(field_rows):
            with field.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=field_columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(unique)
    for filename in ("summary.json", "tournament_summary.json"):
        path = directory / filename
        data = read_json(path)
        if data.get("field_size") == len(rows):
            data["field_size"] = len(clean)
            write_json(path, data)
    from scripts.build_web_assets import write_model_page_snapshot
    write_model_page_snapshot(directory, tour=tour, event_name=entry["event_name"], event_id=event_id, year=year,
                              snapshot_created_utc=entry["initial_snapshot_created_utc"], event_meta=read_json(directory / "meta.json"))
    audit["repaired_sha256"] = {path.name: sha256_file(path) for path in paths}
    write_json(directory / "duplicate_repair.json", audit)
    return len(removed)


def main():
    ap = argparse.ArgumentParser(description="Update archived tournament_summary.json for a completed event.")
    ap.add_argument("--event_id", type=str, required=True, help="Event ID to update")
    ap.add_argument("--tour", type=str, default=None, help="Tour for the event; required when an event ID is shared")
    ap.add_argument("--year", type=str, help="Explicit archive year for duplicate repair")
    ap.add_argument("--deduplicate-players", action="store_true", help="Offline archive repair: retain first saved player row; preserve initial snapshots")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore local upcoming/in-progress status. Used by scheduled previous-week archive updates once a winner is available.",
    )
    args = ap.parse_args()

    event_id = args.event_id
    root = Path(__file__).resolve().parent.parent
    if args.deduplicate_players:
        if not args.tour or not args.year:
            ap.error("--deduplicate-players requires --tour and --year")
        count = deduplicate_archive(root, event_id=event_id, tour=args.tour, year=args.year)
        print(f"Removed {count} duplicate prediction rows from {args.tour}/{args.year}/{event_id}; initial snapshots preserved.")
        return

    # Load upcoming events
    upcoming_data = load_upcoming_events(root)

    # Get event details
    event_details = get_event_details(event_id, upcoming_data, args.tour)
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

    # Fetch winner from API, fallback to upcoming-events.json
    tour = event_details.get("tour")
    winner = fetch_winner_from_api(event_id, year, tour)
    if winner is None:
        winner = event_details.get("winner")
        print(f"DEBUG: Using fallback winner from upcoming-events.json: {winner}")
    if args.force and local_status in {"upcoming", "in_progress", "in-progress"} and winner is None:
        print(f"Skipping: Event {event_id} is locally {local_status!r} and no winner is available yet.")
        return

    # Build archive path, creating a missing archive from saved web assets when possible.
    archive_path = find_archive_summary_path(root, event_details, year)
    if not archive_path.exists() and not materialize_missing_archive(root, event_details, year, archive_path):
        return

    # Update the file
    update_tournament_summary(archive_path, winner)

    print(f"Successfully updated archived event {event_id} ({event_name}) for {year}")


if __name__ == "__main__":
    main()
