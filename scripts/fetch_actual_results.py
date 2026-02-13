#!/usr/bin/env python3
"""Fetch actual tournament results from DataGolf and merge into web archive.

This script iterates over archived events (currently stored under
`web/archive/{year}/{slug}`) and calls the DataGolf API to retrieve the final
leaderboard with finishing positions for each player. The results are saved in
`web/archive/{year}/{slug}/results.json` and also summarized into a single
`web/archive/results_summary.json` for convenience.

Usage:
    python scripts/fetch_actual_results.py [--year 2026]

Environment:
    Requires `DATAGOLF_API_KEY` in your environment (.env) for authentication.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


BASE_URL = "https://feeds.datagolf.com"
RESULTS_ENDPOINT = "historical-raw-data/rounds"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datagolf_client import DataGolfClient  # noqa: E402


def load_archive_index(root: Path) -> list[dict]:
    index_path = root / "web" / "archive" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Archive index not found at {index_path}")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def fetch_results(client: DataGolfClient, tour: str, event_id: str, year: str) -> dict[str, Any]:
    params = {"tour": tour, "event_id": event_id, "year": year}
    return client.get(RESULTS_ENDPOINT, params=params)


def normalize_name(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        return f"{first} {last}" if first else last
    return name


def parse_finish_position(fin_text: str | None) -> int | None:
    if not fin_text:
        return None
    fin_text = fin_text.strip().upper()
    if not fin_text:
        return None
    if fin_text in {"MC", "WD", "DQ"}:
        return None
    fin_text = fin_text.lstrip("T")
    digits = "".join(ch for ch in fin_text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def extract_finishes(data: dict[str, Any]) -> list[dict[str, Any]]:
    players = []
    for row in data.get("scores", []):
        name = normalize_name(row.get("player_name"))
        if not name:
            continue
        player_id = row.get("dg_id")
        fin_text = row.get("fin_text")
        finish_pos = parse_finish_position(fin_text)
        score = row.get("total_score") or row.get("score")
        to_par = row.get("to_par") or row.get("sc_to_par")
        is_cut = row.get("made_cut")
        players.append(
            {
                "player": name,
                "player_id": player_id,
                "finish_text": fin_text,
                "finish_pos": finish_pos,
                "score": score,
                "to_par": to_par,
                "made_cut": is_cut,
            }
        )
    return players


def save_results(event_dir: Path, results: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    event_dir.mkdir(parents=True, exist_ok=True)
    out_path = event_dir / "results.json"
    payload = {"event": metadata, "players": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path.relative_to(event_dir.parents[1])}")


def append_summary(summary: dict[str, Any], key: str, value: Any) -> None:
    summary[key] = value


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch actual tournament results for archived events")
    ap.add_argument("--year", type=str, default=None, help="Filter archive entries by year (e.g. 2026)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    archive_entries = load_archive_index(root)
    if args.year:
        archive_entries = [e for e in archive_entries if str(e.get("year")) == args.year]
    if not archive_entries:
        print("No archive entries found for the specified criteria", file=sys.stderr)
        sys.exit(1)

    client = DataGolfClient(BASE_URL)

    summary: dict[str, Any] = {}
    errors: list[str] = []

    for entry in archive_entries:
        tour = entry.get("tour")
        event_id = str(entry.get("event_id"))
        year = str(entry.get("year"))
        event_name = entry.get("event_name")
        slug = entry.get("slug")
        archive_dir = root / "web" / "archive" / year / slug

        results_out = archive_dir / "results.json"
        summary_path = archive_dir / "tournament_summary.json"
        if summary_path.exists():
            try:
                summary_data = json.load(open(summary_path, encoding="utf-8"))
                if summary_data.get("status") != "completed":
                    print(f"Skipping {event_name} ({year}); event status is {summary_data.get('status')!r}")
                    if results_out.exists():
                        existing = json.load(open(results_out, encoding="utf-8"))
                        metadata = existing.get("event", {})
                        metadata.setdefault("slug", slug)
                        append_summary(summary, f"{year}_{slug}", metadata)
                    continue
            except Exception as exc:
                print(f"Warn: Unable to read summary for {event_name}: {exc}")

        if results_out.exists():
            print(f"Skipping {event_name} ({year}); results already exist")
            existing = json.load(open(results_out, encoding="utf-8"))
            metadata = existing.get("event", {})
            metadata.setdefault("slug", slug)
            append_summary(summary, f"{year}_{slug}", metadata)
            continue

        try:
            print(f"Fetching results for {event_name} ({tour.upper()} {year})...")
            data = fetch_results(client, tour, event_id, year)
        except Exception as exc:
            error_msg = f"Failed to fetch {event_name} ({year}): {exc}"
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            continue

        results = extract_finishes(data)
        if not results:
            error_msg = f"No player results returned for {event_name} ({year})"
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
            continue

        metadata = {
            "event_id": event_id,
            "event_name": event_name,
            "tour": tour,
            "year": year,
            "fetched_at": data.get("last_updated") or data.get("timestamp"),
            "source": RESULTS_ENDPOINT,
        }
        metadata["slug"] = slug
        save_results(archive_dir, results, metadata)

        append_summary(summary, f"{year}_{slug}", metadata)

    if summary:
        summary_path = root / "web" / "archive" / "results_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated summary at {summary_path}")

    if errors:
        print("\nEncountered errors:")
        for msg in errors:
            print(" -", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
