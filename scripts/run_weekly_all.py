#!/usr/bin/env python3
# scripts/run_weekly_all.py
#
# One-command weekly pipeline (hardened to a single event_id):
#   Field + Meta  →  Weather  →  Weather Features  →  Player Data  →  Sigma
#   →  Course Fit/History (optional)  →  Simulate  →  Leaderboard  →  Summary
#
# Key options:
#   --event_id <id>     Force a specific (pinned) event id
#   --pinned            Pinned mode: skip field-updates/parse; use existing meta/field
#   --skip-field        Skip live field-updates/parse
#   --skip-weather      Skip weather fetch/summarize
#   --skip-course       Skip DIY course-fit and course-history merges
#   --skip-html         Do not produce HTML
#   --topN <int>        Top-N leaderboard (default 20)
#   --fast              Alias for --skip-course and --skip-html
#
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

TOUR = "pga"
ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed" / TOUR


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def _parse_ts(iso: str | None) -> float:
    if not iso:
        return 0.0
    s = iso.replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except Exception:
            pass
    return 0.0


def resolve_event_id(cli_event_id: str | None) -> str:
    """Priority: CLI arg → upcoming-events.json (current week by today's date, or closest future event)."""
    if cli_event_id:
        return str(cli_event_id)

    # Load upcoming events
    upcoming_file = ROOT / "upcoming-events.json"
    if not upcoming_file.exists():
        raise FileNotFoundError(f"Upcoming events file not found: {upcoming_file}")

    try:
        with open(upcoming_file, encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("schedule", [])  # Extract the schedule list
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        raise ValueError(f"Failed to load or parse {upcoming_file}: {e}") from e

    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    end_of_week = start_of_week + timedelta(days=6)  # Sunday

    # First, try to find events in the current week
    weekly_events = []
    future_events = []  # Fallback: all events on or after today
    for event in events:
        if not isinstance(event, dict) or "event_id" not in event or "start_date" not in event:
            continue
        try:
            event_date = datetime.fromisoformat(event["start_date"]).date()  # Parse "YYYY-MM-DD"
            if start_of_week <= event_date <= end_of_week:
                weekly_events.append(event)
            elif event_date >= today:
                future_events.append((event_date, event))
        except (ValueError, KeyError):
            continue

    # Use current week if available; otherwise, the closest future event
    if weekly_events:
        return str(weekly_events[0]["event_id"])  # Earliest in week (should be 528 for 2025-11-13)
    elif future_events:
        future_events.sort(key=lambda x: x[0])  # Sort by date
        return str(future_events[0][1]["event_id"])  # Closest future event
    else:
        raise ValueError(f"No current or future events found in {upcoming_file} starting from {today}. Check the file or provide --event_id.")


def main():
    ap = argparse.ArgumentParser(description="Weekly pipeline runner for current/pinned event.")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id; if omitted, use this week's event from upcoming-events.json")
    ap.add_argument("--pinned", action="store_true", help="Pinned mode: skip field-updates/parse; use existing meta/field")
    ap.add_argument("--skip-field", action="store_true", help="Skip field-updates/parse step")
    ap.add_argument("--skip-weather", action="store_true", help="Skip weather fetch/summarize")
    ap.add_argument("--skip-course", action="store_true", help="Skip DIY course-fit and course-history merges")
    ap.add_argument("--skip-html", action="store_true", help="Skip HTML leaderboard output")
    ap.add_argument("--topN", type=int, default=20, help="Top-N leaderboard variant (default 20)")
    ap.add_argument("--fast", action="store_true", help="Fast mode: same as --skip-course and --skip-html")
    args = ap.parse_args()

    skip_course = args.skip_course or args.fast
    skip_html = args.skip_html or args.fast

    try:
        # Resolve a single event_id (prevents drifting to older meta like event_9)
        event_id = resolve_event_id(args.event_id)
        print(f"[info] Using event_id={event_id}")

        # Field + meta
        if args.pinned or args.skip_field:
            print("[info] Pinned/skip-field: skipping field-updates/parse; using existing processed meta/field.")
        else:
            run([sys.executable, str(SCRIPT_DIR / "fetch_field_updates.py")])
            run([sys.executable, str(SCRIPT_DIR / "parse_field_updates.py")])

            # Inline: Merge missing lat/lon/start from upcoming-events.json into meta
            meta_file = PROCESSED / f"event_{event_id}_meta.json"
            upcoming_file = ROOT / "upcoming-events.json"
            if meta_file.exists() and upcoming_file.exists():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        meta = json.load(f)
                    with open(upcoming_file, encoding="utf-8") as f:
                        data = json.load(f)
                    for event in data.get("schedule", []):
                        if str(event.get("event_id")) == event_id:
                            if "latitude" not in meta or not meta.get("latitude"):
                                meta["latitude"] = event.get("latitude")
                            if "longitude" not in meta or not meta.get("longitude"):
                                meta["longitude"] = event.get("longitude")
                            if "start" not in meta or not meta.get("start"):
                                meta["start"] = event.get("start_date")
                            break
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    print(f"[info] Updated meta for event_id={event_id} with latitude/longitude/start from upcoming-events.json")
                except Exception as e:
                    print(f"[warn] Failed to merge meta for event_id={event_id}: {e}")

        # Weather
        if args.skip_weather:
            print("[info] Skipping weather steps (--skip-weather)")
        else:
            run([sys.executable, str(SCRIPT_DIR / "fetch_weather_from_schedule.py"), "--event_id", event_id])
            run([sys.executable, str(SCRIPT_DIR / "summarize_weather_from_schedule.py"), "--event_id", event_id])

        # Weather features
        run([sys.executable, str(SCRIPT_DIR / "build_features_from_weather.py"), "--event_id", event_id])

        # Player data + merge
        run([sys.executable, str(SCRIPT_DIR / "fetch_player_data.py"), "--event_id", event_id])
        run([sys.executable, str(SCRIPT_DIR / "merge_player_data_into_features.py"), "--event_id", event_id])

        # Sigma + merge
        run([sys.executable, str(SCRIPT_DIR / "compute_sigma_from_sg.py"), "--event_id", event_id])
        run([sys.executable, str(SCRIPT_DIR / "merge_sigma_into_features.py"), "--event_id", event_id])

        # Course fit / history (optional)
        if not skip_course:
            # Fetch historical rounds first (if available)
            try:
                run([sys.executable, str(SCRIPT_DIR / "fetch_historical_rounds.py"), "--event_id", event_id])
            except subprocess.CalledProcessError:
                print("[warn] Fetching historical rounds failed or script not found. Continuing with existing data.")

            try:
                run([sys.executable, str(SCRIPT_DIR / "build_course_fit_from_history.py"), "--event_id", event_id])
                run([sys.executable, str(SCRIPT_DIR / "merge_course_fit_diy_into_features.py"), "--event_id", event_id])
            except subprocess.CalledProcessError:
                print("[warn] DIY course-fit step failed or history not available. Continuing.")
            try:
                run([sys.executable, str(SCRIPT_DIR / "build_course_history_from_hist.py"), "--event_id", event_id])
                run([sys.executable, str(SCRIPT_DIR / "merge_course_history_into_features.py"), "--event_id", event_id])
            except subprocess.CalledProcessError:
                print("[warn] Course history stats step failed or unavailable. Continuing.")
        else:
            print("[info] Skipping course-fit/history steps (--skip-course / --fast)")

        # Simulate
        run([sys.executable, str(SCRIPT_DIR / "simulate_event_with_course.py"), "--event_id", event_id])

        # Leaderboard + summary
        lb_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "export_leaderboard.py"),
            "--topN",
            str(args.topN),
            "--event_id",
            event_id,
        ]
        if not skip_html:
            lb_cmd.append("--html")
        run(lb_cmd)

        # Status (always pass event_id so it doesn't drift to another meta)
        run([sys.executable, str(SCRIPT_DIR / "summarize_status.py"), "--event_id", event_id])

        print("[done] Weekly run completed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] Step failed with return code {e.returncode}: {e}", flush=True)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
