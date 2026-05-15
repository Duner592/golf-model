#!/usr/bin/env python3
# scripts/run_weekly_all.py
#
# One-command weekly pipeline (hardened to a single event_id):
#   Field + Meta  →  Weather  →  Weather Features  →  Player Data  →  Sigma
#   →  Course Fit/History (optional)  →  Simulate  →  Leaderboard  →  Summary
#
# Key options:
#   --event_id <id>     Force a specific (pinned) event id
#   --tour <tour>       Tour to process (pga or euro, default pga)
#   --pinned            Pinned mode: skip field-updates/parse; use existing meta/field
#   --skip-field        Skip live field-updates/parse
#   --skip-weather      Skip weather fetch/summarize
#   --skip-course       Skip DIY course-fit and course-history merges
#   --skip-html         Do not produce HTML
#   --topN <int>        Optional Top-N leaderboard variant; 0 disables it (default 0)
#   --fast              Alias for --skip-course and --skip-html
#
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_event import resolve_event_ids  # noqa: E402

def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def main():
    ap = argparse.ArgumentParser(description="Weekly pipeline runner for current/pinned event.")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id; if omitted, use this week's event from upcoming-events.json")
    ap.add_argument("--tour", type=str, default="pga", help="Tour to process (pga or euro)")
    ap.add_argument("--pinned", action="store_true", help="Pinned mode: skip field-updates/parse; use existing meta/field")
    ap.add_argument("--skip-field", action="store_true", help="Skip field-updates/parse step")
    ap.add_argument("--skip-weather", action="store_true", help="Skip weather fetch/summarize")
    ap.add_argument("--skip-course", action="store_true", help="Skip DIY course-fit and course-history merges")
    ap.add_argument("--skip-html", action="store_true", help="Skip HTML leaderboard output")
    ap.add_argument("--topN", type=int, default=0, help="Optional Top-N leaderboard variant; 0 disables it (default 0)")
    ap.add_argument("--fast", action="store_true", help="Fast mode: same as --skip-course and --skip-html")
    args = ap.parse_args()

    TOUR = args.tour
    PROCESSED = ROOT / "data" / "processed" / TOUR

    skip_course = args.skip_course or args.fast
    skip_html = args.skip_html or args.fast

    upcoming_lookup: dict[str, dict] = {}
    start_groups: dict[str, list[str]] = {}
    upcoming_file = ROOT / "upcoming-events.json"
    if upcoming_file.exists():
        try:
            with open(upcoming_file, encoding="utf-8") as f:
                upcoming_data = json.load(f)
            schedule = upcoming_data.get("schedule", [])
            for event in schedule:
                if not isinstance(event, dict):
                    continue
                eid = event.get("event_id")
                if eid is None:
                    continue
                eid_str = str(eid)
                upcoming_lookup[eid_str] = event
                start = event.get("start_date")
                if start:
                    start_groups.setdefault(start, []).append(eid_str)
        except Exception:
            upcoming_lookup = {}
            start_groups = {}
    try:
        cmd_suffix = ["--tour", TOUR]
        event_ids = resolve_event_ids(args.event_id, TOUR)
        if not event_ids:
            raise ValueError(f"No events resolved for tour={TOUR}. Please update upcoming-events.json or pass --event_id.")

        print(f"[info] Events to process for tour={TOUR}: {', '.join(event_ids)}")

        for event_id in event_ids:
            requested_event_id = event_id
            print(f"[info] === Processing event_id={event_id} ===")
            per_event_suffix = ["--event_id", event_id] + cmd_suffix

            event_info = upcoming_lookup.get(event_id, {})
            start_date = event_info.get("start_date")
            same_start_ids = start_groups.get(start_date, []) if start_date else []
            field_update_tour = TOUR
            if TOUR.lower() == "pga" and same_start_ids:
                try:
                    ordered_same_start = sorted(
                        same_start_ids,
                        key=lambda x: int(str(x)) if str(x).isdigit() else str(x),
                    )
                except Exception:
                    ordered_same_start = sorted(same_start_ids)
                if len(ordered_same_start) > 1:
                    if event_id not in ordered_same_start:
                        ordered_same_start.append(event_id)
                    if event_id != ordered_same_start[0]:
                        field_update_tour = "opp"

            # Field + meta
            if args.pinned or args.skip_field:
                print("[info] Pinned/skip-field: skipping field-updates/parse; using existing processed meta/field.")
            else:
                fetch_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "fetch_field_updates.py"),
                    "--event_id",
                    event_id,
                    "--tour",
                    field_update_tour,
                ]
                run(fetch_cmd)

                # Inspect fetched field data before parsing to avoid hard failures
                field_updates_path = SCRIPT_DIR / "field-updates.json"
                skip_event = False
                if field_updates_path.exists():
                    try:
                        with open(field_updates_path, encoding="utf-8") as f:
                            fetched_payload = json.load(f)
                        if isinstance(fetched_payload, dict):
                            payload_event_id = fetched_payload.get("event_id")
                            if payload_event_id is not None and str(payload_event_id) != str(event_id):
                                print(
                                    "[warn] Field updates returned "
                                    f"event_id={payload_event_id} for requested event_id={event_id}; "
                                    "using the returned id for downstream files."
                                )
                                event_id = str(payload_event_id)
                                per_event_suffix = ["--event_id", event_id] + cmd_suffix

                            error_msg = fetched_payload.get("error")
                            field_entries = fetched_payload.get("field")
                            if error_msg:
                                print(f"[warn] Skipping event_id={event_id}: field updates API returned error: {error_msg}")
                                skip_event = True
                            elif not field_entries:
                                print(f"[warn] Skipping event_id={event_id}: field updates payload contains no field entries.")
                                skip_event = True
                    except Exception as exc:
                        print(f"[warn] Unable to inspect field-updates.json for event_id={event_id}: {exc}")
                else:
                    print(f"[warn] Expected field-updates.json not found after fetch for event_id={event_id}; skipping event.")
                    skip_event = True

                if skip_event:
                    continue

                run([sys.executable, str(SCRIPT_DIR / "parse_field_updates.py")] + cmd_suffix)

                # Inline: Merge missing lat/lon/start from upcoming-events.json into meta
                meta_file = PROCESSED / f"event_{event_id}_meta.json"
                if not meta_file.exists():
                    raise FileNotFoundError(
                        f"Processed meta not found after field parsing for event_id={event_id}: {meta_file}"
                    )
                upcoming_file = ROOT / "upcoming-events.json"
                if meta_file.exists() and upcoming_file.exists():
                    try:
                        with open(meta_file, encoding="utf-8") as f:
                            meta = json.load(f)
                        with open(upcoming_file, encoding="utf-8") as f:
                            data = json.load(f)
                        event_id_candidates = {str(event_id), str(requested_event_id)}
                        for event in data.get("schedule", []):
                            if str(event.get("event_id")) in event_id_candidates and event.get("tour") == TOUR:
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
                run([sys.executable, str(SCRIPT_DIR / "fetch_weather_from_schedule.py")] + per_event_suffix)
                run([sys.executable, str(SCRIPT_DIR / "summarize_weather_from_schedule.py")] + per_event_suffix)

            # Weather features
            run([sys.executable, str(SCRIPT_DIR / "build_features_from_weather.py")] + per_event_suffix)

            # Player data + merge
            run([sys.executable, str(SCRIPT_DIR / "fetch_player_data.py")] + per_event_suffix)
            run([sys.executable, str(SCRIPT_DIR / "merge_player_data_into_features.py")] + per_event_suffix)

            # Sigma + merge
            run([sys.executable, str(SCRIPT_DIR / "compute_sigma_from_sg.py")] + per_event_suffix)
            run([sys.executable, str(SCRIPT_DIR / "merge_sigma_into_features.py")] + per_event_suffix)

            # Course fit / history (optional)
            if not skip_course:
                # Fetch historical rounds first (if available)
                try:
                    run([sys.executable, str(SCRIPT_DIR / "fetch_historical_rounds.py")] + per_event_suffix)
                except subprocess.CalledProcessError:
                    print("[warn] Fetching historical rounds failed or script not found. Continuing with existing data.")

                try:
                    run([sys.executable, str(SCRIPT_DIR / "build_course_fit_from_history.py")] + per_event_suffix)
                    run([sys.executable, str(SCRIPT_DIR / "merge_course_fit_diy_into_features.py")] + per_event_suffix)
                except subprocess.CalledProcessError:
                    print("[warn] DIY course-fit step failed or history not available. Continuing.")
                try:
                    run([sys.executable, str(SCRIPT_DIR / "build_course_history_from_hist.py")] + per_event_suffix)
                    run([sys.executable, str(SCRIPT_DIR / "merge_course_history_into_features.py")] + per_event_suffix)
                except subprocess.CalledProcessError:
                    print("[warn] Course history stats step failed or unavailable. Continuing.")
            else:
                print("[info] Skipping course-fit/history steps (--skip-course / --fast)")

            # Simulate
            run([sys.executable, str(SCRIPT_DIR / "simulate_event_with_course.py")] + per_event_suffix)

            # Leaderboard + summary
            lb_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "export_leaderboard.py"),
                "--topN",
                str(args.topN),
                "--event_id",
                event_id,
            ] + cmd_suffix
            if not skip_html:
                lb_cmd.append("--html")
            run(lb_cmd)

            # Status (always pass event_id so it doesn't drift to another meta)
            run([sys.executable, str(SCRIPT_DIR / "summarize_status.py")] + per_event_suffix)

            print(f"[info] Completed pipeline for event_id={event_id}", flush=True)

        print(f"[done] Weekly run completed successfully for events: {', '.join(event_ids)}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] Step failed with return code {e.returncode}: {e}", flush=True)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
