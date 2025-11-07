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
from datetime import datetime
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
    """Priority: CLI arg → scripts/field-updates.json → latest processed meta by saved_at_utc (fallback: file mtime)."""
    if cli_event_id:
        return str(cli_event_id)

    # Prefer current week from field-updates.json (if present)
    fu = SCRIPT_DIR / "field-updates.json"
    if fu.exists():
        try:
            data = json.loads(fu.read_text(encoding="utf-8"))
            eid = data.get("event_id")
            if eid is not None:
                return str(eid)
        except Exception:
            pass

    # Fallback to most-recent processed meta by timestamp (not lexicographic filename order)
    metas = sorted(PROCESSED.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("Cannot resolve event_id (no field-updates.json and no processed meta found).")

    best_eid, best_ts = None, -1.0
    for p in metas:
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            ts = _parse_ts(meta.get("saved_at_utc"))
            if ts > best_ts:
                best_ts = ts
                best_eid = meta.get("event_id")
        except Exception:
            ts = p.stat().st_mtime
            if ts > best_ts:
                best_ts = ts
                try:
                    best_eid = json.loads(p.read_text(encoding="utf-8")).get("event_id")
                except Exception:
                    best_eid = None

    if best_eid is None:
        best_eid = json.loads(metas[-1].read_text(encoding="utf-8")).get("event_id")
    if best_eid is None:
        raise FileNotFoundError("Cannot resolve event_id from meta files.")
    return str(best_eid)


def main():
    ap = argparse.ArgumentParser(description="Weekly pipeline runner for current/pinned event.")
    ap.add_argument("--event_id", type=str, default=None, help="Pinned event id; if omitted, use this week's event")
    ap.add_argument("--pinned", action="store_true", help="Pinned mode: skip live field-updates/parse; use existing meta/field")
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
