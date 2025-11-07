#!/usr/bin/env python3
# scripts/run_weekly_all.py
#
# One-command weekly pipeline (hardened):
#   Field + Meta  →  Weather  →  Weather Features  →  Player Data  →  Sigma  →  Course Fit/History (optional)  →  Simulate  →  Leaderboard  →  Summary
#
# Key options:
#   --event_id <id>     Run all steps against a specific (pinned) event id
#   --pinned            Pinned mode: skip live field-updates/parse; use existing meta/field for the pinned event
#   --skip-field        Skip live field-updates/parse unconditionally
#   --skip-weather      Skip weather fetch/summarize (e.g., if already done)
#   --skip-course       Skip DIY course-fit and course-history merges (recommended for speed/backtests)
#   --skip-html         Do not output HTML leaderboard
#   --topN <int>        Top-N leaderboard variant (default 20)
#   --fast              Convenience alias for --skip-course and --skip-html
#
# Examples:
#   python scripts/run_weekly_all.py
#   python scripts/run_weekly_all.py --event_id 457 --pinned --skip-html
#   python scripts/run_weekly_all.py --event_id 11 --pinned --skip-weather --skip-course
#
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def main():
    ap = argparse.ArgumentParser(
        description="Weekly pipeline runner for current/pinned event."
    )
    ap.add_argument(
        "--event_id",
        type=str,
        default=None,
        help="Pinned event id (forces all steps to use this id)",
    )
    ap.add_argument(
        "--pinned",
        action="store_true",
        help="Pinned mode: skip live field-updates/parse; use existing meta/field",
    )
    ap.add_argument(
        "--skip-field", action="store_true", help="Skip field-updates/parse step"
    )
    ap.add_argument(
        "--skip-weather", action="store_true", help="Skip weather fetch/summarize steps"
    )
    ap.add_argument(
        "--skip-course",
        action="store_true",
        help="Skip DIY course-fit and course-history merges",
    )
    ap.add_argument(
        "--skip-html", action="store_true", help="Skip HTML leaderboard output"
    )
    ap.add_argument(
        "--topN", type=int, default=20, help="Top-N leaderboard variant (default 20)"
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: same as --skip-course and --skip-html",
    )
    args = ap.parse_args()

    # Fast mode alias
    skip_course = args.skip_course or args.fast
    skip_html = args.skip_html or args.fast

    try:
        # Field + meta
        if args.pinned or args.skip_field:
            print(
                "[info] Pinned/skip-field mode: skipping field-updates/parse; using existing meta/field."
            )
        else:
            run([sys.executable, str(SCRIPT_DIR / "fetch_field_updates.py")])
            run([sys.executable, str(SCRIPT_DIR / "parse_field_updates.py")])

        # Weather (fetch + summarize)
        if args.skip_weather:
            print("[info] Skipping weather steps (--skip-weather)")
        else:
            fw_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "fetch_weather_from_schedule.py"),
            ]
            sw_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "summarize_weather_from_schedule.py"),
            ]
            if args.event_id:
                fw_cmd += ["--event_id", args.event_id]
                sw_cmd += ["--event_id", args.event_id]
            run(fw_cmd)
            run(sw_cmd)

        # Weather features (always pass event_id if provided)
        bf_cmd = [sys.executable, str(SCRIPT_DIR / "build_features_from_weather.py")]
        if args.event_id:
            bf_cmd += ["--event_id", args.event_id]
        run(bf_cmd)

        # Player data + merge
        # (fetch_player_data can discover players from processed field)
        fp_cmd = [sys.executable, str(SCRIPT_DIR / "fetch_player_data.py")]
        run(fp_cmd)

        mp_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "merge_player_data_into_features.py"),
        ]
        run(mp_cmd)

        # Sigma + merge (accepts --event_id in your repo; pass when available)
        cs_cmd = [sys.executable, str(SCRIPT_DIR / "compute_sigma_from_sg.py")]
        ms_cmd = [sys.executable, str(SCRIPT_DIR / "merge_sigma_into_features.py")]
        if args.event_id:
            cs_cmd += ["--event_id", args.event_id]
            ms_cmd += ["--event_id", args.event_id]
        run(cs_cmd)
        run(ms_cmd)

        # Course fit + history (optional)
        if not skip_course:
            try:
                # DIY course-fit (regression on venue history; includes driving signals)
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "build_course_fit_from_history.py"),
                    ]
                    + (["--event_id", args.event_id] if args.event_id else [])
                )
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "merge_course_fit_diy_into_features.py"),
                    ]
                    + (["--event_id", args.event_id] if args.event_id else [])
                )
            except subprocess.CalledProcessError:
                print(
                    "[warn] DIY course-fit failed or history unavailable. Continuing."
                )
            try:
                # Course history stats
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "build_course_history_from_hist.py"),
                    ]
                    + (["--event_id", args.event_id] if args.event_id else [])
                )
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "merge_course_history_into_features.py"),
                    ]
                    + (["--event_id", args.event_id] if args.event_id else [])
                )
            except subprocess.CalledProcessError:
                print(
                    "[warn] Course history stats step failed or unavailable. Continuing."
                )
        else:
            print("[info] Skipping course-fit/history steps (--skip-course / --fast)")

        # Simulate (pass event_id)
        sim_cmd = [sys.executable, str(SCRIPT_DIR / "simulate_event_with_course.py")]
        if args.event_id:
            sim_cmd += ["--event_id", args.event_id]
        run(sim_cmd)

        # Leaderboard + summary (pass event_id; optionally skip HTML)
        lb_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "export_leaderboard.py"),
            "--topN",
            str(args.topN),
        ]
        if args.event_id:
            lb_cmd += ["--event_id", args.event_id]
        if not skip_html:
            lb_cmd.append("--html")
        run(lb_cmd)

        # Snapshot summary of artifacts (pass event_id)
        ss_cmd = [sys.executable, str(SCRIPT_DIR / "summarize_status.py")]
        run(ss_cmd)

        print("[done] Weekly run completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print(f"[error] Step failed with return code {e.returncode}: {e}", flush=True)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
