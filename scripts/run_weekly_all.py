#!/usr/bin/env python3
# scripts/run_weekly_all.py
import argparse
import subprocess
from pathlib import Path
import sys
import os

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def run(cmd, check=True, env=None):
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env=env)


def main():
    ap = argparse.ArgumentParser(
        description="Weekly pipeline runner for current event."
    )
    ap.add_argument(
        "--event_id",
        type=str,
        default=None,
        help="Force event_id for weather-features step",
    )
    ap.add_argument(
        "--skip-course",
        action="store_true",
        help="Skip DIY course-fit and course-history merges",
    )
    ap.add_argument(
        "--skip-html", action="store_true", help="Skip HTML leaderboard output"
    )
    ap.add_argument("--topN", type=int, default=20, help="Top-N leaderboard variant")
    ap.add_argument(
        "--fast", action="store_true", help="Fast mode (equivalent to --skip-course)"
    )
    args = ap.parse_args()

    skip_course = args.skip_course or args.fast

    # Inject PYTHONPATH so child scripts can import src.*
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(REPO_ROOT)}{os.pathsep}{env.get('PYTHONPATH','')}"

    try:
        # Field + meta
        run([sys.executable, str(SCRIPT_DIR / "fetch_field_updates.py")], env=env)
        run([sys.executable, str(SCRIPT_DIR / "parse_field_updates.py")], env=env)

        # Weather
        run(
            [sys.executable, str(SCRIPT_DIR / "fetch_weather_from_schedule.py")],
            env=env,
        )
        run(
            [sys.executable, str(SCRIPT_DIR / "summarize_weather_from_schedule.py")],
            env=env,
        )

        # Weather features (allow --event_id)
        bf_cmd = [sys.executable, str(SCRIPT_DIR / "build_features_from_weather.py")]
        if args.event_id:
            bf_cmd += ["--event_id", args.event_id]
        run(bf_cmd, env=env)

        # Player data + merge
        run([sys.executable, str(SCRIPT_DIR / "fetch_player_data.py")], env=env)
        run(
            [sys.executable, str(SCRIPT_DIR / "merge_player_data_into_features.py")],
            env=env,
        )

        # Sigma + merge
        run([sys.executable, str(SCRIPT_DIR / "compute_sigma_from_sg.py")], env=env)
        run([sys.executable, str(SCRIPT_DIR / "merge_sigma_into_features.py")], env=env)

        # DIY course fit + course history (optional)
        if not skip_course:
            try:
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "build_course_fit_from_history.py"),
                    ],
                    env=env,
                )
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "merge_course_fit_diy_into_features.py"),
                    ],
                    env=env,
                )
            except subprocess.CalledProcessError:
                print(
                    "[warn] DIY course-fit failed or history unavailable. Continuing.",
                    flush=True,
                )
            try:
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "build_course_history_from_hist.py"),
                    ],
                    env=env,
                )
                run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "merge_course_history_into_features.py"),
                    ],
                    env=env,
                )
            except subprocess.CalledProcessError:
                print(
                    "[warn] Course history stats step failed or unavailable. Continuing.",
                    flush=True,
                )

        # Simulate
        run(
            [sys.executable, str(SCRIPT_DIR / "simulate_event_with_course.py")], env=env
        )

        # Leaderboard + summary
        lb_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "export_leaderboard.py"),
            "--topN",
            str(args.topN),
        ]
        if not args.skip_html:
            lb_cmd.append("--html")
        run(lb_cmd, env=env)

        run([sys.executable, str(SCRIPT_DIR / "summarize_status.py")], env=env)

        print("\n[done] Weekly run completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print(f"\n[error] Step failed with return code {e.returncode}: {e}", flush=True)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
