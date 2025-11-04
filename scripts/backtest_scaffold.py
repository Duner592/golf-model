#!/usr/bin/env python3
# scripts/backtest_scaffold.py
#
# Minimal backtest scaffold:
# - Iterates over a list of (event_id, r1_date, lat, lon) or fetches via your schedule
# - For each, runs: fetch field -> parse -> fetch weather -> summarize -> build features -> fetch/merge player data -> sigma -> simulate
# NOTE: This is a scaffold; you must provide historical inputs and ensure you use pre-event data only.

import subprocess
from pathlib import Path

TOUR = "pga"


def run(cmd: list[str]):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    script_dir = Path(__file__).resolve().parent

    # For a real backtest, construct a list of historical events and pin their inputs.
    # Here we just run the current pipeline end-to-end as a template.
    run(["python", str(script_dir / "fetch_field_updates.py")])
    run(["python", str(script_dir / "parse_field_updates.py")])
    run(["python", str(script_dir / "fetch_weather_from_schedule.py")])
    run(["python", str(script_dir / "summarize_weather_from_schedule.py")])
    run(["python", str(script_dir / "build_features_from_weather.py")])
    run(["python", str(script_dir / "fetch_player_data.py")])
    run(["python", str(script_dir / "merge_player_data_into_features.py")])
    run(["python", str(script_dir / "compute_sigma_from_sg.py")])
    run(["python", str(script_dir / "merge_sigma_into_features.py")])
    run(["python", str(script_dir / "simulate_event_common_shock.py")])


if __name__ == "__main__":
    main()
