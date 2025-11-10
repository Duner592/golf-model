#!/usr/bin/env python3
# scripts/fetch_historical_rounds.py
#
# Fetch real historical rounds for an event using DataGolf API (via configs/datagolf.yaml).
# Extracts winners (first player in ordered "scores" list) with under-par score and total score.
# Saves rounds to Parquet and winners to JSON.
# Skips invalid years.
# Compatible with run_weekly_all.py.

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_name(s: str) -> str:
    s0 = (s or "").lower()
    s0 = s0.replace(" ", "_")
    return s0


def fetch_real_historical_rounds(event_name: str, event_id: str, tour: str) -> tuple[pd.DataFrame, list]:
    """
    Fetch real historical rounds from DataGolf API.
    Extracts winners (first in "scores" = winner) with under-par and total scores.
    Returns (df_rounds, winners_list).
    """
    # Load config
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "configs" / "datagolf.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = load_yaml(cfg_path)

    base_url = cfg.get("base_url", "https://feeds.datagolf.com")
    endpoint = cfg.get("endpoints", {}).get("historical_rounds", {})
    path = endpoint.get("path", "historical-raw-data/rounds")
    env_var = cfg.get("auth", {}).get("env_var", "DATAGOLF_API_KEY")
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {env_var}")

    history_years_back = cfg.get("defaults", {}).get("history_years_back", 5)
    file_format = cfg.get("defaults", {}).get("file_format", "json")

    records = []
    winners_list = []
    current_year = 2024
    for year in range(current_year - history_years_back, current_year + 1):  # Include current year
        url = f"{base_url}/{path}"
        params = {"tour": tour, "event_id": event_id, "year": str(year), "file_format": file_format, "key": api_key}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                data = response.json()
            else:
                data = response.text

            if isinstance(data, str) and ("not available" in data.lower() or "invalid" in data.lower()):
                print(f"Skipping year {year}: Event not played or invalid.")
                continue

            if not isinstance(data, dict):
                print(f"Skipping year {year}: Unexpected response format.")
                continue

            scores = data.get("scores", [])
            if not scores:
                print(f"Skipping year {year}: No scores data.")
                continue

            # Extract yardage and par
            yardage = data.get("course_yardage")
            course_par = data.get("course_par")
            if course_par is None:
                course_par = 71  # Fallback par for Bermuda Championship (adjust if needed)

            # Extract winner (first in scores)
            winner_item = scores[0]
            player_name = winner_item.get("player_name")
            if player_name:
                total_strokes = 0
                for rnd in [1, 2, 3, 4]:
                    round_data = winner_item.get(f"round_{rnd}")
                    if round_data and "score" in round_data:
                        total_strokes += round_data["score"]
                under_par = total_strokes - (course_par * 4)
                winners_list.append({"year": year, "winner": player_name, "score": under_par, "total_score": total_strokes})

            # Process all players for rounds data
            for item in scores:
                pid = item.get("dg_id")
                yr = data.get("year", year)
                for rnd in [1, 2, 3, 4]:
                    round_key = f"round_{rnd}"
                    if round_key in item:
                        round_data = item[round_key]
                        sg = round_data.get("sg_total")
                        da = round_data.get("driving_acc")
                        dd = round_data.get("driving_dist")
                        if sg is not None:
                            records.append(
                                {
                                    "player_id": str(pid),
                                    "year": yr,
                                    "course_yardage": yardage,
                                    f"round_{rnd}.sg_total": sg,
                                    f"round_{rnd}.driving_acc": da,
                                    f"round_{rnd}.driving_dist": dd,
                                }
                            )
        except requests.exceptions.HTTPError as e:
            print(f"Skipping year {year}: HTTP error - {e}")
            continue
        except Exception as e:
            print(f"Skipping year {year}: Unexpected error - {e}")
            continue

    if not records:
        raise ValueError("No historical round data found for any valid years.")

    df = pd.DataFrame(records)
    agg_funcs = {col: (lambda x: x.dropna().iloc[0] if not x.dropna().empty else None) for col in df.columns if col not in ["player_id", "year"]}
    df = df.groupby(["player_id", "year"], as_index=False).agg(agg_funcs)
    return df, winners_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_id", type=str, required=True)
    ap.add_argument("--tour", type=str, default="pga", help="Tour to process")
    args = ap.parse_args()

    TOUR = args.tour

    processed = Path("data/processed") / TOUR
    metas = sorted(processed.glob("event_*_meta.json"))
    event_name = None
    for p in reversed(metas):
        meta = json.loads(p.read_text(encoding="utf-8"))
        if str(meta.get("event_id")) == str(args.event_id):
            event_name = meta.get("event_name")
            break
    if not event_name:
        raise ValueError(f"Event {args.event_id} not found in meta files.")

    safe_name = normalize_name(event_name)
    out_dir = Path("data/raw/historical") / TOUR
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = out_dir / f"tournament_{safe_name}_rounds_combined.parquet"
    winners_path = out_dir / f"tournament_{safe_name}_winners.json"

    try:
        df, winners_list = fetch_real_historical_rounds(event_name, args.event_id, TOUR)
        df.to_parquet(rounds_path, index=False)
        print(f"Saved real historical rounds to {rounds_path}")
        if winners_list:
            with open(winners_path, "w") as f:
                json.dump(winners_list, f, indent=2)
            print(f"Saved winners to {winners_path}")
    except Exception as e:
        print(f"Failed to fetch/save historical data: {e}")
        exit(1)


if __name__ == "__main__":
    main()
