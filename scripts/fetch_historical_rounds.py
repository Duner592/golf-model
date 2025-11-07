#!/usr/bin/env python3
# scripts/fetch_historical_rounds.py
# Fetch previous years’ round-level data for the current event.
# Event ID discovery order:
#   1) data/processed/{tour}/event_*_meta.json (from parse_field_updates.py)
#   2) scripts/field-updates.json (from fetch_field_updates.py)
#   3) --event_id CLI arg (required if 1/2 missing)
#
# Years:
#   - default years_back from YAML (defaults.history_years_back), fallback 5
#   - excludes the current/unplayed year (max_year = min(today.year, r1_year) - 1)
#   - r1_year from weather_meta if available; else use today.year - 1 logic
#
# Outputs:
#   - data/raw/historical/{tour}/event_{event_id}_{year}_rounds.json
#   - data/raw/historical/{tour}/event_{event_id}_rounds_combined.parquet (if any)


import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def find_event_id(tour: str, root: Path) -> str | None:
    # 1) processed meta
    processed = root / "data" / "processed" / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if metas:
        try:
            meta = json.loads(metas[-1].read_text(encoding="utf-8"))
            eid = meta.get("event_id")
            if eid is not None:
                return str(eid)
        except Exception:
            pass
    # 2) scripts/field-updates.json
    fu = root / "scripts" / "field-updates.json"
    if fu.exists():
        try:
            data = json.loads(fu.read_text(encoding="utf-8"))
            eid = data.get("event_id")
            if eid is not None:
                return str(eid)
        except Exception:
            pass
    return None


def load_round1_year(processed_dir: Path) -> int | None:
    wmetas = sorted(processed_dir.glob("event_*_weather_meta.json"))
    if wmetas:
        try:
            wmeta = json.loads(wmetas[-1].read_text(encoding="utf-8"))
            r1 = wmeta.get("r1_date")
            if r1:
                return datetime.strptime(r1, "%Y-%m-%d").year
        except Exception:
            pass
    return None


def pick_years_to_fetch(r1_year: int | None, years_back: int, min_year: int | None) -> list[int]:
    today_year = datetime.utcnow().year
    max_year = (min(today_year, r1_year) - 1) if r1_year else (today_year - 1)
    years = [max_year - i for i in range(years_back)]
    if min_year is not None:
        years = [y for y in years if y >= min_year]
    years = [y for y in years if 1900 <= y <= today_year - 1]
    return sorted(set(years), reverse=True)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Fetch historical rounds for the current event (previous years only).")
    parser.add_argument(
        "--event_id",
        type=str,
        default=None,
        help="Override event_id (if discovery fails).",
    )
    parser.add_argument(
        "--years_back",
        type=int,
        default=None,
        help="How many past years to fetch (default from YAML or 5).",
    )
    parser.add_argument(
        "--min_year",
        type=int,
        default=None,
        help="Minimum historical year to include (optional).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    cfg = load_yaml(root / "configs" / "datagolf.yaml")

    base_url = cfg["base_url"].rstrip("/")
    key_param = cfg["auth"]["key_param"]
    env_var = cfg["auth"]["env_var"]
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {env_var}")

    tour = cfg["defaults"]["tour"]
    file_fmt = cfg["defaults"].get("file_format", "json")
    default_years_back = int(cfg["defaults"].get("history_years_back", 5))
    years_back = args.years_back if args.years_back is not None else default_years_back

    endpoint = cfg["endpoints"]["historical_rounds"]["path"]

    # Discover event_id (no YAML default)
    event_id = args.event_id or find_event_id(tour, root)
    if not event_id:
        raise ValueError("Could not determine event_id. Run parse_field_updates.py first, or pass --event_id.")

    processed_dir = root / "data" / "processed" / tour
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Determine historical years (exclude current/unplayed)
    r1_year = load_round1_year(processed_dir)
    years_to_fetch = pick_years_to_fetch(r1_year=r1_year, years_back=years_back, min_year=args.min_year)
    if not years_to_fetch:
        print("No valid past years to fetch. Adjust years_back or ensure weather_meta exists for better bounds.")
        return

    print(f"Using event_id={event_id} (tour={tour}); fetching past years: {years_to_fetch}")

    requests_cache.install_cache("dg_cache", expire_after=900)
    session = requests.Session()

    url = f"{base_url}/{endpoint.lstrip('/')}"
    out_dir = root / "data" / "raw" / "historical" / tour
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for year in years_to_fetch:
        params = {
            key_param: api_key,
            "tour": tour,
            "event_id": event_id,
            "year": str(year),
            "file_format": file_fmt,
        }
        try:
            resp = session.get(url, params=params, timeout=120)
            if resp.status_code in (400, 404):
                print(f"Skip year={year}: HTTP {resp.status_code} (no data).")
                continue
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            print(f"Error for year={year}: {e}")
            continue

        out_path = out_dir / f"event_{event_id}_{year}_rounds.json"
        save_json(data, out_path)
        print(f"Saved: {out_path}")

        # Normalize to DataFrame and collect
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            vlist = None
            for v in data.values():
                if isinstance(v, list):
                    vlist = v
                    break
            df = pd.json_normalize(vlist) if vlist is not None else pd.json_normalize(data)
        else:
            df = pd.DataFrame()

        if not df.empty:
            df["event_id"] = event_id
            df["year"] = int(year)
            combined.append(df)

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        combined_out = out_dir / f"event_{event_id}_rounds_combined.parquet"
        combined_df.to_parquet(combined_out, index=False)
        print(f"Saved combined Parquet: {combined_out}")
    else:
        print("No historical rounds saved (no available past years or empty responses).")


if __name__ == "__main__":
    main()
