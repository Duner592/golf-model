#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CFG_PATH = ROOT / "configs" / "datagolf.yaml"
TOUR_DEFAULT = "pga"  # fallback if not in config


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def any_r1_teetime_present(field: list[dict]) -> bool:
    """
    Returns True if any player in 'field' has a non-null R1 tee time.
    field-updates includes r1_teetime..r4_teetime which are null before release [2].
    """
    for p in field or []:
        if p.get("r1_teetime"):
            return True
    return False


def load_existing_meta(tour: str) -> Optional[Dict[str, Any]]:
    # Prefer canonical meta folder; fall back to processed if needed
    meta_dir = ROOT / "data" / "meta" / tour
    processed_dir = ROOT / "data" / "processed" / tour
    candidates = sorted(meta_dir.glob("event_*_meta.json")) or sorted(
        processed_dir.glob("event_*_meta.json")
    )
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def write_file(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    cfg = load_yaml(CFG_PATH)
    base_url = cfg["base_url"].rstrip("/")
    key_param = cfg["auth"]["key_param"]
    env_var = cfg["auth"]["env_var"]
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {env_var}")

    tour = cfg.get("defaults", {}).get("tour", TOUR_DEFAULT)
    endpoint = cfg["endpoints"]["field_updates"]["path"]

    # Short cache (5 min) so repeated checks are cheap but still refresh reasonably fast
    requests_cache.install_cache("dg_cache", expire_after=300)
    session = requests.Session()

    url = f"{base_url}/{endpoint.lstrip('/')}"
    params = {key_param: api_key, "tour": tour}

    # Fetch latest field-updates
    resp = session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # Save a fresh copy next to scripts (the same place fetch_field_updates.py writes)
    field_updates_path = SCRIPT_DIR / "field-updates.json"
    write_file(field_updates_path, data)

    # Inspect if R1 tee times are present now
    field = data.get("field", [])
    r1_present = any_r1_teetime_present(field)
    print(
        f"Tee times present: {r1_present} (event_id={data.get('event_id')}, last_updated={data.get('last_updated')})"
    )

    # Load existing meta to detect state change (optional but nice)
    meta_prev = load_existing_meta(tour)
    prev_flag = None
    if meta_prev is not None:
        prev_flag = meta_prev.get("has_r1_teetimes")

    # If tee times are present now (or state changed), re-parse to build tables
    if r1_present or (prev_flag is False and r1_present is not False):
        print("Rebuilding field tables with tee times...")
        # Reuse your parser which enriches tee times/waves
        # It reads scripts/field-updates.json and writes processed outputs
        os.system(f"python {str(SCRIPT_DIR / 'parse_field_updates.py')}")
    else:
        print("No update needed yet (still no R1 tee times).")


if __name__ == "__main__":
    main()
