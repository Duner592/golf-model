#!/usr/bin/env python3
import json
import os
from pathlib import Path

import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_event_meta(tour: str = "pga") -> dict:
    meta_dir = Path(__file__).resolve().parent.parent / "data" / "meta" / tour
    metas = sorted(meta_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No event meta found. Run make_event_meta.py first.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def main():
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
    meta = load_event_meta(tour)
    event_id = meta["event_id"]

    tee_times_path = cfg["endpoints"].get("tee_times", {}).get("path", None)
    if not tee_times_path:
        raise NotImplementedError("Add 'tee_times' endpoint path in configs/datagolf.yaml")

    requests_cache.install_cache("dg_cache", expire_after=900)
    session = requests.Session()

    url = f"{base_url}/{tee_times_path.lstrip('/')}"
    params = {key_param: api_key, "event_id": event_id}
    resp = session.get(url, params=params, timeout=20)
    if resp.status_code == 404:
        print("Tee times not available yet. Try again later.")
        return
    resp.raise_for_status()
    data = resp.json()

    out_dir = root / "data" / "raw" / tour
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"event_{event_id}_tee-times.json"
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved tee times: {out_file}")


if __name__ == "__main__":
    main()
