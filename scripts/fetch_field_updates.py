#!/usr/bin/env python3
import os
import json
from datetime import datetime
import yaml
import requests
import requests_cache
from dotenv import load_dotenv

load_dotenv()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # Load config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "datagolf.yaml")
    cfg_path = os.path.abspath(cfg_path)
    cfg = load_yaml(cfg_path)

    base_url = cfg["base_url"].rstrip("/")
    key_param = cfg["auth"]["key_param"]
    api_key = os.getenv(cfg["auth"]["env_var"])
    if not api_key:
        raise RuntimeError(f"Missing API key in environment: {cfg['auth']['env_var']}")

    tour = cfg["defaults"]["tour"]
    endpoint = cfg["endpoints"]["field_updates"]["path"]

    # Cache to avoid repeated calls during testing
    requests_cache.install_cache("dg_cache", expire_after=900)
    session = requests.Session()

    url = f"{base_url}/{endpoint}"
    params = {key_param: api_key, "tour": tour}

    try:
        resp = session.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("HTTP error:", e)
        print("Status:", getattr(e.response, "status_code", "unknown"))
        print("Body:", getattr(e.response, "text", ""))
        raise

    data = resp.json()

    # Quick sanity output
    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
    else:
        print(f"Response type: {type(data)}")

    # Save pretty JSON (stable filename beside this script)
    out_path = os.path.join(os.path.dirname(__file__), "field-updates.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

    # Also save a dated copy per tour for history
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", tour)
    os.makedirs(out_dir, exist_ok=True)
    dated_path = os.path.join(out_dir, f"{ts}_field-updates.json")
    with open(dated_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {os.path.abspath(dated_path)}")


if __name__ == "__main__":
    main()
