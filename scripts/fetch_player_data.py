#!/usr/bin/env python3
# scripts/fetch_player_data.py
#
# Fetch DataGolf preds: dg_rankings and skill_ratings.
# Filters to players in the current event's field (processed).
# Saves files under event_{event_id}_*.parquet / .json
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def latest_meta(processed_dir: Path) -> dict:
    metas = sorted(processed_dir.glob("event_*_meta.json"), key=lambda p: p.stat().st_mtime)
    if not metas:
        raise FileNotFoundError("No event meta found.")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def load_field(processed_dir: Path, event_id: str) -> pd.DataFrame:
    for name in [
        f"event_{event_id}_field_teetimes.parquet",
        f"event_{event_id}_field_teetimes.csv",
        f"event_{event_id}_field.parquet",
        f"event_{event_id}_field.csv",
    ]:
        p = processed_dir / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"Field table not found for event {event_id}")


def normalize_payload(obj: Any) -> pd.DataFrame:
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return pd.json_normalize(v)
        return pd.json_normalize(obj)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_id", type=str, default=None, help="Force event_id")
    ap.add_argument("--tour", type=str, default="pga", help="Tour to process")
    args = ap.parse_args()

    TOUR = args.tour

    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    preds_dir = root / "data" / "preds" / TOUR
    preds_dir.mkdir(parents=True, exist_ok=True)

    meta = latest_meta(processed)
    event_id = str(args.event_id) if args.event_id else str(meta["event_id"])

    field_df = load_field(processed, event_id)
    id_col = "dg_id" if "dg_id" in field_df.columns else ("player_id" if "player_id" in field_df.columns else None)
    if id_col is None:
        raise ValueError("No dg_id/player_id in field table.")
    player_ids = sorted(set(field_df[id_col].dropna().astype(str)))

    cfg = load_yaml(root / "configs" / "datagolf.yaml")
    base = cfg["base_url"].rstrip("/")
    keyp = cfg["auth"]["key_param"]
    api_key = os.getenv(cfg["auth"]["env_var"])
    if not api_key:
        raise RuntimeError("Missing DATAGOLF_API_KEY")
    requests_cache.install_cache("dg_cache", expire_after=900)
    session = requests.Session()

    # DG rankings
    r_path = cfg["endpoints"]["dg_rankings"]["path"]
    r_params = {keyp: api_key}
    if "file_format" in cfg["endpoints"]["dg_rankings"].get("optional", []):
        r_params["file_format"] = cfg["defaults"].get("file_format", "json")
    r = session.get(f"{base}/{r_path.lstrip('/')}", params=r_params, timeout=30)
    r.raise_for_status()
    rankings_raw = r.json()
    rankings_df = normalize_payload(rankings_raw)
    if id_col in rankings_df.columns:
        rankings_df = rankings_df[rankings_df[id_col].astype(str).isin(player_ids)]
    else:
        for alt in ["dg_id", "player_id", "id"]:
            if alt in rankings_df.columns:
                rankings_df = rankings_df[rankings_df[alt].astype(str).isin(player_ids)].rename(columns={alt: id_col})
                break

    # Skill ratings
    s_path = cfg["endpoints"]["skill_ratings"]["path"]
    s_params = {keyp: api_key}
    if "file_format" in cfg["endpoints"]["skill_ratings"].get("optional", []):
        s_params["file_format"] = cfg["defaults"].get("file_format", "json")
    s = session.get(f"{base}/{s_path.lstrip('/')}", params=s_params, timeout=45)
    s.raise_for_status()
    skills_raw = s.json()
    skills_df = normalize_payload(skills_raw)
    if id_col not in skills_df.columns:
        for alt in ["dg_id", "player_id", "id"]:
            if alt in skills_df.columns:
                skills_df = skills_df.rename(columns={alt: id_col})
                break
    skills_df = skills_df[skills_df[id_col].astype(str).isin(player_ids)]

    # Save under the correct event_id
    (processed / f"event_{event_id}_dg_rankings.json").write_text(json.dumps(rankings_raw, indent=2), encoding="utf-8")
    rankings_df.to_parquet(processed / f"event_{event_id}_dg_rankings.parquet", index=False)
    (processed / f"event_{event_id}_skill_ratings.json").write_text(json.dumps(skills_raw, indent=2), encoding="utf-8")
    skills_df.to_parquet(processed / f"event_{event_id}_skill_ratings.parquet", index=False)

    print("Saved player data under:", processed)
    print(
        "-",
        processed / f"event_{event_id}_dg_rankings.parquet",
        f"(rows={len(rankings_df)})",
    )
    print(
        "-",
        processed / f"event_{event_id}_skill_ratings.parquet",
        f"(rows={len(skills_df)})",
    )


if __name__ == "__main__":
    main()
