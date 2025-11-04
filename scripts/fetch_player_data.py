#!/usr/bin/env python3
# scripts/fetch_player_data.py
#
# Refactored to use configs/datagolf.yaml with:
# - endpoints.dg_rankings (preds/get-dg-rankings)
# - endpoints.skill_ratings (preds/skill-ratings)
#
# It:
# - loads current event_id and field table (to get dg_id)
# - fetches DG rankings + skill ratings
# - normalizes and filters to this week's field
# - saves raw JSON and normalized Parquet under data/processed/{tour}/

import os
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()

TOUR_DEFAULT = "pga"


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_event_meta(tour: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(
            "No event meta found. Run parse_field_updates.py first."
        )
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def load_field_teetimes(tour: str, event_id: str) -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / tour
    for ext in (".parquet", ".csv"):
        p = processed / f"event_{event_id}_field_teetimes{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    # Fallback to base field
    for ext in (".parquet", ".csv"):
        p = processed / f"event_{event_id}_field{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError("Field tables not found. Run parse_field_updates.py first.")


def normalize_payload(obj: Any) -> pd.DataFrame:
    """
    Robust normalizer:
    - list[dict] -> DataFrame
    - dict with a list-valued key -> that list -> DataFrame
    - dict of scalars -> single-row DF
    """
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        # try to find the first list value
        for v in obj.values():
            if isinstance(v, list):
                return pd.json_normalize(v)
        return pd.json_normalize(obj)
    return pd.DataFrame()


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

    tour = cfg.get("defaults", {}).get("tour", TOUR_DEFAULT)
    default_file_format = cfg.get("defaults", {}).get("file_format", "json")

    eps = cfg["endpoints"]
    path_rankings = eps["dg_rankings"]["path"]
    path_skills = eps["skill_ratings"]["path"]

    # Load event + field to know current dg_id list
    meta = load_event_meta(tour)
    event_id = str(meta["event_id"])
    field_df = load_field_teetimes(tour, event_id)

    # Choose identifier from field (prefer dg_id)
    id_col = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in field_df.columns:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(
            "No player identifier found in field table (expected dg_id / player_id / id)."
        )

    player_ids = set(field_df[id_col].dropna().astype(str).tolist())
    print(f"Players in field (unique): {len(player_ids)} using id_col={id_col}")

    # HTTP session with cache
    requests_cache.install_cache("dg_cache", expire_after=900)
    session = requests.Session()

    processed_dir = root / "data" / "processed" / tour
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -------- Fetch DG Rankings --------
    # Params: only file_format is optional per your YAML
    r_params = {key_param: api_key}
    if "file_format" in eps["dg_rankings"].get("optional", []):
        r_params["file_format"] = default_file_format

    url_r = f"{base_url}/{path_rankings.lstrip('/')}"
    r_resp = session.get(url_r, params=r_params, timeout=30)
    r_resp.raise_for_status()
    rankings_raw = r_resp.json()
    rankings_df = normalize_payload(rankings_raw)

    # Align identifier
    if id_col not in rankings_df.columns:
        for alt in ["dg_id", "player_id", "id"]:
            if alt in rankings_df.columns:
                rankings_df = rankings_df.rename(columns={alt: id_col})
                break

    # Filter to this week's field if id present
    if id_col in rankings_df.columns:
        rankings_df = rankings_df[
            rankings_df[id_col].astype(str).isin(player_ids)
        ].copy()

    # Save raw + normalized
    (processed_dir / f"event_{event_id}_dg_rankings.json").write_text(
        json.dumps(rankings_raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rankings_df.to_parquet(
        processed_dir / f"event_{event_id}_dg_rankings.parquet", index=False
    )

    # -------- Fetch Skill Ratings --------
    # Optional params: display, file_format
    s_params = {key_param: api_key}
    if "file_format" in eps["skill_ratings"].get("optional", []):
        s_params["file_format"] = default_file_format
    # If you want a specific "display" (per DataGolf docs), set it here:
    # s_params["display"] = "by_player"  # example only; adjust to your account/docs

    url_s = f"{base_url}/{path_skills.lstrip('/')}"
    s_resp = session.get(url_s, params=s_params, timeout=45)
    s_resp.raise_for_status()
    skills_raw = s_resp.json()
    skills_df = normalize_payload(skills_raw)

    if id_col not in skills_df.columns:
        for alt in ["dg_id", "player_id", "id"]:
            if alt in skills_df.columns:
                skills_df = skills_df.rename(columns={alt: id_col})
                break

    if id_col in skills_df.columns:
        skills_df = skills_df[skills_df[id_col].astype(str).isin(player_ids)].copy()

    (processed_dir / f"event_{event_id}_skill_ratings.json").write_text(
        json.dumps(skills_raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    skills_df.to_parquet(
        processed_dir / f"event_{event_id}_skill_ratings.parquet", index=False
    )

    print("Saved player data under:", processed_dir)
    print(
        f"- {processed_dir / f'event_{event_id}_dg_rankings.parquet'} "
        f"(rows={len(rankings_df)})"
    )
    print(
        f"- {processed_dir / f'event_{event_id}_skill_ratings.parquet'} "
        f"(rows={len(skills_df)})"
    )


if __name__ == "__main__":
    main()
