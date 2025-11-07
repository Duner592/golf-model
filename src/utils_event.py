#!/usr/bin/env python3
# src/utils_event.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TOUR_DEFAULT = "pga"
ROOT = Path(__file__).resolve().parents[1]  # repo root


# ---------- Meta / event id ----------


def list_meta(tour: str = TOUR_DEFAULT) -> list[Path]:
    return sorted((ROOT / "data" / "processed" / tour).glob("event_*_meta.json"))


def load_latest_meta(tour: str = TOUR_DEFAULT) -> dict:
    metas = list_meta(tour)
    if not metas:
        raise FileNotFoundError(f"No meta under data/processed/{tour}")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def resolve_event_id(cli_event_id: str | None = None, tour: str = TOUR_DEFAULT) -> str:
    """
    Resolve event_id with priority:
      1) cli_event_id if provided
      2) scripts/field-updates.json (current week)
      3) latest processed meta
    """
    if cli_event_id:
        return str(cli_event_id)

    fu = ROOT / "scripts" / "field-updates.json"
    if fu.exists():
        try:
            data = json.loads(fu.read_text(encoding="utf-8"))
            eid = data.get("event_id")
            if eid is not None:
                return str(eid)
        except Exception:
            pass

    meta = load_latest_meta(tour)
    eid = meta.get("event_id")
    if eid is None:
        raise ValueError("event_id missing in latest meta")
    return str(eid)


# ---------- Field / weather loaders ----------


def load_field_table(event_id: str, tour: str = TOUR_DEFAULT) -> pd.DataFrame:
    """
    Prefer tee-time enriched field; fallback to base field.
    """
    processed = ROOT / "data" / "processed" / tour
    candidates = [
        processed / f"event_{event_id}_field_teetimes.parquet",
        processed / f"event_{event_id}_field_teetimes.csv",
        processed / f"event_{event_id}_field.parquet",
        processed / f"event_{event_id}_field.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"No field table found for event_{event_id}")


def weather_paths(event_id: str, tour: str = TOUR_DEFAULT) -> tuple[Path, Path]:
    processed = ROOT / "data" / "processed" / tour
    neu = processed / f"event_{event_id}_weather_round_neutral.parquet"
    wav = processed / f"event_{event_id}_weather_round_wave.parquet"
    return neu, wav


def load_weather_neutral(event_id: str, tour: str = TOUR_DEFAULT) -> pd.DataFrame:
    neu, _ = weather_paths(event_id, tour)
    if not neu.exists():
        raise FileNotFoundError(f"Missing neutral weather summary: {neu}")
    return pd.read_parquet(neu)


def try_load_weather_wave(event_id: str, tour: str = TOUR_DEFAULT) -> pd.DataFrame | None:
    _, wav = weather_paths(event_id, tour)
    return pd.read_parquet(wav) if wav.exists() else None


# ---------- Join-key helper ----------


def choose_join_key(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    """
    Pick a join key and auto-align by renaming 'b' if needed.
    """
    if "dg_id" in a.columns and "dg_id" in b.columns:
        return "dg_id"
    if "dg_id" in a.columns and "player_id" in b.columns:
        b.rename(columns={"player_id": "dg_id"}, inplace=True)
        return "dg_id"
    if "player_id" in a.columns and "player_id" in b.columns:
        return "player_id"
    if "player_id" in a.columns and "dg_id" in b.columns:
        b.rename(columns={"dg_id": "player_id"}, inplace=True)
        return "player_id"
    if "player_name" in a.columns and "player_name" in b.columns:
        return "player_name"
    return None
