# src/utils_event.py
#!/usr/bin/env python3
# Centralized event utilities (hardened resolver)

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

TOUR_DEFAULT = "pga"
ROOT = Path(__file__).resolve().parents[1]  # repo root: .../personal/golf-model


def _parse_ts(iso: str | None) -> float:
    """Parse common ISO-like timestamps to epoch seconds; return 0.0 if unknown."""
    if not iso:
        return 0.0
    s = iso.replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except Exception:
            pass
    return 0.0


def list_meta(tour: str = TOUR_DEFAULT) -> list[Path]:
    return sorted((ROOT / "data" / "processed" / tour).glob("event_*_meta.json"))


def load_latest_meta(tour: str = TOUR_DEFAULT) -> dict:
    metas = list_meta(tour)
    if not metas:
        raise FileNotFoundError(f"No meta under data/processed/{tour}")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def _parse_cli_event_ids(cli_event_ids: str | None) -> list[str]:
    if not cli_event_ids:
        return []
    parts = [p.strip() for p in re.split(r"[\s,]+", str(cli_event_ids)) if p.strip()]
    return [str(p) for p in parts]


def _resolve_single_event_id(cli_event_id: str | None = None, tour: str = TOUR_DEFAULT) -> str:
    """
    Priority:
      1) CLI --event_id
      2) scripts/field-updates.json (current week)
      3) Latest processed meta by saved_at_utc (fallback: file mtime)

    Prevents drift to e.g. event_9 by lexicographic filename order.
    """
    if cli_event_id:
        return str(cli_event_id)

    # Prefer current week from field-updates.json (written by fetch_field_updates.py),
    # but only when it belongs to the requested tour. This file is shared across
    # tours, so using a stale PGA payload for a Euro run can drift to the wrong id.
    fu = ROOT / "scripts" / "field-updates.json"
    if fu.exists():
        try:
            data = json.loads(fu.read_text(encoding="utf-8"))
            eid = data.get("event_id")
            payload_tour = data.get("tour")
            if eid is not None and (not tour or not payload_tour or str(payload_tour).lower() == str(tour).lower()):
                return str(eid)
        except Exception:
            pass

    # Fallback: most recent processed meta by saved_at_utc (or file mtime)
    processed = ROOT / "data" / "processed" / tour
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta files under {processed}")

    best_eid, best_ts = None, -1.0
    for p in metas:
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            ts = _parse_ts(meta.get("saved_at_utc"))
            if ts > best_ts:
                best_ts = ts
                best_eid = meta.get("event_id")
        except Exception:
            ts = p.stat().st_mtime
            if ts > best_ts:
                best_ts = ts
                try:
                    best_eid = json.loads(p.read_text(encoding="utf-8")).get("event_id")
                except Exception:
                    best_eid = None

    if best_eid is None:
        meta = json.loads(metas[-1].read_text(encoding="utf-8"))
        best_eid = meta.get("event_id")
    if best_eid is None:
        raise ValueError("Could not determine event_id from meta files.")
    return str(best_eid)


def resolve_event_ids(cli_event_ids: str | None = None, tour: str = TOUR_DEFAULT) -> list[str]:
    parsed = _parse_cli_event_ids(cli_event_ids)
    if parsed:
        seen = set()
        ordered = []
        for eid in parsed:
            if eid in seen:
                continue
            seen.add(eid)
            ordered.append(eid)
        return ordered

    upcoming_file = ROOT / "upcoming-events.json"
    events_in_week: list[tuple[datetime, str]] = []
    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    if upcoming_file.exists():
        try:
            data = json.loads(upcoming_file.read_text(encoding="utf-8"))
            schedule = data.get("schedule", [])
            if tour:
                schedule = [e for e in schedule if e.get("tour") == tour]
            for event in schedule:
                if not isinstance(event, dict):
                    continue
                eid = event.get("event_id")
                start_date = event.get("start_date")
                if eid is None or start_date is None:
                    continue
                try:
                    event_date = datetime.fromisoformat(start_date).date()
                except Exception:
                    continue
                if start_of_week <= event_date <= end_of_week:
                    events_in_week.append((datetime.fromisoformat(start_date), str(eid)))
        except Exception:
            events_in_week = []

    def _sort_key_pair(pair: tuple[datetime, str]) -> tuple[datetime, int | str]:
        dt, eid = pair
        try:
            return dt, int(str(eid))
        except Exception:
            return dt, str(eid)

    if events_in_week:
        events_in_week.sort(key=_sort_key_pair)
        ordered = [eid for _, eid in events_in_week]
        seen = set()
        unique = []
        for eid in ordered:
            if eid not in seen:
                seen.add(eid)
                unique.append(eid)
        return unique

    single = _resolve_single_event_id(None, tour)
    return [single] if single is not None else []


def resolve_event_id(cli_event_id: str | None = None, tour: str = TOUR_DEFAULT) -> str:
    ids = resolve_event_ids(cli_event_id, tour)
    if not ids:
        raise ValueError("Could not determine event_id.")
    return ids[0]


def load_field_table(event_id: str, tour: str = TOUR_DEFAULT):
    """
    Prefer tee-time enriched field; fallback to base field.
    """
    import pandas as pd

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


def load_weather_neutral(event_id: str, tour: str = TOUR_DEFAULT):
    import pandas as pd

    neu, _ = weather_paths(event_id, tour)
    if not neu.exists():
        raise FileNotFoundError(f"Missing neutral weather summary: {neu}")
    return pd.read_parquet(neu)


def try_load_weather_wave(event_id: str, tour: str = TOUR_DEFAULT):
    import pandas as pd

    _, wav = weather_paths(event_id, tour)
    return pd.read_parquet(wav) if wav.exists() else None


def choose_join_key(a, b) -> str | None:
    """
    Pick a join key and auto-align by renaming b if needed.
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
