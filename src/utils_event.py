# src/utils_event.py
#!/usr/bin/env python3
# Centralized event utilities (hardened resolver)

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

TOUR_DEFAULT = "pga"
ROOT = Path(__file__).resolve().parents[1]  # repo root: .../personal/golf-model
UNRESOLVED_EVENT_IDS = {"", "tbd", "none", "null", "nan"}


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


def is_resolved_event_id(event_id: object) -> bool:
    if event_id is None:
        return False
    return str(event_id).strip().lower() not in UNRESOLVED_EVENT_IDS


def current_week_bounds(reference_date: date | None = None, *, include_next_week_on_sunday: bool = False) -> tuple[date, date]:
    today = reference_date or datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    if include_next_week_on_sunday and today.weekday() == 6:
        end_of_week = start_of_week + timedelta(days=13)
    return start_of_week, end_of_week


def current_week_events(
    tour: str = TOUR_DEFAULT,
    *,
    reference_date: date | None = None,
    schedule_path: Path | None = None,
    require_resolved_id: bool = True,
    include_next_week_on_sunday: bool = False,
) -> list[dict]:
    upcoming_file = schedule_path or (ROOT / "upcoming-events.json")
    if not upcoming_file.exists():
        return []

    try:
        data = json.loads(upcoming_file.read_text(encoding="utf-8"))
    except Exception:
        return []

    schedule = data.get("schedule", [])
    if not isinstance(schedule, list):
        return []

    start_of_week, end_of_week = current_week_bounds(
        reference_date,
        include_next_week_on_sunday=include_next_week_on_sunday,
    )
    events: list[tuple[date, int | str, dict]] = []
    for event in schedule:
        if not isinstance(event, dict):
            continue
        if tour and str(event.get("tour", "")).lower() != str(tour).lower():
            continue

        event_id = event.get("event_id")
        if require_resolved_id and not is_resolved_event_id(event_id):
            continue

        start_date = event.get("start_date")
        if not start_date:
            continue
        try:
            event_dt = datetime.fromisoformat(str(start_date).replace("Z", "+00:00"))
        except Exception:
            continue
        event_date = event_dt.date()
        if start_of_week <= event_date <= end_of_week:
            try:
                sort_id: int | str = int(str(event_id))
            except Exception:
                sort_id = str(event_id)
            events.append((event_date, sort_id, event))

    return [event for _, _, event in sorted(events, key=lambda row: (row[0], row[1]))]


def current_week_event_ids(
    tour: str = TOUR_DEFAULT,
    *,
    reference_date: date | None = None,
    schedule_path: Path | None = None,
    require_resolved_id: bool = True,
    include_next_week_on_sunday: bool = False,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for event in current_week_events(
        tour,
        reference_date=reference_date,
        schedule_path=schedule_path,
        require_resolved_id=require_resolved_id,
        include_next_week_on_sunday=include_next_week_on_sunday,
    ):
        eid = str(event.get("event_id", "")).strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        ids.append(eid)
    return ids


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

    current_ids = current_week_event_ids(tour, require_resolved_id=True)
    if current_ids:
        return current_ids

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
