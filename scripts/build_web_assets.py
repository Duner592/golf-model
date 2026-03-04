#!/usr/bin/env python3
# scripts/build_web_assets.py
#
# Build static web assets from the latest run:
#   web/{tour}/leaderboard.json
#   web/{tour}/summary.json
#   web/{tour}/meta.json
#   web/{tour}/weather_round_neutral.json
#   web/{tour}/weather_round_wave.json
#   web/{tour}/weather_meta.json
#   web/{tour}/course_fit_weights.json           (if available)
#   web/{tour}/course_history_summary.json       (if available)
#   web/{tour}/tournament_summary.json           (course, yardage, location, start date, field size, last 5 winners)
#   web/{tour}/field_teetimes.csv                (if available)
#   web/{tour}/downloads/<stamped leaderboard CSV/HTML>

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests  # Added for API calls

# ensure repo root is importable when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils_event import resolve_event_ids

MPH_PER_MPS = 2.237
KMH_TO_MPH = 0.621371


# ---------- basic I/O ----------
def latest_meta(processed_dir: Path) -> dict:
    """
    Retrieve the latest processed meta file for any event.

    Args:
        processed_dir (Path): Directory containing processed event data.

    Returns:
        dict: Contents of the most recent event_*_meta.json file.

    Raises:
        FileNotFoundError: If no meta files are found.
    """
    metas = sorted(processed_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta in {processed_dir}")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def pick_latest_timestamped_leaderboard(preds_dir: Path, event_id: str) -> tuple[Path | None, Path | None]:
    """
    Select the latest timestamped leaderboard CSV and optional HTML for a given event.
    Returns None, None if no files are found (instead of raising).
    """
    stamped = sorted(preds_dir.glob(f"event_{event_id}_*_leaderboard.csv"))
    html = None
    if stamped:
        lb = stamped[-1]
        candidate_html = lb.with_suffix(".html")
        if candidate_html.exists():
            html = candidate_html
        return lb, html
    lb = preds_dir / f"event_{event_id}_leaderboard.csv"
    if lb.exists():
        candidate_html = lb.with_suffix(".html")
        if candidate_html.exists():
            html = candidate_html
        return lb, html
    return None, None  # Return None instead of raising FileNotFoundError


def pick_matching_summary(preds_dir: Path, csv_path: Path) -> Path | None:
    """
    Find a matching summary JSON file based on the CSV path.

    Args:
        preds_dir (Path): Directory containing prediction data.
        csv_path (Path): Path to the leaderboard CSV.

    Returns:
        Path or None: Path to the summary JSON if found.
    """
    base = csv_path.name.replace("_leaderboard.csv", "")
    candidate = preds_dir / f"{base}_summary.json"
    if candidate.exists():
        return candidate
    cand = sorted(preds_dir.glob("*_leaderboard_summary.json"))
    return cand[-1] if cand else None


def _sanitize_jsonable(obj):
    """
    Recursively replace NaN/Inf with None for JSON compliance.

    Args:
        obj: Any Python object.

    Returns:
        Sanitized object suitable for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_jsonable(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # numpy scalars
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return obj


def write_json(path: Path, obj) -> None:
    """
    Write a sanitized JSON object to a file.

    Args:
        path (Path): Output file path.
        obj: Object to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_obj = _sanitize_jsonable(obj)
    path.write_text(json.dumps(safe_obj, indent=2, allow_nan=False), encoding="utf-8")


# ---------- formatting helpers ----------
def normalize_utc_str(s: str | None, fallback: str) -> str:
    """
    Normalize a UTC timestamp string to a readable format.

    Args:
        s (str or None): Input timestamp.
        fallback (str): Fallback if parsing fails.

    Returns:
        str: Formatted timestamp.
    """
    if not s:
        return fallback
    candidates = [
        ("%Y-%m-%dT%HM%SZ", s.replace(":", "").replace("-", "")),
        ("%Y-%m-%dT%H:%M:%SZ", s),
        ("%Y-%m-%d %H:%M:%S", s.replace("T", " ").replace("Z", "")),
        ("%Y-%m-%dT%H:%M:%S", s.replace("Z", "")),
    ]
    for fmt, val in candidates:
        try:
            dt = datetime.strptime(val, fmt)
            return dt.strftime("%d-%b-%Y %H:%M:%S")
        except Exception:
            pass
    return fallback


def _coerce_wind_fields(rec: dict) -> dict:
    """
    Coerce wind fields to standardized units (MPH).

    Args:
        rec (dict): Weather record.

    Returns:
        dict: Updated record with wind_mph and gust_mph.
    """
    w = rec.get("wind_mph")
    g = rec.get("gust_mph")
    if w is None:
        if rec.get("wind_mps") is not None:
            w = float(rec["wind_mps"]) * MPH_PER_MPS
        elif rec.get("wind_kmh") is not None:
            w = float(rec["wind_kmh"]) * KMH_TO_MPH
    if g is None:
        if rec.get("gust_mps") is not None:
            g = float(rec["gust_mps"]) * MPH_PER_MPS
        elif rec.get("gust_kmh") is not None:
            g = float(rec["gust_kmh"]) * KMH_TO_MPH
    p = rec.get("precip_pct")
    if p is None and rec.get("precip_prob") is not None:
        p = float(rec["precip_prob"])
    out = dict(rec)
    out["wind_mph"] = round(w, 2) if w is not None else None
    out["gust_mph"] = round(g, 2) if g is not None else None
    out["precip_pct"] = round(p, 0) if p is not None else None
    return out


def _time_only(val: str | None) -> str:
    """
    Extract time-only string from a value.

    Args:
        val (str or None): Input value.

    Returns:
        str: Time string or empty.
    """
    if not val or not isinstance(val, str):
        return ""
    m = re.search(r"\b(\d{1,2}:\d{2})\b", val.strip())
    return m.group(1) if m else ""


def _norm_name(s: str | None) -> str:
    """
    Normalize a name for keying.

    Args:
        s (str or None): Input name.

    Returns:
        str: Normalized name.
    """
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _numeric_mode(series: pd.Series) -> int | None:
    """
    Compute the mode of a numeric series.

    Args:
        series (pd.Series): Input series.

    Returns:
        int or None: Mode value.
    """
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        modes = s.mode()
        if not modes.empty:
            return int(round(modes.iloc[0]))
        return int(round(s.median()))
    except Exception:
        return None


# ---------- weather parquet -> json ----------
def neutral_parquet_to_json(neutral_pq: Path, out_json: Path) -> bool:
    """
    Convert neutral weather parquet to JSON.

    Args:
        neutral_pq (Path): Input parquet file.
        out_json (Path): Output JSON file.

    Returns:
        bool: True if successful.
    """
    if not neutral_pq.exists():
        return False
    df = pd.read_parquet(neutral_pq)
    records = df.to_dict(orient="records")
    records = [_coerce_wind_fields(r) for r in records]
    write_json(out_json, records)
    return True


def wave_parquet_to_json(wave_pq: Path, out_json: Path) -> bool:
    """
    Convert wave weather parquet to JSON.

    Args:
        wave_pq (Path): Input parquet file.
        out_json (Path): Output JSON file.

    Returns:
        bool: True if successful.
    """
    if not wave_pq.exists():
        return False
    df = pd.read_parquet(wave_pq)
    records = df.to_dict(orient="records")
    records = [_coerce_wind_fields(r) for r in records]
    write_json(out_json, records)
    return True


# ---------- course history summary ----------
def build_history_summary(stats_path: Path, out_json: Path) -> bool:
    """
    Build course history summary from stats parquet.

    Args:
        stats_path (Path): Input parquet file.
        out_json (Path): Output JSON file.

    Returns:
        bool: True if successful.
    """
    if not stats_path.exists():
        return False
    df = pd.read_parquet(stats_path)
    payload = {
        "rows": int(len(df)),
        "rounds_course_avg": (float(df["rounds_course"].mean()) if "rounds_course" in df else None),
        "sg_course_mean_shrunk_avg": (float(df["sg_course_mean_shrunk"].mean()) if "sg_course_mean_shrunk" in df else None),
        "top_rounds": (df.sort_values("rounds_course", ascending=False).head(10).to_dict(orient="records") if "rounds_course" in df else []),
        "top_sg_course": (df.sort_values("sg_course_mean_shrunk", ascending=False).head(10).to_dict(orient="records") if "sg_course_mean_shrunk" in df else []),
    }
    write_json(out_json, payload)
    return True


# ---------- winners + yardage helpers (robust) ----------
def _winner_from_event_json(path: Path) -> tuple[str | None, float | None]:
    """
    Extract winner from an event JSON file.

    Args:
        path (Path): JSON file path.

    Returns:
        tuple: (winner name, score) or (None, None).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
        if not rows:
            return None, None
        df = pd.json_normalize(rows)
        if "fin_text" in df.columns:
            mask = df["fin_text"].astype(str).str.upper().str.replace("^T", "", regex=True).isin(["1", "W", "WIN"])
            if mask.any():
                r = df.loc[mask].iloc[0]
                score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
                total = float(pd.to_numeric(r[score_cols], errors="coerce").sum()) if score_cols else None
                return str(r.get("player_name", "")), total
        score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
        if score_cols:
            sc = df[score_cols].apply(pd.to_numeric, errors="coerce")
            df["__total"] = sc.sum(axis=1, min_count=1)
            df2 = df.dropna(subset=["__total"])
            if not df2.empty:
                r = df2.sort_values("__total", ascending=True).iloc[0]
                return str(r.get("player_name", "")), float(r["__total"])
    except Exception:
        return None, None
    return None, None


def _clean_int_series_from_any(s: pd.Series) -> pd.Series:
    """
    Clean and convert series to numeric.

    Args:
        s (pd.Series): Input series.

    Returns:
        pd.Series: Cleaned numeric series.
    """
    if s.dtype.kind in ("i", "u", "f"):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _pick_course_yardage(field: pd.DataFrame) -> int | None:
    """
    Pick course yardage from field data.

    Args:
        field (pd.DataFrame): Field data.

    Returns:
        int or None: Yardage value.
    """
    include_regexes = [
        r"(^|^.*\b)(course_)?(total_)?yardage($|\b.*$)",
        r"(^|^.*\b)(course_)?(total_)?yards($|\b.*$)",
        r"(^|^)yardage($|$)",
        r"(^|^)yards($|$)",
    ]
    exclude_regexes = [
        r"driv",
        r"carry",
        r"avg",
        r"gain",
        r"putt",
        r"scram",
        r"approach",
        r"chip",
        r"per_",
        r"_per",
        r"prox",
    ]

    def include_col(c: str) -> bool:
        cl = c.lower()
        if any(re.search(rx, cl) for rx in exclude_regexes):
            return False
        return any(re.search(rx, cl) for rx in include_regexes)

    candidates = []
    for c in field.columns:
        if not include_col(c):
            continue
        s = _clean_int_series_from_any(field[c])
        s = s[(s >= 5800) & (s <= 8200)]
        if not s.dropna().empty:
            candidates.append(int(round(s.median())))
    if candidates:
        ser = pd.Series(candidates)
        m = ser.mode()
        return int(m.iloc[0]) if not m.empty else int(round(ser.median()))

    for c in ["total_yardage", "course_yardage", "yardage", "yards", "course_total_yards"]:
        if c in field.columns:
            s = _clean_int_series_from_any(field[c])
            s = s[(s >= 5800) & (s <= 8200)]
            if not s.dropna().empty:
                return int(round(s.median()))
    return None


def _detect_year_column(df: pd.DataFrame) -> pd.Series | None:
    """
    Detect year column in DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series or None: Year series.
    """
    for c in ["year", "season", "event_year", "tournament_year"]:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors="coerce")
            if y.notna().any():
                return y.astype("Int64")
    # parse from dates if available
    for c in ["event_date", "tournament_date", "start_date", "date"]:
        if c in df.columns:
            try:
                dt = pd.to_datetime(df[c], errors="coerce", utc=False)
                if dt.notna().any():
                    return dt.dt.year.astype("Int64")
            except Exception:
                pass
    return None


def _is_winner_fin(fin_val) -> bool:
    """
    Check if fin_val indicates a winner.

    Args:
        fin_val: Fin value.

    Returns:
        bool: True if winner.
    """
    if pd.isna(fin_val):
        return False
    s = str(fin_val).strip().upper()
    if s.startswith("T"):
        s = s[1:]
    return s in {"1", "W", "WIN"}


def _compute_total_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute total score from round scores.

    Args:
        df (pd.DataFrame): DataFrame with round scores.

    Returns:
        pd.Series: Total scores.
    """
    score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
    if not score_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    tmp = df[score_cols].apply(pd.to_numeric, errors="coerce")
    return tmp.sum(axis=1, min_count=1)


def _winners_from_df(df: pd.DataFrame) -> list[dict]:
    """
    Extract winners from DataFrame.

    Args:
        df (pd.DataFrame): Historical data.

    Returns:
        list[dict]: List of winners.
    """
    year = _detect_year_column(df)
    if year is None:
        return []
    df = df.copy()
    df["__year"] = year
    df["__total"] = _compute_total_score(df)
    if "fin_text" in df.columns:
        df["__is_win"] = df["fin_text"].map(_is_winner_fin)
    else:
        df["__is_win"] = False

    winners = []
    for y, g in df.groupby("__year"):
        if pd.isna(y):
            continue
        g2 = g
        if g2["__is_win"].any():
            row = g2[g2["__is_win"]].iloc[0]
        else:
            g2 = g2.dropna(subset=["__total"])
            if g2.empty:
                continue
            row = g2.sort_values("__total", ascending=True).iloc[0]

        name = None
        for nc in ["player_name", "name", "Player", "player"]:
            if nc in row and pd.notna(row[nc]):
                name = str(row[nc])
                break
        total = float(row["__total"]) if pd.notna(row["__total"]) else None
        if name:
            winners.append({"year": int(y), "winner": name, "score": total})

    winners = sorted(winners, key=lambda r: r["year"], reverse=True)
    return winners


def _slug_event(s: str | None) -> str:
    """
    Slugify event name.

    Args:
        s (str or None): Event name.

    Returns:
        str: Slug.
    """
    # normalize + underscore; matches saved combined parquet
    return _norm_name(s or "").replace(" ", "_")


def _load_hist_combined(raw_hist_dir: Path, event_name: str) -> pd.DataFrame | None:
    """
    Load combined historical parquet.

    Args:
        raw_hist_dir (Path): Historical data directory.
        event_name (str): Event name.

    Returns:
        pd.DataFrame or None: Loaded DataFrame.
    """
    slug = _slug_event(event_name)
    p = raw_hist_dir / f"tournament_{slug}_rounds_combined.parquet"
    print(f"DEBUG: looking for combined hist parquet at: {p}")
    if p.exists():
        try:
            df = pd.read_parquet(p)
            print(f"DEBUG: loaded combined hist parquet rows={len(df)}")
            return df
        except Exception as e:
            print(f"Warn: failed to read {p}: {e}")
            return None
    print("DEBUG: combined hist parquet not found")
    return None


def _collect_winners_from_files(raw_hist_dir: Path, event_id: str) -> list[dict]:
    """
    Collect winners from historical files.

    Args:
        raw_hist_dir (Path): Historical data directory.
        event_id (str): Event ID.

    Returns:
        list[dict]: List of winners.
    """
    pats = [
        f"event_{event_id}_*_rounds.json",
        f"event_{event_id}_*_results.json",
        f"event_{event_id}_*_leaderboard.json",
    ]
    files = []
    for pat in pats:
        files.extend(raw_hist_dir.glob(pat))

    def _year_from_name(p: Path) -> int:
        m = re.match(rf"event_{re.escape(str(event_id))}_(\d{{4}})_", p.name)
        return int(m.group(1)) if m else -1

    winners = []
    for f in sorted(files, key=_year_from_name, reverse=True):
        y = _year_from_name(f)
        if y < 0:
            continue
        name, total = _winner_from_event_json(f)
        if name:
            winners.append({"year": y, "winner": name, "score": total})

    # de-dup by year
    out = {}
    for w in winners:
        out.setdefault(w.get("year"), w)
    winners = sorted(out.values(), key=lambda r: r["year"], reverse=True)
    return winners


def _read_json(path: Path) -> dict | None:
    """
    Read JSON file.

    Args:
        path (Path): File path.

    Returns:
        dict or None: JSON content.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_course_catalog(root: Path) -> dict:
    """
    Load course catalog.

    Args:
        root (Path): Project root.

    Returns:
        dict: Catalog data.
    """
    for rel in ["data/static/course_catalog.json", "static/course_catalog.json"]:
        p = root / rel
        if p.exists():
            data = _read_json(p)
            if isinstance(data, dict):
                return data
    return {}


def _lookup_course_catalog(course_name: str | None, catalog: dict) -> dict:
    """
    Lookup course in catalog.

    Args:
        course_name (str or None): Course name.
        catalog (dict): Catalog dict.

    Returns:
        dict: Course entry.
    """
    if not isinstance(catalog, dict) or not course_name:
        return {}
    key = _norm_name(course_name)
    if key in catalog and isinstance(catalog[key], dict):
        return catalog[key]
    if course_name in catalog and isinstance(catalog[course_name], dict):
        return catalog[course_name]
    return {}


def _pick_yardage_from_meta(meta_proc: dict) -> int | None:
    """
    Pick yardage from meta.

    Args:
        meta_proc (dict): Meta data.

    Returns:
        int or None: Yardage.
    """
    for k in ["total_yardage", "course_yardage", "yardage", "yards"]:
        v = meta_proc.get(k)
        if v is None:
            continue
        try:
            if isinstance(v, (int, float)):
                n = int(round(float(v)))
            else:
                n = int(float(re.sub(r"[^0-9.]", "", str(v))))
            if 5800 <= n <= 8200:
                return n
        except Exception:
            continue
    return None


def _pick_yardage_from_hist(df_hist: pd.DataFrame) -> int | None:
    """
    Pick yardage from historical data.

    Args:
        df_hist (pd.DataFrame): Historical DataFrame.

    Returns:
        int or None: Yardage.
    """
    yard_cols = [c for c in df_hist.columns if re.search(r"(yardage|yards)$", c.lower())]
    vals = []
    for c in yard_cols:
        try:
            s = _clean_int_series_from_any(pd.Series(df_hist[c]))
            s = s[(s >= 5800) & (s <= 8200)]
            if s.notna().any():
                vals.append(int(round(s.median())))
        except Exception:
            pass
    if vals:
        ser = pd.Series(vals)
        m = ser.mode()
        return int(m.iloc[0]) if not m.empty else int(round(ser.median()))
    return None


# ---------- tournament summary (robust) ----------
def build_tournament_summary(processed_dir: Path, raw_hist_dir: Path, event_id: str, meta_proc: dict, out_json: Path) -> None:
    """
    Build tournament summary JSON.

    Args:
        processed_dir (Path): Processed data directory.
        raw_hist_dir (Path): Raw historical data directory.
        event_id (str): Event ID.
        meta_proc (dict): Processed meta.
        out_json (Path): Output JSON path.
    """
    root = Path(__file__).resolve().parent.parent
    upcoming_file = root / "upcoming-events.json"

    course_name = None
    location = None
    start_date_str = None
    upcoming_yardage = None
    upcoming_winners = None
    event_status = None

    if upcoming_file.exists():
        try:
            upcoming_data = json.loads(upcoming_file.read_text(encoding="utf-8"))
            for event in upcoming_data.get("schedule", []):
                if str(event.get("event_id")) == str(event_id):
                    course_name = event.get("course")
                    location = event.get("location")
                    start_date_str = event.get("start_date")
                    upcoming_yardage = event.get("yardage")
                    if isinstance(event.get("previous_winners"), list):
                        upcoming_winners = event.get("previous_winners")
                    event_status = event.get("status")
                    break
        except Exception as e:
            print(f"Warn: Failed to load upcoming-events.json: {e}")

    # If event is completed, try to fetch winner from DataGolf API
    current_winner = None
    if event_status == "completed" and start_date_str:
        try:
            year = start_date_str.split("-")[0]
            response = requests.get(f"https://datagolf.ca/api/get-schedule?season={year}")
            if response.status_code == 200:
                schedule_data = response.json()
                for sched_event in schedule_data.get("schedule", []):
                    if str(sched_event.get("event_id")) == str(event_id):
                        current_winner = sched_event.get("winner")
                        break
        except Exception as e:
            print(f"Warn: Failed to fetch winner from API: {e}")

    # field: try tee-times first, then field
    field = None
    for name in [
        f"event_{event_id}_field_teetimes.parquet",
        f"event_{event_id}_field_teetimes.csv",
        f"event_{event_id}_field.parquet",
        f"event_{event_id}_field.csv",
    ]:
        p = processed_dir / name
        if p.exists():
            field = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            break
    field_size = int(len(field)) if field is not None else None

    # yardage candidates
    yardage = None

    # 1) from field tables
    if field is not None:
        yardage = _pick_course_yardage(field)
        print(f"DEBUG: yardage from field tables: {yardage}")

    # course name override if field has a reasonable value
    if field is not None:
        for col in ["course", "course_name", "venue", "course_title"]:
            if col in field.columns and field[col].notna().any():
                try:
                    val = str(field[col].mode().iloc[0]).strip()
                    if val and val.lower() not in {"n/a", "na", "unknown"}:
                        course_name = val
                        break
                except Exception:
                    pass

    # 2) from meta_proc keys
    if yardage is None:
        yardage = _pick_yardage_from_meta(meta_proc)
        print(f"DEBUG: yardage from meta_proc: {yardage}")

    # 3) from upcoming-events.json
    if yardage is None and upcoming_yardage is not None:
        try:
            if isinstance(upcoming_yardage, (int, float)):
                n = int(round(float(upcoming_yardage)))
            else:
                n = int(float(re.sub(r"[^0-9.]", "", str(upcoming_yardage))))
            if 5800 <= n <= 8200:
                yardage = n
        except Exception:
            pass
        print(f"DEBUG: yardage from upcoming-events.json: {yardage}")

    # 4) from course catalog
    catalog = _load_course_catalog(root)
    catalog_entry = _lookup_course_catalog(course_name, catalog)
    if yardage is None and isinstance(catalog_entry.get("yardage"), (int, float, str)):
        try:
            n = catalog_entry["yardage"]
            if not isinstance(n, (int, float)):
                n = float(re.sub(r"[^0-9.]", "", str(n)))
            n = int(round(n))
            if 5800 <= n <= 8200:
                yardage = n
        except Exception:
            pass
        print(f"DEBUG: yardage from course catalog: {yardage}")

    # 5) from historical combined parquet, if such column exists
    df_hist = _load_hist_combined(raw_hist_dir, meta_proc.get("event_name", ""))
    if yardage is None and df_hist is not None and not df_hist.empty:
        yd_hist = _pick_yardage_from_hist(df_hist)
        if yd_hist is not None:
            yardage = yd_hist
        print(f"DEBUG: yardage from historical combined parquet: {yardage}")

    lat = meta_proc.get("lat")
    lon = meta_proc.get("lon")
    course_location = location
    if not course_location and lat is not None and lon is not None:
        try:
            course_location = f"{float(lat):.4f}, {float(lon):.4f}"
        except Exception:
            pass

    start_date = None
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%d-%b-%Y")
        except Exception:
            start_date = start_date_str

    # winners: prefer combined parquet; fallback to per-year files -> winners.json -> upcoming -> catalog
    winners = []
    if df_hist is not None and not df_hist.empty:
        winners = _winners_from_df(df_hist)
        print(f"DEBUG: winners from combined parquet (first 5): {winners[:5]}")
    if not winners:
        winners = _collect_winners_from_files(raw_hist_dir, event_id)
        print(f"DEBUG: winners from per-year files (first 5): {winners[:5]}")

    # Load winners from winners.json if available
    winners_json_path = raw_hist_dir / f"tournament_{_slug_event(meta_proc.get('event_name', ''))}_winners.json"
    if winners_json_path.exists() and not winners:
        try:
            winners_data = json.loads(winners_json_path.read_text(encoding="utf-8"))
            winners = [{"year": w["year"], "winner": w["winner"], "score": w["score"]} for w in winners_data]
            winners = sorted(winners, key=lambda r: r["year"], reverse=True)
            print(f"DEBUG: winners from winners.json (first 5): {winners[:5]}")
        except Exception as e:
            print(f"Warn: Failed to load winners from {winners_json_path}: {e}")

    # top-up from upcoming-events.json
    if (len(winners) < 5) and isinstance(upcoming_winners, list):
        seen_years = {w.get("year") for w in winners}
        for w in upcoming_winners:
            if isinstance(w, dict) and ("year" in w) and (w["year"] not in seen_years):
                winners.append({"year": int(w["year"]), "winner": w.get("winner"), "score": w.get("score")})
        winners = sorted(winners, key=lambda r: r["year"], reverse=True)

    # top-up from course catalog
    if (len(winners) < 5) and isinstance(catalog_entry.get("previous_winners"), list):
        seen_years = {w.get("year") for w in winners}
        for w in catalog_entry["previous_winners"]:
            if isinstance(w, dict) and ("year" in w) and (w["year"] not in seen_years):
                winners.append({"year": int(w["year"]), "winner": w.get("winner"), "score": w.get("score")})
        winners = sorted(winners, key=lambda r: r["year"], reverse=True)

    winners = winners[:5]  # last five

    payload = {
        "event_name": meta_proc.get("event_name", "Unknown Event"),
        "event_id": event_id,
        "course": course_name or "Unknown Course",
        "total_yardage": yardage,
        "course_location": course_location or "Unknown Location",
        "start_date": start_date,
        "field_size": field_size,
        "status": event_status or "unknown",
        "winner": current_winner,
        "previous_winners": winners,
    }
    write_json(out_json, payload)


# ---------- load start-holes from processed field (robust) ----------
# ---------- load start-holes from processed field (robust) ----------
def load_start_holes(processed_dir: Path, event_id: str) -> pd.DataFrame | None:
    """
    Load start holes DataFrame.

    Args:
        processed_dir (Path): Processed data directory.
        event_id (str): Event ID.

    Returns:
        pd.DataFrame or None: Start holes data.
    """
    for name in [
        f"event_{event_id}_field_teetimes.parquet",
        f"event_{event_id}_field_teetimes.csv",
        f"event_{event_id}_field.parquet",
        f"event_{event_id}_field.csv",
    ]:
        p = processed_dir / name
        if not p.exists():
            continue

        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        # canonicalize player_name
        if "player_name" not in df.columns:
            for c in ["name", "Player", "player"]:
                if c in df.columns:
                    df = df.rename(columns={c: "player_name"})
                    break
        if "player_name" not in df.columns:
            continue

        # Candidate columns for R1/R2 start holes
        cands_r1 = [
            "r1_start_hole",
            "start_hole_r1",
            "r1_start",
            "r1_starttee",
            "start_hole",
        ]
        cands_r2 = [
            "r2_start_hole",
            "start_hole_r2",
            "r2_start",
            "r2_starttee",
        ]  # Removed "start_hole" to avoid picking the same for R2

        frame_columns = set(df.columns)

        def pick(row: dict, cols: list[str], frame_columns: set) -> str | None:
            for c in cols:
                if c in frame_columns:
                    v = row.get(c)
                    if pd.notna(v):
                        return v
            return None

        r1, r2 = [], []
        for _, row in df.iterrows():
            r1.append(pick(row, cands_r1, frame_columns))
            r2.append(pick(row, cands_r2, frame_columns))

        # Infer R2 start hole as opposite of R1 if R2 is missing
        for i in range(len(r1)):
            if r1[i] and not r2[i]:
                try:
                    start = int(r1[i])
                    if start == 1:
                        r2[i] = "10"
                    elif start == 10:
                        r2[i] = "1"
                except Exception:
                    pass

        out = pd.DataFrame(
            {
                "player_name": df["player_name"].astype(str),
                "r1_start_hole": pd.Series(r1, index=df.index),
                "r2_start_hole": pd.Series(r2, index=df.index),
            }
        )

        out["name_key"] = out["player_name"].map(_norm_name)
        return out.drop_duplicates(subset=["name_key"])

    return None


def scan_latest_stamped_leaderboard(preds_dir: Path) -> tuple[Path | None, str | None]:
    """
    Scan for latest stamped leaderboard.

    Args:
        preds_dir (Path): Predictions directory.

    Returns:
        tuple: (CSV path, event_id) or (None, None).
    """
    stamped = sorted(preds_dir.glob("event_*_*_leaderboard.csv"), key=lambda p: p.stat().st_mtime)
    if not stamped:
        return None, None
    p = stamped[-1]
    m = re.match(r"event_(\d+)_.*_leaderboard\.csv$", p.name)
    eid = m.group(1) if m else None
    return p, eid


def load_meta_for_event(processed_dir: Path, event_id: str) -> dict:
    """
    Load meta for event.

    Args:
        processed_dir (Path): Processed data directory.
        event_id (str): Event ID.

    Returns:
        dict: Meta data.

    Raises:
        FileNotFoundError: If no meta found.
    """
    p = processed_dir / f"event_{event_id}_meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    w = processed_dir / f"event_{event_id}_weather_meta.json"
    if w.exists():
        return json.loads(w.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No meta for event_id={event_id} (looked for meta and weather_meta).")


# ---------- new helper for current week events ----------
def has_current_week_events(tour: str) -> bool:
    """
    Check if there are any events for the given tour in the current week (or next week if run on Sunday).

    Args:
        tour (str): Tour name (e.g., 'pga').

    Returns:
        bool: True if events exist in the current week (or next week on Sunday).
    """
    root = Path(__file__).resolve().parent.parent
    upcoming_file = root / "upcoming-events.json"
    if not upcoming_file.exists():
        return False

    try:
        with open(upcoming_file, encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("schedule", [])
        if tour:
            events = [e for e in events if e.get("tour") == tour]
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return False

    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    end_of_week = start_of_week + timedelta(days=6)  # Sunday

    # If today is Sunday, extend to next Sunday (include next week)
    if today.weekday() == 6:
        end_of_week = start_of_week + timedelta(days=13)

    for event in events:
        if not isinstance(event, dict) or "event_id" not in event or "start_date" not in event:
            continue
        try:
            event_date = datetime.fromisoformat(event["start_date"]).date()
            if start_of_week <= event_date <= end_of_week:
                return True
        except (ValueError, KeyError):
            continue
    return False


# ---------- new helper to clear old event assets in no-event mode ----------
def clear_old_event_assets(web_dir: Path) -> None:
    """
    Delete old event-specific assets to ensure no lingering data in no-event mode.

    Args:
        web_dir (Path): Web directory (e.g., web/pga).
    """
    files_to_remove = [
        "weather_round_neutral.json",
        "weather_round_wave.json",
        "weather_meta.json",
        "course_fit_weights.json",
        "course_history_summary.json",
        "tournament_summary.json",
        "field_teetimes.csv",
    ]
    for f in files_to_remove:
        p = web_dir / f
        if p.exists():
            try:
                p.unlink()
                print(f"[info] Removed old asset: {p}")
            except Exception as e:
                print(f"[warn] Failed to remove {p}: {e}")
    events_dir = web_dir / "events"
    if events_dir.exists():
        try:
            shutil.rmtree(events_dir)
            print(f"[info] Cleared events directory: {events_dir}")
        except Exception as e:
            print(f"[warn] Failed to clear events directory {events_dir}: {e}")


def publish_primary_assets(event_dir: Path, web_dir: Path) -> None:
    """
    Sync the primary event's assets to the legacy single-event locations for backwards compatibility.
    """
    files_to_publish = [
        "leaderboard.json",
        "summary.json",
        "weather_round_neutral.json",
        "weather_round_wave.json",
        "weather_meta.json",
        "course_fit_weights.json",
        "course_history_summary.json",
        "tournament_summary.json",
        "field_teetimes.csv",
    ]
    for name in files_to_publish:
        src = event_dir / name
        dest = web_dir / name
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
        elif dest.exists():
            dest.unlink()


# ---------- build schedule from upcoming-events.json ----------
def build_schedule_json(root: Path, tour: str, out_json: Path) -> None:
    """
    Build schedule JSON from upcoming events (only for the current year, including past/completed within that year).

    Args:
        root (Path): Project root.
        tour (str): Tour name.
        out_json (Path): Output JSON path.
    """
    upcoming_file = root / "upcoming-events.json"
    if not upcoming_file.exists():
        return
    upcoming_data = json.loads(upcoming_file.read_text(encoding="utf-8"))
    schedule = []
    current_year = 2026  # Hardcoded to 2026 as per user request
    for event in upcoming_data.get("schedule", []):
        if event.get("tour", "").lower() == tour.lower():
            date_str = event.get("start_date") or event.get("date")
            if date_str:
                try:
                    event_year = datetime.strptime(date_str, "%Y-%m-%d").year
                    if event_year == current_year:
                        # Include only if in current year
                        event_name = event.get("event_name") or event.get("name") or event.get("event")
                        formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m-%Y")
                        winner = event.get("winner", "N/A")
                        # Clean up winner name: remove (ID) and reorder to First Last
                        winner = event.get("winner", "N/A")
                        # Clean up winner name: remove (ID) and reorder to First Last
                        if winner != "N/A":
                            try:
                                parts = winner.split(", ")
                                if len(parts) == 2:
                                    last, first_id = parts
                                    first = first_id.split(" (")[0]  # Remove (ID)
                                    winner = f"{first} {last}"
                            except Exception:  # Specify Exception instead of bare except
                                pass  # Keep original if parsing fails
                        schedule.append(
                            {
                                "date": formatted_date,
                                "event": event_name,
                                "course": event.get("course"),
                                "location": event.get("location"),
                                "winner": winner,
                            }
                        )
                except ValueError:
                    # Skip if date parsing fails
                    continue
    # Sort schedule by date ascending (chronological order: earliest first)
    schedule.sort(key=lambda x: datetime.strptime(x["date"], "%d-%m-%Y"))
    write_json(out_json, schedule)


# ---------- archive event predictions ----------
def archive_event_predictions(root: Path, tour: str, event_name: str, event_id: str, r1_date: str | None, lb_csv: Path | None, source_dir: Path) -> None:
    """
    Archive the current event's prediction data into web/archive/{year}/.
    """
    # Get year from r1_date or current year
    year = None
    if r1_date:
        try:
            year = r1_date.split("-")[0]
        except Exception:
            pass
    if not year:
        year = str(datetime.now().year)

    archive_dir = root / "web" / "archive" / year
    event_slug = _slug_event(event_name)
    event_archive_dir = archive_dir / event_slug
    event_archive_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy
    files_to_copy = [
        "leaderboard.json",
        "meta.json",
        "tournament_summary.json",
        "summary.json",
    ]
    if lb_csv and lb_csv.exists():
        # Copy CSV to archive
        shutil.copy(lb_csv, event_archive_dir / "leaderboard.csv")
        csv_available = True
    else:
        csv_available = False

    for file_name in files_to_copy:
        src = source_dir / file_name
        if src.exists():
            shutil.copy(src, event_archive_dir / file_name)

    # Update index.json (in root archive dir)
    index_file = root / "web" / "archive" / "index.json"
    index_data = []
    if index_file.exists():
        try:
            index_data = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            index_data = []

    # Add/update this event
    event_entry = {
        "event_id": event_id,
        "event_name": event_name,
        "tour": tour,
        "slug": event_slug,
        "year": year,
        "csv_available": csv_available,
        "archived_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    # Remove existing entry for this event
    index_data = [e for e in index_data if e.get("event_id") != event_id or e.get("tour") != tour]
    index_data.append(event_entry)

    # Sort by date descending
    index_data.sort(key=lambda x: x.get("archived_at", ""), reverse=True)

    write_json(index_file, index_data)
    print(f"Archived predictions for {event_name} ({year}) in {event_archive_dir}")


def process_event(
    event_id: str,
    *,
    root: Path,
    tour: str,
    processed_dir: Path,
    preds_dir: Path,
    web_dir: Path,
    downloads_dir: Path,
    raw_hist_dir: Path,
    primary: bool,
) -> dict | None:
    """
    Build assets for a single event and return metadata summary for aggregation.
    """
    event_dir = web_dir / "events" / f"event_{event_id}"
    if event_dir.exists():
        shutil.rmtree(event_dir)
    event_dir.mkdir(parents=True, exist_ok=True)

    try:
        meta_proc = load_meta_for_event(processed_dir, event_id)
    except FileNotFoundError:
        print(f"[warn] Skipping event_id={event_id}: meta not found in {processed_dir}")
        return None

    event_name = meta_proc.get("event_name", f"event_{event_id}")
    lat = meta_proc.get("lat")
    lon = meta_proc.get("lon")
    start_date = meta_proc.get("start") or meta_proc.get("start_date")

    weather_meta_path = processed_dir / f"event_{event_id}_weather_meta.json"
    r1_date = None
    if weather_meta_path.exists():
        try:
            weather_meta = json.loads(weather_meta_path.read_text(encoding="utf-8"))
            r1_date = weather_meta.get("r1_date")
        except Exception:
            pass

    lb_csv, lb_html = pick_latest_timestamped_leaderboard(preds_dir, event_id)
    summary_json = pick_matching_summary(preds_dir, lb_csv) if lb_csv else None
    generated_utc = datetime.utcnow().strftime("%d-%b-%Y %H:%M:%S")

    has_predictions = lb_csv is not None

    if has_predictions:
        df_lb = pd.read_csv(lb_csv)
        name_col = "player_name" if "player_name" in df_lb.columns else ("Player" if "Player" in df_lb.columns else None)
        if not name_col:
            raise ValueError(f"Leaderboard CSV for event_id={event_id} missing player name column")
        if name_col != "player_name":
            df_lb = df_lb.rename(columns={name_col: "player_name"})
        df_lb["name_key"] = df_lb["player_name"].map(_norm_name)
        df_lb["player_name"] = df_lb["player_name"].apply(lambda name: " ".join(reversed(name.split(", "))) if ", " in name else name)

        start_df = load_start_holes(processed_dir, event_id)
        if start_df is not None:
            df_lb = df_lb.merge(
                start_df[["name_key", "r1_start_hole", "r2_start_hole"]],
                on="name_key",
                how="left",
            )

        def tee_time_with_tee(time_val, start_val):
            base = _time_only(time_val)
            try:
                ten = (str(start_val).strip() == "10") or (int(str(start_val).strip() or "0") == 10)
            except Exception:
                ten = False
            return f"{base}*" if base and ten else base

        for r in [1, 2]:
            time_col = f"r{r}_teetime"
            sh_col = f"r{r}_start_hole"
            if time_col in df_lb.columns:
                if sh_col in df_lb.columns:
                    df_lb[time_col] = [tee_time_with_tee(t, s) for t, s in zip(df_lb[time_col], df_lb[sh_col], strict=False)]
                else:
                    df_lb[time_col] = df_lb[time_col].apply(_time_only)

        drop_cols = [c for c in ("name_key", "r1_start_hole", "r2_start_hole", "start_hole") if c in df_lb.columns]
        if drop_cols:
            df_lb = df_lb.drop(columns=drop_cols)

        write_json(event_dir / "leaderboard.json", df_lb.to_dict(orient="records"))
    else:
        write_json(event_dir / "leaderboard.json", [])

    summary_data = {
        "event_id": event_id,
        "event_name": event_name,
        "source_csv": str(lb_csv.relative_to(root)) if lb_csv else None,
        "generated_utc": generated_utc,
        "metrics": None,
    }
    if summary_json and summary_json.exists():
        try:
            sj = json.loads(summary_json.read_text(encoding="utf-8"))
            summary_data["metrics"] = sj.get("summary")
            summary_data["generated_utc"] = normalize_utc_str(sj.get("generated_utc"), generated_utc)
        except Exception:
            pass
    write_json(event_dir / "summary.json", summary_data)

    downloads_csv = None
    if lb_csv:
        dst = downloads_dir / lb_csv.name
        if not dst.exists():
            dst.write_bytes(lb_csv.read_bytes())
        downloads_csv = dst
    downloads_html = None
    if lb_html and lb_html.exists():
        dst = downloads_dir / lb_html.name
        if not dst.exists():
            dst.write_bytes(lb_html.read_bytes())
        downloads_html = dst

    wn_ok = neutral_parquet_to_json(
        processed_dir / f"event_{event_id}_weather_round_neutral.parquet",
        event_dir / "weather_round_neutral.json",
    )
    ww_ok = wave_parquet_to_json(
        processed_dir / f"event_{event_id}_weather_round_wave.parquet",
        event_dir / "weather_round_wave.json",
    )
    if weather_meta_path.exists():
        (event_dir / "weather_meta.json").write_bytes(weather_meta_path.read_bytes())

    cf_weights = processed_dir / f"event_{event_id}_course_fit_weights.json"
    if cf_weights.exists():
        (event_dir / "course_fit_weights.json").write_bytes(cf_weights.read_bytes())
    build_history_summary(
        processed_dir / f"event_{event_id}_course_history_stats.parquet",
        event_dir / "course_history_summary.json",
    )

    tournament_summary_path = event_dir / "tournament_summary.json"
    build_tournament_summary(processed_dir, raw_hist_dir, event_id, meta_proc, tournament_summary_path)

    src_teetimes = processed_dir / f"event_{event_id}_field_teetimes.csv"
    if src_teetimes.exists():
        shutil.copy(src_teetimes, event_dir / "field_teetimes.csv")

    if primary:
        publish_primary_assets(event_dir, web_dir)

    archive_event_predictions(root, tour, event_name, event_id, r1_date, lb_csv, event_dir)

    event_base = f"{tour}/events/event_{event_id}"
    resources = {
        "leaderboard": f"{event_base}/leaderboard.json",
        "summary": f"{event_base}/summary.json",
        "meta": f"{event_base}/meta.json",
        "downloads_csv": f"{tour}/downloads/{downloads_csv.name}" if downloads_csv else None,
        "downloads_html": f"{tour}/downloads/{downloads_html.name}" if downloads_html else None,
        "weather_meta": f"{event_base}/weather_meta.json" if (event_dir / "weather_meta.json").exists() else None,
        "weather_round_neutral": f"{event_base}/weather_round_neutral.json" if wn_ok else None,
        "weather_round_wave": f"{event_base}/weather_round_wave.json" if ww_ok else None,
        "course_fit_weights": f"{event_base}/course_fit_weights.json" if (event_dir / "course_fit_weights.json").exists() else None,
        "course_history_summary": f"{event_base}/course_history_summary.json" if (event_dir / "course_history_summary.json").exists() else None,
        "tournament_summary": f"{event_base}/tournament_summary.json" if tournament_summary_path.exists() else None,
        "field_teetimes": f"{event_base}/field_teetimes.csv" if (event_dir / "field_teetimes.csv").exists() else None,
        "schedule": f"{tour}/schedule.json" if (web_dir / "schedule.json").exists() else None,
    }

    event_meta = {
        "tour": tour,
        "event_id": event_id,
        "event_name": event_name,
        "lat": lat,
        "lon": lon,
        "r1_date": r1_date,
        "start_date": start_date,
        "generated_utc": summary_data["generated_utc"],
        "source_csv": summary_data["source_csv"],
        "has_predictions": has_predictions,
        "resources": resources,
    }
    write_json(event_dir / "meta.json", event_meta)

    return {
        "event_id": event_id,
        "event_name": event_name,
        "r1_date": r1_date,
        "start_date": start_date,
        "generated_utc": summary_data["generated_utc"],
        "has_predictions": has_predictions,
        "resources": resources,
        "source_csv": summary_data["source_csv"],
        "event_dir": event_base,
        "lat": lat,
        "lon": lon,
    }

# ---------- main ----------
def main():
    """
    Build static web assets for one or more current events.
    """
    ap = argparse.ArgumentParser(description="Build static web assets from the latest run.")
    ap.add_argument("--event_id", type=str, default=None, help="Force a specific event id for web assets (comma separated for multiples)")
    ap.add_argument("--tour", type=str, default="pga", help="Tour to process")
    args = ap.parse_args()

    TOUR = args.tour

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR
    preds_dir = root / "data" / "preds" / TOUR
    raw_hist_dir = root / "data" / "raw" / "historical" / TOUR

    web_dir = root / "web" / TOUR
    web_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = web_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    events_root = web_dir / "events"
    events_root.mkdir(parents=True, exist_ok=True)

    build_schedule_json(root, TOUR, web_dir / "schedule.json")

    # Resolve events (preserve order, remove duplicates)
    raw_event_ids = resolve_event_ids(args.event_id, TOUR)
    seen_ids = set()
    event_ids = []
    for eid in raw_event_ids:
        if eid not in seen_ids:
            seen_ids.add(eid)
            event_ids.append(eid)

    if not args.event_id and not has_current_week_events(TOUR):
        event_ids = []
        print(f"[info] No events in current week (or next week if Sunday) for tour={TOUR}; generating placeholder assets.")

    reports = []

    if not event_ids:
        clear_old_event_assets(web_dir)
        events_root.mkdir(parents=True, exist_ok=True)

        generated_utc = datetime.utcnow().strftime("%d-%b-%Y %H:%M:%S")
        write_json(web_dir / "leaderboard.json", [])
        write_json(
            web_dir / "summary.json",
            {
                "event_id": "0",
                "event_name": "No Event",
                "source_csv": None,
                "generated_utc": generated_utc,
                "metrics": None,
            },
        )
        resources_placeholder = {
            "downloads_csv": None,
            "downloads_html": None,
            "weather_meta": None,
            "weather_round_neutral": None,
            "weather_round_wave": None,
            "course_fit_weights": None,
            "course_history_summary": None,
            "tournament_summary": None,
            "field_teetimes": None,
            "leaderboard": f"{TOUR}/leaderboard.json",
            "summary": f"{TOUR}/summary.json",
            "schedule": f"{TOUR}/schedule.json" if (web_dir / "schedule.json").exists() else None,
            "primary_event_dir": None,
        }
        meta_out = {
            "tour": TOUR,
            "event_id": "0",
            "event_name": "No Event",
            "lat": None,
            "lon": None,
            "r1_date": None,
            "generated_utc": generated_utc,
            "resources": resources_placeholder,
            "no_event_message": "No upcoming events during the holiday season. Stay tuned for 2026!",
            "active_events": [],
            "primary_event_dir": None,
        }
        write_json(web_dir / "meta.json", meta_out)
        reports.append("No events processed; placeholder assets generated.")
    else:
        aggregated_events = []
        for idx, eid in enumerate(event_ids):
            summary = process_event(
                eid,
                root=root,
                tour=TOUR,
                processed_dir=processed_dir,
                preds_dir=preds_dir,
                web_dir=web_dir,
                downloads_dir=downloads_dir,
                raw_hist_dir=raw_hist_dir,
                primary=(idx == 0),
            )
            if summary:
                aggregated_events.append(summary)

        if not aggregated_events:
            generated_utc = datetime.utcnow().strftime("%d-%b-%Y %H:%M:%S")
            write_json(web_dir / "leaderboard.json", [])
            write_json(
                web_dir / "summary.json",
                {
                    "event_id": "0",
                    "event_name": "No Event",
                    "source_csv": None,
                    "generated_utc": generated_utc,
                    "metrics": None,
                },
            )
            meta_out = {
                "tour": TOUR,
                "event_id": "0",
                "event_name": "No Event",
                "lat": None,
                "lon": None,
                "r1_date": None,
                "generated_utc": generated_utc,
                "resources": {
                    "leaderboard": f"{TOUR}/leaderboard.json",
                    "summary": f"{TOUR}/summary.json",
                    "schedule": f"{TOUR}/schedule.json" if (web_dir / "schedule.json").exists() else None,
                    "primary_event_dir": None,
                },
                "no_event_message": "No event data available.",
                "active_events": [],
                "primary_event_dir": None,
            }
            write_json(web_dir / "meta.json", meta_out)
            reports.append("Event data missing; reverted to placeholder assets.")
        else:
            primary = aggregated_events[0]
            generated_utc = primary["generated_utc"]
            resources_primary = {
                "downloads_csv": primary["resources"].get("downloads_csv"),
                "downloads_html": primary["resources"].get("downloads_html"),
                "weather_meta": f"{TOUR}/weather_meta.json" if (web_dir / "weather_meta.json").exists() else None,
                "weather_round_neutral": f"{TOUR}/weather_round_neutral.json" if (web_dir / "weather_round_neutral.json").exists() else None,
                "weather_round_wave": f"{TOUR}/weather_round_wave.json" if (web_dir / "weather_round_wave.json").exists() else None,
                "course_fit_weights": f"{TOUR}/course_fit_weights.json" if (web_dir / "course_fit_weights.json").exists() else None,
                "course_history_summary": f"{TOUR}/course_history_summary.json" if (web_dir / "course_history_summary.json").exists() else None,
                "tournament_summary": f"{TOUR}/tournament_summary.json" if (web_dir / "tournament_summary.json").exists() else None,
                "field_teetimes": f"{TOUR}/field_teetimes.csv" if (web_dir / "field_teetimes.csv").exists() else None,
                "leaderboard": f"{TOUR}/leaderboard.json" if (web_dir / "leaderboard.json").exists() else None,
                "summary": f"{TOUR}/summary.json" if (web_dir / "summary.json").exists() else None,
                "schedule": f"{TOUR}/schedule.json" if (web_dir / "schedule.json").exists() else None,
                "primary_event_dir": primary["event_dir"],
            }
            meta_out = {
                "tour": TOUR,
                "event_id": primary["event_id"],
                "event_name": primary["event_name"],
                "lat": primary.get("lat"),
                "lon": primary.get("lon"),
                "r1_date": primary.get("r1_date"),
                "generated_utc": generated_utc,
                "resources": resources_primary,
                "active_events": aggregated_events,
                "primary_event_dir": primary["event_dir"],
            }
            write_json(web_dir / "meta.json", meta_out)
            reports.append(f"Processed events: {', '.join(e['event_id'] for e in aggregated_events)}")

    print("Wrote web assets under web/:")
    for rel_path in [
        "leaderboard.json",
        "summary.json",
        "meta.json",
        "weather_round_neutral.json",
        "weather_round_wave.json",
        "weather_meta.json",
        "course_fit_weights.json",
        "course_history_summary.json",
        "tournament_summary.json",
        "field_teetimes.csv",
        "schedule.json",
    ]:
        out_path = web_dir / rel_path
        print("-", out_path if out_path.exists() else f"- (not generated) {out_path}")

    for message in reports:
        print(f"[info] {message}")


if __name__ == "__main__":
    main()
