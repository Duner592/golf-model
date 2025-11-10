#!/usr/bin/env python3
# scripts/build_web_assets.py
#
# Build static web assets from the latest run:
#   web/leaderboard.json
#   web/summary.json
#   web/meta.json
#   web/weather_round_neutral.json
#   web/weather_round_wave.json
#   web/weather_meta.json
#   web/course_fit_weights.json           (if available)
#   web/course_history_summary.json       (if available)
#   web/tournament_summary.json           (course, yardage, location, start date, field size, last 5 winners)
#   web/field_teetimes.csv                (if available)
#   web/downloads/<stamped leaderboard CSV/HTML>
#
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ensure repo root is importable when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

TOUR = "pga"
MPH_PER_MPS = 2.237
KMH_TO_MPH = 0.621371


# ---------- basic I/O ----------
def latest_meta(processed_dir: Path) -> dict:
    metas = sorted(processed_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta in {processed_dir}")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def pick_latest_timestamped_leaderboard(preds_dir: Path, event_id: str) -> tuple[Path, Path | None]:
    stamped = sorted(preds_dir.glob(f"event_{event_id}_*_leaderboard.csv"))
    html = None
    if stamped:
        lb = stamped[-1]
        candidate_html = lb.with_suffix(".html")
        if candidate_html.exists():
            html = candidate_html
        return lb, html
    lb = preds_dir / f"event_{event_id}_leaderboard.csv"
    if not lb.exists():
        raise FileNotFoundError(f"No leaderboard CSV found under {preds_dir}")
    html = preds_dir / f"event_{event_id}_leaderboard.html"
    if not html.exists():
        html = None
    return lb, html


def pick_matching_summary(preds_dir: Path, csv_path: Path) -> Path | None:
    base = csv_path.name.replace("_leaderboard.csv", "")
    candidate = preds_dir / f"{base}_summary.json"
    if candidate.exists():
        return candidate
    cand = sorted(preds_dir.glob("*_leaderboard_summary.json"))
    return cand[-1] if cand else None


def _sanitize_jsonable(obj):
    """
    Recursively replace NaN/Inf with None so JSON is standards-compliant.
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
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_obj = _sanitize_jsonable(obj)
    path.write_text(json.dumps(safe_obj, indent=2, allow_nan=False), encoding="utf-8")


# ---------- formatting helpers ----------
def normalize_utc_str(s: str | None, fallback: str) -> str:
    if not s:
        return fallback
    candidates = [
        ("%Y-%m-%dT%H%M%SZ", s.replace(":", "").replace("-", "")),
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
    if not val or not isinstance(val, str):
        return ""
    m = re.search(r"\b(\d{1,2}:\d{2})\b", val.strip())
    return m.group(1) if m else ""


def _norm_name(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _numeric_mode(series: pd.Series) -> int | None:
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
    if not neutral_pq.exists():
        return False
    df = pd.read_parquet(neutral_pq)
    records = df.to_dict(orient="records")
    records = [_coerce_wind_fields(r) for r in records]
    write_json(out_json, records)
    return True


def wave_parquet_to_json(wave_pq: Path, out_json: Path) -> bool:
    if not wave_pq.exists():
        return False
    df = pd.read_parquet(wave_pq)
    records = df.to_dict(orient="records")
    records = [_coerce_wind_fields(r) for r in records]
    write_json(out_json, records)
    return True


# ---------- course history summary ----------
def build_history_summary(stats_path: Path, out_json: Path) -> bool:
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
    if s.dtype.kind in ("i", "u", "f"):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _pick_course_yardage(field: pd.DataFrame) -> int | None:
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
    # explicit numeric columns first
    for c in ["year", "season", "event_year", "tournament_year"]:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors="coerce")
            if y.notna().any():
                return y.astype("Int64")

    # parse from dates
    for c in ["event_date", "tournament_date", "start_date", "date", "r1_date"]:
        if c in df.columns:
            try:
                dt = pd.to_datetime(df[c], errors="coerce", utc=False)
                if dt.notna().any():
                    return dt.dt.year.astype("Int64")
            except Exception:
                pass

    # parse from textual columns like "Butterfield Bermuda Championship 2024"
    text_cols = [c for c in ["tournament", "event_name", "event", "name"] if c in df.columns]
    for c in text_cols:
        ser = df[c].astype(str)
        # extract last occurrence of a 4-digit year
        y = ser.str.extract(r"((?:19|20)\d{2})", expand=False)
        y = pd.to_numeric(y, errors="coerce")
        if y.notna().any():
            return y.astype("Int64")

    return None


def _is_winner_fin(fin_val) -> bool:
    if pd.isna(fin_val):
        return False
    s = str(fin_val).strip().upper()
    if s.startswith("T"):
        s = s[1:]
    return s in {"1", "W", "WIN"}


def _compute_total_score(df: pd.DataFrame) -> pd.Series:
    score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
    if not score_cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    tmp = df[score_cols].apply(pd.to_numeric, errors="coerce")
    return tmp.sum(axis=1, min_count=1)


def _winners_from_df(df: pd.DataFrame) -> list[dict]:
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
    # normalize + underscore; matches saved combined parquet
    return _norm_name(s or "").replace(" ", "_")


def _load_hist_combined(raw_hist_dir: Path, event_name: str) -> pd.DataFrame | None:
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
        out.setdefault(w["year"], w)
    winners = sorted(out.values(), key=lambda r: r["year"], reverse=True)
    return winners


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_course_catalog(root: Path) -> dict:
    for rel in ["data/static/course_catalog.json", "static/course_catalog.json"]:
        p = root / rel
        if p.exists():
            data = _read_json(p)
            if isinstance(data, dict):
                return data
    return {}


def _lookup_course_catalog(course_name: str | None, catalog: dict) -> dict:
    if not isinstance(catalog, dict) or not course_name:
        return {}
    key = _norm_name(course_name)
    if key in catalog and isinstance(catalog[key], dict):
        return catalog[key]
    if course_name in catalog and isinstance(catalog[course_name], dict):
        return catalog[course_name]
    return {}


def _pick_yardage_from_meta(meta_proc: dict) -> int | None:
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
    root = Path(__file__).resolve().parent.parent
    upcoming_file = root / "upcoming-events.json"

    course_name = None
    location = None
    start_date_str = None
    upcoming_yardage = None
    upcoming_winners = None

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
                    break
        except Exception as e:
            print(f"Warn: Failed to load upcoming-events.json: {e}")

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

    # winners priority: combined parquet -> per-year files -> winners.json -> upcoming -> catalog
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
        "course": course_name or "Unknown Course",
        "total_yardage": yardage,
        "course_location": course_location or "Unknown Location",
        "start_date": start_date,
        "field_size": field_size,
        "previous_winners": winners,
    }
    write_json(out_json, payload)


# ---------- load start-holes from processed field (robust) ----------
def load_start_holes(processed_dir: Path, event_id: str) -> pd.DataFrame | None:
    """
    Return a small DataFrame with columns:
      [player_name, r1_start_hole, r2_start_hole, name_key]
    Built from the processed field/teetimes tables if available.
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
            "start_hole",
        ]

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
    Return (latest_stamped_csv, event_id) by file mtime.
    Looks for event_{eid}_{slug}_{date}_leaderboard.csv
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
    Prefer processed event_{event_id}_meta.json; fallback to weather_meta.
    """
    p = processed_dir / f"event_{event_id}_meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    w = processed_dir / f"event_{event_id}_weather_meta.json"
    if w.exists():
        return json.loads(w.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No meta for event_id={event_id} (looked for meta and weather_meta).")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build static web assets from the latest run.")
    ap.add_argument("--event_id", type=str, default=None, help="Force a specific event id for web assets")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR
    preds_dir = root / "data" / "preds" / TOUR

    from src.utils_event import resolve_event_id as _resolve  # centralized resolver

    event_id = None

    if args.event_id:
        event_id = str(args.event_id)

    if not event_id:
        latest_lb, eid_from_lb = scan_latest_stamped_leaderboard(preds_dir)
        if eid_from_lb:
            event_id = str(eid_from_lb)

    if not event_id:
        event_id = _resolve(None)

    meta_proc = load_meta_for_event(processed_dir, event_id)
    event_name = meta_proc.get("event_name", f"event_{event_id}")
    lat = meta_proc.get("lat")
    lon = meta_proc.get("lon")

    # Load r1_date from weather_meta.json if available
    weather_meta_path = processed_dir / f"event_{event_id}_weather_meta.json"
    r1_date = None
    if weather_meta_path.exists():
        try:
            weather_meta = json.loads(weather_meta_path.read_text(encoding="utf-8"))
            r1_date = weather_meta.get("r1_date")
        except Exception:
            pass

    # Pick leaderboard CSV/HTML for this event
    lb_csv, lb_html = pick_latest_timestamped_leaderboard(preds_dir, event_id)
    summary_json = pick_matching_summary(preds_dir, lb_csv)

    web_dir = root / "web"
    dl_dir = web_dir / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Leaderboard JSON: merge start holes, format times with "*" for 10th tee
    df_lb = pd.read_csv(lb_csv)

    # find and normalize player name column
    name_col = "player_name" if "player_name" in df_lb.columns else ("Player" if "Player" in df_lb.columns else None)
    if not name_col:
        raise ValueError("Leaderboard CSV missing player name column")
    if name_col != "player_name":
        df_lb = df_lb.rename(columns={name_col: "player_name"})
    df_lb["name_key"] = df_lb["player_name"].map(_norm_name)

    # merge start-holes on normalized name
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

    # apply time-only + asterisk using merged start holes (if present)
    for r in [1, 2]:
        time_col = f"r{r}_teetime"
        sh_col = f"r{r}_start_hole"
        if time_col in df_lb.columns:
            if sh_col in df_lb.columns:
                df_lb[time_col] = [tee_time_with_tee(t, s) for t, s in zip(df_lb[time_col], df_lb[sh_col], strict=False)]
            else:
                df_lb[time_col] = df_lb[time_col].apply(_time_only)

    # DROP helper/start_hole columns to avoid NaN in JSON
    drop_cols = [c for c in ("name_key", "r1_start_hole", "r2_start_hole", "start_hole") if c in df_lb.columns]
    if drop_cols:
        df_lb = df_lb.drop(columns=drop_cols)

    write_json(web_dir / "leaderboard.json", df_lb.to_dict(orient="records"))

    # summary
    generated_utc_formatted = datetime.utcnow().strftime("%d-%b-%Y %H:%M:%S")
    summary = {
        "event_id": event_id,
        "event_name": event_name,
        "source_csv": str(lb_csv.relative_to(root)),
        "generated_utc": generated_utc_formatted,
        "metrics": None,
    }
    if summary_json and summary_json.exists():
        try:
            sj = json.loads(summary_json.read_text(encoding="utf-8"))
            summary["metrics"] = sj.get("summary")
            summary["generated_utc"] = normalize_utc_str(sj.get("generated_utc"), generated_utc_formatted)
        except Exception:
            pass
    write_json(web_dir / "summary.json", summary)

    # downloads
    dl_csv = dl_dir / lb_csv.name
    if not dl_csv.exists():
        dl_csv.write_bytes(lb_csv.read_bytes())
    dl_html = None
    if lb_html and lb_html.exists():
        dl_html = dl_dir / lb_html.name
        if not dl_html.exists():
            dl_html.write_bytes(lb_html.read_bytes())

    # weather
    wn_ok = neutral_parquet_to_json(
        processed_dir / f"event_{event_id}_weather_round_neutral.parquet",
        web_dir / "weather_round_neutral.json",
    )
    ww_ok = wave_parquet_to_json(
        processed_dir / f"event_{event_id}_weather_round_wave.parquet",
        web_dir / "weather_round_wave.json",
    )
    wm = processed_dir / f"event_{event_id}_weather_meta.json"
    if wm.exists():
        (web_dir / "weather_meta.json").write_bytes(wm.read_bytes())

    # course-fit + history
    cf_weights = processed_dir / f"event_{event_id}_course_fit_weights.json"
    if cf_weights.exists():
        (web_dir / "course_fit_weights.json").write_bytes(cf_weights.read_bytes())
    build_history_summary(
        processed_dir / f"event_{event_id}_course_history_stats.parquet",
        web_dir / "course_history_summary.json",
    )

    # tournament summary
    raw_hist_dir = root / "data" / "raw" / "historical" / TOUR
    ts_path = web_dir / "tournament_summary.json"
    build_tournament_summary(processed_dir, raw_hist_dir, event_id, meta_proc, ts_path)

    # Copy field_teetimes.csv to web (guard if missing)
    import shutil

    src_teetimes = processed_dir / f"event_{event_id}_field_teetimes.csv"
    if src_teetimes.exists():
        shutil.copy(src_teetimes, web_dir / "field_teetimes.csv")

    meta_out = {
        "tour": TOUR,
        "event_id": event_id,
        "event_name": event_name,
        "lat": lat,
        "lon": lon,
        "r1_date": r1_date,
        "generated_utc": summary["generated_utc"],
        "resources": {
            "downloads_csv": f"downloads/{lb_csv.name}",
            "downloads_html": f"downloads/{lb_html.name}" if dl_html else None,
            "weather_meta": ("weather_meta.json" if (web_dir / "weather_meta.json").exists() else None),
            "weather_round_neutral": "weather_round_neutral.json" if wn_ok else None,
            "weather_round_wave": "weather_round_wave.json" if ww_ok else None,
            "course_fit_weights": ("course_fit_weights.json" if (web_dir / "course_fit_weights.json").exists() else None),
            "course_history_summary": ("course_history_summary.json" if (web_dir / "course_history_summary.json").exists() else None),
            "tournament_summary": ("tournament_summary.json" if ts_path.exists() else None),
        },
    }
    write_json(web_dir / "meta.json", meta_out)

    print("Wrote web assets under web/:")
    for p in [
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
    ]:
        q = web_dir / p
        print("-", q if q.exists() else f"- (not generated) {q}")


if __name__ == "__main__":
    main()
