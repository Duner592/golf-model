#!/usr/bin/env python3
# scripts/build_web_assets.py
#
# Build static web assets from the latest run:
#   web/leaderboard.json
#   web/summary.json
#   web/meta.json
#   web/weather_round_neutral.json  (mph/%)
#   web/weather_round_wave.json     (mph/%)
#   web/weather_meta.json
#   web/course_fit_weights.json     (if available)
#   web/course_history_summary.json (if available)
#   web/tournament_summary.json     (course, yardage, location, start date, field size, last 5 winners)
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


# ---------- last 5 winners ----------
def _winner_from_event_json(path: Path) -> tuple[str | None, float | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
        if not rows:
            return None, None
        df = pd.json_normalize(rows)
        if "fin_text" in df.columns:
            mask = df["fin_text"].astype(str).str.upper().isin(["1", "W", "WIN"])
            if mask.any():
                r = df.loc[mask].iloc[0]
                score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
                total = float(r[score_cols].sum()) if score_cols else None
                return str(r.get("player_name", "")), total
        score_cols = [c for c in df.columns if re.match(r"^round_\d+\.score$", c)]
        if score_cols:
            df["__total"] = df[score_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            df2 = df.dropna(subset=["__total"])
            if not df2.empty:
                r = df2.sort_values("__total", ascending=True).iloc[0]
                return str(r.get("player_name", "")), float(r["__total"])
    except Exception:
        return None, None
    return None, None


# ---------- tournament summary ----------
def build_tournament_summary(
    processed_dir: Path,
    raw_hist_dir: Path,
    event_id: str,
    meta_proc: dict,
    out_json: Path,
) -> None:
    field = None
    for name in [f"event_{event_id}_field.parquet", f"event_{event_id}_field.csv"]:
        p = processed_dir / name
        if p.exists():
            field = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            break
    field_size = int(len(field)) if field is not None else None
    course_name = None
    course_yardage = None
    if field is not None:
        for col in ["total_yardage", "course_yardage", "yardage"]:
            if col in field.columns and field[col].notna().any():
                course_yardage = _numeric_mode(field[col])
                if course_yardage is not None:
                    break
        if course_yardage is None:
            yard_cols = [c for c in field.columns if "yard" in c.lower()]
            for yc in yard_cols:
                if field[yc].notna().any():
                    course_yardage = _numeric_mode(field[yc])
                    if course_yardage is not None:
                        break
    if field is not None:
        for col in ["course", "course_name", "venue", "course_title"]:
            if col in field.columns and field[col].notna().any():
                try:
                    course_name = str(field[col].mode().iloc[0])
                    break
                except Exception:
                    pass
    lat = meta_proc.get("lat")
    lon = meta_proc.get("lon")
    course_location = f"{lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else None
    start_date_raw = meta_proc.get("r1_date")
    start_date = None
    if start_date_raw:
        try:
            start_date = datetime.strptime(start_date_raw, "%Y-%m-%d").strftime("%d-%b-%Y")
        except Exception:
            start_date = start_date_raw
    winners = []
    files = sorted(raw_hist_dir.glob(f"event_{event_id}_*_rounds.json"))

    def _year_from_name(p: Path) -> int:
        m = re.search(rf"event_{re.escape(str(event_id))}_(\d{{4}})_rounds\.json", p.name)
        return int(m.group(1)) if m else -1

    files = sorted(files, key=_year_from_name, reverse=True)[:5]
    for f in files:
        year = _year_from_name(f)
        name, total = _winner_from_event_json(f)
        if name:
            winners.append({"year": year, "winner": name, "score": total})
    payload = {
        "course": course_name,
        "total_yardage": course_yardage,
        "course_location": course_location,
        "start_date": start_date,
        "field_size": field_size,
        "previous_winners": winners,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        # Use dict-like access for row to be robust across CSV/Parquet
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

        # normalize names for merge with leaderboard
        def _norm_name(s: str | None) -> str:
            if not isinstance(s, str):
                return ""
            t = s.lower().strip()
            t = re.sub(r"[^a-z0-9]+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        out["name_key"] = out["player_name"].map(_norm_name)
        # Keep unique per player
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
    # NEW: allow --event_id and fallback to the most recent stamped leaderboard
    ap = argparse.ArgumentParser(description="Build static web assets from the latest run.")
    ap.add_argument("--event_id", type=str, default=None, help="Force a specific event id for web assets")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR
    preds_dir = root / "data" / "preds" / TOUR

    # Resolve event_id
    from src.utils_event import resolve_event_id as _resolve  # centralized resolver

    event_id = None

    # 1) explicit CLI
    if args.event_id:
        event_id = str(args.event_id)

    # 2) latest stamped leaderboard (most recent export)
    if not event_id:
        latest_lb, eid_from_lb = scan_latest_stamped_leaderboard(preds_dir)
        if eid_from_lb:
            event_id = str(eid_from_lb)

    # 3) fallback to centralized resolver (this week)
    if not event_id:
        event_id = _resolve(None)

    # Load meta for that event
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

    # Leaderboard JSON (apply HH:MM + '*' for 10th tee starts, then drop start_hole cols)
    df_lb = pd.read_csv(lb_csv)

    def tee_time_with_tee(time_val, start_val):
        base = _time_only(time_val)
        try:
            ten = (str(start_val).strip() == "10") or (int(str(start_val).strip() or "0") == 10)
        except Exception:
            ten = False
        return f"{base}*" if base and ten else base

    for r in [1, 2]:
        time_col = f"r{r}_teetime"
        # prefer round-specific start hole if present
        start_col = f"r{r}_start_hole" if f"r{r}_start_hole" in df_lb.columns else "start_hole"
        if time_col in df_lb.columns:
            if start_col in df_lb.columns:
                df_lb[time_col] = [tee_time_with_tee(t, s) for t, s in zip(df_lb[time_col], df_lb[start_col], strict=False)]
            else:
                df_lb[time_col] = df_lb[time_col].apply(_time_only)

    # DROP start_hole columns to avoid NaN in JSON (browser-unsafe)
    drop_cols = [c for c in ("r1_start_hole", "r2_start_hole", "start_hole") if c in df_lb.columns]
    if drop_cols:
        df_lb = df_lb.drop(columns=drop_cols)

    # Optional: ensure any remaining NaN/Inf are removed before dict conversion
    df_lb = df_lb.replace({np.nan: None, np.inf: None, -np.inf: None})

    write_json(web_dir / "leaderboard.json", df_lb.to_dict(orient="records"))
    # find player name column (your CSV typically has 'player_name' already)
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

    # remove helper key
    if "name_key" in df_lb.columns:
        df_lb = df_lb.drop(columns=["name_key"], errors="ignore")

    # write leaderboard.json with asterisks
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

    # Copy field_teetimes.csv to web
    import shutil

    shutil.copy(processed_dir / f"event_{event_id}_field_teetimes.csv", web_dir / "field_teetimes.csv")

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
