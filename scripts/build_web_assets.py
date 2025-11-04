#!/usr/bin/env python3
# scripts/build_web_assets.py
#
# Collect latest leaderboard + summary + resources for web/:
#   web/leaderboard.json
#   web/summary.json
#   web/meta.json (includes generated_utc and resource links)
#   web/weather_round_neutral.json  (wind_mph, gust_mph, precip_pct)
#   web/weather_round_wave.json     (wind_mph, gust_mph, precip_pct)
#   web/weather_meta.json           (copy of processed meta)
#   web/course_fit_weights.json     (if available)
#   web/course_history_summary.json (if available)
#   web/downloads/<original CSV/HTML>

from pathlib import Path
import json
from datetime import datetime
import pandas as pd

TOUR = "pga"
MPH_PER_MPS = 2.237
KMH_TO_MPH = 0.621371


def latest_meta(processed_dir: Path) -> dict:
    metas = sorted(processed_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta in {processed_dir}")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def pick_latest_timestamped_leaderboard(preds_dir: Path, event_id: str):
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
    # fallback: latest any summary
    cand = sorted(preds_dir.glob("*_leaderboard_summary.json"))
    return cand[-1] if cand else None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _coerce_wind_fields(rec: dict) -> dict:
    # Normalize any legacy fields to mph and %.
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


def build_history_summary(stats_path: Path, out_json: Path) -> bool:
    if not stats_path.exists():
        return False
    df = pd.read_parquet(stats_path)
    payload = {
        "rows": int(len(df)),
        "rounds_course_avg": (
            float(df["rounds_course"].mean()) if "rounds_course" in df else None
        ),
        "sg_course_mean_shrunk_avg": (
            float(df["sg_course_mean_shrunk"].mean())
            if "sg_course_mean_shrunk" in df
            else None
        ),
        "top_rounds": (
            df.sort_values("rounds_course", ascending=False)
            .head(10)
            .to_dict(orient="records")
            if "rounds_course" in df
            else []
        ),
        "top_sg_course": (
            df.sort_values("sg_course_mean_shrunk", ascending=False)
            .head(10)
            .to_dict(orient="records")
            if "sg_course_mean_shrunk" in df
            else []
        ),
    }
    write_json(out_json, payload)
    return True


def main():
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR
    preds_dir = root / "data" / "preds" / TOUR

    meta_proc = latest_meta(processed_dir)
    event_id = str(meta_proc["event_id"])
    event_name = meta_proc.get("event_name", f"event_{event_id}")
    lat = meta_proc.get("lat")
    lon = meta_proc.get("lon")

    lb_csv, lb_html = pick_latest_timestamped_leaderboard(preds_dir, event_id)
    summary_json = pick_matching_summary(preds_dir, lb_csv)

    # Prepare web dir
    web_dir = root / "web"
    dl_dir = web_dir / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Leaderboard JSON (enhance tee times to DATETIME local if time is present)
    df_lb = pd.read_csv(lb_csv)
    r1_date = meta_proc.get("r1_date")
    r2_date = meta_proc.get("r2_date")
    if r1_date and "r1_teetime" in df_lb.columns:
        df_lb["r1_datetime_local"] = df_lb["r1_teetime"].apply(
            lambda t: f"{r1_date} {t}" if pd.notna(t) and str(t).strip() else ""
        )
    if r2_date and "r2_teetime" in df_lb.columns:
        df_lb["r2_datetime_local"] = df_lb["r2_teetime"].apply(
            lambda t: f"{r2_date} {t}" if pd.notna(t) and str(t).strip() else ""
        )
    write_json(web_dir / "leaderboard.json", df_lb.to_dict(orient="records"))

    # Summary JSON (include generated_utc)
    generated_utc = datetime.utcnow().strftime("%d-%b-%Y %H:%M:%S")
    summary = {
        "event_id": event_id,
        "event_name": event_name,
        "source_csv": str(lb_csv.relative_to(root)),
        "generated_utc": generated_utc,
        "metrics": None,
    }
    if summary_json and summary_json.exists():
        try:
            sj = json.loads(summary_json.read_text(encoding="utf-8"))
            summary["metrics"] = sj.get("summary")
            summary["generated_utc"] = sj.get("generated_utc", generated_utc)
        except Exception:
            pass
    write_json(web_dir / "summary.json", summary)

    # Copy downloads
    dl_csv = dl_dir / lb_csv.name
    if not dl_csv.exists():
        dl_csv.write_bytes(lb_csv.read_bytes())
    dl_html = None
    if lb_html and lb_html.exists():
        dl_html = dl_dir / lb_html.name
        if not dl_html.exists():
            dl_html.write_bytes(lb_html.read_bytes())

    # Weather JSON (mph/%)
    wn_ok = neutral_parquet_to_json(
        processed_dir / f"event_{event_id}_weather_round_neutral.parquet",
        web_dir / "weather_round_neutral.json",
    )
    ww_ok = wave_parquet_to_json(
        processed_dir / f"event_{event_id}_weather_round_wave.parquet",
        web_dir / "weather_round_wave.json",
    )
    # Copy weather meta if exists
    wm = processed_dir / f"event_{event_id}_weather_meta.json"
    if wm.exists():
        (web_dir / "weather_meta.json").write_bytes(wm.read_bytes())

    # Course fit and history summaries
    cf_weights = processed_dir / f"event_{event_id}_course_fit_weights.json"
    if cf_weights.exists():
        (web_dir / "course_fit_weights.json").write_bytes(cf_weights.read_bytes())
    build_history_summary(
        processed_dir / f"event_{event_id}_course_history_stats.parquet",
        web_dir / "course_history_summary.json",
    )

    # Meta for page: resource links + lat/lon + generated_utc
    meta_out = {
        "tour": TOUR,
        "event_id": event_id,
        "event_name": event_name,
        "lat": lat,
        "lon": lon,
        "generated_utc": summary["generated_utc"],
        "resources": {
            "downloads_csv": f"downloads/{lb_csv.name}",
            "downloads_html": f"downloads/{lb_html.name}" if dl_html else None,
            "weather_meta": (
                "weather_meta.json"
                if (web_dir / "weather_meta.json").exists()
                else None
            ),
            "weather_round_neutral": "weather_round_neutral.json" if wn_ok else None,
            "weather_round_wave": "weather_round_wave.json" if ww_ok else None,
            "course_fit_weights": (
                "course_fit_weights.json"
                if (web_dir / "course_fit_weights.json").exists()
                else None
            ),
            "course_history_summary": (
                "course_history_summary.json"
                if (web_dir / "course_history_summary.json").exists()
                else None
            ),
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
    ]:
        q = web_dir / p
        print("-", q if q.exists() else f"- (not generated) {q}")


if __name__ == "__main__":
    main()
