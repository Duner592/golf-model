#!/usr/bin/env python3
# scripts/build_combined_history.py
#
# Build a single "combined historical rounds" parquet for a tournament,
# saved as: data/raw/historical/{tour}/tournament_{slug}_rounds_combined.parquet
#
# How it works:
# - Determine the "base" tournament (from latest processed meta, or via --event_id/--name).
# - For selected years, fetch the schedule; match all events by name similarity.
# - Ensure event_{event_id}_{year}_rounds.json exists (download if missing).
# - Normalize each JSON to long rows (one row per round); keep sg_total and common driving columns if present.
# - Concatenate and save combined parquet for your course-fit/history steps.
#
# Requirements:
# - configs/datagolf.yaml includes 'schedule' and 'historical_rounds' endpoints.
# - DATAGOLF_API_KEY set in environment (.env).
# - pandas, requests, requests_cache, pyyaml

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
import requests_cache
import yaml
from dotenv import load_dotenv

load_dotenv()

TOUR = "pga"


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def slugify(s: str) -> str:
    s0 = (s or "").lower()
    s0 = re.sub(r"[^a-z0-9]+", " ", s0)
    return re.sub(r"\s+", " ", s0).strip().replace(" ", "_")


def tokenize(s: str) -> set:
    return set(slugify(s).split())


def jaccard(a: str, b: str) -> float:
    A, B = tokenize(a), tokenize(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def latest_meta(processed_dir: Path) -> dict:
    metas = sorted(processed_dir.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta under {processed_dir}")
    return json.loads(metas[-1].read_text(encoding="utf-8"))


def schedule_events(base_url: str, key_param: str, api_key: str, sched_path: str, tour: str, year: int) -> list[dict]:
    url = f"{base_url}/{sched_path.lstrip('/')}"
    r = requests.get(url, params={key_param: api_key, "tour": tour, "year": str(year)}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("events", "schedule", "data"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
    return []


def ensure_rounds_json(
    base_url: str,
    key_param: str,
    api_key: str,
    hist_path: str,
    tour: str,
    event_id: str,
    year: int,
    out_dir: Path,
) -> Path | None:
    """
    Guarantee data/raw/historical/{tour}/event_{eid}_{year}_rounds.json exists; download if not.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"event_{event_id}_{year}_rounds.json"
    if out.exists():
        return out
    url = f"{base_url}/{hist_path.lstrip('/')}"
    params = {
        key_param: api_key,
        "tour": tour,
        "event_id": str(event_id),
        "year": str(year),
        "file_format": "json",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code in (400, 404):
            return None
        r.raise_for_status()
        out.write_text(json.dumps(r.json(), indent=2), encoding="utf-8")
        return out
    except requests.HTTPError:
        return None


def extract_rows_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
    return rows or []


def json_wide_to_long(df: pd.DataFrame, event_id: str, year: int) -> pd.DataFrame:
    """
    Convert a "wide" per-event df with columns like round_1.sg_total, round_1.driving_acc to "long" rows per round.
    Keep player_name and dg_id if present.
    """
    keep_cols = []
    if "player_name" in df.columns:
        keep_cols.append("player_name")
    if "dg_id" in df.columns:
        keep_cols.append("dg_id")
    # If only player_id is present, rename for downstream consistency
    if "dg_id" not in df.columns and "player_id" in df.columns:
        df = df.rename(columns={"player_id": "dg_id"})
        keep_cols.append("dg_id")
    round_nums = sorted({int(m.group(1)) for c in df.columns for m in [re.match(r"^round_(\d+)\.", c)] if m})
    recs = []
    for _, row in df.iterrows():
        base = {c: row.get(c) for c in keep_cols}
        for r in round_nums:
            sg = row.get(f"round_{r}.sg_total", None)
            # keep row only if sg_total present (otherwise skip that round)
            if pd.notna(sg):
                rec = dict(base)
                rec["round"] = r
                rec["sg_total"] = float(sg)
                # optional driving fields
                for dr in ("driving_acc", "driving_dist"):
                    v = row.get(f"round_{r}.{dr}", None)
                    if pd.notna(v):
                        rec[dr] = float(v)
                rec["event_id"] = str(event_id)
                rec["year"] = int(year)
                recs.append(rec)
    return pd.DataFrame(recs)


def pick_base_event_name(meta: dict, override_id: str | None, override_name: str | None) -> tuple[str, str | None]:
    """
    Return (base_event_name, base_event_id_for_name_only).
    If override_name is provided, use it and leave id None unless override_id given too.
    If override_id is provided, try to get event_name from latest meta or leave id-only.
    """
    if override_name:
        return override_name, override_id
    if override_id and str(meta.get("event_id")) != str(override_id):
        # can't rely on meta name; use id-only for matching (will pull schedule names by id each year)
        return f"event_{override_id}", str(override_id)
    return meta.get("event_name") or f"event_{meta.get('event_id')}", str(meta.get("event_id"))


def main():
    ap = argparse.ArgumentParser(description="Build combined historical rounds parquet for a tournament.")
    ap.add_argument(
        "--event_id",
        type=str,
        default=None,
        help="Pin by event_id (helps schedule matching)",
    )
    ap.add_argument(
        "--name",
        type=str,
        default=None,
        help='Pin by tournament name (e.g., "Arnold Palmer Invitational presented by Mastercard")',
    )
    ap.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to include (e.g., 2019,2020,2021)",
    )
    ap.add_argument(
        "--years_back",
        type=int,
        default=5,
        help="If --years omitted, look back this many years from current year",
    )
    ap.add_argument(
        "--sim_threshold",
        type=float,
        default=0.45,
        help="Name similarity threshold (0..1)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed" / TOUR
    hist_dir = root / "data" / "raw" / "historical" / TOUR
    hist_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(root / "configs" / "datagolf.yaml")
    base_url = cfg["base_url"].rstrip("/")
    keyp = cfg["auth"]["key_param"]
    api_key = os.getenv(cfg["auth"]["env_var"])
    if not api_key:
        raise RuntimeError("Missing API key in env var")
    sched_path = cfg["endpoints"]["schedule"]["path"]
    hist_path = cfg["endpoints"]["historical_rounds"]["path"]
    tour = cfg["defaults"]["tour"]

    # Determine base event name/id
    meta = latest_meta(processed_dir)
    base_name, base_id_for_name = pick_base_event_name(meta, args.event_id, args.name)
    base_slug = slugify(base_name)
    print(f"[base] name={base_name!r} slug={base_slug} event_id_hint={base_id_for_name}")

    # Determine years to process
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    else:
        from datetime import datetime

        this_year = datetime.utcnow().year
        years = list(range(this_year - args.years_back, this_year + 1))
    print(f"[years] {years}")

    requests_cache.install_cache("dg_cache", expire_after=600)
    combined_frames: list[pd.DataFrame] = []

    for yr in years:
        # Pull the schedule for the year
        try:
            events = schedule_events(base_url, keyp, api_key, sched_path, tour, yr)
        except requests.HTTPError as e:
            print(f"[warn] schedule fetch failed for {yr}: {e}")
            continue
        if not events:
            continue

        # Score each event by similarity to base name; if event_id hint provided, give bonus
        scored = []
        for ev in events:
            name = ev.get("event_name") or ev.get("name") or ""
            sim = jaccard(base_name, name)
            if base_id_for_name and str(ev.get("event_id")) == str(base_id_for_name):
                sim = max(sim, 0.99)
            scored.append((sim, ev))
        scored.sort(key=lambda x: x[0], reverse=True)
        # pick all above threshold
        matches = [ev for sim, ev in scored if sim >= args.sim_threshold]
        if not matches and scored:
            # if nothing above, pick top-1 (most likely rename)
            best_sim, best_ev = scored[0]
            if best_sim >= 0.25:
                matches = [best_ev]

        if not matches:
            print(f"[{yr}] no schedule matches for {base_name!r}")
            continue

        for ev in matches:
            ev_id = str(ev.get("event_id"))
            ev_name = ev.get("event_name") or ev.get("name") or f"event_{ev_id}"
            print(f"[{yr}] match: {ev_id} • {ev_name}")

            # Ensure rounds JSON exists
            jpath = hist_dir / f"event_{ev_id}_{yr}_rounds.json"
            if not jpath.exists():
                jpath = ensure_rounds_json(base_url, keyp, api_key, hist_path, tour, ev_id, yr, hist_dir)
                if jpath is None:
                    print(f"  -> skip: no rounds JSON available for {ev_id} {yr}")
                    continue

            rows = extract_rows_from_json(jpath)
            if not rows:
                print("  -> skip empty rows in JSON")
                continue
            df_ev = pd.json_normalize(rows)
            df_long = json_wide_to_long(df_ev, ev_id, yr)
            if not df_long.empty:
                combined_frames.append(df_long)

    if not combined_frames:
        raise FileNotFoundError("No historical rounds parsed/matched; nothing to combine.")

    combined = pd.concat(combined_frames, ignore_index=True)
    # Basic cleanup: ensure numeric types where possible
    for col in ("sg_total", "driving_acc", "driving_dist"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    # Save combined parquet
    out_path = hist_dir / f"tournament_{base_slug}_rounds_combined.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"[done] saved combined: {out_path} rows={len(combined)}")


if __name__ == "__main__":
    main()
