#!/usr/bin/env python3
# scripts/summarize_status.py
#
# Summarize artifacts for a specific event. If --event_id is not provided,
# resolves the most recent event that has predictions in data/preds/{tour}.
#
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TOUR = "pga"


def scan_pred_events(preds_dir: Path) -> list[str]:
    ids = set()
    for p in preds_dir.glob("event_*_preds_*.parquet"):
        m = re.match(r"event_(\d+)_preds_", p.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def resolve_event_id_arg_or_preds(arg_eid: str | None, tour: str) -> str:
    if arg_eid:
        return str(arg_eid)
    preds_dir = Path("data/preds") / tour
    cand = scan_pred_events(preds_dir)
    if not cand:
        # fall back to latest meta if there are no preds files at all
        processed = Path("data/processed") / tour
        metas = sorted(processed.glob("event_*_meta.json"))
        if not metas:
            raise FileNotFoundError("No predictions or meta found. Run the weekly pipeline first.")
        meta = json.loads(metas[-1].read_text(encoding="utf-8"))
        return str(meta["event_id"])
    return cand[-1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def head_cols(df: pd.DataFrame, k=8) -> list:
    return list(df.columns[:k])


def exists(p: Path) -> bool:
    return p.exists()


def main():
    ap = argparse.ArgumentParser(description="Summarize artifacts for an event.")
    ap.add_argument("--tour", default=TOUR)
    ap.add_argument("--event_id", type=str, default=None, help="Event id to summarize")
    args = ap.parse_args()

    tour = args.tour
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / tour
    features = root / "data" / "features" / tour
    preds = root / "data" / "preds" / tour
    raw_hist = root / "data" / "raw" / "historical" / tour

    # Resolve event_id (prefer CLI; else most recent that has preds)
    event_id = resolve_event_id_arg_or_preds(args.event_id, tour)

    # Meta (by event id)
    meta_path = processed / f"event_{event_id}_meta.json"
    if not meta_path.exists():
        # fallback: use latest meta if specific one missing (but print a warning)
        metas = sorted(processed.glob("event_*_meta.json"))
        if not metas:
            raise FileNotFoundError("No meta found. Run parse_field_updates.py first.")
        meta_path = metas[-1]
        print(f"[warn] Specific meta for event_{event_id} not found. Using latest: {meta_path.name}")
    meta = load_json(meta_path)
    event_name = meta.get("event_name")
    print("\n" + "=" * 80)
    print(f"Event Meta: {event_name} (event_id={meta.get('event_id')})")
    print("=" * 80)
    print(json.dumps(meta, indent=2))

    # Field + tee-times (by event id)
    print("\n" + "=" * 80)
    print("Field / Tee Times")
    print("=" * 80)
    fld_parquet = processed / f"event_{event_id}_field.parquet"
    fld_csv = processed / f"event_{event_id}_field.csv"
    tt_parquet = processed / f"event_{event_id}_field_teetimes.parquet"
    tt_csv = processed / f"event_{event_id}_field_teetimes.csv"

    if exists(fld_parquet) or exists(fld_csv):
        df_field = pd.read_parquet(fld_parquet) if fld_parquet.exists() else pd.read_csv(fld_csv)
        print(f"Field rows: {len(df_field)}, cols: {len(df_field.columns)}; sample cols: {head_cols(df_field)}")
    else:
        print("Field table missing.")

    if exists(tt_parquet) or exists(tt_csv):
        df_tt = pd.read_parquet(tt_parquet) if tt_parquet.exists() else pd.read_csv(tt_csv)
        have_r1 = "r1_teetime" in df_tt.columns
        print(f"Tee-time table rows: {len(df_tt)}, cols: {len(df_tt.columns)}; R1 present: {have_r1}")
    else:
        print("Tee-time table missing (may be null until release).")

    # Weather
    print("\n" + "=" * 80)
    print("Weather")
    print("=" * 80)
    wh = processed / f"event_{event_id}_weather_hourly.json"
    wn = processed / f"event_{event_id}_weather_round_neutral.parquet"
    ww = processed / f"event_{event_id}_weather_round_wave.parquet"
    if wh.exists():
        j = load_json(wh)
        print(f"Hourly weather keys: {list(j.keys())}")
    else:
        print("Hourly weather not found.")

    if wn.exists():
        df_wn = pd.read_parquet(wn)
        print(f"Neutral round weather: {len(df_wn)} rows; cols: {head_cols(df_wn)}")
    else:
        print("Neutral round weather summary missing.")

    if ww.exists():
        df_ww = pd.read_parquet(ww)
        print(f"Wave-aware round weather: {len(df_ww)} rows; cols: {head_cols(df_ww)}")
    else:
        print("Wave-aware round weather summary missing.")

    # Player data
    print("\n" + "=" * 80)
    print("Player Ratings / Rankings")
    print("=" * 80)
    dr = processed / f"event_{event_id}_dg_rankings.parquet"
    sr = processed / f"event_{event_id}_skill_ratings.parquet"
    if dr.exists():
        df_dr = pd.read_parquet(dr)
        print(f"DG rankings: {len(df_dr)} rows; cols: {head_cols(df_dr)}")
    else:
        print("DG rankings missing.")
    if sr.exists():
        df_sr = pd.read_parquet(sr)
        print(f"Skill ratings: {len(df_sr)} rows; cols: {head_cols(df_sr)}")
    else:
        print("Skill ratings missing.")

    # Sigma
    print("\n" + "=" * 80)
    print("Volatility (Sigma)")
    print("=" * 80)
    sp = processed / f"event_{event_id}_player_sigma.parquet"
    if sp.exists():
        df_sigma = pd.read_parquet(sp)
        print(f"Player sigma: {len(df_sigma)} rows; cols: {head_cols(df_sigma)}")
    else:
        print("Sigma parquet missing.")

    # Course fit / weights
    print("\n" + "=" * 80)
    print("Course Fit (DIY) + Weights")
    print("=" * 80)
    diy = processed / f"event_{event_id}_course_fit_diy.parquet"
    wts = processed / f"event_{event_id}_course_fit_weights.json"
    if diy.exists():
        df_diy = pd.read_parquet(diy)
        print(f"DIY course fit: {len(df_diy)} rows; cols: {head_cols(df_diy)}")
        has_da = "da_input" in df_diy.columns or "da_z" in df_diy.columns
        has_dd = "dd_input" in df_diy.columns or "dd_z" in df_diy.columns
        print(f"Driving in DIY: DA present={has_da}, DD present={has_dd}")
    else:
        print("DIY course fit parquet missing.")
    if wts.exists():
        jw = load_json(wts)
        print("Course fit weights:", json.dumps(jw, indent=2))
    else:
        print("Course fit weights json missing.")

    # Features
    print("\n" + "=" * 80)
    print("Features Tables")
    print("=" * 80)
    fw = features / f"event_{event_id}_features_weather.parquet"
    ff = features / f"event_{event_id}_features_full.parquet"
    fc = features / f"event_{event_id}_features_course.parquet"
    if fw.exists():
        df_fw = pd.read_parquet(fw)
        print(f"Features (weather): {len(df_fw)} rows; cols: {head_cols(df_fw)}")
    else:
        print("Features (weather) missing.")
    if ff.exists():
        df_ff = pd.read_parquet(ff)
        print(f"Features (full): {len(df_ff)} rows; cols: {head_cols(df_ff)}")
    else:
        print("Features (full) missing.")
    if fc.exists():
        df_fc = pd.read_parquet(fc)
        print(f"Features (course): {len(df_fc)} rows; cols: {head_cols(df_fc)}")
    else:
        print("Features (course) missing (might still be equal to full).")

    # Predictions (by event id)
    print("\n" + "=" * 80)
    print("Predictions")
    print("=" * 80)
    any_preds = False
    for stem in ["with_course", "common_shock", "baseline"]:
        pf = preds / f"event_{event_id}_preds_{stem}.parquet"
        if pf.exists():
            any_preds = True
            dfp = pd.read_parquet(pf)
            print(f"{pf.name}: rows={len(dfp)}; cols: {head_cols(dfp)}")
            if "p_win" in dfp.columns:
                print("Top 10 by p_win:")
                print(dfp.sort_values("p_win", ascending=False).head(10).to_string(index=False))
    if not any_preds:
        print("No predictions found yet for this event.")

    # Historical combined (optional)
    print("\n" + "=" * 80)
    print("Historical Combined (optional)")
    print("=" * 80)
    hist_glob = list(raw_hist.glob("tournament_*_rounds_combined.parquet"))
    if hist_glob:
        print("Found historical combined parquet(s):")
        for p in hist_glob[:3]:
            print("-", p.name)
        if len(hist_glob) > 3:
            print(f"... and {len(hist_glob) - 3} more")
    else:
        print("No historical combined parquet found.")


if __name__ == "__main__":
    main()
