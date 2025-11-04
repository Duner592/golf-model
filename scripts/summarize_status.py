#!/usr/bin/env python3
# scripts/summarize_status.py
#
# Summarizes the current event’s artifacts:
# - processed: field, meta, weather (hourly + summaries), player data, sigma, course fit DIY (+ driving inside DIY)
# - features: features_weather, features_full, features_course
# - preds: baseline / common_shock / with_course
#
# Note:
# - Driving features are already included in DIY course fit (da_input, dd_input, da_z, dd_z).
# - We no longer check (or warn) about a standalone driving_features parquet.

from pathlib import Path
import json
import pandas as pd

TOUR = "pga"


def exists(p: Path) -> bool:
    return p.exists()


def head_cols(df: pd.DataFrame, k=8) -> list:
    return list(df.columns[:k])


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR
    preds = root / "data" / "preds" / TOUR
    raw_hist = root / "data" / "raw" / "historical" / TOUR

    # Meta
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        print("No meta found. Run parse_field_updates.py first.")
        return
    meta = load_json(metas[-1])
    event_id = str(meta.get("event_id"))
    event_name = meta.get("event_name")
    print_section(f"Event Meta: {event_name} (event_id={event_id})")
    print(json.dumps(meta, indent=2))

    # Field + tee times
    print_section("Field / Tee Times")
    fld_parquet = processed / f"event_{event_id}_field.parquet"
    fld_csv = processed / f"event_{event_id}_field.csv"
    tt_parquet = processed / f"event_{event_id}_field_teetimes.parquet"
    tt_csv = processed / f"event_{event_id}_field_teetimes.csv"

    if exists(fld_parquet) or exists(fld_csv):
        df_field = (
            pd.read_parquet(fld_parquet)
            if fld_parquet.exists()
            else pd.read_csv(fld_csv)
        )
        print(
            f"Field rows: {len(df_field)}, cols: {len(df_field.columns)}; sample cols: {head_cols(df_field)}"
        )
    else:
        print("Field table missing.")

    if exists(tt_parquet) or exists(tt_csv):
        df_tt = (
            pd.read_parquet(tt_parquet) if tt_parquet.exists() else pd.read_csv(tt_csv)
        )
        have_r1 = "r1_teetime" in df_tt.columns
        print(
            f"Tee-time table rows: {len(df_tt)}, cols: {len(df_tt.columns)}; R1 present: {have_r1}"
        )
    else:
        print("Tee-time table missing (may be null until release).")

    # Weather (hourly + summaries)
    print_section("Weather")
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

    # Player data: rankings / skill ratings
    print_section("Player Ratings / Rankings")
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
    print_section("Volatility (Sigma)")
    sp = processed / f"event_{event_id}_player_sigma.parquet"
    if sp.exists():
        df_sigma = pd.read_parquet(sp)
        print(f"Player sigma: {len(df_sigma)} rows; cols: {head_cols(df_sigma)}")
    else:
        print("Sigma parquet missing.")

    # DIY course fit + weights (includes driving inputs already)
    print_section("Course Fit (DIY) + Weights")
    diy = processed / f"event_{event_id}_course_fit_diy.parquet"
    wts = processed / f"event_{event_id}_course_fit_weights.json"
    if diy.exists():
        df_diy = pd.read_parquet(diy)
        print(f"DIY course fit: {len(df_diy)} rows; cols: {head_cols(df_diy)}")
        # Highlight presence of driving inputs within DIY
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

    # NOTE: We no longer check (or warn) about a standalone driving_features parquet.

    # Features tables
    print_section("Features Tables")
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

    # Predictions
    print_section("Predictions")
    pred_files = [
        preds / f"event_{event_id}_preds_baseline.parquet",
        preds / f"event_{event_id}_preds_common_shock.parquet",
        preds / f"event_{event_id}_preds_with_course.parquet",
    ]
    any_preds = False
    for pf in pred_files:
        if pf.exists():
            any_preds = True
            dfp = pd.read_parquet(pf)
            print(f"{pf.name}: rows={len(dfp)}; cols: {head_cols(dfp)}")
            if "p_win" in dfp.columns:
                print("Top 10 by p_win:")
                print(
                    dfp.sort_values("p_win", ascending=False)
                    .head(10)
                    .to_string(index=False)
                )
    if not any_preds:
        print("No predictions found yet.")

    # Historical combined parquet presence (optional)
    print_section("Historical Combined (optional)")
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
