#!/usr/bin/env python3
# scripts/merge_course_fit_diy_into_features.py
#
# Merge DIY course fit (with driving features) into features_full.
# Inputs:
#   - data/processed/{tour}/event_{event_id}_course_fit_diy.parquet
#       columns: [dg_id/player_id, player_name?, course_fit_score, da_input, dd_input, da_z, dd_z, ...cats]
#   - data/features/{tour}/event_{event_id}_features_full.parquet
# Outputs (in-place update + snapshot):
#   - data/features/{tour}/event_{event_id}_features_full.parquet  (updated with DIY columns)
#   - data/features/{tour}/event_{event_id}_features_course.parquet (snapshot with same content, for simulator)

from pathlib import Path
import json
import pandas as pd

TOUR = "pga"

DIY_COLS_PREFERRED = [
    "course_fit_score",  # main DIY course fit metric
    "da_input",
    "dd_input",  # venue/overall inputs used
    "da_z",
    "dd_z",  # standardized vs venue field for model
    # any category columns present will be merged too (e.g., sg_ott, sg_app, ...)
]


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    features = root / "data" / "features" / TOUR
    features.mkdir(parents=True, exist_ok=True)

    # Load current event meta
    metas = sorted(processed.glob("event_*_meta.json"))
    if not metas:
        raise FileNotFoundError("No meta found. Run parse_field_updates.py first.")
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    event_id = str(meta["event_id"])

    # Load base features
    feats_path = features / f"event_{event_id}_features_full.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(
            "Missing features_full; run merge_player_data_into_features.py first."
        )
    feats = pd.read_parquet(feats_path)

    # Load DIY course fit
    diy_path = processed / f"event_{event_id}_course_fit_diy.parquet"
    if not diy_path.exists():
        raise FileNotFoundError(
            "Missing DIY course fit parquet. Run build_course_fit_from_history.py first."
        )
    diy = pd.read_parquet(diy_path)

    # Align ID column
    key = None
    for cand in ["dg_id", "player_id", "id"]:
        if cand in feats.columns and cand in diy.columns:
            key = cand
            break
    if key is None:
        # try to rename diy id to feats key
        for feats_key in ["dg_id", "player_id", "id"]:
            if feats_key in feats.columns:
                for diy_key in ["dg_id", "player_id", "id"]:
                    if diy_key in diy.columns:
                        diy = diy.rename(columns={diy_key: feats_key})
                        key = feats_key
                        break
            if key:
                break
    if key is None:
        raise ValueError(
            "Could not align ID columns between features_full and DIY course fit."
        )

    # Select DIY columns to merge (keep whatever exists)
    diy_cols_available = [c for c in DIY_COLS_PREFERRED if c in diy.columns]
    # Include any skill category columns if they exist (sg_ott, sg_app, sg_arg, sg_putt or sg_t2g)
    extra_skill_cols = [
        c
        for c in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g"]
        if c in diy.columns
    ]
    merge_cols = [key] + list(
        dict.fromkeys(diy_cols_available + extra_skill_cols)
    )  # preserve order, dedupe

    diy_small = diy[merge_cols].drop_duplicates(subset=[key]).copy()

    # If features already have a course_fit_score, prefer DIY by overwriting nulls and conflicting values
    if "course_fit_score" in feats.columns and "course_fit_score" in diy_small.columns:
        # we'll merge DIY and then fillna with existing where DIY missing
        pass  # handled by merge + fill below

    # Merge
    out = feats.merge(diy_small, on=key, how="left", suffixes=("", "_diy"))

    # Fill sensible defaults
    # course_fit_score: fill missing with median (neutral effect)
    if "course_fit_score" in out.columns:
        out["course_fit_score"] = out["course_fit_score"].astype(float)
        if out["course_fit_score"].isna().any():
            out["course_fit_score"] = out["course_fit_score"].fillna(
                out["course_fit_score"].median()
            )

    # driving z-scores default to 0 (neutral vs field)
    for zcol in ["da_z", "dd_z"]:
        if zcol in out.columns:
            out[zcol] = out[zcol].astype(float).fillna(0.0)

    # inputs: keep as-is; if missing, leave NaN or fill with feature-wise mean
    for incol in ["da_input", "dd_input"]:
        if incol in out.columns:
            # If you prefer to fill, uncomment:
            # out[incol] = out[incol].astype(float).fillna(out[incol].mean())
            out[incol] = out[incol].astype(float)

    # Persist updates
    out.to_parquet(feats_path, index=False)

    # Also write a "course" snapshot (used by simulator variant)
    course_snapshot = features / f"event_{event_id}_features_course.parquet"
    out.to_parquet(course_snapshot, index=False)

    print("Merged DIY course fit (and driving) into features:")
    print(f"- Updated: {feats_path}")
    print(f"- Snapshot: {course_snapshot}")

    # Quick preview
    preview_cols = [key, "player_name", "course_fit_score", "da_z", "dd_z"]
    preview_cols = [c for c in preview_cols if c in out.columns]
    if preview_cols:
        print(out[preview_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
