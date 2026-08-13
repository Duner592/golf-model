#!/usr/bin/env python3
"""
Combine archived prediction leaderboards with actual tournament results
to evaluate model reliability across events.

Outputs:
    data/analytics/prediction_vs_actual.csv   (detailed player-level joins)
    data/analytics/prediction_vs_actual.parquet (optional, if --parquet)
    data/analytics/prediction_accuracy_summary.json (tour-level metrics)

Usage:
    python scripts/build_prediction_accuracy.py [--tour pga] [--year 2026] [--fetch-missing]

Notes:
    - Requires predictions archived under web/archive/{year}/{slug}.
    - For completed events, expects results.json created by fetch_actual_results.py.
      Pass --fetch-missing to download missing results via DataGolf API on the fly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge archived predictions with actual results for evaluation.")
    ap.add_argument("--tour", action="append", help="Limit to one or more tours (e.g. --tour pga --tour euro).")
    ap.add_argument("--year", help="Limit to a specific season year (e.g. 2026).")
    ap.add_argument("--fetch-missing", action="store_true", help="Run fetch_actual_results.py before merging to backfill results.")
    ap.add_argument("--parquet", action="store_true", help="Also write Parquet output alongside CSV.")
    ap.add_argument("--out-dir", default="data/analytics", help="Directory to store aggregated outputs.")
    ap.add_argument("--verbose", action="store_true", help="Print details while processing.")
    return ap.parse_args()


def load_archive_index() -> list[dict[str, Any]]:
    index_path = ROOT / "web" / "archive" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Archive index not found: {index_path}")
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Archive index format unexpected (expected list): {index_path}")
    return data


def normalize_prediction_name(name: str | None) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        return f"{first} {last}".strip() if first else last
    return name


PLAYER_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_tokens(name: str | None) -> list[str]:
    name = normalize_prediction_name(name)
    if not name:
        return []
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[a-z0-9]+", ascii_name.lower())
    while tokens and tokens[-1] in PLAYER_SUFFIXES:
        tokens.pop()
    return tokens


def clean_name_key(name: str | None) -> str:
    return "".join(name_tokens(name))


def name_key_candidates(name: str | None) -> list[str]:
    tokens = name_tokens(name)
    keys: list[str] = []
    if tokens:
        keys.append("".join(tokens))
    if len(tokens) >= 3:
        keys.append(f"{tokens[0]}{tokens[-1]}")
    return list(dict.fromkeys(keys))


def coerce_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return bool(int(val))
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in {"", "na", "null", "none"}:
            return None
        if cleaned in {"true", "t", "yes", "y", "1"}:
            return True
        if cleaned in {"false", "f", "no", "n", "0"}:
            return False
    return bool(val)


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_predictions(event_dir: Path) -> pd.DataFrame:
    csv_path = event_dir / "leaderboard.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        json_path = event_dir / "leaderboard.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No leaderboard file found in {event_dir}")
        df = pd.DataFrame(_read_json(json_path))
    # Standardize column names
    rename_map = {
        "player_name": "player",
        "Player": "player",
        "p_win_%": "p_win_pct",
        "p_top5_%": "p_top5_pct",
        "p_top10_%": "p_top10_pct",
        "p_mc_%": "p_mc_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "player" not in df.columns:
        raise ValueError(f"Leaderboard missing player column in {event_dir}")
    df["player"] = df["player"].apply(normalize_prediction_name)
    df["player_key"] = df["player"].apply(clean_name_key)
    df["_match_keys"] = df["player"].apply(name_key_candidates)
    for col in ("p_win_pct", "p_top5_pct", "p_top10_pct", "p_mc_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_actual_results(event_dir: Path) -> pd.DataFrame | None:
    results_path = event_dir / "results.json"
    if not results_path.exists():
        return None
    data = _read_json(results_path)
    players = data.get("players") or []
    if not players:
        return None
    df = pd.DataFrame(players)
    if "player" not in df.columns:
        raise ValueError(f"results.json missing 'player' entries in {event_dir}")
    df["player"] = df["player"].apply(normalize_prediction_name)
    df["player_key"] = df["player"].apply(clean_name_key)
    df["_match_keys"] = df["player"].apply(name_key_candidates)
    return df


def event_completed(event_dir: Path) -> bool:
    summary_path = event_dir / "tournament_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = _read_json(summary_path)
    except Exception:
        return False
    status = (summary.get("status") or "").strip().lower()
    return status in {"completed", "finished"}


def brier_score(probs: Iterable[float], outcomes: Iterable[float]) -> float | None:
    probs_series = pd.Series(list(probs), dtype="float64")
    outcomes_series = pd.Series(list(outcomes), dtype="float64")
    mask = probs_series.notna() & outcomes_series.notna()
    if not mask.any():
        return None
    diff = probs_series[mask] / 100.0 - outcomes_series[mask]
    return float((diff ** 2).mean())


def log_loss(probs: Iterable[float], outcomes: Iterable[float]) -> float | None:
    """Mean binary log loss for percentage probabilities, with safe clipping."""
    probs_series = pd.Series(list(probs), dtype="float64")
    outcomes_series = pd.Series(list(outcomes), dtype="float64")
    mask = probs_series.notna() & outcomes_series.notna()
    if not mask.any():
        return None
    clipped = (probs_series[mask] / 100.0).clip(lower=1e-6, upper=1 - 1e-6)
    values = -(outcomes_series[mask] * clipped.map(math.log) + (1 - outcomes_series[mask]) * (1 - clipped).map(math.log))
    return float(values.mean())


def calibration_buckets(probs: Iterable[float], outcomes: Iterable[float], *, bucket_width: int = 5) -> list[dict[str, Any]]:
    """Return player-level calibration buckets without silently dropping missing outcomes."""
    frame = pd.DataFrame({"probability_pct": list(probs), "outcome": list(outcomes)})
    frame["probability_pct"] = pd.to_numeric(frame["probability_pct"], errors="coerce")
    frame["outcome"] = pd.to_numeric(frame["outcome"], errors="coerce")
    frame = frame.dropna(subset=["probability_pct", "outcome"])
    if frame.empty:
        return []
    frame["probability_pct"] = frame["probability_pct"].clip(lower=0, upper=100)
    frame["bucket_start_pct"] = (frame["probability_pct"] // bucket_width * bucket_width).astype(int)
    frame.loc[frame["bucket_start_pct"] == 100, "bucket_start_pct"] = 100 - bucket_width

    buckets: list[dict[str, Any]] = []
    for bucket_start, group in frame.groupby("bucket_start_pct", sort=True):
        mean_prediction = float(group["probability_pct"].mean())
        actual_rate = float(group["outcome"].mean() * 100)
        buckets.append(
            {
                "range": f"{bucket_start}-{bucket_start + bucket_width}%",
                "n": int(len(group)),
                "mean_predicted_pct": mean_prediction,
                "actual_rate_pct": actual_rate,
                "gap_pct_points": actual_rate - mean_prediction,
            }
        )
    return buckets


def uniform_field_probabilities(group: pd.DataFrame, *, target_places: int) -> pd.Series:
    """A transparent no-skill baseline: equal probability for every listed player."""
    counts = group.groupby("event_id")["event_id"].transform("size").clip(lower=1)
    return (100.0 * target_places / counts).clip(upper=100.0)


def merge_predictions_with_actuals(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    actual_cols = [
        "player_key",
        "player",
        "finish_text",
        "finish_pos",
        "score",
        "to_par",
        "made_cut",
    ]
    key_to_indices: dict[str, list[int]] = {}
    for actual_idx, actual_row in actual_df.iterrows():
        for key in actual_row.get("_match_keys") or [actual_row.get("player_key")]:
            if key:
                key_to_indices.setdefault(str(key), []).append(actual_idx)

    unique_key_to_index = {
        key: indices[0]
        for key, indices in key_to_indices.items()
        if len(set(indices)) == 1
    }

    matched_records: list[dict[str, Any]] = []
    for _, pred_row in pred_df.iterrows():
        matched_idx = None
        for key in pred_row.get("_match_keys") or [pred_row.get("player_key")]:
            matched_idx = unique_key_to_index.get(str(key))
            if matched_idx is not None:
                break

        if matched_idx is None:
            matched_records.append({col: None for col in actual_cols})
            continue

        matched_row = actual_df.loc[matched_idx]
        matched_records.append({col: matched_row.get(col) for col in actual_cols})

    actual_matches = pd.DataFrame(matched_records)
    actual_matches = actual_matches.rename(
        columns={
            "player_key": "player_key_actual",
            "player": "player_actual",
        }
    )
    return pd.concat(
        [
            pred_df.drop(columns=["_match_keys"], errors="ignore").reset_index(drop=True),
            actual_matches.reset_index(drop=True),
        ],
        axis=1,
    )


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_entries = load_archive_index()
    if args.year:
        archive_entries = [e for e in archive_entries if str(e.get("year")) == str(args.year)]
    archive_entries = [e for e in archive_entries if str(e.get("tour")).lower() != "liv"]
    if args.tour:
        wanted = {t.lower() for t in args.tour}
        archive_entries = [e for e in archive_entries if str(e.get("tour")).lower() in wanted]

    if not archive_entries:
        print("No archive entries match the provided filters.")
        return

    rows: list[dict[str, Any]] = []
    skipped_events: list[str] = []

    if args.fetch_missing:
        cmd = [sys.executable, str(ROOT / "scripts" / "fetch_actual_results.py")]
        if args.year:
            cmd += ["--year", str(args.year)]
        if args.verbose:
            print(f"[info] Running {' '.join(cmd)} to backfill results...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[warn] fetch_actual_results.py exited with {exc.returncode}: {exc}")

    for entry in archive_entries:
        tour = str(entry.get("tour"))
        year = str(entry.get("year"))
        event_id = str(entry.get("event_id"))
        slug = entry.get("slug")
        event_name = entry.get("event_name")
        event_dir = ROOT / "web" / "archive" / year / slug

        if not event_dir.exists():
            skipped_events.append(f"{year}-{event_name} (archive dir missing)")
            continue

        if not event_completed(event_dir):
            if args.verbose:
                print(f"[info] Skipping {event_name} ({tour} {year}) - event not completed.")
            continue

        try:
            pred_df = load_predictions(event_dir)
        except Exception as exc:
            skipped_events.append(f"{year}-{event_name} (predictions load failed: {exc})")
            continue

        actual_df = load_actual_results(event_dir)
        if actual_df is None or actual_df.empty:
            skipped_events.append(f"{year}-{event_name} (no actual results)")
            continue

        merged = merge_predictions_with_actuals(pred_df, actual_df)

        merged["event_id"] = event_id
        merged["event_name"] = event_name
        merged["tour"] = tour
        merged["year"] = year
        merged["finish_pos"] = pd.to_numeric(merged["finish_pos"], errors="coerce")
        merged["made_cut"] = merged["made_cut"].apply(coerce_bool)
        merged["actual_win"] = merged["finish_pos"].apply(lambda x: float(x == 1) if pd.notna(x) else None)
        merged["actual_top10"] = merged["finish_pos"].apply(lambda x: float(x <= 10) if pd.notna(x) else None)
        merged["actual_made_cut"] = merged["made_cut"].apply(lambda x: float(x) if x is not None else None)

        # Probability columns -> ensure float and keep as percentage
        for col in ("p_win_pct", "p_top10_pct", "p_mc_pct"):
            if col not in merged.columns:
                merged[col] = None

        rows.extend(merged.to_dict(orient="records"))

    if not rows:
        print("No merged prediction/actual rows produced. Nothing to write.")
        if skipped_events:
            print("Skipped events:")
            for msg in skipped_events:
                print(" -", msg)
        return

    df = pd.DataFrame(rows)
    columns = ["event_id", "event_name", "tour", "year"]
    if "rank" in df.columns:
        columns.append("rank")
    columns.extend(
        [
            "player",
            "p_win_pct",
            "p_top10_pct",
            "p_mc_pct",
        ]
    )
    if "course_fit_score" in df.columns:
        columns.append("course_fit_score")
    columns.extend(
        [
            "finish_pos",
            "finish_text",
            "made_cut",
            "actual_win",
            "actual_top10",
            "actual_made_cut",
        ]
    )
    df = df[columns]
    df = df.dropna(axis=1, how="all")  # drop entirely empty columns

    # Save detailed rows
    csv_path = out_dir / "prediction_vs_actual.csv"
    df.to_csv(csv_path, index=False)
    if args.verbose:
        print(f"[info] Wrote {len(df)} rows to {csv_path}")
    if args.parquet:
        parquet_path = out_dir / "prediction_vs_actual.parquet"
        df.to_parquet(parquet_path, index=False)
        if args.verbose:
            print(f"[info] Wrote Parquet to {parquet_path}")

    # Build event-level scores first. These are the correct unit for a
    # chronological out-of-sample review: an event only enters once its final
    # results are archived, and no aggregate can hide a single bad tournament.
    event_rows: list[dict[str, Any]] = []
    for (tour, year, event_id, event_name), group in df.groupby(["tour", "year", "event_id", "event_name"], dropna=False):
        record: dict[str, Any] = {
            "tour": tour,
            "year": year,
            "event_id": event_id,
            "event_name": event_name,
            "players": int(len(group)),
        }
        for label, probability_col, outcome_col, target_places in (
            ("win", "p_win_pct", "actual_win", 1),
            ("top10", "p_top10_pct", "actual_top10", 10),
            ("make_cut", "p_mc_pct", "actual_made_cut", 0),
        ):
            if {probability_col, outcome_col}.issubset(group.columns):
                record[f"brier_{label}"] = brier_score(group[probability_col], group[outcome_col])
                record[f"log_loss_{label}"] = log_loss(group[probability_col], group[outcome_col])
                if target_places:
                    baseline_probs = uniform_field_probabilities(group, target_places=target_places)
                    baseline_brier = brier_score(baseline_probs, group[outcome_col])
                    record[f"uniform_brier_{label}"] = baseline_brier
                    record[f"brier_skill_{label}"] = (
                        1 - record[f"brier_{label}"] / baseline_brier
                        if baseline_brier not in (None, 0) and record[f"brier_{label}"] is not None
                        else None
                    )
        event_rows.append(record)

    event_summary = pd.DataFrame(event_rows).sort_values(["year", "tour", "event_name"], kind="stable")
    event_summary.to_csv(out_dir / "prediction_accuracy_by_event.csv", index=False)

    # Build aggregated summary by tour, including a no-skill equal-field
    # reference. Market/odds comparisons belong in a separate input contract:
    # do not manufacture a market baseline from untimestamped spreadsheet data.
    summary: dict[str, Any] = {}
    for tour, group in df.groupby("tour"):
        available_cols = set(group.columns)
        win_brier = (
            brier_score(group["p_win_pct"], group["actual_win"])
            if {"p_win_pct", "actual_win"}.issubset(available_cols)
            else None
        )
        top10_brier = (
            brier_score(group["p_top10_pct"], group["actual_top10"])
            if {"p_top10_pct", "actual_top10"}.issubset(available_cols)
            else None
        )
        mc_brier = (
            brier_score(group["p_mc_pct"], group["actual_made_cut"])
            if {"p_mc_pct", "actual_made_cut"}.issubset(available_cols)
            else None
        )
        win_uniform_brier = (
            brier_score(uniform_field_probabilities(group, target_places=1), group["actual_win"])
            if {"event_id", "actual_win"}.issubset(available_cols)
            else None
        )
        top10_uniform_brier = (
            brier_score(uniform_field_probabilities(group, target_places=10), group["actual_top10"])
            if {"event_id", "actual_top10"}.issubset(available_cols)
            else None
        )
        summary[tour] = {
            "events": int(group["event_id"].nunique()),
            "rows": int(len(group)),
            "brier_win": win_brier,
            "brier_top10": top10_brier,
            "brier_mc": mc_brier,
            "log_loss_win": log_loss(group["p_win_pct"], group["actual_win"]) if {"p_win_pct", "actual_win"}.issubset(available_cols) else None,
            "log_loss_top10": log_loss(group["p_top10_pct"], group["actual_top10"]) if {"p_top10_pct", "actual_top10"}.issubset(available_cols) else None,
            "log_loss_mc": log_loss(group["p_mc_pct"], group["actual_made_cut"]) if {"p_mc_pct", "actual_made_cut"}.issubset(available_cols) else None,
            "uniform_brier_win": win_uniform_brier,
            "uniform_brier_top10": top10_uniform_brier,
            "brier_skill_win": 1 - win_brier / win_uniform_brier if win_brier is not None and win_uniform_brier not in (None, 0) else None,
            "brier_skill_top10": 1 - top10_brier / top10_uniform_brier if top10_brier is not None and top10_uniform_brier not in (None, 0) else None,
            "mean_predicted_win_pct": float(group["p_win_pct"].dropna().mean()) if "p_win_pct" in available_cols else None,
            "mean_actual_win_rate": float(group["actual_win"].dropna().mean()) if "actual_win" in available_cols else None,
            "mean_predicted_top10_pct": float(group["p_top10_pct"].dropna().mean()) if "p_top10_pct" in available_cols else None,
            "mean_actual_top10_rate": float(group["actual_top10"].dropna().mean()) if "actual_top10" in available_cols else None,
        }

    calibration = {
        tour: {
            "win": calibration_buckets(group["p_win_pct"], group["actual_win"]) if {"p_win_pct", "actual_win"}.issubset(group.columns) else [],
            "top10": calibration_buckets(group["p_top10_pct"], group["actual_top10"]) if {"p_top10_pct", "actual_top10"}.issubset(group.columns) else [],
            "make_cut": calibration_buckets(group["p_mc_pct"], group["actual_made_cut"]) if {"p_mc_pct", "actual_made_cut"}.issubset(group.columns) else [],
        }
        for tour, group in df.groupby("tour")
    }

    summary_path = out_dir / "prediction_accuracy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "prediction_calibration_buckets.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    if args.verbose:
        print(f"[info] Wrote summary metrics to {summary_path}")

    if skipped_events:
        print("Completed with some skips:")
        for msg in skipped_events:
            print(" -", msg)


if __name__ == "__main__":
    main()
