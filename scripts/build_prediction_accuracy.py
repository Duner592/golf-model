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
import re
import subprocess
import sys
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


def clean_name_key(name: str | None) -> str:
    name = normalize_prediction_name(name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


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

        merged = pred_df.merge(
            actual_df[
                [
                    "player_key",
                    "player",
                    "finish_text",
                    "finish_pos",
                    "score",
                    "to_par",
                    "made_cut",
                ]
            ],
            on="player_key",
            how="left",
            suffixes=("", "_actual"),
        )

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

    # Build aggregated summary by tour
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
        summary[tour] = {
            "events": int(group["event_id"].nunique()),
            "rows": int(len(group)),
            "brier_win": win_brier,
            "brier_top10": top10_brier,
            "brier_mc": mc_brier,
            "mean_predicted_win_pct": float(group["p_win_pct"].dropna().mean()) if "p_win_pct" in available_cols else None,
            "mean_actual_win_rate": float(group["actual_win"].dropna().mean()) if "actual_win" in available_cols else None,
            "mean_predicted_top10_pct": float(group["p_top10_pct"].dropna().mean()) if "p_top10_pct" in available_cols else None,
            "mean_actual_top10_rate": float(group["actual_top10"].dropna().mean()) if "actual_top10" in available_cols else None,
        }

    summary_path = out_dir / "prediction_accuracy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.verbose:
        print(f"[info] Wrote summary metrics to {summary_path}")

    if skipped_events:
        print("Completed with some skips:")
        for msg in skipped_events:
            print(" -", msg)


if __name__ == "__main__":
    main()
