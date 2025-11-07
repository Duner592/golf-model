#!/usr/bin/env python3
# scripts/analyze_backtest_2025.py
#
# Analyze the 2025 backtest recap with optional exclusions.
# - Loads data/preds/pga/backtest_2025_recap.json
# - Applies exclusions from configs/experiments.yaml (backtest.exclude_event_ids)
#   and/or CLI --exclude (comma-separated list)
# - Prints aggregate metrics, best/worst tables, outliers, and saves a sortable CSV.
#
from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd

TOUR = "pga"
RECAP_PATH = Path(f"data/preds/{TOUR}/backtest_2025_recap.json")
TABLE_OUT = Path(f"data/preds/{TOUR}/backtest_2025_table.csv")
EXPERIMENTS_YAML = Path("configs/experiments.yaml")


def load_recap() -> pd.DataFrame:
    if not RECAP_PATH.exists():
        raise FileNotFoundError(f"Missing recap: {RECAP_PATH}")
    raw = json.loads(RECAP_PATH.read_text(encoding="utf-8"))
    rows = raw.get("events", raw.get("results", []))  # support both keys
    df = pd.DataFrame(rows)
    for c in ("log_loss", "brier"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].astype(str)
    return df


def load_exclusions() -> set[str]:
    """
    Read exclude_event_ids from configs/experiments.yaml (if present).
    """
    if not EXPERIMENTS_YAML.exists():
        return set()
    try:
        import yaml  # local import to avoid hard dependency elsewhere

        cfg = yaml.safe_load(EXPERIMENTS_YAML.read_text(encoding="utf-8")) or {}
        ids = cfg.get("backtest", {}).get("exclude_event_ids", []) or []
        return {str(x) for x in ids}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser(
        description="Analyze backtest_2025_recap.json with optional exclusions."
    )
    ap.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated event_ids to drop in addition to config exclusions (e.g., '60,27')",
    )
    args = ap.parse_args()

    # Load recap and exclusions
    df = load_recap()
    exclude = load_exclusions()
    if args.exclude:
        extra = {x.strip() for x in args.exclude.split(",") if x.strip()}
        exclude |= extra

    # Apply exclusions
    if exclude and "event_id" in df.columns:
        df = df[~df["event_id"].isin(exclude)].copy()
        df = df.reset_index(drop=True)
        print(f"[info] Excluding events: {sorted(exclude)}")

    if df.empty:
        print("No events in recap after exclusions.")
        return

    # Aggregate metrics
    agg = {
        "events": int(len(df)),
        "log_loss_mean": float(df["log_loss"].mean()),
        "log_loss_median": float(df["log_loss"].median()),
        "brier_mean": float(df["brier"].mean()),
        "brier_median": float(df["brier"].median()),
        "log_loss_std": float(df["log_loss"].std()),
    }
    print("Aggregate:")
    for k, v in agg.items():
        print(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}")

    # Best/Worst tables
    print("\nBest 10 events (lowest log-loss):")
    print(df.sort_values("log_loss").head(10).to_string(index=False))

    print("\nWorst 10 events (highest log-loss):")
    print(df.sort_values("log_loss").tail(10).to_string(index=False))

    # Outliers: > mean + 2*std
    if agg["log_loss_std"] > 0:
        thr = agg["log_loss_mean"] + 2 * agg["log_loss_std"]
        out = df[df["log_loss"] > thr]
        print(f"\nOutliers (log_loss > {thr:.6f}): {len(out)}")
        if not out.empty:
            print(out.sort_values("log_loss", ascending=False).to_string(index=False))

    # Save sortable table
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("log_loss").to_csv(TABLE_OUT, index=False)
    print("Saved:", TABLE_OUT)


if __name__ == "__main__":
    main()
