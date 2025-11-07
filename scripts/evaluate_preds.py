#!/usr/bin/env python3
# scripts/evaluate_preds.py
#
# Evaluate predictions for a specific event (log-loss, Brier).
# Resolution order for event_id:
#   1) --event_id
#   2) Most recent event that has predictions in data/preds/{tour}
#
# Outputs:
#   data/preds/{tour}/event_{event_id}_eval_summary.json

from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

TOUR = "pga"


def scan_pred_events(preds_dir: Path) -> list[str]:
    """Return sorted unique event_ids that have preds files."""
    ids = set()
    for p in preds_dir.glob("event_*_preds_*.parquet"):
        m = re.match(r"event_(\d+)_preds_", p.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def resolve_event_id(preds_dir: Path, arg_event_id: str | None) -> str:
    if arg_event_id:
        return str(arg_event_id)
    cand = scan_pred_events(preds_dir)
    if not cand:
        raise FileNotFoundError(f"No prediction files found under {preds_dir}")
    return cand[-1]  # most recent event_id by numeric sort


def load_preds_for_event(preds_dir: Path, event_id: str) -> pd.DataFrame:
    """Load predictions for event_id with preference order."""
    for stem in ["with_course", "common_shock", "baseline"]:
        p = preds_dir / f"event_{event_id}_preds_{stem}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"No predictions found for event_id={event_id} in {preds_dir}"
    )


def load_results_for_event(processed_dir: Path, event_id: str) -> pd.DataFrame:
    p = processed_dir / f"event_{event_id}_results.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing results CSV: {p}")
    return pd.read_csv(p)


def pick_join_key(preds: pd.DataFrame, results: pd.DataFrame) -> str:
    if "dg_id" in preds.columns and "dg_id" in results.columns:
        return "dg_id"
    return "player_name"


def evaluate(y_true: pd.Series, p: pd.Series) -> dict:
    """Compute metrics, handling the 'single label' case gracefully."""
    y = y_true.astype(int).to_numpy()
    ps = np.clip(p.to_numpy().astype(float), 1e-12, 1 - 1e-12)

    # Default metrics
    out = {
        "p_win_sum": float(ps.sum()),
        "p_win_max": float(ps.max()),
        "p_win_median": float(np.median(ps)),
        "n": int(len(y)),
    }

    unique = np.unique(y)
    if unique.size == 1:
        # Incomplete results (all zeros) or no winner flagged: report log_loss with labels=[0,1]
        out["log_loss"] = float(log_loss(y, ps, labels=[0, 1]))
        out["brier"] = float(brier_score_loss(y, ps))
        out["note"] = (
            "winner_flag has a single class; used labels=[0,1] (event likely not completed)"
        )
    else:
        out["log_loss"] = float(log_loss(y, ps))
        out["brier"] = float(brier_score_loss(y, ps))

    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions for an event.")
    ap.add_argument("--tour", default=TOUR)
    ap.add_argument(
        "--event_id",
        type=str,
        default=None,
        help="Specific event_id (recommended in backtests)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    preds_dir = root / "data" / "preds" / args.tour
    processed_dir = root / "data" / "processed" / args.tour

    event_id = resolve_event_id(preds_dir, args.event_id)
    preds = load_preds_for_event(preds_dir, event_id)
    results = load_results_for_event(processed_dir, event_id)

    key = pick_join_key(preds, results)
    preds[key] = preds[key].astype(str)
    results[key] = results[key].astype(str)

    merged = preds.merge(results[[key, "winner_flag"]], on=key, how="inner")
    if merged.empty or "p_win" not in merged.columns:
        raise ValueError(
            f"No overlap or missing p_win for event_id={event_id} (key={key})"
        )

    metrics = evaluate(merged["winner_flag"], merged["p_win"])

    # Write summary
    out = preds_dir / f"event_{event_id}_eval_summary.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Diagnostics
    print(
        json.dumps(
            {
                "event_id": event_id,
                "join_key": key,
                "preds_rows": len(preds),
                "results_rows": len(results),
                "merged_rows": len(merged),
                **metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
