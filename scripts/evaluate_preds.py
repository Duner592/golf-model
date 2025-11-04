#!/usr/bin/env python3
# scripts/evaluate_preds.py
#
# Evaluate predictions for one event using realized outcomes.
# Inputs:
#   - data/preds/{tour}/event_{event_id}_preds_with_course.parquet (or fallback)
#   - data/processed/{tour}/event_{event_id}_results.csv   (columns: player_name or dg_id/player_id, winner_flag [0/1])
# Outputs:
#   - data/preds/{tour}/event_{event_id}_eval_summary.json
#   - prints Brier, log-loss, and calibration by deciles

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

TOUR = "pga"


def load_preds(root: Path, event_id: str) -> pd.DataFrame:
    preds_dir = root / "data" / "preds" / TOUR
    for name in [
        f"event_{event_id}_preds_with_course.parquet",
        f"event_{event_id}_preds_common_shock.parquet",
        f"event_{event_id}_preds_baseline.parquet",
    ]:
        p = preds_dir / name
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError("No predictions found.")


def load_results(root: Path, event_id: str) -> pd.DataFrame:
    processed = root / "data" / "processed" / TOUR
    # Expected columns: winner_flag plus a join key
    # Example schema: player_name, winner_flag (1=winner, else 0)
    p = processed / f"event_{event_id}_results.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing results file: {p}. Create it with columns [player_name, winner_flag]."
        )
    return pd.read_csv(p)


def best_key_match(df1: pd.DataFrame, df2: pd.DataFrame):
    # Try to join on dg_id or player_id or player_name
    for k in ["dg_id", "player_id", "player_name"]:
        if k in df1.columns and k in df2.columns:
            return k
    # Try fallback: rename in results if needed
    for k in ["dg_id", "player_id", "player_name"]:
        for r in ["dg_id", "player_id", "player_name"]:
            if k in df1.columns and r in df2.columns:
                d2 = df2.rename(columns={r: k})
                return k, d2
    return None


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins=10) -> pd.DataFrame:
    cuts = np.quantile(p, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = 0.0, 1.0
    idx = np.digitize(p, cuts[1:-1], right=False)
    df = pd.DataFrame({"y": y_true, "p": p, "bin": idx})
    out = (
        df.groupby("bin")
        .agg(obs_rate=("y", "mean"), avg_p=("p", "mean"), count=("p", "size"))
        .reset_index()
    )
    out["abs_gap"] = (out["obs_rate"] - out["avg_p"]).abs()
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed" / TOUR
    preds_dir = root / "data" / "preds" / TOUR

    # Load meta to get event_id
    meta = json.loads(
        sorted(processed.glob("event_*_meta.json"))[-1].read_text(encoding="utf-8")
    )
    event_id = str(meta["event_id"])

    preds = load_preds(root, event_id)
    results = load_results(root, event_id)

    # Pick join key
    key = None
    if "dg_id" in preds.columns and "dg_id" in results.columns:
        key = "dg_id"
    elif "dg_id" in preds.columns and "player_id" in results.columns:
        results = results.rename(columns={"player_id": "dg_id"})
        key = "dg_id"
    elif "player_name" in preds.columns and "player_name" in results.columns:
        key = "player_name"
    else:
        raise ValueError("Could not align join key between predictions and results.")

    # Merge
    df = preds.merge(results[[key, "winner_flag"]], on=key, how="inner")
    if df.empty:
        raise ValueError("No overlap between predictions and results after merge.")

    # Metrics
    y = df["winner_flag"].astype(int).to_numpy()
    p = df["p_win"].astype(float).to_numpy()
    eps = 1e-12
    p_clip = np.clip(p, eps, 1 - eps)

    brier = float(brier_score_loss(y, p_clip))
    ll = float(log_loss(y, p_clip))

    cal = calibration_table(y, p)
    eval_summary = {
        "event_id": event_id,
        "field_size": int(len(df)),
        "brier": brier,
        "log_loss": ll,
        "p_win_sum": float(p.sum()),
        "p_win_max": float(p.max()),
        "p_win_median": float(np.median(p)),
        "calibration_bins": cal.to_dict(orient="records"),
    }

    out = preds_dir / f"event_{event_id}_eval_summary.json"
    out.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    print("Brier:", round(brier, 6), "  Log-loss:", round(ll, 6))
    print(
        "Sum p_win:",
        round(eval_summary["p_win_sum"], 6),
        "  Max p_win:",
        round(eval_summary["p_win_max"], 4),
    )
    print("\nCalibration by decile (obs vs avg_p):")
    print(cal.to_string(index=False))


if __name__ == "__main__":
    main()
