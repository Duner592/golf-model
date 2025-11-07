#!/usr/bin/env python3
# scripts/eval_utils.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def eval_basic(y_true: pd.Series, p: pd.Series) -> dict:
    y = y_true.astype(int).to_numpy()
    ps = np.clip(p.to_numpy().astype(float), 1e-12, 1 - 1e-12)
    return {
        "log_loss": float(log_loss(y, ps)),
        "brier": float(brier_score_loss(y, ps)),
        "p_sum": float(ps.sum()),
        "p_max": float(ps.max()),
        "p_median": float(np.median(ps)),
        "n": int(len(y)),
    }


def calibration_table(y_true: pd.Series, p: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "p": p.astype(float)})
    cuts = np.quantile(df["p"], np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = 0.0, 1.0
    idx = np.digitize(df["p"], cuts[1:-1], right=False)
    grp = df.groupby(idx)
    out = grp.agg(obs_rate=("y", "mean"), avg_p=("p", "mean"), count=("p", "size")).reset_index().rename(columns={"index": "bin"})
    out["abs_gap"] = np.abs(out["obs_rate"] - out["avg_p"])
    return out
