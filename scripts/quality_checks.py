#!/usr/bin/env python3
# scripts/quality_checks.py
from __future__ import annotations

import pandas as pd


def check_features(df: pd.DataFrame) -> list[str]:
    issues = []
    if df.empty:
        issues.append("Features table is empty")
        return issues
    if df["player_name"].isna().any():
        issues.append("Missing player_name values")
    if "p_win" in df.columns and (df["p_win"] < 0).any():
        issues.append("Negative probabilities in p_win (should not happen)")
    for col in ["r1_teetime", "r2_teetime"]:
        if col in df.columns and df[col].astype(str).str.contains(":").mean() < 0.5:
            issues.append(f"Low fraction of valid times in {col}")
    return issues


def check_weather(df_neutral: pd.DataFrame) -> list[str]:
    issues = []
    needed = {"round", "wind_mph", "gust_mph", "precip_pct"}
    if not needed.issubset(df_neutral.columns):
        issues.append(f"Neutral weather missing cols: {needed - set(df_neutral.columns)}")
        return issues
    if (df_neutral["wind_mph"] > 70).any():
        issues.append("Implausibly high wind_mph (>70)")
    if (df_neutral["precip_pct"] > 100).any() or (df_neutral["precip_pct"] < 0).any():
        issues.append("precip_pct outside 0..100")
    return issues
