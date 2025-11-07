#!/usr/bin/env python3
# scripts/backtest_runner.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from eval_utils import eval_basic

TOUR = "pga"


def load_cfg() -> dict:
    cfg = yaml.safe_load((Path("configs") / "experiments.yaml").read_text(encoding="utf-8"))
    return cfg


def list_backtest_events(years: list[int], max_events_per_year: int) -> list[dict]:
    # Use your schedule snapshots or curated list; here we rely on existing historical rounds present
    root = Path("data/raw/historical") / TOUR
    events = []
    for p in root.glob("event_*_*_rounds.json"):
        eid = p.name.split("_")[1]
        yr = int(p.name.split("_")[2])
        if years and yr not in years:
            continue
        events.append({"event_id": eid, "year": yr})
    # group by year/event, cap per year
    df = pd.DataFrame(events).drop_duplicates().sort_values(["year", "event_id"])
    out = []
    for _yr, g in df.groupby("year"):
        g_sorted = g.head(max_events_per_year) if max_events_per_year else g
        out.extend(g_sorted.to_dict(orient="records"))
    return out


def load_preds(event_id: str) -> pd.DataFrame | None:
    p = Path(f"data/preds/{TOUR}/event_{event_id}_preds_with_course.parquet")
    if p.exists():
        return pd.read_parquet(p)
    p2 = Path(f"data/preds/{TOUR}/event_{event_id}_preds_common_shock.parquet")
    if p2.exists():
        return pd.read_parquet(p2)
    p3 = Path(f"data/preds/{TOUR}/event_{event_id}_preds_baseline.parquet")
    if p3.exists():
        return pd.read_parquet(p3)
    return None


def load_results(event_id: str) -> pd.DataFrame:
    p = Path(f"data/processed/{TOUR}/event_{event_id}_results.csv")
    return pd.read_csv(p)


def align(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    key = "dg_id" if "dg_id" in preds.columns and "dg_id" in results.columns else "player_name"
    if key not in preds.columns or key not in results.columns:
        # fallback: try rename
        if key == "dg_id" and "player_id" in results.columns:
            results = results.rename(columns={"player_id": "dg_id"})
        elif key == "player_name" and "player_name" not in results.columns:
            raise ValueError("No join key found for results")
    m = preds.merge(results[[key, "winner_flag"]], on=key, how="inner")
    if "p_win" not in m.columns:
        raise ValueError("Preds missing p_win")
    return m


def main():
    cfg = load_cfg()
    years = cfg["backtest"]["years"]
    max_per_year = int(cfg["backtest"]["max_events_per_year"])
    events = list_backtest_events(years, max_per_year)
    rows = []
    by_event = []
    for ev in events:
        eid = str(ev["event_id"])
        # assume you pinned & ran your pipeline for this event beforehand, or adjust to call scripts
        preds = load_preds(eid)
        if preds is None:
            continue
        try:
            res = load_results(eid)
        except FileNotFoundError:
            continue
        merged = align(preds, res)
        mb = eval_basic(merged["winner_flag"], merged["p_win"])
        mb["event_id"] = eid
        mb["year"] = ev["year"]
        by_event.append(mb)
        rows.append(mb)
    if not rows:
        print("No events evaluated.")
        return
    df = pd.DataFrame(rows)
    agg = df[["log_loss", "brier", "p_sum", "p_max", "p_median"]].mean().to_dict()
    print("Backtest aggregated:")
    print(agg)
    Path("data/preds") / TOUR
    out_dir = Path("data/preds") / TOUR
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "backtest_summary.json").write_text(json.dumps({"aggregate": agg, "by_event": by_event}, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "backtest_summary.json")


if __name__ == "__main__":
    main()
