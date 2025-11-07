#!/usr/bin/env python3
# scripts/tune_simulator_optuna.py
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
import optuna
import yaml
import pandas as pd

from eval_utils import eval_basic

TOUR = "pga"


def load_base_yaml() -> dict:
    base = yaml.safe_load(
        (Path("configs") / "simulator.yaml").read_text(encoding="utf-8")
    )
    return base


def write_yaml(d: dict, path: Path):
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


def run_simulate() -> None:
    # run simulate only; assumes features were already built via run_weekly_all or your own orchestration
    subprocess.run(["python", "scripts/simulate_event_with_course.py"], check=True)


def eval_current_event() -> float:
    # Use latest meta and preds
    meta = json.loads(
        next(
            sorted((Path("data/processed") / TOUR).glob("event_*_meta.json"))
        ).read_text(encoding="utf-8")
    )
    eid = str(meta["event_id"])
    preds_path = Path(f"data/preds/{TOUR}/event_{eid}_preds_with_course.parquet")
    results_path = Path(f"data/processed/{TOUR}/event_{eid}_results.csv")
    if not (preds_path.exists() and results_path.exists()):
        return 9999.0
    preds = pd.read_parquet(preds_path)
    res = pd.read_csv(results_path)
    key = (
        "dg_id"
        if "dg_id" in preds.columns and "dg_id" in res.columns
        else "player_name"
    )
    if key not in preds.columns or key not in res.columns:
        return 9998.0
    m = preds.merge(res[[key, "winner_flag"]], on=key, how="inner")
    if m.empty:
        return 9997.0
    met = eval_basic(m["winner_flag"], m["p_win"])
    return met["log_loss"]


def objective(trial: optuna.Trial) -> float:
    base = load_base_yaml()
    # Sample params
    base["weights"]["beta_course_hist"] = trial.suggest_float(
        "beta_course_hist", -0.6, 0.0
    )
    base["weights"]["beta_course_fit"] = trial.suggest_float(
        "beta_course_fit", -0.4, 0.0
    )
    base["shocks"]["round_sd"] = trial.suggest_float("round_sd", 0.00, 0.35)
    base["shocks"]["wave_sd"] = trial.suggest_float("wave_sd", 0.00, 0.25)
    base["sigma"]["default"] = trial.suggest_float("sigma_default", 2.2, 3.2)
    base["sigma"]["min"] = trial.suggest_float("sigma_min", 1.6, 2.2)
    base["sigma"]["max"] = trial.suggest_float("sigma_max", 3.2, 4.2)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "simulator.yaml"
        write_yaml(base, tmp)
        # Swap the active config temporarily
        real = Path("configs") / "simulator.yaml"
        backup = Path(td) / "simulator.backup.yaml"
        backup.write_bytes(real.read_bytes())
        try:
            real.write_bytes(tmp.read_bytes())
            run_simulate()
            score = eval_current_event()
        finally:
            real.write_bytes(backup.read_bytes())
    trial.set_user_attr("score", score)
    return score


def main():
    cfg = yaml.safe_load(
        (Path("configs") / "experiments.yaml").read_text(encoding="utf-8")
    )
    n_trials = int(cfg["tune"]["n_trials"])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    Path(f"data/preds/{TOUR}/tuning_best.json").write_text(
        json.dumps(
            {"best_value": study.best_value, "best_params": study.best_params}, indent=2
        ),
        encoding="utf-8",
    )
    print("Saved:", f"data/preds/{TOUR}/tuning_best.json")


if __name__ == "__main__":
    main()
