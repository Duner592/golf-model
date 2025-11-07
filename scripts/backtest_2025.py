#!/usr/bin/env python3
# scripts/backtest_2025.py
#
# Robust 2025 backtest (with exclusions):
# - Reads 2025 schedule snapshot and filters to completed events that have rounds JSON available
# - Excludes event_ids from configs/experiments.yaml (backtest.exclude_event_ids) and/or --exclude CLI
# - Writes meta directly from schedule (avoids name-matching drift)
# - Runs pinned weekly pipeline with --event_id and --skip-course for speed/stability
# - Evaluates predictions with --event_id and aggregates results
#
# Usage:
#   python scripts/backtest_2025.py
#   python scripts/backtest_2025.py --exclude 60,27
#
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

YEAR = 2025
TOUR = "pga"

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HIST = ROOT / "data" / "raw" / "historical" / TOUR
PROCESSED = ROOT / "data" / "processed" / TOUR
PREDS = ROOT / "data" / "preds" / TOUR


def run(cmd: list[str], timeout: int | None = None) -> bool:
    print(">>>", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        return True
    except subprocess.CalledProcessError as e:
        print("[warn] Command failed:", e, flush=True)
        return False
    except subprocess.TimeoutExpired:
        print("[warn] Command timed out:", " ".join(cmd), flush=True)
        return False


def schedule_2025() -> list[dict]:
    p = RAW / f"schedule_{TOUR}_{YEAR}.json"
    if not p.exists():
        print(f"[error] Missing schedule snapshot: {p}. Run fetch_schedule_and_rounds_2025.py first.")
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as ex:
        print("[error] Could not read schedule JSON:", ex)
        return []


def is_completed(start: str) -> bool:
    try:
        r1 = datetime.strptime(start, "%Y-%m-%d").date()
        r4 = r1 + timedelta(days=3)
        return r4 < datetime.utcnow().date()
    except Exception:
        return False


def has_rounds(eid: str) -> bool:
    return (HIST / f"event_{eid}_{YEAR}_rounds.json").exists()


def load_exclusions() -> set[str]:
    cfg_p = ROOT / "configs" / "experiments.yaml"
    if not cfg_p.exists():
        return set()
    try:
        import yaml  # local import to avoid hard dependency elsewhere

        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
        ids = cfg.get("backtest", {}).get("exclude_event_ids", []) or []
        return {str(x) for x in ids}
    except Exception:
        return set()


def write_meta_from_schedule(e: dict) -> Path | None:
    """
    Create processed meta for the event (event_id, event_name, lat, lon, R1..R4 dates).
    """
    try:
        eid = str(e.get("event_id"))
        ename = e.get("event_name") or e.get("name") or f"event_{eid}"
        lat = e.get("latitude") or e.get("lat") or e.get("course_lat")
        lon = e.get("longitude") or e.get("lon") or e.get("course_lon")
        start = e.get("start") or e.get("start_date")
        if not (eid and start and lat is not None and lon is not None):
            print(
                "[skip] meta fields missing:",
                {"event_id": eid, "start": start, "lat": lat, "lon": lon},
            )
            return None
        r1 = datetime.strptime(start, "%Y-%m-%d").date()
        dates = {
            "r1_date": r1.isoformat(),
            "r2_date": (r1 + timedelta(days=1)).isoformat(),
            "r3_date": (r1 + timedelta(days=2)).isoformat(),
            "r4_date": (r1 + timedelta(days=3)).isoformat(),
        }
        meta = {
            "event_id": int(eid),
            "event_name": ename,
            "lat": float(lat),
            "lon": float(lon),
            **dates,
            "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": f"schedule pin ({YEAR})",
        }
        out = PROCESSED / f"event_{eid}_meta.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("Pinned event meta:", out)
        return out
    except Exception as ex:
        print("[skip] failed to write meta:", ex)
        return None


def build_field_and_results(eid: str) -> bool:
    rounds_json = HIST / f"event_{eid}_{YEAR}_rounds.json"
    if not rounds_json.exists():
        print("[skip] rounds JSON missing:", rounds_json)
        return False
    return run(
        [
            "python",
            "scripts/build_field_and_results_from_hist.py",
            "--year",
            str(YEAR),
            "--event_id",
            eid,
        ]
    )


def weather_for_event(eid: str) -> bool:
    if not run(["python", "scripts/fetch_weather_from_schedule.py", "--event_id", eid]):
        return False
    if not run(["python", "scripts/summarize_weather_from_schedule.py", "--event_id", eid]):
        return False
    return True


def weekly_pipeline(eid: str) -> bool:
    # pinned, event-specific run; skip course for speed/stability
    return run(
        [
            "python",
            "scripts/run_weekly_all.py",
            "--pinned",
            "--event_id",
            eid,
            "--skip-course",
            "--skip-html",
        ],
        timeout=900,
    )


def evaluate(eid: str) -> bool:
    # evaluate for this event explicitly (hardened evaluate uses --event_id)
    return run(["python", "scripts/evaluate_preds.py", "--event_id", eid])


def main():
    ap = argparse.ArgumentParser(description="Backtest 2025 completed events with exclusions.")
    ap.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated event_ids to exclude (overrides configs/experiments.yaml)",
    )
    args = ap.parse_args()

    exclude = load_exclusions()
    if args.exclude:
        extra = {x.strip() for x in args.exclude.split(",") if x.strip()}
        exclude |= extra

    events = schedule_2025()
    if not events:
        return

    # Filter to completed events with rounds JSON present and not excluded
    todo = []
    for e in events:
        start = e.get("start") or e.get("start_date")
        eid = str(e.get("event_id"))
        if not eid or not start:
            continue
        if eid in exclude:
            print(f"[info] Skipping excluded event_id={eid}")
            continue
        if not is_completed(start):
            continue
        if not has_rounds(eid):
            print(f"[skip] no rounds JSON for event_id={eid}")
            continue
        todo.append(e)

    print(f"[info] Backtesting {len(todo)} completed events from {YEAR} (excluded: {sorted(exclude)})")

    recap = []
    for i, e in enumerate(todo, 1):
        eid = str(e.get("event_id"))
        name = e.get("event_name") or e.get("name") or f"event_{eid}"
        start = e.get("start") or e.get("start_date")
        print(f"=== [{i}/{len(todo)}] event_id={eid} • {name} • start={start} ===")

        if not write_meta_from_schedule(e):
            continue

        if not build_field_and_results(eid):
            continue

        if not weather_for_event(eid):
            continue

        if not weekly_pipeline(eid):
            continue

        if not evaluate(eid):
            print("[warn] evaluation failed; missing winner_flag?")
            continue

        eval_path = PREDS / f"event_{eid}_eval_summary.json"
        if eval_path.exists():
            try:
                rj = json.loads(eval_path.read_text(encoding="utf-8"))
                recap.append(
                    {
                        "event_id": eid,
                        "log_loss": rj.get("log_loss"),
                        "brier": rj.get("brier"),
                    }
                )
            except Exception as ex:
                print("[warn] could not read eval:", ex)

    out = PREDS / f"backtest_{YEAR}_recap.json"
    out.write_text(json.dumps({"year": YEAR, "events": recap}, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
