#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd

TOUR = "pga"
meta = json.loads(next(sorted((Path("data/processed") / TOUR).glob("event_*_meta.json"))).read_text())
eid = str(meta["event_id"])
pp = Path(f"data/preds/{TOUR}/event_{eid}_preds_with_course.parquet")
if not pp.exists():
    print("[warn] predictions parquet not found")
    sys.exit(0)

df = pd.read_parquet(pp)
s = float(df["p_win"].sum())
m = float(df["p_win"].max())
issues = []
if not (0.98 <= s <= 1.02):
    issues.append(f"sum(p_win)={s:.4f} out of [0.98,1.02]")
if m > 0.20:
    issues.append(f"max p_win={m:.4f} > 0.20")
if issues:
    raise SystemExit("[fail] smoke test: " + "; ".join(issues))
print("Smoke OK:", {"sum": round(s, 4), "max": round(m, 4), "rows": len(df)})
