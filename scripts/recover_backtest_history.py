#!/usr/bin/env python3
"""Offline, reproducible backtest recovery from Git; never reruns predictions.

Default is a dry run. --apply writes separate historical snapshots and identity
sidecars, preserving all existing prediction files and their provenance.
"""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.player_integrity import validate_unique_players


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    return result.stdout if result.returncode == 0 else None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity_payload(raw, event, source):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows or any(str(row.get("event_id")) != str(event["event_id"]) for row in rows):
        return None
    return {"event_id":str(event["event_id"]), "tour":event["tour"], "year":str(event["year"]),
            "source":source, "source_sha256":digest(raw),
            "players":[{"player_name":r["player_name"], "player_id":r["player_id"]} for r in rows if r.get("player_name") and r.get("player_id")]}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def recover(event):
    if event.get("reconstruction") or event.get("prediction_snapshot") == "initial" or "zurich classic" in event["event_name"].lower():
        return None
    base = f"web/archive/{event['year']}/{event['slug']}"
    path = f"{base}/leaderboard.json"
    cutoff = datetime.fromisoformat(event["start_date"]).replace(tzinfo=timezone.utc)
    history = git("log", "--format=%H %cI", "--before=" + cutoff.isoformat(), "--", path)
    for line in (history or b"").decode().splitlines():
        commit, stamp = line.split(" ", 1)
        if datetime.fromisoformat(stamp) >= cutoff:
            continue
        raw = git("show", f"{commit}:{path}")
        index_raw = git("show", f"{commit}:web/archive/index.json")
        summary_raw = git("show", f"{commit}:{base}/tournament_summary.json")
        if not raw or not index_raw or not summary_raw:
            continue
        index, summary, rows = json.loads(index_raw), json.loads(summary_raw), json.loads(raw)
        matching = [e for e in index if all(str(e.get(k)) == str(event[k]) for k in ("event_id", "tour", "year", "slug"))]
        try:
            date = datetime.strptime(summary.get("start_date", ""), "%d-%b-%Y").date().isoformat()
        except ValueError:
            date = summary.get("start_date")
        if len(matching) != 1 or date != event["start_date"] or not isinstance(rows, list) or not rows:
            continue
        try:
            validate_unique_players(rows, context=f"Historical {event['slug']}")
        except ValueError:
            continue
        identity = None
        for field_path in (f"{base}/field_teetimes.csv", f"web/{event['tour']}/field_teetimes.csv"):
            field = git("show", f"{commit}:{field_path}")
            if field:
                identity = identity_payload(field, event, f"git:{commit}:{field_path}")
                if identity:
                    break
        descriptor = {"snapshot_type":"historical", "event_id":str(event["event_id"]), "tour":event["tour"], "year":str(event["year"]),
                      "start_date":event["start_date"], "commit":commit, "committed_utc":datetime.fromisoformat(stamp).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                      "source_path":path, "leaderboard_sha256":digest(raw), "summary_sha256":digest(summary_raw),
                      "evidence":"Prediction file present in a pre-event Git commit; not a reconstructed model run."}
        return descriptor, raw, summary_raw, identity
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    index_path = ROOT / "web/archive/index.json"
    entries = json.loads(index_path.read_text())
    recovered = identities = 0
    for event in entries:
        directory = ROOT / "web/archive" / str(event["year"]) / event["slug"]
        field = directory / "field_teetimes.csv"
        if field.exists():
            payload = identity_payload(field.read_bytes(), event, str(field.relative_to(ROOT)))
            if payload:
                identities += 1
                if args.apply:
                    dump(directory / "player_ids.json", payload)
                    event["player_ids_available"] = True
                    event["player_ids_sha256"] = digest((directory / "player_ids.json").read_bytes())
        found = recover(event)
        if found:
            descriptor, raw, summary, identity = found
            recovered += 1
            print(f"{event['event_name']}: {descriptor['committed_utc']} {descriptor['commit'][:12]}")
            if args.apply:
                target = directory / "historical"
                target.mkdir(parents=True, exist_ok=True)
                (target / "leaderboard.json").write_bytes(raw)
                (target / "tournament_summary.json").write_bytes(summary)
                if identity:
                    dump(target / "player_ids.json", identity)
                    descriptor["player_ids_sha256"] = digest((target / "player_ids.json").read_bytes())
                descriptor["player_ids_available"] = bool(identity)
                dump(target / "snapshot.json", descriptor)
                event["historical_snapshot"] = descriptor
    if args.apply:
        dump(index_path, entries)
    print(f"{'Applied' if args.apply else 'Dry run'}: {recovered} Git-recovered snapshots; {identities} archived identity sidecars.")


if __name__ == "__main__":
    main()
