#!/usr/bin/env python3
# scripts/project_snapshot.py
#
# Create a clean, shareable snapshot of your project for review:
# - project tree (sizes), env info, redacted configs, latest artifacts summary
# - optional small archive (excludes large raw data and secrets)
#
# Usage:
#   python scripts/project_snapshot.py

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "snapshot"
SNAP.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "snapshot",
    "dg_cache",
    ".idea",
    ".vscode",
}
EXCLUDE_TOP = {"data/raw"}  # exclude heavy raw by default


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return res.stdout.strip()


def human_size(p: Path) -> str:
    try:
        sz = p.stat().st_size
    except Exception:
        return "-"
    for unit in ["B", "KB", "MB", "GB"]:
        if sz < 1024:
            return f"{sz:.0f}{unit}"
        sz /= 1024
    return f"{sz:.1f}TB"


def build_tree(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        parts = rel.parts
        if any(part in EXCLUDE_DIRS for part in parts):
            continue
        if len(parts) > 0 and str(parts[0]) in EXCLUDE_TOP:
            continue
        indent = "  " * (len(parts) - 1)
        name = parts[-1]
        if path.is_dir():
            lines.append(f"{indent}📁 {name}/")
        else:
            lines.append(f"{indent}📄 {name}  ({human_size(path)})")
    return "\n".join(lines)


def redacted_yaml_text(text: str) -> str:
    # simple redaction of likely keys
    text = re.sub(r"(env_var:\s*)([A-Z0-9_]+)", r"\1DATAGOLF_API_KEY", text)
    text = re.sub(r"(key:\s*)([a-zA-Z0-9]+)", r"\1***REDACTED***", text)
    return text


def redact_configs():
    out_dir = SNAP / "configs_redacted"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = ROOT / "configs"
    for p in cfg_dir.glob("*.yaml"):
        t = p.read_text(encoding="utf-8")
        (out_dir / p.name).write_text(redacted_yaml_text(t), encoding="utf-8")
    print(f"Saved redacted configs -> {out_dir}")


def write_env_info():
    info = []
    info.append(run([sys.executable, "-V"]))
    info.append(run([sys.executable, "-c", "import platform; print(platform.platform())"]))
    info.append("\n# pip freeze\n" + run([sys.executable, "-m", "pip", "freeze"]))
    (SNAP / "env_info.txt").write_text("\n".join(info), encoding="utf-8")
    print("Saved env_info.txt")


def latest_meta(tour: str = "pga") -> Path | None:
    metas = sorted((ROOT / "data" / "processed" / tour).glob("event_*_meta.json"))
    return metas[-1] if metas else None


def collect_artifacts() -> dict[str, Any]:
    tours = ["pga"]
    out: dict[str, Any] = {}
    for tour in tours:
        tdir = ROOT / "data" / "processed" / tour
        pdir = ROOT / "data" / "preds" / tour
        fdir = ROOT / "data" / "features" / tour
        rec: dict[str, Any] = {}
        if tdir.exists():
            m = latest_meta(tour)
            if m:
                meta = json.loads(m.read_text(encoding="utf-8"))
                eid = str(meta.get("event_id"))
                rec["event_id"] = eid
                rec["event_name"] = meta.get("event_name")
                rec["meta_path"] = str(m)
                # features/preds presence
                rec["features_weather"] = str(
                    next(
                        (fdir / f"event_{eid}_features_weather.parquet").glob("**/*"),
                        fdir / f"event_{eid}_features_weather.parquet",
                    )
                )
                rec["features_course"] = str(fdir / f"event_{eid}_features_course.parquet")
                # preds
                preds = {}
                for stem in ["with_course", "common_shock", "baseline"]:
                    p = pdir / f"event_{eid}_preds_{stem}.parquet"
                    preds[stem] = {"exists": p.exists(), "path": str(p)}
                rec["preds"] = preds
        out[tour] = rec
    return out


def write_tree():
    tree = build_tree(ROOT)
    (SNAP / "project_tree.txt").write_text(tree, encoding="utf-8")
    print("Saved project_tree.txt")


def write_artifacts():
    arts = collect_artifacts()
    (SNAP / "artifacts.json").write_text(json.dumps(arts, indent=2), encoding="utf-8")
    print("Saved artifacts.json")


def make_archive():
    # pack selected dirs, exclude raw data and caches
    archive = SNAP / "project_snapshot.tar.gz"
    include = [
        "scripts",
        "src",
        "configs",
        "snapshot/project_tree.txt",
        "snapshot/env_info.txt",
        "snapshot/configs_redacted",
        "snapshot/artifacts.json",
        "README.md",
        "requirements.txt",
    ]
    include_paths = [str(ROOT / p) for p in include if (ROOT / p).exists()]
    cmd = ["tar", "-czf", str(archive)] + include_paths
    try:
        run(cmd)
        print("Saved archive:", archive)
    except Exception as e:
        print("[warn] Could not create tar archive:", e)


def main():
    write_tree()
    write_env_info()
    redact_configs()
    write_artifacts()
    make_archive()
    print("\nSnapshot complete. Share files under ./snapshot/ or the tarball for review.")


if __name__ == "__main__":
    main()
