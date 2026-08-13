"""Small, dependency-free provenance helpers for frozen model artifacts."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot_provenance(root: Path, snapshot_dir: Path) -> dict[str, object]:
    """Capture hashes needed to identify the code, config, and frozen artifacts."""
    code_paths = [
        root / "scripts" / "simulate_event_with_course.py",
        root / "scripts" / "build_course_fit_from_history.py",
        root / "scripts" / "build_web_assets.py",
        root / "src" / "utils_event.py",
    ]
    config_paths = [root / "pyproject.toml", root / "configs" / "datagolf.yaml", root / "configs" / "event_rules.yaml"]
    artifact_hashes = {
        path.name: digest
        for path in sorted(snapshot_dir.iterdir())
        if (digest := sha256_file(path)) is not None
    }
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "python_version": sys.version.split()[0],
        "code_sha256": {str(path.relative_to(root)): digest for path in code_paths if (digest := sha256_file(path)) is not None},
        "config_sha256": {str(path.relative_to(root)): digest for path in config_paths if (digest := sha256_file(path)) is not None},
        "artifact_sha256": artifact_hashes,
    }
