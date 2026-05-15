#!/usr/bin/env bash
set -euo pipefail

YEAR="${YEAR:-$(date -u +%Y)}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "::error::Neither python nor python3 is available on PATH."
    exit 1
  fi
fi

if [[ -z "${DATAGOLF_API_KEY:-}" ]]; then
  echo "::error::DATAGOLF_API_KEY is not set. Add it as a GitHub Actions repository secret."
  exit 1
fi

"$PYTHON_BIN" scripts/update_upcoming_events.py

echo "::group::Fetch actual results for $YEAR"
"$PYTHON_BIN" scripts/fetch_actual_results.py --year "$YEAR"
echo "::endgroup::"

echo "::group::Build prediction accuracy for $YEAR"
"$PYTHON_BIN" scripts/build_prediction_accuracy.py --year "$YEAR"
echo "::endgroup::"

"$PYTHON_BIN" scripts/update_web_status.py --schedule-refreshed --sync-assets --workflow "actual-results"
