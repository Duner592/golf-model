#!/usr/bin/env bash
set -euo pipefail

TOUR="${TOUR:-both}"
EVENT_ID="${EVENT_ID:-}"
FAST="${FAST:-false}"
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

checksum_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

if [[ -z "${DATAGOLF_API_KEY:-}" ]]; then
  echo "::error::DATAGOLF_API_KEY is not set. Add it as a GitHub Actions repository secret."
  exit 1
fi

case "$TOUR" in
  pga|euro)
    tours=("$TOUR")
    ;;
  both|"")
    if [[ -n "$EVENT_ID" ]]; then
      echo "::error::When EVENT_ID is set, set TOUR to pga or euro instead of both."
      exit 1
    fi
    tours=("pga" "euro")
    ;;
  *)
    echo "::error::Unsupported TOUR '$TOUR'. Expected pga, euro, or both."
    exit 1
    ;;
esac

"$PYTHON_BIN" scripts/update_upcoming_events.py
"$PYTHON_BIN" scripts/update_web_status.py --schedule-refreshed --sync-assets --workflow "scheduled-model"

spreadsheet_checksum=""
if [[ -f web/spreadsheet_data.csv ]]; then
  read -r spreadsheet_checksum _ < <(checksum_file web/spreadsheet_data.csv)
fi

run_args=()
web_args=()

if [[ -n "$EVENT_ID" ]]; then
  run_args+=(--event_id "$EVENT_ID")
  web_args+=(--event_id "$EVENT_ID")
fi

if [[ "$FAST" == "true" ]]; then
  run_args+=(--fast)
fi

for tour in "${tours[@]}"; do
  echo "::group::Run weekly pipeline for $tour"
  "$PYTHON_BIN" scripts/run_weekly_all.py --tour "$tour" "${run_args[@]}"
  echo "::endgroup::"

  echo "::group::Build web assets for $tour"
  "$PYTHON_BIN" scripts/build_web_assets.py --tour "$tour" "${web_args[@]}"
  echo "::endgroup::"

  "$PYTHON_BIN" scripts/update_web_status.py --model-run --tour "$tour" --workflow "scheduled-model"
done

if [[ -n "$spreadsheet_checksum" ]]; then
  if [[ ! -f web/spreadsheet_data.csv ]]; then
    echo "::error::web/spreadsheet_data.csv was removed during automation. Keep it as a manual input."
    exit 1
  fi
  read -r after_checksum _ < <(checksum_file web/spreadsheet_data.csv)
  if [[ "$after_checksum" != "$spreadsheet_checksum" ]]; then
    echo "::error::web/spreadsheet_data.csv changed during automation. Keep it as a manual input."
    exit 1
  fi
fi
