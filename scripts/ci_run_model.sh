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
    allow_no_event_skip="false"
    ;;
  both|"")
    if [[ -n "$EVENT_ID" ]]; then
      echo "::error::When EVENT_ID is set, set TOUR to pga or euro instead of both."
      exit 1
    fi
    tours=("pga" "euro")
    allow_no_event_skip="true"
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

run_grouped() {
  local title="$1"
  shift
  local rc
  echo "::group::$title"
  set +e
  "$@"
  rc=$?
  set -e
  echo "::endgroup::"
  return "$rc"
}

has_current_week_event() {
  local tour="$1"
  "$PYTHON_BIN" -c 'from src.utils_event import current_week_event_ids; import sys; sys.exit(0 if current_week_event_ids(sys.argv[1], require_resolved_id=True) else 1)' "$tour"
}

run_tour() {
  local tour="$1"
  local rc

  run_grouped "Run weekly pipeline for $tour" "$PYTHON_BIN" scripts/run_weekly_all.py --tour "$tour" "${run_args[@]}"
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi

  run_grouped "Build web assets for $tour" "$PYTHON_BIN" scripts/build_web_assets.py --tour "$tour" "${web_args[@]}"
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi

  "$PYTHON_BIN" scripts/update_web_status.py --model-run --tour "$tour" --workflow "scheduled-model"
}

tour_successes=()
tour_skips=()
tour_failures=()

for tour in "${tours[@]}"; do
  if [[ -z "$EVENT_ID" && "$allow_no_event_skip" == "true" ]] && ! has_current_week_event "$tour"; then
    echo "::notice::No current-week event found for $tour in upcoming-events.json. Generating no-event web assets and skipping the model pipeline for this tour."
    if run_grouped "Build no-event web assets for $tour" "$PYTHON_BIN" scripts/build_web_assets.py --tour "$tour"; then
      "$PYTHON_BIN" scripts/update_web_status.py --model-run --tour "$tour" --workflow "scheduled-model"
      tour_skips+=("$tour")
      continue
    else
      rc=$?
      tour_failures+=("$tour")
      echo "::error::No-event web asset build failed for $tour with exit code $rc."
      continue
    fi
  fi

  if run_tour "$tour"; then
    tour_successes+=("$tour")
  else
    rc=$?
    tour_failures+=("$tour")
    echo "::error::Model refresh failed for $tour with exit code $rc. Refusing to deploy a partial Pages artifact."
  fi
done

if [[ "${#tour_successes[@]}" -eq 0 && "${#tour_skips[@]}" -eq 0 ]]; then
  echo "::error::No tour model or no-event assets were built successfully."
  exit 1
fi

if [[ "${#tour_failures[@]}" -gt 0 ]]; then
  echo "::error::Partial model refresh. Successful tours: ${tour_successes[*]}; failed tours: ${tour_failures[*]}"
  echo "::error::Generated model assets are not committed to master, so deploying now could overwrite the last good Pages site with stale checked-in assets."
  exit 1
fi

if [[ "${#tour_skips[@]}" -gt 0 ]]; then
  echo "::notice::Skipped model pipeline for tours with no current-week event: ${tour_skips[*]}"
fi

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
