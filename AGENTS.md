# Codex Project Instructions

This repository is a weekly golf tournament win-probability model for PGA and DP World Tour events. It combines DataGolf inputs, weather, course features, simulation outputs, betting/ROI tracking, and static web assets.

## Working Principles

- Treat this repo as the durable project context. Update `.codex/PROJECT_CONTEXT.md` when decisions, assumptions, model behavior, data contracts, or operational workflows change.
- Prefer the existing pipeline scripts and data layout over adding new entry points unless there is a clear reason.
- Keep generated data and hand-authored source changes separate. If a task requires refreshing artifacts, use the pipeline scripts rather than manually editing generated outputs.
- Do not commit or expose secrets. `DATAGOLF_API_KEY` belongs in `.env` only.
- Preserve user-created changes in the worktree. Do not reset, checkout, or overwrite unrelated files.

## Common Commands

- Run weekly pipeline: `python scripts/run_weekly_all.py --tour $TOUR`
- Refresh upcoming events: `python scripts/update_upcoming_events.py`
- Run CI-style model wrapper: `TOUR=pga FAST=true DATAGOLF_API_KEY=... bash scripts/ci_run_model.sh`
- List previous-week archive event IDs: `python scripts/update_previous_week_archives.py --dry-run`
- Fetch actual results in CI style: `YEAR=2026 DATAGOLF_API_KEY=... bash scripts/ci_fetch_actual_results.sh`
- Update archived event: `python scripts/update_archived_event.py --event_id $EVENT_ID`
- Fetch actual results: `python scripts/fetch_actual_results.py --year $YEAR`
- Build web assets: `python scripts/build_web_assets.py --tour $TOUR`
- Summarize artifacts/status: `python scripts/summarize_status.py`
- Test local web output: `cd web/ && python -m http.server 8000`

## Automation

- GitHub Actions workflow files must use `.yml` or `.yaml`.
- `.github/workflows/deploy-pages.yml` deploys `web/` to GitHub Pages on manual runs and pushes that touch `web/**`.
- `.github/workflows/refresh-upcoming-events.yml` runs hourly at minute 7 UTC, refreshes `upcoming-events.json` from DataGolf, and commits only that file when it changes.
- `.github/workflows/weekly-model.yml` runs every 2 hours Monday-Wednesday at minute 23 UTC, then deploys `web/` as a Pages artifact.
- Scheduled model runs always export full-field leaderboards. Do not add a leaderboard-size input to the workflow.
- `.github/workflows/archive-update.yml` runs Monday at 12:00 and 21:00 UTC, resolves previous-week PGA/Euro event IDs, updates archived summaries, then deploys `web/`.
- `.github/workflows/actual-results.yml` runs daily at 02:17 UTC, refreshes actual results for the selected/current year, rebuilds prediction accuracy, then deploys `web/`.
- `web/spreadsheet_data.csv` is a manual source-of-truth input. Do not overwrite it from automation.

## Project Layout

- `scripts/`: pipeline, archive, export, and utility scripts.
- `src/`: shared Python helpers.
- `configs/`: configuration such as DataGolf endpoint/default settings.
- `data/processed/`: normalized fields, weather, course metadata, and related intermediate inputs.
- `data/features/`: feature tables.
- `data/preds/`: predictions, leaderboards, and summaries.
- `data/analytics/`: prediction-vs-actual and accuracy tracking.
- `web/`: generated/static site and archive output.
- `GolfBetting.xlsx` and spreadsheet-derived CSV data: betting and ROI tracking inputs.

## Validation Expectations

- For pipeline changes, run the narrowest relevant script first, then broaden to the weekly pipeline if the change affects shared behavior.
- For web changes, serve `web/` locally and inspect the resulting page when practical.
- For data changes, verify row counts, key columns, event IDs, tour names, and date/year assumptions explicitly.

## Domain Notes

- Weather is sourced from schedule/location data via Open-Meteo.
- Course fit is DIY/historical-regression based.
- Simulation uses common shock, player sigma, and wave-aware weather components.
- Actual results come from DataGolf historical raw rounds and populate `web/archive/<year>/<slug>/results.json`.
