# Automation

This repo is set up to run the weekly model and publish the static `web/` site with GitHub Actions.

## Required GitHub Setup

1. In GitHub, add a repository secret named `DATAGOLF_API_KEY`.
2. In GitHub Pages settings, set the build/deployment source to GitHub Actions.
3. Keep manual betting/spreadsheet edits in `web/spreadsheet_data.csv` committed to `master`.

## Workflows

- `.github/workflows/refresh-upcoming-events.yml`
  - Runs hourly at minute 7 UTC.
  - Runs `python scripts/update_upcoming_events.py`.
  - Commits `upcoming-events.json` back to `master` only when the file changes.
  - Defaults to refreshing PGA and Euro from DataGolf while preserving existing non-refreshed tours in the file.

- `.github/workflows/deploy-pages.yml`
  - Deploys the current `web/` directory to GitHub Pages.
  - Runs when files under `web/` are pushed to `master`.
  - Can also be run manually from the Actions tab.

- `.github/workflows/weekly-model.yml`
  - Runs every 2 hours on Monday, Tuesday, and Wednesday at minute 23 UTC.
  - Can be run manually from the Actions tab.
  - Installs the Python package and dependencies from `pyproject.toml`.
  - Runs `scripts/ci_run_model.sh`, then deploys the generated `web/` directory to GitHub Pages.
  - Scheduled runs process both tours:
    - `python scripts/run_weekly_all.py --tour pga`
    - `python scripts/build_web_assets.py --tour pga`
    - `python scripts/run_weekly_all.py --tour euro`
    - `python scripts/build_web_assets.py --tour euro`

- `.github/workflows/archive-update.yml`
  - Runs Monday at 12:00 and 21:00 UTC.
  - Finds PGA and Euro events that started in the previous Monday-Sunday window.
  - Runs `python scripts/update_archived_event.py --event_id <event_id> --force` for each matching event.

- `.github/workflows/actual-results.yml`
  - Runs once per day at 02:17 UTC.
  - Runs `python scripts/fetch_actual_results.py --year <year>`.
  - Defaults to the current UTC year unless a manual workflow input overrides it.

## Manual Run Inputs

- `tour`: `both`, `pga`, or `euro`.
- `event_id`: optional pinned event ID. If set, choose `pga` or `euro`, not `both`.
- `fast`: skips slower course/history and HTML leaderboard work.
- `top_n`: leaderboard size for `export_leaderboard.py`.

## Previous-Week Event IDs

To see which event IDs will be archived on a Monday run:

```sh
python scripts/update_previous_week_archives.py --dry-run
```

To test a specific Monday reference date:

```sh
python scripts/update_previous_week_archives.py --date 2026-05-18 --dry-run
```

The script reads `upcoming-events.json`, finds `pga` and `euro` events whose `start_date` falls in the previous Monday-Sunday window, prints the event IDs, and prints the exact `update_archived_event.py` command it will run.

Archive automation uses `--force` because the checked-in `upcoming-events.json` may still show a previous-week event as `upcoming`. The updater still skips forced updates when no winner is available yet.

To update only one tour:

```sh
python scripts/update_previous_week_archives.py --tour pga --dry-run
```

## Spreadsheet Guardrail

`web/spreadsheet_data.csv` is treated as a manual input. The scheduled model wrapper hashes it before and after the run and fails the workflow if the file changes or disappears.

Manual spreadsheet updates should be committed directly to `master`. The static deploy workflow will publish the updated `web/` folder without running the model pipeline.

## First Test Run

Use the Actions tab to run `Scheduled model run` manually with:

- `tour`: `pga`
- `fast`: `true`
- `event_id`: blank

If that passes, run `euro` the same way, then try a full non-fast run.
