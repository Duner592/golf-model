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
  - Does not deploy GitHub Pages. Live model assets are generated in Actions and are not committed to `master`, so deploying this workflow's checked-out `web/` tree can overwrite the last good model deployment with stale files.

- `.github/workflows/deploy-pages.yml`
  - Refreshes model assets, then deploys the generated `web/` directory to GitHub Pages.
  - Runs when files under `web/` are pushed to `master`.
  - Can also be run manually from the Actions tab.
  - This avoids overwriting a scheduled model deploy with stale checked-in `web/` assets.

- `.github/workflows/weekly-model.yml`
  - Runs every 2 hours on Monday, Tuesday, and Wednesday at minute 23 UTC.
  - Can be run manually from the Actions tab.
  - Installs the Python package and dependencies from `pyproject.toml`.
  - Runs `scripts/ci_run_model.sh`, then deploys the generated `web/` directory to GitHub Pages.
  - Commits durable prediction artifacts (`web/archive/**` and `web/{tour}/initial/**`) back to `master` on full runs so archive snapshots survive later workflow checkouts. Live `web/{tour}/...` assets still stay artifact-only.
  - Scheduled runs process both tours:
    - `python scripts/run_weekly_all.py --tour pga`
    - `python scripts/build_web_assets.py --tour pga`
    - `python scripts/run_weekly_all.py --tour euro`
    - `python scripts/build_web_assets.py --tour euro`
  - In automatic `tour=both` runs, if a tour has no runnable current-week event in `upcoming-events.json`, the wrapper skips that tour's model pipeline, builds its no-event web placeholder, and continues with the other tour.
  - If any requested tour with a runnable event fails, the workflow fails before Pages deploy. Explicit single-tour and pinned-event runs remain strict. This preserves the last good Pages deployment instead of uploading a partial artifact with stale checked-in assets for a failed tour.

- `.github/workflows/archive-update.yml`
  - Runs Monday at 12:00 and 21:00 UTC.
  - Finds PGA and Euro events that started in the previous Monday-Sunday window.
  - Runs `python scripts/update_archived_event.py --event_id <event_id> --force` for each matching event.
  - If an archive entry is missing, the updater first tries to create it from `web/{tour}/initial/{year}/event_{event_id}/`, then from checked-in `web/{tour}/events/event_{event_id}/` assets.
  - Refreshes actual results with `--allow-missing` and rebuilds prediction accuracy after archive updates, so completed events appear in the accuracy dashboards as soon as DataGolf results are available.
  - Commits changed `web/archive/**` and `data/analytics/**` files back to `master` instead of deploying Pages directly. The resulting push triggers `.github/workflows/deploy-pages.yml`, which rebuilds active model assets before publishing the archive changes.

- `.github/workflows/actual-results.yml`
  - Runs once per day at 02:17 UTC.
  - Runs `python scripts/fetch_actual_results.py --year <year> --allow-missing`.
  - Defaults to the current UTC year unless a manual workflow input overrides it.
  - Commits changed `web/archive/**` results and `data/analytics/**` outputs back to `master`, even if DataGolf has not published every latest event yet. The resulting push triggers `.github/workflows/deploy-pages.yml`, which rebuilds active model assets before publishing.

## Manual Run Inputs

- `tour`: `both`, `pga`, or `euro`.
- `event_id`: optional pinned event ID. If set, choose `pga` or `euro`, not `both`.
- `fast`: skips slower course/history and HTML leaderboard work.

Scheduled model runs always export the full field leaderboard. `export_leaderboard.py --topN` remains available for ad hoc local Top-N files, but it is not exposed in the GitHub workflow.

## Site Status

The site status card is injected by `web/menu.js` at the bottom of every page that loads the shared menu. It reads `web/status.json` and checks the published `web/spreadsheet_data.csv` metadata for the betting-data timestamp.

- Model workflows update the model run timestamp, tour, and event IDs after web assets are rebuilt.
- Local `build_web_assets.py` runs also update the model run status for the tour they rebuild.
- Running `run_weekly_all.py` alone does not update the web page status until the web assets are rebuilt.
- Schedule-refresh workflows update the schedule refresh timestamp after `upcoming-events.json` is refreshed.
- Betting-data timing appears in the same status card as `Betting Data Updated`.
- Other workflows that deploy `web/` should run `scripts/update_web_status.py --sync-assets` before uploading the Pages artifact, then `scripts/guard_pages_model_assets.py` unless they rebuilt all active live model assets in that job.

## Initial Prediction Snapshots

`scripts/build_web_assets.py` keeps two versions of each active model output:

- Latest assets stay under `web/{tour}/...` and are replaced by each scheduled model run.
- The first successful prediction build for each tour/event/year is frozen under `web/{tour}/initial/{year}/event_{event_id}/`.

The frozen initial snapshot is left untouched by later scheduled runs. Prediction archives are copied from that initial snapshot, so archive accuracy remains tied to the first model view of the week rather than a later refreshed model.

Scheduled model runs and deploy-time model refreshes commit these initial snapshot and archive files back to `master`. They do not commit the refreshed live model assets under `web/{tour}/...`.

The home page links to both versions:

- `pga.html` / `euro.html` for latest model results.
- `pga.html?snapshot=initial` / `euro.html?snapshot=initial` for initial-run results.

## Previous-Week Event IDs

To see which event IDs will be archived on a Monday run:

```sh
python scripts/update_previous_week_archives.py --dry-run
```

To test a specific Monday reference date:

```sh
python scripts/update_previous_week_archives.py --date 2026-05-18 --dry-run
```

To scan more than one completed week, use a lookback:

```sh
python scripts/update_previous_week_archives.py --lookback-weeks 6 --completed-only --dry-run
```

The script reads `upcoming-events.json`, finds `pga` and `euro` events whose `start_date` falls in the selected completed Monday-Sunday window(s), prints the event IDs, and prints the exact `update_archived_event.py` command it will run.

Archive automation uses `--force` because the checked-in `upcoming-events.json` may still show a previous-week event as `upcoming`. The updater still skips forced updates when no winner is available yet. If `web/archive/{year}/{slug}/` is missing, it attempts to materialize the archive from saved initial snapshots or checked-in event assets before updating status and winner.

To update only one tour:

```sh
python scripts/update_previous_week_archives.py --tour pga --dry-run
```

## Spreadsheet Guardrail

`web/spreadsheet_data.csv` is treated as a manual input. The scheduled model wrapper hashes it before and after the run and fails the workflow if the file changes or disappears.

Manual spreadsheet updates should be committed directly to `master`. The static deploy workflow refreshes model assets before publishing so it does not replace a scheduled model deploy with stale checked-in assets.

For local edits, prefer the safe commit helper so scheduled workflow commits are rebased in before pushing:

```sh
scripts/commit_and_push.sh "updating SS"
```

The helper stages current edits, ignores `.env` and Office lock files, rebases over `origin/master`, commits, rebases once more in case automation landed during the commit, then pushes.

## First Test Run

Use the Actions tab to run `Scheduled model run` manually with:

- `tour`: `pga`
- `fast`: `true`
- `event_id`: blank

If that passes, run `euro` the same way, then try a full non-fast run.
