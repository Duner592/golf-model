# Golf Model Project Context

Last updated: 2026-07-08

## Purpose

Build and maintain a weekly golf tournament win-probability model for PGA and DP World Tour events, with generated leaderboards, archive pages, actual-result tracking, and betting ROI support.

Hosted site: https://duner592.github.io/golf-model/

## Current Operating Model

- Data source: DataGolf, with `DATAGOLF_API_KEY` stored in `.env`.
- Weather source: Open-Meteo via event schedule/location metadata.
- Course fit: internal DIY method using historical regression/course history.
- Simulation: common-shock simulator with player sigma and wave-aware weather adjustments.
- Output: predictions and summaries in `data/preds/`, archive output under `web/archive/`, and static web assets under `web/`.

## Important Commands

```sh
python scripts/run_weekly_all.py --tour $TOUR
python scripts/update_upcoming_events.py
python scripts/update_web_status.py --sync-assets
TOUR=pga FAST=true DATAGOLF_API_KEY=... bash scripts/ci_run_model.sh
python scripts/update_previous_week_archives.py --dry-run
python scripts/update_previous_week_archives.py --lookback-weeks 6 --completed-only --dry-run
YEAR=2026 DATAGOLF_API_KEY=... bash scripts/ci_fetch_actual_results.sh
python scripts/update_archived_event.py --event_id $EVENT_ID
python scripts/fetch_actual_results.py --year $YEAR
python scripts/build_web_assets.py --tour $TOUR
python scripts/summarize_status.py
python scripts/check_site_integrity.py
```

Local web check:

```sh
cd web/ && python -m http.server 8000
```

## Data And Artifacts

- `data/processed/`: normalized event inputs, including field, tee times, weather summaries, rankings, course fit, and metadata.
- `data/features/`: model feature tables.
- `data/preds/`: prediction outputs, leaderboards, and summaries.
- `data/analytics/`: prediction accuracy and actual-vs-predicted tracking.
- `web/archive/`: generated archive pages and event result JSON.
- `web/index.html` reads each tour's `meta.json`; when `active_events` contains more than one event, the homepage shows an event switcher and loads the selected preview from that event's `resources` paths under `web/{tour}/events/event_{event_id}/`.
- `web/player.html` is a static player drilldown page. PGA/DP World leaderboard player names link to it with `tour`, `event_id`, `player`, and optional `snapshot=initial`; the page reads served model JSON, tour-agnostic archive appearance JSON, and `web/spreadsheet_data.csv`.
- `web/course_details.html` is a static course drilldown page. It supports `tour`, `event_id`, `course`, and optional `snapshot=initial`; PGA/DP World tournament summaries and player drilldown course links point to it. The page combines served model assets (`tournament_summary.json`, course-fit weights, course-history summary, weather, leaderboard, and optional `field_teetimes.csv`) with the best matching spreadsheet course note via `web/assets/js/course-notes.js`. The Event Context yardage row uses the parsed course-note `Length` field when available, then falls back to tournament `total_yardage`. It includes directional first-round-leader analysis using model strength, course fit, R1 wave weather, tee-time data where available, and matched historical FRL bets; this is not a true FRL probability model. Course notes strip the retired `Key Attribute` and `Insight` sections before rendering.
- Course-history summaries should include `player_name` where historical source data provides it. `scripts/build_course_history_from_hist.py` carries names into `event_{event_id}_course_history_stats.parquet`, and `scripts/build_web_assets.py` can backfill names from the historical combined parquet when older stats files contain only player IDs. The course drilldown page displays names in the course-history side card and only falls back to current field/leaderboard lookup when summaries lack names.
- `web/{tour}/initial/{year}/event_{event_id}/`: frozen first successful prediction snapshot for an event/week; later scheduled runs keep live assets updated without overwriting this initial copy. Each snapshot should include `model_page.html`, a self-contained event-specific backup page with embedded leaderboard, tournament, weather, course-fit, and provenance data.
- `web/model_health.html`: browser-rendered operational health page for deployed site assets. It reads only files served from `web/` (`status.json`, tour `meta.json`, tour `schedule.json`, `archive/index.json`, and `archive/results_summary.json`) and mirrors the main integrity concerns: stale status, current schedule vs model metadata, status vs model metadata, archive leaderboard availability, and expected result-file coverage.
- `web/odds_value.html`: browser-rendered Odds & Value page comparing betting-sheet prices with model probabilities. It is presented as a beta initiative that started with the 2026 season; value labels should be treated as decision support while archive samples, result links, and place assumptions mature. The page has two view modes: `Pending` for current pending bets against live model files, and `2026 archive` for bets matched to archived 2026 event snapshots. Pending mode loads every `active_events` model for each tour so alternate-field/co-sanctioned weeks can match bets to the correct tournament instead of only the primary tour event.
- `web/odds_value.html` force-refreshes `web/spreadsheet_data.csv` on load, versions its `web/assets/js/betting-data.js` script tag, and resets its browser controls to Pending/default filters so newly added pending bets are not hidden by stale `localStorage` or restored form state. Pending view should include all pending included bets even when the current model does not yet match the event; pending bets should prefer current models over archive models and must display `Pending`/`Event not started` in the Actual Result column even if an archive match has historical results. Shared `web/assets/js/betting-data.js` should fetch fresh CSV data when response metadata is unavailable, falling back to stored data only if a fresh fetch fails; it includes a local CSV parser fallback so betting pages can still load rows if the PapaParse CDN is unavailable.
- `web/archive_accuracy.html`: browser-rendered prediction accuracy dashboard for completed archive events with archived `results.json`. It uses compact week/tour x-axis labels to keep charts readable, keeps full tournament names in chart hover text, supports All/PGA/Euro tour filters, lists completed events with missing actual results as pending instead of plotting partial data, and auto-scales the trend chart y-axis to the selected data instead of forcing a fixed 0-100% range.
- `web/archive_accuracy.html` also checks served PGA/Euro schedule files and lists completed events that have no prediction archive as "prediction archive missing" so missed model snapshots do not silently disappear from the accuracy dashboard.
- `web/archive.html` displays compact status badges. Reconstructed backfills are labelled `Reconstructed`; completed archives are labelled `Completed`; missing summaries/results show `Summary pending`, `Awaiting results`, or `Results unknown` instead of silently looking like normal archives.
- Player matching for prediction accuracy uses compact normalized name keys plus a first-name/final-surname fallback in both `web/archive_accuracy.html` and `scripts/build_prediction_accuracy.py`. Keep those behaviors aligned so DataGolf display-name differences such as `Eugenio Lopez-Chacarra` versus `Eugenio Chacarra` do not drop winner probabilities or actual-result matches.
- `web/archive/2026/the_players_championship/leaderboard.json` was repaired from its archived CSV after it had been overwritten to an empty array while `results.json` and `leaderboard.csv` were valid. If an archive has a valid CSV but empty JSON, rebuild the JSON from CSV rather than treating actual results as missing.
- `web/calibration_dashboard.html`: browser-rendered player-level calibration dashboard. It is presented as a beta initiative that started with the 2026 season and includes a visible explainer for calibration curves, bucket groups, Brier/log loss, and Avg Gap. It must use files served under `web/` (`web/archive/index.json`, archived leaderboards, and `results.json`) rather than `data/analytics/**`, which is not directly served by GitHub Pages. Make-cut calibration derives outcomes from finish position and cut status when explicit `made_cut` flags are missing.
- `scripts/check_site_integrity.py`: read-only local/CI diagnostic for generated site assets. It validates current PGA/Euro events against `web/{tour}/meta.json`, checks active prediction presence, flags stale or mismatched `web/status.json`, validates archive index entry files, and warns on recent completed events missing archives or archived results. It runs before Pages artifact upload in `deploy-pages.yml` and `weekly-model.yml`. Use `--strict` to fail on warnings, `--archive-lookback-days 0` to scan all completed scheduled PGA/Euro events, and `--status-age-hours` to tune status freshness.

Generated artifacts should usually be regenerated through scripts rather than manually edited.

## Known Guardrails

- Never expose or commit `.env` values.
- Be careful with event IDs and tour naming. PGA and DPWT/Euro data can coexist in parallel folder structures.
- Use explicit years for historical/actual result tasks.
- When comparing predictions to actuals, confirm whether the event has final results available before treating missing results as model failure.
- `web/spreadsheet_data.csv` is manually maintained and should not be overwritten by scheduled automation.

## Automation Notes

- GitHub Actions workflows were added for Pages deploys and scheduled/manual model runs.
- Schedule refresh: hourly at minute 7 UTC. It refreshes `upcoming-events.json` from DataGolf and commits it to `master` only when changed; it does not deploy Pages or modify published `web/status.json`.
- Model refresh schedule: every 2 hours Monday-Wednesday at minute 23 UTC.
- Shared site status card is injected by `web/menu.js`, appears at the bottom of pages that load the menu, and reads `web/status.json` generated by `scripts/update_web_status.py`.
- Archive update schedule: Monday at 12:00 and 21:00 UTC. Previous-week event IDs come from `scripts/update_previous_week_archives.py --dry-run`; manual runs can set `lookback_weeks` to scan multiple completed Monday-Sunday windows. Archive updates are committed to `master` so later Pages deploys do not revert completed archive status.
- `scripts/update_archived_event.py` now materializes a missing `web/archive/{year}/{slug}/` entry from saved `web/{tour}/initial/{year}/event_{event_id}/` files, falling back to checked-in `web/{tour}/events/event_{event_id}/` assets. Forced updates still skip locally upcoming/in-progress events when no winner is available.
- `archive-update.yml` and the daily actual-results wrapper refresh actual results with `--allow-missing` and rebuild prediction accuracy after archive updates, then commit `web/archive/**` and `data/analytics/**` changes.
- Actual-results schedule: daily at 02:17 UTC, defaulting to current UTC year. It commits changed `web/archive/**` results and `data/analytics/**` outputs to `master`; the normal Pages deploy workflow publishes them after rebuilding active model assets.
- `scripts/ci_fetch_actual_results.sh` runs `scripts/update_previous_week_archives.py --lookback-weeks 6 --completed-only` before fetching results so the daily actual-results job can self-heal recent archive entries missed by the Monday archive workflow. This can only materialize events that have saved initial snapshots or checked-in event assets.
- GitHub repo setup still needs `DATAGOLF_API_KEY` added as an Actions secret.
- GitHub Pages should be configured to deploy from GitHub Actions.
- Scheduled model runs deploy the generated `web/` directory directly as a Pages artifact. Full scheduled runs and deploy-time model refreshes also commit durable prediction snapshots (`web/archive/**` and `web/{tour}/initial/**`) back to `master`; refreshed live `web/{tour}/...` assets remain artifact-only.
- Because live model assets are generated in Actions and not committed to `master`, non-model workflows must not deploy the checked-out `web/` tree unless they first rebuild all active model assets or pass `scripts/guard_pages_model_assets.py`. The hourly schedule refresh intentionally does not deploy Pages.
- In automatic multi-tour runs, `scripts/ci_run_model.sh` skips the model pipeline for a tour that has no runnable current-week event in `upcoming-events.json`, builds that tour's no-event web placeholder, and still deploys other successful tours. Explicit single-tour and pinned-event runs remain strict. Real requested-tour failures still fail the workflow because a partial deploy can overwrite the last good Pages deployment with stale checked-in assets for the failed tour.
- `scripts/run_weekly_all.py` refuses to continue if DataGolf `field-updates` returns an `event_id` different from the requested event. This prevents a pinned or scheduled run from writing current-event predictions under a missed historical event.
- When two PGA events share a start date, DataGolf's `field-updates` feed can return the active event for `tour=pga` or `tour=opp` regardless of the requested `event_id`. `scripts/run_weekly_all.py` now retries the alternate field feed only when the mismatched payload event is one of the same-start PGA events, then still refuses to parse unresolved mismatches.
- New `web/{tour}/schedule.json` builds include `event_id`, `tour`, `status`, and ISO `start_date` alongside the existing display fields so browser pages can match scheduled events back to archive entries.
- The 2026 Charles Schwab Challenge / Charles Schwab Classic archive was reconstructed on 2026-06-15 because the original first-run snapshot was not recoverable from the repo or Pages artifacts. Its provenance is recorded in `data/processed/pga/event_21_reconstruction_meta.json` and propagated to `web/pga/initial/2026/event_21/` and `web/archive/index.json`. Treat it as a reconstructed backfill, not the original scheduled model output: field/results came from DataGolf historical rounds, weather from Open-Meteo archive, and player ratings/skills from the nearest prior checked-in PGA Championship snapshot (`event_33`).
- The 2026 Austrian Alpine Open archive (`event_id=2026120`) was reconstructed on 2026-06-26 because no original first-run snapshot or checked-in event assets were available. Its provenance is recorded in `data/processed/euro/event_2026120_reconstruction_meta.json` and propagated to `web/euro/initial/2026/event_2026120/` and `web/archive/2026/austrian_alpine_open/`. Treat it as a reconstructed backfill: field/results came from DataGolf historical raw rounds, weather from Open-Meteo archive, player ratings/skills from the reconstruction run, and course-fit/course-history were unavailable so course-fit scores are neutral.
- `scripts/fetch_historical_rounds_single.py` supports explicit `--event_id`/`--tour` arguments for targeted backfills, and pinned `scripts/fetch_weather_from_schedule.py` can fall back to `upcoming-events.json` metadata when processed event meta is missing.
- Local edits should usually be committed with `scripts/commit_and_push.sh "message"`, which stages edits, ignores `.env` and Office lock files, rebases over workflow commits on `origin/master`, commits, rebases once more, and pushes.
- `scripts/build_web_assets.py` now preserves the first successful model output for each tour/event/year as an initial snapshot, including a self-contained `model_page.html` backup. Prediction archives should use that snapshot rather than later refreshed live assets. Homepage links expose latest results and `?snapshot=initial` initial-run results.
- Archive writers must fail on invalid `web/archive/index.json` rather than falling back to an empty list. A deploy on 2026-06-15 exposed why: conflict markers in the index caused `scripts/build_web_assets.py` to publish a one-event U.S. Open index, and the archive page then showed no displayable completed events because U.S. Open was still upcoming.
- `scripts/check_site_integrity.py` currently distinguishes errors from warnings: stale/mismatched live tour assets are errors, while stale status metadata and recent missing archive/results are warnings unless `--strict` is used. This keeps local diagnostics useful even when checked-in live assets intentionally lag scheduled model deployments.

## Open Notes

- Expand this file as we make modeling decisions, discover data quirks, or settle on recurring workflows.
- Track any future calibration choices, feature changes, and archive publishing steps here.
- Page/model improvement ideas are tracked in `docs/page_model_roadmap.md`, split into near-term, next, and longer-term work.
