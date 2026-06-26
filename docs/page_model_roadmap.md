# Page And Model Improvement Roadmap

This is a backlog for improvements to the golf model site and prediction pipeline. The aim is to separate quick wins from bigger model-quality work so ideas do not get lost between weekly runs.

## Completed

### Archive Page Status Labels

Completed on 2026-06-26.

Implemented in `web/archive.html`.

Current features:

- Displays compact status badges instead of raw status text.
- Marks completed archives as `Completed`.
- Marks reconstructed archives as `Reconstructed`.
- Shows `Summary pending` when an archive index entry has no tournament summary.
- Shows `Awaiting results` or `Results unknown` when completed archived events do not have result data.

### Site Integrity Check

Completed on 2026-06-26.

Implemented in `scripts/check_site_integrity.py`.

Current features:

- Validates `upcoming-events.json`, `web/status.json`, `web/{tour}/meta.json`, and `web/archive/index.json` parse correctly.
- Checks current-week PGA/Euro schedule events against published tour metadata.
- Fails when an active published event is missing predictions.
- Warns when status metadata is stale or points at different model events than the current schedule.
- Checks archive entries for missing directories and leaderboard files.
- Warns on recent completed events missing from the archive index or archived events missing `results.json`.
- Supports `--strict`, `--status-age-hours`, and `--archive-lookback-days` for CI/local tuning.

### Odds And Value View

Completed on 2026-05-19.

Implemented in `web/odds_value.html` and linked from the Betting History menu.

Current features:

- Loads bets from `web/spreadsheet_data.csv`.
- Includes only pre-tournament `E/W` and `Outright` bets.
- Joins bets to current PGA/DP World model leaderboards.
- Loads 2026 archived prediction leaderboards from `web/archive/index.json`.
- Adds a source filter for current week, 2026 archive, and all 2026 bets.
- Calculates bookmaker implied probability, model edge, win EV, place EV, and combined each-way EV.
- Uses visible value labels with documented thresholds.
- Links player names and tournament names back to filtered betting history.
- Includes actual result and profit/loss columns.
- Links to the prediction accuracy dashboard.

Future follow-ups for this view belong under the existing `Closing-Line Value Tracking` item unless they require a separate page.

## Do Soon

### Model Health Page

Add a small internal-facing page showing the operational state of the site.

Useful fields:

- Latest successful model run per tour.
- Current event name, event ID, year, and tour.
- Latest archive update time.
- Latest actual-results update time.
- Whether live model assets, archive assets, and status assets agree.
- Any missing or stale files from the site integrity check.

This page can be plain and functional. It is mainly for diagnosing workflow and publishing issues quickly.

### Homepage Recent Improvements

Add a compact recent-improvements section to the homepage showing the last 3-4 meaningful site/model changes.

Useful fields:

- Date.
- Short title.
- One-line description.
- Optional link to the changed page or relevant dashboard.

The section should be manually curated or generated from a small JSON file rather than inferred from every Git commit, so routine data refreshes and workflow commits do not create noise.

## Do Next

### Calibration Dashboard

Use `data/analytics/prediction_vs_actual.csv` to track whether model probabilities are well calibrated.

Useful views:

- Win probability calibration.
- Top-10 probability calibration.
- Make-cut probability calibration if available.
- PGA vs DP World Tour split.
- Brier score and log loss by tour.
- Calibration by probability bucket.

Potential implementation starting point: `scripts/eval_utils.py` already has calibration-table style utilities.

### Backtest Summary By Event Type

Break model performance down by context.

Possible splits:

- Tour.
- Field strength.
- Major vs regular event.
- Course type.
- Weather volatility.
- Small vs full field.
- Before and after calibration.

This should help identify where the model is genuinely strong or weak.

### Player Drilldown Page

Make leaderboard player names clickable.

Player detail could include:

- Current event win, top-10, and make-cut probabilities.
- Model rank.
- Course-fit contribution.
- Weather/wave adjustment.
- Recent archive performance.
- Betting history and ROI where available.

This would make the site easier to investigate player by player.

### Course Fit Confidence

Expose whether course fit is strong, weak, fallback, or unavailable.

Possible labels:

- `High confidence`
- `Limited course history`
- `Fallback`
- `Unavailable`

This is especially useful when a `course_fit_score` is `0.0`, because that can mean missing/neutral data rather than a meaningful zero.

## Longer-Term

### Weather Sensitivity View

Show baseline predictions next to weather-adjusted predictions.

Useful fields:

- Baseline win/top-10 probability.
- Weather-adjusted probability.
- Net weather impact.
- Wave advantage/disadvantage.
- Forecast confidence.

This would make weather effects easier to trust and explain.

### Variant And Feature Tracking

Automate regular comparison of model variants.

Examples:

- Baseline model.
- Course-fit enabled.
- Weather enabled.
- Common-shock simulation enabled.
- Calibrated probabilities.

Store variant results per event so it is possible to see whether each feature improves historical performance.

Potential implementation starting point: `scripts/compare_variants.py`.

### Closing-Line Value Tracking

If odds snapshots are available, track whether model-backed bets beat the closing price.

Metrics:

- Opening price.
- Bet price.
- Closing price.
- Closing-line value.
- Result.
- ROI.

This gives a better read on process quality than short-term betting results alone.

### Better Course Data Coverage

Expand course metadata used by course fit.

Useful course features:

- Yardage.
- Par.
- Grass type.
- Altitude.
- Coastal/wind exposure.
- Firmness if available.
- Historical scoring profile.
- Driving accuracy/approach/short-game emphasis.

The goal is to reduce fallback course-fit cases and make course-fit scores more interpretable.

### Workflow Dashboard From GitHub Actions

If useful later, pull recent GitHub Actions status into the site or a local diagnostic command.

Useful fields:

- Last scheduled model run.
- Last archive update.
- Last actual-results run.
- Last failed workflow.
- Commit SHA deployed to Pages.

This is optional because the site integrity check and health page should cover most local diagnostics first.

## Suggested Build Order

1. Add `scripts/check_site_integrity.py`.
2. Add a basic model health page using the integrity-check output.
3. Improve archive labels so incomplete events are visible.
4. Add calibration and backtest dashboards.
5. Add player drilldowns and course-fit confidence.
6. Add weather sensitivity and variant tracking.
