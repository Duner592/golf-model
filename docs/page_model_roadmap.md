# Page And Model Improvement Roadmap

This is a backlog for improvements to the golf model site and prediction pipeline. The aim is to separate quick wins from bigger model-quality work so ideas do not get lost between weekly runs.

## Do Soon

### Site Integrity Check

Add a script such as `scripts/check_site_integrity.py` that can be run locally and in GitHub Actions before deploys.

Checks to include:

- Current PGA/Euro events in `upcoming-events.json` agree with `web/{tour}/meta.json`.
- No active tour page is deployed as `No Event` when DataGolf has a current event.
- `web/status.json` is fresh enough and agrees with current tour metadata.
- Completed archive events are not still marked as `upcoming`.
- Completed archive events have `results.json` when actual results are available.
- `web/archive/index.json` includes the event summaries that should be visible on the archive page.

Why this matters: it should catch stale deploys, missing DP World pages, hidden archives, and incomplete actual-results updates before they reach the public site.

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

### Archive Page Status Labels

Make archive entries visible even when they are incomplete, instead of silently hiding them.

Potential labels:

- `Completed`
- `Awaiting results`
- `Summary pending`
- `Accuracy pending`
- `Results unavailable`

This would make archive problems easier to see from the page itself.

### Odds And Value View

Connect model probabilities to the manually maintained betting data in `web/spreadsheet_data.csv`.

Show:

- Model probability.
- Bookmaker implied probability.
- Estimated edge.
- Bet/result.
- ROI.
- Optional stake band or confidence label.

This turns the model output into a clearer decision view.

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
4. Build the odds/value page.
5. Add calibration and backtest dashboards.
6. Add player drilldowns and course-fit confidence.
7. Add weather sensitivity and variant tracking.
