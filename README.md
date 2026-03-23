# golf-model

Purpose
- Weekly tournament win-probability model (PGA/DPWT) using DataGolf + weather + course features.
- Hosted @https://duner592.github.io/golf-model/
- Bet tracking, ROI data utilising speadsheet_data.csv

Quick start
- python scripts/run_weekly_all.py --tour $TOUR
- python scripts/update_archived_event.py --event_id $EVENT_ID
- python scripts/fetch_actual_results.py --year $YEAR
- python scripts/build_web_assets.py --tour $TOUR

Key scripts
- scripts/run_weekly_all.py – end-to-end weekly pipeline
- scripts/export_leaderboard.py – CSV/HTML leaderboard export
- scripts/summarize_status.py – snapshot of current artifacts

Data layout
- data/processed/ – normalized inputs (field, weather summaries, meta)
- data/features/ – features tables (weather/full/course)
- data/preds/ – predictions + leaderboard + summary

Testing
- cd web/ && python -m http.server 8000 - Test webpage from local


Actual results
- scripts/fetch_actual_results.py – pulls final finishes via DataGolf's `historical-raw-data/rounds` endpoint and populates `web/archive/<year>/<slug>/results.json`.
  - Usage: `python scripts/fetch_actual_results.py [--year 2026]` (requires `DATAGOLF_API_KEY` in .env).
Archiving
- web/archive - Generated predictions archive

Actual results
- scripts/fetch_actual_results.py – pulls final finishes via DataGolf's `historical-raw-data/rounds` endpoint and populates `web/archive/<year>/<slug>/results.json`.
  - Usage: `python scripts/fetch_actual_results.py [--year 2026]` (requires `DATAGOLF_API_KEY` in .env).

Config
- configs/datagolf.yaml (redacted) – endpoints and defaults
- Secrets: .env with DATAGOLF_API_KEY (never commit)

Notes
- Weather from schedule (Open-Meteo); course fit DIY via historical regression; common-shock simulator with sigma and wave-aware weather.
