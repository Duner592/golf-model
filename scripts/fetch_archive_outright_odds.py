#!/usr/bin/env python3
"""Publish Bet365/Sky Bet historical outright prices for archive appearances."""
from __future__ import annotations
import json, os, time
from datetime import datetime, timezone
from pathlib import Path
import requests, yaml
from dotenv import load_dotenv
from request_safety import raise_for_status_safely

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ("bet365", "skybet")

def main():
    load_dotenv(ROOT / '.env')
    config = yaml.safe_load((ROOT / 'configs/datagolf.yaml').read_text())
    key = os.getenv(config['auth']['env_var'])
    if not key: raise RuntimeError(f"Missing API key in environment: {config['auth']['env_var']}")
    index = json.loads((ROOT / 'web/archive/index.json').read_text())
    endpoint = config['endpoints']['historical_outright_odds']['path']
    url = f"{config['base_url'].rstrip('/')}/{endpoint}"
    events = []
    for entry in index:
        if not all(entry.get(field) for field in ('tour','event_id','year','slug')): continue
        if not (ROOT / 'web' / 'archive' / str(entry['year']) / str(entry['slug']) / 'results.json').exists():
            continue
        players = {}
        for book in BOOKS:
            params = {config['auth']['key_param']: key, 'tour': entry['tour'], 'event_id': entry['event_id'], 'year': entry['year'], 'market': 'win', 'book': book, 'odds_format': 'fraction'}
            response = requests.get(url, params=params, timeout=30)
            try:
                raise_for_status_safely(response)
            except requests.HTTPError as error:
                print(f"Skipping {entry['event_name']} ({book}): {error}")
                continue
            for row in response.json().get('odds', []):
                name = str(row.get('player_name') or '').strip()
                if name:
                    players.setdefault(name, {})[book] = {'open': row.get('open_odds'), 'close': row.get('close_odds')}
            time.sleep(config.get('rate_limits', {}).get('min_sleep_seconds', 0.5))
        if players:
            events.append({'tour': entry['tour'], 'event_id': str(entry['event_id']), 'year': str(entry['year']), 'slug': entry['slug'], 'event_name': entry.get('event_name',''), 'players': players})
    output = {'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'books': list(BOOKS), 'events': events}
    path = ROOT / 'web/archive/historical_odds.json'; path.write_text(json.dumps(output, indent=2) + '\n')
    print(f'Wrote {path.relative_to(ROOT)} for {len(events)} archive events.')
if __name__ == '__main__': main()
