#!/usr/bin/env python3
"""Publish Bet365/Sky Bet historical outright prices for archive appearances."""
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime, timezone
from pathlib import Path
import requests, yaml
from dotenv import load_dotenv
from request_safety import raise_for_status_safely

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ("bet365", "skybet")

def write_output(path, events):
    output = {'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'books': list(BOOKS), 'events': list(events.values())}
    path.write_text(json.dumps(output, indent=2) + '\n')


def historical_odds_rows(payload):
    """Normalise DataGolf historical-odds responses to player-price rows.

    The endpoint returns an ``odds`` mapping keyed by player name.  Keeping
    support for list-shaped responses also makes this resilient to older API
    exports and avoids a malformed item aborting an otherwise resumable batch.
    """
    odds = payload.get('odds', []) if isinstance(payload, dict) else []
    if isinstance(odds, dict):
        for name, prices in odds.items():
            if isinstance(prices, dict):
                yield {
                    'player_name': name,
                    'open_odds': prices.get('open_odds', prices.get('open')),
                    'close_odds': prices.get('close_odds', prices.get('close')),
                }
            elif isinstance(prices, (list, tuple)):
                yield {
                    'player_name': name,
                    'open_odds': prices[0] if prices else None,
                    'close_odds': prices[-1] if prices else None,
                }
            else:
                yield {'player_name': name, 'open_odds': None, 'close_odds': prices}
        return
    if isinstance(odds, list):
        yield from (row for row in odds if isinstance(row, dict))


def main():
    parser = argparse.ArgumentParser(description='Backfill archived outright odds without exceeding DataGolf limits.')
    parser.add_argument('--delay-seconds', type=float, default=12.0, help='Minimum wait between requests (default: 12).')
    parser.add_argument('--max-events', type=int, default=0, help='Limit missing archive events for a resumable batch.')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    config = yaml.safe_load((ROOT / 'configs/datagolf.yaml').read_text())
    key = os.getenv(config['auth']['env_var'])
    if not key: raise RuntimeError(f"Missing API key in environment: {config['auth']['env_var']}")
    index = json.loads((ROOT / 'web/archive/index.json').read_text())
    endpoint = config['endpoints']['historical_outright_odds']['path']
    url = f"{config['base_url'].rstrip('/')}/{endpoint}"
    path = ROOT / 'web/archive/historical_odds.json'
    existing = json.loads(path.read_text()).get('events', []) if path.exists() else []
    events = {f"{item['tour']}:{item['year']}:{item['event_id']}": item for item in existing if isinstance(item, dict) and item.get('event_id')}
    pending = []
    for entry in index:
        if not all(entry.get(field) for field in ('tour','event_id','year','slug')): continue
        if not (ROOT / 'web' / 'archive' / str(entry['year']) / str(entry['slug']) / 'results.json').exists():
            continue
        event_key = f"{entry['tour']}:{entry['year']}:{entry['event_id']}"
        known = events.get(event_key, {})
        known_players = known.get('players', {}) if isinstance(known.get('players'), dict) else {}
        if all(any(book in prices for prices in known_players.values()) for book in BOOKS):
            continue
        pending.append(entry)
    if args.max_events:
        pending = pending[:args.max_events]
    print(f'Fetching {len(pending)} archive event(s) at >= {args.delay_seconds:g}s per request.')
    for position, entry in enumerate(pending, 1):
        event_key = f"{entry['tour']}:{entry['year']}:{entry['event_id']}"
        players = events.get(event_key, {}).get('players', {})
        for book in BOOKS:
            if any(book in prices for prices in players.values()):
                continue
            params = {config['auth']['key_param']: key, 'tour': entry['tour'], 'event_id': entry['event_id'], 'year': entry['year'], 'market': 'win', 'book': book, 'odds_format': 'fraction'}
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = float(response.headers.get('Retry-After', args.delay_seconds * 5))
                print(f'Rate limited; waiting {retry_after:g}s before retrying {entry["event_name"]} ({book}).', flush=True)
                time.sleep(retry_after)
                response = requests.get(url, params=params, timeout=30)
            try: raise_for_status_safely(response)
            except requests.HTTPError as error:
                print(f"Skipping {entry['event_name']} ({book}): {error}", flush=True); continue
            for row in historical_odds_rows(response.json()):
                name = str(row.get('player_name') or '').strip()
                if name:
                    players.setdefault(name, {})[book] = {'open': row.get('open_odds'), 'close': row.get('close_odds')}
            time.sleep(max(args.delay_seconds, config.get('rate_limits', {}).get('min_sleep_seconds', 0.5)))
        if players:
            events[event_key] = {'tour': entry['tour'], 'event_id': str(entry['event_id']), 'year': str(entry['year']), 'slug': entry['slug'], 'event_name': entry.get('event_name',''), 'players': players}
            write_output(path, events)
        print(f'[{position}/{len(pending)}] {entry["event_name"]}', flush=True)
    write_output(path, events)
    print(f'Wrote {path.relative_to(ROOT)} for {len(events)} archive events.')
if __name__ == '__main__': main()
