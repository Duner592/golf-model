import json
import unittest
import sys
from unittest.mock import patch
from pathlib import Path

from scripts.recover_backtest_history import recover, identity_payload, digest, git


class RecoveryTests(unittest.TestCase):
    def test_identity_requires_same_event_and_keeps_ambiguous_rows_for_consumer(self):
        raw = b'player_id,player_name,event_id\n1,A Smith,6\n2,A Smith,6\n'
        event = {"event_id":"6", "tour":"pga", "year":"2026"}
        self.assertEqual(len(identity_payload(raw, event, "test")["players"]), 2)
        self.assertIsNone(identity_payload(raw, {**event, "event_id":"7"}, "test"))

    def test_post_event_commit_not_recovered(self):
        event = {"event_id":"6", "tour":"pga", "year":"2026", "slug":"sony", "event_name":"Sony", "start_date":"2026-01-15"}
        with patch('scripts.recover_backtest_history.git', return_value=b'a'*40+b' 2026-01-16T00:00:00Z\n'):
            self.assertIsNone(recover(event))

    def test_saved_files_are_byte_exact_pre_event_git_evidence(self):
        root = Path(__file__).resolve().parents[1]
        entries = json.loads((root / 'web/archive/index.json').read_text())
        recovered = [e for e in entries if e.get('historical_snapshot')]
        self.assertGreater(len(recovered), 0)
        for event in recovered:
            proof = event['historical_snapshot']
            raw = git('show', f"{proof['commit']}:{proof['source_path']}")
            self.assertEqual(digest(raw), proof['leaderboard_sha256'])
            target = root / 'web/archive' / event['year'] / event['slug'] / 'historical'
            self.assertEqual((target / 'leaderboard.json').read_bytes(), raw)
            self.assertLess(proof['committed_utc'], event['start_date']+'T00:00:00Z')
            if proof.get('player_ids_available'):
                payload = json.loads((target/'player_ids.json').read_text())
                _, commit, source = payload['source'].split(':', 2)
                field = git('show', f'{commit}:{source}')
                self.assertEqual(identity_payload(field, event, payload['source']), payload)

    def test_future_exports_preserve_player_ids(self):
        import pandas as pd
        with patch('sys.path', [str(Path(__file__).resolve().parents[1] / 'scripts'), *sys.path]):
            from scripts.export_leaderboard import build_display_table
        rows = pd.DataFrame([{'dg_id':123, 'player_name':'A Smith', 'p_win':.5, 'p_top10':.8, 'p_mc':.9}])
        self.assertEqual(build_display_table(rows, None).iloc[0]['dg_id'], 123)


if __name__ == '__main__':
    unittest.main()
