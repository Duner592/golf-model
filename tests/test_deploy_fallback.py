import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_site_integrity import Finding, saved_data_findings
from scripts import update_web_status as status


class DeploymentFallbackTests(unittest.TestCase):
    def test_only_freshness_failures_are_downgraded(self):
        codes = ['active-event-mismatch', 'active-event-no-predictions',
                 'invalid-json', 'missing-file', 'archive-player-integrity',
                 'archive-final-snapshot-invalid']
        original = [Finding('error', code, code) for code in codes]
        result = saved_data_findings(original)
        self.assertEqual([f.severity for f in result], ['warning', 'warning'] + ['error'] * 4)
        self.assertTrue(all(f.severity == 'error' for f in original))

    def test_failed_refresh_preserves_model_dates_and_updates_betting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'web').mkdir()
            path = root / 'web/status.json'
            path.write_text(json.dumps({'model_runs': {'pga': {'last_run_utc': '2026-10-01'}},
                                        'schedule': {'last_refreshed_utc': '2026-10-01'}}))
            (root / 'web/spreadsheet_data.csv').write_text('Date,Player\n')
            with patch.object(status, 'ROOT', root), patch.object(status, 'STATUS_PATH', path), \
                 patch.object(status, 'git_file_is_dirty', return_value=False), \
                 patch.object(status, 'git_last_commit_utc', return_value='2026-11-10'), \
                 patch('sys.argv', ['status', '--model-refresh', 'failure', '--sync-assets']):
                status.main()
                result = json.loads(path.read_text())
                self.assertEqual(result['model_runs']['pga']['last_run_utc'], '2026-10-01')
                self.assertEqual(result['schedule']['last_refreshed_utc'], '2026-10-01')
                self.assertEqual(result['betting_data']['last_modified'], '2026-11-10')
                self.assertEqual(result['model_refresh']['outcome'], 'failure')
                with patch('sys.argv', ['status', '--model-refresh', 'success']):
                    status.main()
                self.assertEqual(json.loads(path.read_text())['model_refresh']['outcome'], 'success')

    def test_workflow_keeps_failed_outputs_isolated(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        workflow = yaml.safe_load((root / '.github/workflows/deploy-pages.yml').read_text())
        steps = {s['name']: s for s in workflow['jobs']['deploy']['steps']}
        refresh = steps['Refresh model assets']
        self.assertEqual(refresh['working-directory'], 'model-refresh')
        self.assertTrue(refresh['continue-on-error'])
        self.assertIn('PYTHONPATH="$PWD', refresh['run'])
        for name in ['Adopt successful model assets', 'Commit durable prediction snapshots']:
            self.assertEqual(steps[name]['if'], "steps.model.outcome == 'success'")
        self.assertIn('--exclude=spreadsheet_data.csv', steps['Adopt successful model assets']['run'])
        self.assertNotIn('continue-on-error', steps['Check site integrity'])


if __name__ == '__main__':
    unittest.main()
