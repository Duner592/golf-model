import unittest
from unittest.mock import Mock, patch
from scripts.update_upcoming_events import fetch_schedule, sort_key


class ScheduleSeasonTests(unittest.TestCase):
    @patch('scripts.update_upcoming_events.requests.get')
    def test_documented_season_parameter_and_cross_year_events(self, get):
        events = [{'event_id':'TBD', 'start_date':'2026-11-26'}]
        get.return_value = Mock(status_code=200)
        get.return_value.json.return_value = {'season':2027, 'schedule':events}
        self.assertEqual(fetch_schedule('https://example.test', 'get-schedule', 'key', 'test', 'euro', 2027), events)
        params = get.call_args.kwargs['params']
        self.assertEqual(params['season'], '2027')
        self.assertNotIn('year', params)

    @patch('scripts.update_upcoming_events.requests.get')
    def test_rejects_wrong_or_empty_season(self, get):
        get.return_value = Mock(status_code=200)
        for payload in [{'season':2026, 'schedule':[{}]}, {'season':2027, 'schedule':[]}]:
            get.return_value.json.return_value = payload
            with self.assertRaises(ValueError):
                fetch_schedule('https://example.test', 'get-schedule', 'key', 'test', 'pga', 2027)

    def test_numeric_and_provisional_ids_sort_together(self):
        events = [{'event_id':id, 'start_date':'2027-01-01', 'tour':'pga'} for id in ['TBD','12']]
        self.assertEqual(sorted(events,key=sort_key)[0]['event_id'],'12')
