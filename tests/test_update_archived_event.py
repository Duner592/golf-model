import unittest

from scripts.update_archived_event import get_event_details


class ArchivedEventResolverTests(unittest.TestCase):
    def setUp(self):
        self.schedule = {
            "schedule": [
                {"event_id": "27", "tour": "kft", "event_name": "Albertsons Boise Open"},
                {"event_id": "27", "tour": "pga", "event_name": "FedEx St. Jude Championship"},
            ]
        }

    def test_duplicate_event_ids_require_tour_and_select_the_matching_event(self):
        event = get_event_details("27", self.schedule, "pga")
        self.assertEqual(event["event_name"], "FedEx St. Jude Championship")

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            get_event_details("27", self.schedule)

