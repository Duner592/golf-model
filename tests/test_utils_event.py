from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.utils_event import current_week_event_ids, is_resolved_event_id


class EventResolverTests(unittest.TestCase):
    def test_current_week_filters_tour_and_unresolved_ids(self) -> None:
        schedule = {
            "schedule": [
                {"event_id": "101", "tour": "pga", "start_date": "2026-08-13"},
                {"event_id": "TBD", "tour": "pga", "start_date": "2026-08-13"},
                {"event_id": "2026131", "tour": "euro", "start_date": "2026-08-13"},
                {"event_id": "old", "tour": "pga", "start_date": "2026-08-06"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            path.write_text(json.dumps(schedule), encoding="utf-8")
            self.assertEqual(current_week_event_ids("pga", reference_date=date(2026, 8, 13), schedule_path=path), ["101"])
            self.assertEqual(current_week_event_ids("euro", reference_date=date(2026, 8, 13), schedule_path=path), ["2026131"])

    def test_unresolved_ids_are_rejected(self) -> None:
        self.assertFalse(is_resolved_event_id(None))
        self.assertFalse(is_resolved_event_id(" TBD "))
        self.assertTrue(is_resolved_event_id("2026131"))


if __name__ == "__main__":
    unittest.main()
