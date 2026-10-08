import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.final_snapshot import capture_final_snapshot, capture_window


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class FinalSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.event = dict(event_id="554", tour="pga", year="2026", slug="utah", start_date="2026-10-01")
        self.write("leaderboard.json", [{"player_name": "A Smith", "p_win_%": 100}])
        self.write("tournament_summary.json", {"event_id":"554", "status":"upcoming", "field_size":1})
        self.write("weather_meta.json", {"event_id":554, "r1_date":"2026-10-01", "timezone":"America/Denver"})
        self.csv = self.source / "leaderboard.csv"
        self.csv.write_text("player_name,p_win_%\nA Smith,100\n")

    def write(self, name, data):
        (self.source / name).write_text(json.dumps(data))

    def capture(self, generated="2026-10-01T02:00:00Z", now="2026-10-01T02:10:00Z"):
        return capture_final_snapshot(self.root, event=self.event, source_dir=self.source,
                                      leaderboard_csv=self.csv, generated_utc=generated, now=utc(now))

    def test_local_evening_not_uk_wednesday(self):
        opening, cutoff = capture_window("2026-10-01", "America/Denver")
        self.assertEqual(opening, utc("2026-10-01T00:00:00Z"))
        self.assertEqual(cutoff, utc("2026-10-01T06:00:00Z"))
        self.assertEqual(capture_window("2026-10-01", "Asia/Tokyo")[1], utc("2026-09-30T15:00:00Z"))

    def test_wednesday_start_and_winter_timezone(self):
        self.assertEqual(capture_window("2026-10-07", "Europe/London")[0], utc("2026-10-06T17:00:00Z"))
        self.assertEqual(capture_window("2026-12-03", "Europe/London")[0], utc("2026-12-02T18:00:00Z"))

    def test_latest_fresh_snapshot_replaces_earlier_then_freezes(self):
        first = self.capture()
        self.assertEqual(first["snapshot_type"], "final")
        final = self.capture("2026-10-01T040000Z", "2026-10-01T04:10:00Z")
        self.assertEqual(final["prediction_generated_utc"], "2026-10-01T04:00:00Z")
        self.assertEqual(self.capture(), final)
        path = self.root / "web/archive/2026/utah/final/snapshot.json"
        before = path.read_bytes()
        self.assertIsNone(self.capture("2026-10-01T05:59:00Z", "2026-10-01T06:00:00Z"))
        self.assertEqual(before, path.read_bytes())
        self.assertFalse((self.root / "web/pga/initial").exists())

    def test_missing_naive_stale_or_future_timestamp_is_rejected(self):
        for generated in (None, "bad", "2026-10-01T01:00:00", "2026-09-28T02:00:00Z", "2026-10-01T03:00:00Z"):
            self.assertIsNone(self.capture(generated))
        self.assertIsNone(self.capture("2026-10-01T00:00:00Z", "2026-10-01T05:00:00Z"))

    def test_missing_timezone_or_wrong_weather_event_is_rejected(self):
        self.write("weather_meta.json", {})
        self.assertIsNone(self.capture())
        self.write("weather_meta.json", {"event_id":123, "r1_date":"2026-10-01", "timezone":"America/Denver"})
        self.assertIsNone(self.capture())

    def test_reconstruction_and_duplicate_players_are_rejected(self):
        self.event["reconstruction"] = True
        self.assertIsNone(self.capture())
        self.event.pop("reconstruction")
        self.write("leaderboard.json", [{"player_name":"A Smith"}] * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.capture()

    def test_initial_archive_refresh_preserves_final_reference(self):
        from scripts.build_web_assets import archive_event_predictions
        archive = self.root / "web/archive"
        archive.mkdir(parents=True)
        entry = {**self.event, "final_snapshot": {"snapshot_type":"final"}}
        (archive / "index.json").write_text(json.dumps([entry]))
        archive_event_predictions(self.root, "pga", "Utah", "554", "2026-10-01", self.csv, self.source,
                                  archived_at="2026-09-28T01:00:00Z", snapshot_type="initial")
        self.assertEqual(json.loads((archive / "index.json").read_text())[0]["final_snapshot"], entry["final_snapshot"])

    def test_site_integrity_validates_final_hashes(self):
        from scripts.check_site_integrity import IntegrityCheck
        metadata = self.capture()
        entry = {**self.event, "final_snapshot":{key:value for key,value in metadata.items() if key != "provenance"}}
        with patch("scripts.check_site_integrity.ROOT", self.root):
            check = IntegrityCheck(strict_status_age_hours=48, archive_lookback_days=30)
            check.check_archive_entry_files(entry)
            self.assertFalse(any(f.code == "archive-final-snapshot-invalid" for f in check.findings))
            (self.root / "web/archive/2026/utah/final/leaderboard.json").write_text("[]")
            check.check_archive_entry_files(entry)
            self.assertTrue(any(f.code == "archive-final-snapshot-invalid" for f in check.findings))


if __name__ == "__main__":
    unittest.main()
