import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.player_integrity import validate_unique_players


class PlayerIntegrityTests(unittest.TestCase):
    def test_unique_field(self):
        validate_unique_players([{"player_id": 1, "player_name": "A Smith"}, {"player_id": 2, "player_name": "B Smith"}], context="test", require_ids=True)

    def test_duplicate_id_even_with_different_name(self):
        with self.assertRaisesRegex(ValueError, "duplicate players"):
            validate_unique_players([{"player_id": 1, "player_name": "A"}, {"player_id": "1.0", "player_name": "B"}], context="test")

    def test_name_only_export_normalization(self):
        with self.assertRaisesRegex(ValueError, "duplicate players"):
            validate_unique_players([{"player_name": "Smith, José"}, {"player_name": "Jose Smith"}], context="test")

    def test_missing_id_rejected_before_simulation(self):
        for value in (None, "", float("nan")):
            with self.assertRaisesRegex(ValueError, "missing player ID"):
                validate_unique_players([{"dg_id": value, "player_name": "A"}], context="test", require_ids=True)

    def test_original_archives_are_detected(self):
        root = Path(__file__).resolve().parents[1]
        for event_id in (525, 522):
            with (root / f"web/pga/initial/2026/event_{event_id}/leaderboard.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            with self.assertRaisesRegex(ValueError, "duplicate players"):
                validate_unique_players(rows, context=f"snapshot {event_id}")

    def test_field_parser_rejects_duplicate_provider_rows(self):
        from scripts.parse_field_updates import normalize_field
        with self.assertRaisesRegex(ValueError, "duplicate players"):
            normalize_field({"event_id": "525", "field": [{"dg_id": 32457, "player_name": "Johnny Keefer"}] * 2})

    def test_simulator_rejects_duplicate_inputs(self):
        from scripts.simulate_event_with_course import simulate
        with self.assertRaisesRegex(ValueError, "duplicate players"):
            simulate(ids=[1, 1], names=["A", "A"], mu_base=None, sigma=None, weather=None,
                     r1_wave=None, r2_wave=None, n_sims=1, cut_top=65, seed=42, round_sd=.2, wave_sd=.2)

    def test_archive_repair_preserves_initial_snapshot_and_is_idempotent(self):
        from scripts.update_archived_event import deduplicate_archive
        from src.provenance import sha256_file
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = root / "web/pga/initial/2026/event_525"
            original = work / "web/pga/initial/2026/event_525"
            archive = work / "web/archive/2026/3m_open"
            shutil.copytree(source, original)
            shutil.copytree(source, archive)
            entries = json.loads((root / "web/archive/index.json").read_text())
            entry = next(e for e in entries if e["slug"] == "3m_open")
            (work / "web/archive/index.json").write_text(json.dumps([entry]))
            before = {p.name: sha256_file(p) for p in original.iterdir()}
            self.assertEqual(deduplicate_archive(work, event_id="525", tour="pga", year="2026"), 2)
            self.assertEqual(before, {p.name: sha256_file(p) for p in original.iterdir()})
            rows = json.loads((archive / "leaderboard.json").read_text())
            self.assertEqual(len(rows), 143)
            self.assertEqual(next(r for r in rows if r["player_name"] == "Johnny Keefer")["p_win_%"], .09)
            self.assertEqual(next(r for r in rows if r["player_name"] == "Aldrich Potgieter")["p_win_%"], .02)
            audit = json.loads((archive / "duplicate_repair.json").read_text())
            for name, digest in audit["repaired_sha256"].items():
                self.assertEqual(sha256_file(archive / name), digest)
            self.assertEqual(deduplicate_archive(work, event_id="525", tour="pga", year="2026"), 0)


if __name__ == "__main__":
    unittest.main()
