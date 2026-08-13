from __future__ import annotations

import unittest

from scripts.build_prediction_accuracy import brier_score, calibration_buckets, log_loss, uniform_field_probabilities

import pandas as pd


class PredictionAccuracyTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_loss(self) -> None:
        self.assertEqual(brier_score([0, 100], [0, 1]), 0.0)
        self.assertLess(log_loss([0, 100], [0, 1]) or 1.0, 0.00001)

    def test_calibration_buckets_report_rate_and_gap(self) -> None:
        buckets = calibration_buckets([1, 4, 6], [0, 1, 0], bucket_width=5)
        self.assertEqual(buckets[0]["range"], "0-5%")
        self.assertEqual(buckets[0]["n"], 2)
        self.assertEqual(buckets[0]["actual_rate_pct"], 50.0)
        self.assertEqual(buckets[1]["range"], "5-10%")

    def test_uniform_baseline_respects_each_event_field_size(self) -> None:
        frame = pd.DataFrame({"event_id": ["small", "small", "large", "large", "large", "large"]})
        self.assertEqual(uniform_field_probabilities(frame, target_places=1).tolist(), [50.0, 50.0, 25.0, 25.0, 25.0, 25.0])


if __name__ == "__main__":
    unittest.main()
