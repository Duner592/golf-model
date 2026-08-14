from __future__ import annotations

import unittest

import requests

from scripts.build_prediction_accuracy import brier_score, calibration_buckets, log_loss, uniform_field_probabilities
from scripts.fetch_historical_rounds import resolve_course_par
from scripts.request_safety import redact_sensitive_text, raise_for_status_safely

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

    def test_verified_historical_course_par_override_beats_placeholder(self) -> None:
        self.assertEqual(resolve_course_par({"course_par": None}, "FedEx St. Jude Championship", "pga"), 70)
        self.assertIsNone(resolve_course_par({"course_par": None}, "Unknown event", "pga"))

    def test_request_errors_redact_api_credentials(self) -> None:
        message = "403 for https://feeds.datagolf.com/api?tour=pga&key=not-a-real-key"
        self.assertEqual(
            redact_sensitive_text(message),
            "403 for https://feeds.datagolf.com/api?tour=pga&key=[REDACTED]",
        )

        class FailingResponse:
            def raise_for_status(self) -> None:
                raise requests.HTTPError(message)

        with self.assertRaisesRegex(requests.HTTPError, r"key=\[REDACTED\]") as caught:
            raise_for_status_safely(FailingResponse())
        self.assertNotIn("not-a-real-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
