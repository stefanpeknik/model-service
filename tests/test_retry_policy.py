import unittest
from datetime import timedelta

from execution_scheduler.domain import RetryPolicy


class RetryPolicyTests(unittest.TestCase):
    def test_sustained_rejections_saturate_before_duration_overflow(self) -> None:
        policy = RetryPolicy(3, timedelta(seconds=5), timedelta(seconds=300))
        for attempt in (45, 1000, 10**9):
            with self.subTest(attempt=attempt):
                self.assertEqual(policy.backoff(attempt), timedelta(seconds=300))

    def test_initial_and_uncapped_delays_are_preserved(self) -> None:
        policy = RetryPolicy(3, timedelta(seconds=5), timedelta(seconds=300))
        self.assertEqual(policy.backoff(0), timedelta(seconds=5))
        self.assertEqual(policy.backoff(3), timedelta(seconds=20))
        self.assertEqual(policy.backoff(7), timedelta(seconds=300))

    def test_zero_delay_remains_zero_for_large_attempt_counts(self) -> None:
        policy = RetryPolicy(3, timedelta(0), timedelta(seconds=300))
        self.assertEqual(policy.backoff(10**9), timedelta(0))

    def test_maximum_representable_duration_does_not_overflow(self) -> None:
        policy = RetryPolicy(3, timedelta(microseconds=1), timedelta.max)
        self.assertEqual(policy.backoff(1000), timedelta.max)
