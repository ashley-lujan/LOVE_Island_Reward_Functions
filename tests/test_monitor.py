"""Unit tests for monitor.py — pattern detection and query triggering."""

import os
import sys
import types
import unittest
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eureka"))

from utils.love_island.buffer import EpisodeRecord, EvaluationBuffer
from utils.love_island.monitor import DisagreementMonitor


def _make_cfg(**overrides) -> types.SimpleNamespace:
    defaults = dict(
        high_threshold=0.3,
        low_threshold=0.1,
        bad_threshold=0.3,
        good_threshold=0.7,
        warmup_iterations=2,
        persistence_threshold=2,
        spike_window=3,
        spike_k=2.0,
        stagnation_window=5,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_monitor(**cfg_overrides) -> DisagreementMonitor:
    buf = EvaluationBuffer(capacity=8, random_ratio=0.25)
    cfg = _make_cfg(**cfg_overrides)
    return DisagreementMonitor(eval_buffer=buf, cfg=cfg)


def _make_record(sigma: float, iteration: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=str(uuid.uuid4()),
        joint_pos_snap=torch.zeros(1, 2),
        joint_vel_snap=torch.zeros(1, 2),
        sigma=sigma,
        iteration_cached=iteration,
    )


def _pump(monitor: DisagreementMonitor, sigma: float, reward: float) -> None:
    """Inject a synthetic σ into both the buffer (random slots) and internal history."""
    rec = _make_record(sigma)
    monitor.eval_buffer._random_slots = [rec]
    monitor._record_sigma(sigma, reward)


class TestWarmupGuard(unittest.TestCase):
    def test_no_query_during_warmup(self):
        m = _make_monitor(warmup_iterations=3)
        for _ in range(3):
            _pump(m, sigma=0.9, reward=0.1)
            should_query, _, _ = m.evaluate_mini_iteration()
            self.assertFalse(should_query)

    def test_query_allowed_after_warmup(self):
        m = _make_monitor(warmup_iterations=1, persistence_threshold=1)
        _pump(m, 0.9, 0.1)
        m.evaluate_mini_iteration()  # iter 1 = last warmup iter

        _pump(m, 0.9, 0.1)
        should_query, _, _ = m.evaluate_mini_iteration()  # iter 2 = post-warmup
        self.assertTrue(should_query)


class TestPatternDetection(unittest.TestCase):
    def test_high_disagreement_bad_performance(self):
        m = _make_monitor()
        self.assertEqual(m._detect_pattern(0.5, 0.1), "high_disagreement_bad_performance")

    def test_high_disagreement_ambiguous(self):
        m = _make_monitor()
        self.assertEqual(m._detect_pattern(0.5, 0.6), "high_disagreement_ambiguous")

    def test_stagnation_detection(self):
        m = _make_monitor(stagnation_window=5)
        for _ in range(5):
            m._record_sigma(0.05, 0.5)
        self.assertEqual(m._detect_pattern(0.05, 0.5), "stagnation")

    def test_spike_detection(self):
        # sudden_divergence is only reached when σ is below high_threshold (elif chain).
        # History: very low (0.02), current: moderate spike (0.25) but still < 0.3.
        m = _make_monitor(spike_window=3, spike_k=1.5)
        for _ in range(4):
            m._record_sigma(0.02, 0.5)
        self.assertEqual(m._detect_pattern(0.25, 0.5), "sudden_divergence")

    def test_no_pattern_normal(self):
        m = _make_monitor()
        self.assertIsNone(m._detect_pattern(0.15, 0.5))


class TestPatternToFeedbackMapping(unittest.TestCase):
    MAPPING = [
        ("high_disagreement_bad_performance", "corrective"),
        ("high_disagreement_ambiguous", "comparative"),
        ("sudden_divergence", "guidance"),
        ("stagnation", "guidance"),
        ("confident_positive_outlier", "explanatory"),
        ("unknown_pattern", "binary"),
    ]

    def test_all_mappings(self):
        m = _make_monitor()
        for pattern, expected in self.MAPPING:
            with self.subTest(pattern=pattern):
                self.assertEqual(m._map_pattern_to_feedback(pattern), expected)


class TestPersistenceCounter(unittest.TestCase):
    def test_query_triggered_after_persistence_threshold(self):
        m = _make_monitor(warmup_iterations=0, persistence_threshold=2)
        should_query = False
        for _ in range(2):
            _pump(m, sigma=0.5, reward=0.1)
            should_query, _, _ = m.evaluate_mini_iteration()
        self.assertTrue(should_query)

    def test_persistence_resets_after_query(self):
        m = _make_monitor(warmup_iterations=0, persistence_threshold=2)
        for _ in range(2):
            _pump(m, sigma=0.5, reward=0.1)
            m.evaluate_mini_iteration()
        _pump(m, sigma=0.5, reward=0.1)
        should_query, _, _ = m.evaluate_mini_iteration()
        self.assertFalse(should_query)

    def test_pattern_change_resets_counter(self):
        m = _make_monitor(warmup_iterations=0, persistence_threshold=3)
        _pump(m, sigma=0.5, reward=0.1)
        m.evaluate_mini_iteration()  # high_disagreement_bad → counter=1
        _pump(m, sigma=0.05, reward=0.5)
        m.evaluate_mini_iteration()  # no pattern → counter reset
        self.assertEqual(m._persistence_counter, 1)


if __name__ == "__main__":
    unittest.main()
