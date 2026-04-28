"""Unit tests for buffer.py — EvaluationBuffer slot management."""

import os
import sys
import unittest
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eureka"))

from utils.love_island.buffer import (
    EpisodeRecord,
    EvaluationBuffer,
    FeedbackRecord,
    make_episode_records,
)


def _make_record(sigma: float = 0.5, iteration: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=str(uuid.uuid4()),
        joint_pos_snap=torch.zeros(1, 2),
        joint_vel_snap=torch.zeros(1, 2),
        sigma=sigma,
        iteration_cached=iteration,
    )


def _make_buffer(**kwargs) -> EvaluationBuffer:
    defaults = dict(capacity=8, random_ratio=0.25, staleness_penalty=0.02, high_sigma_threshold=0.2)
    defaults.update(kwargs)
    return EvaluationBuffer(**defaults)


class TestRandomSlots(unittest.TestCase):
    def test_refresh_fills_random_slots(self):
        buf = _make_buffer()
        buf.refresh_random_slots([_make_record(0.3) for _ in range(10)])
        self.assertEqual(len(buf._random_slots), buf.n_random)

    def test_refresh_replaces_all_slots(self):
        buf = _make_buffer()
        records_a = [_make_record(0.3) for _ in range(10)]
        buf.refresh_random_slots(records_a)
        old_ids = {r.episode_id for r in buf._random_slots}
        records_b = [_make_record(0.3) for _ in range(10)]
        buf.refresh_random_slots(records_b)
        new_ids = {r.episode_id for r in buf._random_slots}
        self.assertTrue(old_ids.isdisjoint(new_ids))

    def test_reset_clears_random_slots(self):
        buf = _make_buffer()
        buf.refresh_random_slots([_make_record(0.3) for _ in range(5)])
        buf.reset_for_new_mini_iteration(1)
        self.assertEqual(len(buf._random_slots), 0)


class TestInterestingSlots(unittest.TestCase):
    def test_high_sigma_enters_interesting_slots(self):
        buf = _make_buffer()
        rec = _make_record(sigma=0.5)
        buf.update_interesting_slots(rec)
        self.assertIn(rec, buf._interesting_slots)

    def test_low_sigma_rejected(self):
        buf = _make_buffer(high_sigma_threshold=0.3)
        rec = _make_record(sigma=0.1)
        buf.update_interesting_slots(rec)
        self.assertNotIn(rec, buf._interesting_slots)

    def test_interesting_slots_capped(self):
        buf = _make_buffer(capacity=4, random_ratio=0.25)  # n_interesting=3
        for i in range(10):
            buf.update_interesting_slots(_make_record(sigma=0.5 + i * 0.01))
        self.assertLessEqual(len(buf._interesting_slots), buf.n_interesting)

    def test_staleness_penalty_allows_replacement(self):
        buf = _make_buffer(capacity=4, random_ratio=0.25, staleness_penalty=0.1)
        for _ in range(buf.n_interesting):
            buf.update_interesting_slots(_make_record(sigma=0.5, iteration=0))
        buf.reset_for_new_mini_iteration(10)
        new_rec = _make_record(sigma=0.6, iteration=10)
        buf.update_interesting_slots(new_rec)
        self.assertIn(new_rec, buf._interesting_slots)

    def test_interesting_slots_persist_across_reset(self):
        buf = _make_buffer()
        rec = _make_record(sigma=0.5)
        buf.update_interesting_slots(rec)
        buf.reset_for_new_mini_iteration(1)
        self.assertIn(rec, buf._interesting_slots)


class TestMeanSigma(unittest.TestCase):
    def test_mean_sigma_includes_both_partitions(self):
        buf = _make_buffer(capacity=8, random_ratio=0.5)
        buf.refresh_random_slots([_make_record(sigma=0.2) for _ in range(4)])
        buf.update_interesting_slots(_make_record(sigma=0.8))
        mean = buf.get_mean_sigma()
        self.assertGreater(mean, 0.2)
        self.assertLess(mean, 0.8)

    def test_mean_sigma_zero_for_empty_buffer(self):
        self.assertEqual(_make_buffer().get_mean_sigma(), 0.0)


class TestFeedbackTracking(unittest.TestCase):
    def test_mark_episode_queried_stores_record(self):
        buf = _make_buffer()
        rec = _make_record(sigma=0.5)
        buf.update_interesting_slots(rec)
        buf.mark_episode_queried(rec.episode_id, "binary", "looks good", 1, "increased upright", "more upright")
        history = buf.get_context_for_reflection(rec.episode_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].feedback_type, "binary")
        self.assertEqual(history[0].feedback_text, "looks good")

    def test_has_been_queried_false_before_marking(self):
        buf = _make_buffer()
        rec = _make_record(sigma=0.5)
        buf.update_interesting_slots(rec)
        self.assertFalse(buf.has_been_queried(rec.episode_id))

    def test_has_been_queried_true_after_marking(self):
        buf = _make_buffer()
        rec = _make_record(sigma=0.5)
        buf.update_interesting_slots(rec)
        buf.mark_episode_queried(rec.episode_id, "binary", "ok", 1, "c", "d")
        self.assertTrue(buf.has_been_queried(rec.episode_id))

    def test_feedback_history_persists_after_reset(self):
        buf = _make_buffer()
        rec = _make_record(sigma=0.5)
        buf.update_interesting_slots(rec)
        buf.mark_episode_queried(rec.episode_id, "binary", "ok", 1, "c", "d")
        buf.reset_for_new_mini_iteration(2)
        self.assertEqual(len(buf.get_context_for_reflection(rec.episode_id)), 1)


class TestCandidateSelection(unittest.TestCase):
    def test_candidates_from_interesting_only(self):
        buf = _make_buffer()
        interesting = _make_record(sigma=0.5)
        buf.update_interesting_slots(interesting)
        random_rec = _make_record(sigma=0.9)
        buf.refresh_random_slots([random_rec])
        candidates = buf.get_candidates_for_feedback("binary")
        ids = {r.episode_id for r in candidates}
        self.assertIn(interesting.episode_id, ids)
        self.assertNotIn(random_rec.episode_id, ids)


class TestMakeEpisodeRecords(unittest.TestCase):
    def test_creates_one_record_per_state(self):
        n = 6
        states = {"joint_pos": torch.zeros(n, 2), "joint_vel": torch.zeros(n, 2)}
        sigmas = np.ones(n) * 0.5
        records = make_episode_records(states, sigmas, iteration=0)
        self.assertEqual(len(records), n)
        for r in records:
            self.assertAlmostEqual(r.sigma, 0.5)
            self.assertEqual(r.joint_pos_snap.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
