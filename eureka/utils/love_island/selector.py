"""EpisodeSelector: chooses which episode(s) from the buffer to show humans."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from .buffer import EpisodeRecord, EvaluationBuffer


class EpisodeSelector:
    def __init__(self, max_sigma: float = 1.0, persistence_window: int = 3):
        self.max_sigma = max_sigma
        self.persistence_window = persistence_window

    def select(
        self,
        eval_buffer: EvaluationBuffer,
        feedback_type: str,
        pattern: str,
        reward_fns: Optional[List[Callable]] = None,
    ) -> List[EpisodeRecord]:
        candidates = eval_buffer.get_candidates_for_feedback(feedback_type)
        if not candidates:
            logging.warning("[selector] no candidates in interesting slots")
            return []

        if feedback_type == "corrective":
            return [self._select_corrective(candidates)]

        if feedback_type == "comparative":
            pair = self._select_comparative_pair(candidates, reward_fns)
            return pair if pair else [candidates[0]]

        if feedback_type == "guidance" and pattern == "stagnation":
            return [self._select_stagnation(candidates)]

        if feedback_type == "guidance" and pattern == "sudden_divergence":
            return self._select_divergence_pair(candidates, eval_buffer)

        if feedback_type == "explanatory":
            return [self._select_outlier(candidates)]

        # binary (default)
        return [self._select_highest_sigma(candidates)]

    # ------------------------------------------------------------------ #
    # Per-type selection
    # ------------------------------------------------------------------ #

    def _select_corrective(self, candidates: List[EpisodeRecord]) -> EpisodeRecord:
        """Highest σ + lowest mean reward + most query-history recurrence."""
        def score(r: EpisodeRecord) -> float:
            recurrence = len(r.query_history)
            return 0.5 * r.sigma + 0.3 * recurrence - 0.2 * r.sigma

        return max(candidates, key=lambda r: r.sigma)

    def _select_comparative_pair(
        self,
        candidates: List[EpisodeRecord],
        reward_fns: Optional[List[Callable]],
    ) -> Optional[List[EpisodeRecord]]:
        """Two episodes with most inverted island reward rankings."""
        if len(candidates) < 2:
            return None
        if reward_fns is None or all(fn is None for fn in reward_fns):
            # Fallback: highest and lowest σ
            sorted_c = sorted(candidates, key=lambda r: r.sigma, reverse=True)
            return [sorted_c[0], sorted_c[-1]]

        import torch
        from scipy.stats import spearmanr

        active_fns = [fn for fn in reward_fns if fn is not None]

        def island_rankings(record: EpisodeRecord) -> np.ndarray:
            rewards = []
            with torch.no_grad():
                for fn in active_fns:
                    try:
                        r, _ = fn(record.joint_pos_snap, record.joint_vel_snap)
                        rewards.append(float(r.mean()))
                    except Exception:
                        rewards.append(0.0)
            return np.array(rewards)

        # Score all candidate pairs by rank inversion
        best_pair = None
        best_inversion = -2.0
        subset = candidates[:min(10, len(candidates))]  # limit comparison cost

        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                r_i = island_rankings(subset[i])
                r_j = island_rankings(subset[j])
                if len(r_i) < 2 or len(r_j) < 2:
                    continue
                corr, _ = spearmanr(r_i, r_j)
                if corr < best_inversion or best_pair is None:
                    best_inversion = corr
                    best_pair = [subset[i], subset[j]]

        return best_pair

    def _select_stagnation(self, candidates: List[EpisodeRecord]) -> EpisodeRecord:
        """Representative episode from middle of the candidate list (median σ)."""
        sorted_c = sorted(candidates, key=lambda r: r.sigma)
        return sorted_c[len(sorted_c) // 2]

    def _select_divergence_pair(
        self,
        candidates: List[EpisodeRecord],
        eval_buffer: EvaluationBuffer,
    ) -> List[EpisodeRecord]:
        """One episode from before the spike, one from after."""
        if len(candidates) < 2:
            return candidates[:1]
        sorted_by_iter = sorted(candidates, key=lambda r: r.iteration_cached)
        return [sorted_by_iter[0], sorted_by_iter[-1]]

    def _select_outlier(self, candidates: List[EpisodeRecord]) -> EpisodeRecord:
        """Episode with lowest σ (confident ensemble) and highest reward proxy."""
        return min(candidates, key=lambda r: r.sigma)

    def _select_highest_sigma(self, candidates: List[EpisodeRecord]) -> EpisodeRecord:
        return max(candidates, key=lambda r: r.sigma)

    # ------------------------------------------------------------------ #
    # Scoring utility
    # ------------------------------------------------------------------ #

    def score_episode(
        self,
        record: EpisodeRecord,
        current_iteration: int,
        recurrence_weight: float = 0.6,
        sigma_weight: float = 0.4,
    ) -> float:
        age = current_iteration - record.iteration_cached
        recurrence = len(record.query_history)
        recurrence_score = age / max(self.persistence_window, 1)
        sigma_score = record.sigma / max(self.max_sigma, 1e-6)
        return recurrence_weight * recurrence_score + sigma_weight * sigma_score
