"""EvaluationBuffer and supporting dataclasses for LOVE Island.

Two-partition buffer:
  Random slots  (25%): refreshed every mini-iteration with current-batch states.
  Interesting slots (75%): high-σ episodes, persist across iterations.

Random slots pull the mean σ down when training improves, preventing
over-querying on stale interesting-slot signal.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class FeedbackRecord:
    feedback_type: str
    feedback_text: str
    iteration_given: int
    reward_update_summary: str
    expected_direction: str
    reflection_count: int = 0


@dataclass
class EpisodeRecord:
    episode_id: str
    joint_pos_snap: torch.Tensor   # CPU tensor [K, J]
    joint_vel_snap: torch.Tensor   # CPU tensor [K, J]
    sigma: float
    iteration_cached: int
    was_queried: bool = False
    query_history: List[FeedbackRecord] = field(default_factory=list)
    sigma_after_update: Optional[float] = None
    performance_after: Optional[float] = None
    reflection_triggered: bool = False
    traj_key: Optional[str] = None  # "{step}_{env_idx}" for trajectory file lookup


class EvaluationBuffer:
    def __init__(
        self,
        capacity: int = 20,
        random_ratio: float = 0.25,
        staleness_penalty: float = 0.02,
        high_sigma_threshold: float = 0.2,
    ):
        self.capacity = capacity
        self.n_random = max(1, int(capacity * random_ratio))
        self.n_interesting = capacity - self.n_random
        self.staleness_penalty = staleness_penalty
        self.high_sigma_threshold = high_sigma_threshold

        self._random_slots: List[EpisodeRecord] = []
        self._interesting_slots: List[EpisodeRecord] = []
        self._feedback_index: Dict[str, EpisodeRecord] = {}
        self._current_iteration: int = 0

    # ------------------------------------------------------------------ #
    # Slot management
    # ------------------------------------------------------------------ #

    def refresh_random_slots(
        self,
        records: List[EpisodeRecord],
    ) -> None:
        """Replace all random slots with a sample from the current batch."""
        if not records:
            return
        n = min(self.n_random, len(records))
        indices = np.random.choice(len(records), size=n, replace=False)
        self._random_slots = [records[i] for i in indices]

    def update_interesting_slots(self, record: EpisodeRecord) -> None:
        """Try to add record to interesting slots using staleness-adjusted σ."""
        if record.sigma < self.high_sigma_threshold:
            return

        def adjusted_sigma(r: EpisodeRecord) -> float:
            age = self._current_iteration - r.iteration_cached
            return r.sigma - self.staleness_penalty * age

        if len(self._interesting_slots) < self.n_interesting:
            self._interesting_slots.append(record)
            self._feedback_index[record.episode_id] = record
            return

        worst_idx = min(
            range(len(self._interesting_slots)),
            key=lambda i: adjusted_sigma(self._interesting_slots[i]),
        )
        if record.sigma > adjusted_sigma(self._interesting_slots[worst_idx]):
            old = self._interesting_slots[worst_idx]
            if old.episode_id in self._feedback_index and not old.was_queried:
                del self._feedback_index[old.episode_id]
            self._interesting_slots[worst_idx] = record
            self._feedback_index[record.episode_id] = record

    # ------------------------------------------------------------------ #
    # Sigma accessors
    # ------------------------------------------------------------------ #

    def get_mean_sigma(self) -> float:
        all_slots = self._random_slots + self._interesting_slots
        if not all_slots:
            return 0.0
        return float(np.mean([r.sigma for r in all_slots]))

    def get_sigma_breakdown(self) -> Dict[str, float]:
        rand_sigmas = [r.sigma for r in self._random_slots]
        int_sigmas = [r.sigma for r in self._interesting_slots]
        return {
            "mean_sigma_random": float(np.mean(rand_sigmas)) if rand_sigmas else 0.0,
            "mean_sigma_interesting": float(np.mean(int_sigmas)) if int_sigmas else 0.0,
            "mean_sigma_overall": self.get_mean_sigma(),
            "n_random": len(self._random_slots),
            "n_interesting": len(self._interesting_slots),
        }

    # ------------------------------------------------------------------ #
    # Feedback tracking
    # ------------------------------------------------------------------ #

    def mark_episode_queried(
        self,
        episode_id: str,
        feedback_type: str,
        feedback_text: str,
        iteration: int,
        reward_update_summary: str,
        expected_direction: str,
    ) -> None:
        record = self._feedback_index.get(episode_id)
        if record is None:
            return
        fb = FeedbackRecord(
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            iteration_given=iteration,
            reward_update_summary=reward_update_summary,
            expected_direction=expected_direction,
            reflection_count=len([q for q in record.query_history if q.feedback_type == feedback_type]),
        )
        record.query_history.append(fb)
        record.was_queried = True

    def get_context_for_reflection(self, episode_id: str) -> List[FeedbackRecord]:
        record = self._feedback_index.get(episode_id)
        if record is None:
            return []
        return list(record.query_history)

    def has_been_queried(self, episode_id: str) -> bool:
        record = self._feedback_index.get(episode_id)
        return record is not None and record.was_queried

    def get_similar_feedback_history(self, episode: EpisodeRecord) -> List[FeedbackRecord]:
        """Return feedback records from previously queried episodes."""
        history = []
        for record in self._feedback_index.values():
            if record.was_queried and record.episode_id != episode.episode_id:
                history.extend(record.query_history)
        return history

    # ------------------------------------------------------------------ #
    # Candidate selection
    # ------------------------------------------------------------------ #

    def get_candidates_for_feedback(self, feedback_type: str) -> List[EpisodeRecord]:
        """Interesting slots only — random slots are never shown to humans."""
        candidates = [r for r in self._interesting_slots if not self.has_been_queried(r.episode_id)]
        if not candidates:
            candidates = list(self._interesting_slots)
        candidates.sort(key=lambda r: r.sigma, reverse=True)
        return candidates

    # ------------------------------------------------------------------ #
    # Iteration housekeeping
    # ------------------------------------------------------------------ #

    def reset_for_new_mini_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration
        self._random_slots = []


def make_episode_records(
    states: Dict[str, torch.Tensor],
    sigmas: np.ndarray,
    iteration: int,
    traj_keys: Optional[List[str]] = None,
) -> List[EpisodeRecord]:
    """Create one EpisodeRecord per terminal state from a states dict."""
    records = []
    n = states["joint_pos"].shape[0]
    for i in range(n):
        tk = traj_keys[i] if (traj_keys is not None and i < len(traj_keys)) else None
        records.append(
            EpisodeRecord(
                episode_id=str(uuid.uuid4()),
                joint_pos_snap=states["joint_pos"][i : i + 1],
                joint_vel_snap=states["joint_vel"][i : i + 1],
                sigma=float(sigmas[i]),
                iteration_cached=iteration,
                traj_key=tk,
            )
        )
    return records
