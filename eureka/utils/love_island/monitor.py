"""DisagreementMonitor: pattern detection and query triggering for LOVE Island."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .buffer import EvaluationBuffer, EpisodeRecord, make_episode_records
from .sigma import (
    compute_sigma_batch,
    load_reward_fn,
    load_terminal_states,
    tb_sigma_prefilter,
)


class DisagreementMonitor:
    def __init__(
        self,
        eval_buffer: EvaluationBuffer,
        cfg,
    ):
        self.eval_buffer = eval_buffer
        self.cfg = cfg

        self._mini_iteration: int = 0
        self._sigma_history: List[float] = []
        self._mean_reward_history: List[float] = []
        self._pattern_history: List[Optional[str]] = []
        self._persistence_counter: int = 0
        self._current_pattern: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Per-iteration σ computation
    # ------------------------------------------------------------------ #

    def compute_sigma_for_iteration(
        self,
        reward_fn_paths: List[str],
        states_dirs: List[str],
        tb_dirs: Optional[List[str]] = None,
    ) -> Dict:
        """Load terminal states and reward fns; compute σ; update buffer.

        Returns summary dict for logging.
        """
        # Cheap TensorBoard pre-filter: skip .pt loading if σ_tb is trivially low
        if tb_dirs:
            sigma_tb = tb_sigma_prefilter(tb_dirs, low_threshold=self.cfg.low_threshold)
            if sigma_tb < self.cfg.low_threshold:
                logging.info(
                    f"[monitor] iter {self._mini_iteration}: σ_tb={sigma_tb:.4f} "
                    f"below threshold — skipping state loading"
                )
                self.eval_buffer.reset_for_new_mini_iteration(self._mini_iteration)
                self._record_sigma(0.0, 0.0)
                return {"mean_sigma": 0.0, "mean_reward": 0.0, "n_states": 0, "skipped": True}

        # Load all terminal states across all island runs
        # Load all terminal states across all island runs
        all_states: List[Dict] = []
        for states_dir in states_dirs:
            if not states_dir:
                continue
            states = load_terminal_states(states_dir)
            if states is not None:
                all_states.append(states)

        if not all_states:
            logging.info(f"[monitor] iter {self._mini_iteration}: no terminal states found")
            self.eval_buffer.reset_for_new_mini_iteration(self._mini_iteration)
            self._record_sigma(0.0, 0.0)
            return {"mean_sigma": 0.0, "mean_reward": 0.0, "n_states": 0, "skipped": False}

        import torch
        merged: Dict[str, list] = {}
        for s in all_states:
            for k, v in s.items():
                merged.setdefault(k, []).append(v)
        full_state_dict = {k: torch.cat(v, dim=0) for k, v in merged.items()}

        joint_pos = full_state_dict["joint_pos"]
        joint_vel = full_state_dict["joint_vel"]

        reward_fns = [load_reward_fn(p) for p in reward_fn_paths]
        active = sum(1 for fn in reward_fns if fn is not None)
        logging.info(f"[monitor] iter {self._mini_iteration}: {active}/{len(reward_fns)} reward fns loaded")
        if active < 2:
            logging.warning("[monitor] fewer than 2 reward fns — σ will be 0")

        sigmas, mean_rewards = compute_sigma_batch(
            reward_fns, joint_pos, joint_vel, state_dict=full_state_dict
        )

        mean_sigma = float(sigmas.mean())
        mean_reward = float(mean_rewards.mean())
        self._record_sigma(mean_sigma, mean_reward)

        # Build episode records and populate buffer
        records = make_episode_records(
            full_state_dict,   # was {"joint_pos": joint_pos, "joint_vel": joint_vel}
            sigmas,
            self._mini_iteration,
        )
        self.eval_buffer.refresh_random_slots(records)
        for rec in records:
            self.eval_buffer.update_interesting_slots(rec)

        logging.info(
            f"[monitor] iter {self._mini_iteration}: "
            f"mean_σ={mean_sigma:.4f} mean_μ={mean_reward:.4f} n_states={len(sigmas)}"
        )
        return {
            "mean_sigma": mean_sigma,
            "mean_reward": mean_reward,
            "n_states": int(len(sigmas)),
            "skipped": False,
        }

    # ------------------------------------------------------------------ #
    # Query decision
    # ------------------------------------------------------------------ #

    def evaluate_mini_iteration(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Return (should_query, feedback_type, pattern_name).

        Warmup guard: suppresses all queries for the first
        cfg.warmup_iterations iterations. Buffer accumulation still happens.
        """
        self._mini_iteration += 1

        if self._mini_iteration <= self.cfg.warmup_iterations:
            logging.info(
                f"[monitor] iter {self._mini_iteration}: warmup "
                f"({self._mini_iteration}/{self.cfg.warmup_iterations})"
            )
            return False, None, None

        mean_sigma = self.eval_buffer.get_mean_sigma()
        mean_reward = self._mean_reward_history[-1] if self._mean_reward_history else 0.0

        pattern = self._detect_pattern(mean_sigma, mean_reward)

        if pattern == self._current_pattern and pattern is not None:
            self._persistence_counter += 1
        else:
            self._persistence_counter = 1
            self._current_pattern = pattern

        self._pattern_history.append(pattern)

        logging.info(
            f"[monitor] iter {self._mini_iteration}: "
            f"pattern={pattern} persistence={self._persistence_counter}"
        )

        if (
            pattern is not None
            and self._persistence_counter >= self.cfg.persistence_threshold
        ):
            feedback_type = self._map_pattern_to_feedback(pattern)
            self._persistence_counter = 0
            logging.info(
                f"[monitor] QUERY TRIGGERED: pattern={pattern} → feedback_type={feedback_type}"
            )
            return True, feedback_type, pattern

        return False, None, None

    def get_sigma_breakdown(self) -> Dict:
        breakdown = self.eval_buffer.get_sigma_breakdown()
        breakdown["sigma_history"] = list(self._sigma_history[-10:])
        breakdown["current_pattern"] = self._current_pattern
        breakdown["persistence_counter"] = self._persistence_counter
        return breakdown

    # ------------------------------------------------------------------ #
    # Pattern detection
    # ------------------------------------------------------------------ #

    def _detect_pattern(self, mean_sigma: float, mean_reward: float) -> Optional[str]:
        hi = self.cfg.high_threshold
        lo = self.cfg.low_threshold
        bad = self.cfg.bad_threshold
        good = self.cfg.good_threshold

        if mean_sigma > hi and mean_reward < bad:
            return "high_disagreement_bad_performance"

        if mean_sigma > hi:
            return "high_disagreement_ambiguous"

        if self._detect_spike(mean_sigma):
            return "sudden_divergence"

        if self._detect_stagnation():
            return "stagnation"

        if mean_sigma < lo and self._is_positive_outlier(mean_reward):
            return "confident_positive_outlier"

        return None

    def _map_pattern_to_feedback(self, pattern: str) -> str:
        mapping = {
            "high_disagreement_bad_performance": "corrective",
            "high_disagreement_ambiguous": "comparative",
            "sudden_divergence": "guidance",
            "stagnation": "guidance",
            "confident_positive_outlier": "explanatory",
        }
        return mapping.get(pattern, "binary")

    def _detect_spike(self, current_sigma: float) -> bool:
        window = self.cfg.spike_window
        k = self.cfg.spike_k
        history = self._sigma_history[-(window + 1) : -1]
        if len(history) < 3:
            return False
        mu = np.mean(history)
        std = np.std(history)
        return current_sigma > mu + k * std

    def _detect_stagnation(self) -> bool:
        window = self.cfg.stagnation_window
        if len(self._sigma_history) < window or len(self._mean_reward_history) < window:
            return False
        recent_sigma = self._sigma_history[-window:]
        recent_reward = self._mean_reward_history[-window:]
        sigma_low = float(np.mean(recent_sigma)) < self.cfg.low_threshold
        reward_slope = np.polyfit(range(window), recent_reward, 1)[0]
        return sigma_low and abs(reward_slope) < 0.001

    def _is_positive_outlier(self, mean_reward: float) -> bool:
        if len(self._mean_reward_history) < 5:
            return False
        hist = self._mean_reward_history[-10:]
        mu = np.mean(hist)
        std = np.std(hist)
        return mean_reward > mu + 0.2 and std > 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _record_sigma(self, mean_sigma: float, mean_reward: float) -> None:
        self._sigma_history.append(mean_sigma)
        self._mean_reward_history.append(mean_reward)
