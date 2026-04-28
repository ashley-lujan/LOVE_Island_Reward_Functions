"""RewardUpdateManager: Signals 2 and 3 for LOVE Island.

Signal 2: After reward update, compute σ_after on saved terminal states.
          If σ spikes above baseline, retrain from scratch; else continue
          from checkpoint.

Signal 3: After reflection_window mini-iterations, compare μ_reflect to
          μ_at_update. If no improvement, retry Code Agent with full
          feedback context (up to max_reflection_retries).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .buffer import EvaluationBuffer, FeedbackRecord
from .sigma import compute_sigma_batch, load_reward_fn, load_terminal_states


def _build_reflection_prompt(
    feedback_history: List[FeedbackRecord],
    mu_at_update: float,
    mu_reflect: float,
    sigma_reflect: float,
    retry_count: int,
    improvement_threshold: float,
    reflection_window: int,
) -> str:
    history_text = "\n".join(
        f"  Attempt {i+1} (iter {fb.iteration_given}):\n"
        f"    Feedback: '{fb.feedback_text}'\n"
        f"    Change made: {fb.reward_update_summary}\n"
        f"    Expected: {fb.expected_direction}"
        for i, fb in enumerate(feedback_history)
    )
    return (
        f"REWARD REFLECTION - RETRY ATTEMPT {retry_count + 1}\n\n"
        f"FEEDBACK HISTORY:\n{history_text}\n\n"
        f"CURRENT RESULTS (after {reflection_window} training iterations):\n"
        f"  Mean reward before update: {mu_at_update:.4f}\n"
        f"  Mean reward after training: {mu_reflect:.4f}\n"
        f"  Ensemble disagreement: {sigma_reflect:.4f}\n"
        f"  Improvement: {mu_reflect - mu_at_update:.4f} "
        f"(required: >{improvement_threshold})\n\n"
        "The policy has not improved in the expected direction.\n"
        "Previous implementation approaches have not worked.\n"
        "Please try a fundamentally different approach to implementing "
        "this feedback in the reward function."
    )


class RewardUpdateManager:
    def __init__(
        self,
        eval_buffer: EvaluationBuffer,
        cfg,
    ):
        self.eval_buffer = eval_buffer
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Signal 2: post-update σ check
    # ------------------------------------------------------------------ #

    def check_retrain(
        self,
        sigma_before: float,
        new_reward_fn_paths: List[str],
        states_dirs: List[str],
    ) -> Tuple[bool, float]:
        """Compute σ_after on saved terminal states with the new reward functions.

        Returns (should_retrain, sigma_after).
          should_retrain=True  → full restart from random init
          should_retrain=False → continue from checkpoint (value fn adapts)
        """
        # Load terminal states from all island runs
        import torch

        all_jp, all_jv = [], []
        for sd in states_dirs:
            if not sd:
                continue
            states = load_terminal_states(sd)
            if states is not None:
                all_jp.append(states["joint_pos"])
                all_jv.append(states["joint_vel"])

        if not all_jp:
            logging.warning("[reward_manager] Signal 2: no terminal states — defaulting to no retrain")
            return False, sigma_before

        joint_pos = torch.cat(all_jp, dim=0)
        joint_vel = torch.cat(all_jv, dim=0)

        # Subsample to cfg.n_probe to keep this fast
        n = joint_pos.shape[0]
        if n > self.cfg.n_probe:
            idx = np.random.choice(n, size=self.cfg.n_probe, replace=False)
            joint_pos = joint_pos[idx]
            joint_vel = joint_vel[idx]

        reward_fns = [load_reward_fn(p) for p in new_reward_fn_paths]
        sigmas, _ = compute_sigma_batch(reward_fns, joint_pos, joint_vel)
        sigma_after = float(sigmas.mean())

        should_retrain = sigma_after > sigma_before + self.cfg.retrain_delta
        logging.info(
            f"[reward_manager] Signal 2: σ_before={sigma_before:.4f} "
            f"σ_after={sigma_after:.4f} retrain={should_retrain}"
        )
        return should_retrain, sigma_after

    # ------------------------------------------------------------------ #
    # Signal 3: behavioral reflection check
    # ------------------------------------------------------------------ #

    def check_reflection(
        self,
        episode_id: str,
        mu_at_update: float,
        tb_dirs: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """Check if training has improved after a reward update.

        Reads TensorBoard gpt_reward logs from the most recent training runs.
        Returns (passed, reflection_prompt_or_None).
        """
        mu_reflect, sigma_reflect = self._read_recent_metrics(tb_dirs)

        delta = mu_reflect - mu_at_update
        logging.info(
            f"[reward_manager] Signal 3: μ_update={mu_at_update:.4f} "
            f"μ_reflect={mu_reflect:.4f} Δ={delta:.4f}"
        )

        if delta > self.cfg.improvement_threshold:
            logging.info("[reward_manager] Signal 3: reflection PASSED")
            return True, None

        feedback_history = self.eval_buffer.get_context_for_reflection(episode_id)
        retry_count = len(feedback_history)

        reflection_prompt = _build_reflection_prompt(
            feedback_history=feedback_history,
            mu_at_update=mu_at_update,
            mu_reflect=mu_reflect,
            sigma_reflect=sigma_reflect,
            retry_count=retry_count,
            improvement_threshold=self.cfg.improvement_threshold,
            reflection_window=self.cfg.reflection_window,
        )

        logging.info(
            f"[reward_manager] Signal 3: reflection FAILED "
            f"(retry_count={retry_count})"
        )
        return False, reflection_prompt

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_recent_metrics(self, tb_dirs: List[str]) -> Tuple[float, float]:
        """Return (mean gpt_reward, std gpt_reward) across recent TensorBoard logs."""
        from utils.file_utils import load_tensorboard_logs

        final_means = []
        for tb_dir in tb_dirs:
            if not tb_dir or not os.path.isdir(tb_dir):
                continue
            try:
                logs = load_tensorboard_logs(tb_dir)
                vals = logs.get("gpt_reward", [])
                if vals:
                    final_means.append(float(vals[-1]))
            except Exception as e:
                logging.debug(f"[reward_manager] tb read failed for {tb_dir}: {e}")

        if not final_means:
            return 0.0, 0.0
        return float(np.mean(final_means)), float(np.std(final_means))
