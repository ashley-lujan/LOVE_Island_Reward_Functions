"""FeedbackAgent: two LLM calls per feedback cycle.

Call 1 (populate_schema): generates a structured human-facing query.
Call 2 (contextualize_response): translates raw human input into a
  reward-update instruction for the Code Agent.

Reuses _get_client() and _chat_call() / _extract_json() from multi_agent.py.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .buffer import EpisodeRecord, FeedbackRecord


def _load_prompt(prompts_dir: str, name: str) -> str:
    path = os.path.join(prompts_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[feedback_agent] prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_episode_summary(episode: EpisodeRecord) -> str:
    jp = episode.joint_pos_snap
    jv = episode.joint_vel_snap
    n = jp.shape[0]
    return (
        f"Episode snapshot: {n} terminal state(s)\n"
        f"  joint_pos mean: {jp.mean().item():.4f}  std: {jp.std().item():.4f}\n"
        f"  joint_vel mean: {jv.mean().item():.4f}  std: {jv.std().item():.4f}\n"
        f"  σ (ensemble disagreement): {episode.sigma:.4f}\n"
        f"  query history entries: {len(episode.query_history)}"
    )


def _format_feedback_history(history: List[FeedbackRecord]) -> str:
    if not history:
        return "None"
    lines = []
    for i, fb in enumerate(history):
        lines.append(
            f"  [{i+1}] type={fb.feedback_type} iter={fb.iteration_given}\n"
            f"       feedback: {fb.feedback_text}\n"
            f"       change made: {fb.reward_update_summary}\n"
            f"       expected: {fb.expected_direction}"
        )
    return "\n".join(lines)


class FeedbackAgent:
    def __init__(
        self,
        prompts_dir: str,
        model: str = "gpt-5.4-mini",
        temperature: float = 0.3,
    ):
        self.prompts_dir = os.path.join(prompts_dir, "love_island")
        self.model = model
        self.temperature = temperature

    def populate_schema(
        self,
        feedback_type: str,
        episode_data: List[EpisodeRecord],
        sigma_breakdown: Dict[str, Any],
        similar_history: List[FeedbackRecord],
    ) -> Dict[str, Any]:
        """LLM Call 1: generate human-facing query schema."""
        from utils.multi_agent import _chat_call, _extract_json

        template_name = f"feedback_schema_{feedback_type}.txt"
        try:
            system_prompt = _load_prompt(self.prompts_dir, template_name)
        except FileNotFoundError:
            logging.warning(f"[feedback_agent] missing template {template_name}, using binary")
            system_prompt = _load_prompt(self.prompts_dir, "feedback_schema_binary.txt")

        episode_summary = "\n\n".join(_format_episode_summary(ep) for ep in episode_data)
        history_text = _format_feedback_history(similar_history)

        user_content = (
            f"FEEDBACK TYPE: {feedback_type}\n\n"
            f"EPISODE SUMMARY:\n{episode_summary}\n\n"
            f"DISAGREEMENT SIGNAL:\n{json.dumps(sigma_breakdown, indent=2)}\n\n"
            f"SIMILAR PAST FEEDBACK:\n{history_text}\n\n"
            "Please populate the query schema as JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = _chat_call(self.model, messages, self.temperature, n=1)
            raw = response["choices"][0]["message"]["content"]
            schema = _extract_json(raw)
        except Exception as e:
            logging.error(f"[feedback_agent] populate_schema failed: {e}")
            schema = {}

        # Ensure required keys are present
        schema.setdefault("question_text", f"Please review the following {feedback_type} episode:")
        schema.setdefault("input_type", "text" if feedback_type in ("corrective", "guidance", "explanatory") else "binary")
        schema.setdefault("options", ["Good", "Bad"] if schema["input_type"] == "binary" else [])
        schema.setdefault("context_for_human", f"This query was triggered by ensemble disagreement signal: σ={sigma_breakdown.get('mean_sigma_overall', 0):.4f}")
        schema["feedback_type"] = feedback_type
        schema["episode_summary"] = episode_summary
        return schema

    def contextualize_response(
        self,
        raw_feedback: str,
        feedback_type: str,
        episode_data: List[EpisodeRecord],
        sigma_breakdown: Dict[str, Any],
        query_history: List[FeedbackRecord],
    ) -> Dict[str, Any]:
        """LLM Call 2: translate human response into a reward-update instruction."""
        from utils.multi_agent import _chat_call, _extract_json

        try:
            system_prompt = _load_prompt(self.prompts_dir, "contextualize_response.txt")
        except FileNotFoundError:
            system_prompt = (
                "You are a reward-function engineering assistant. "
                "Given a human's feedback on an RL episode and the context of why the query "
                "was triggered, produce a structured JSON object with reward update instructions."
            )

        episode_summary = "\n\n".join(_format_episode_summary(ep) for ep in episode_data)
        history_text = _format_feedback_history(query_history)

        user_content = (
            f"HUMAN FEEDBACK (raw): {raw_feedback}\n\n"
            f"FEEDBACK TYPE: {feedback_type}\n\n"
            f"EPISODE SUMMARY:\n{episode_summary}\n\n"
            f"DISAGREEMENT SIGNAL:\n{json.dumps(sigma_breakdown, indent=2)}\n\n"
            f"QUERY HISTORY (same episode):\n{history_text}\n\n"
            "Produce a JSON object with keys: "
            "feedback_summary, expected_direction, relates_to_previous (bool), "
            "previous_feedback_context (str or null), reward_update_instruction."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = _chat_call(self.model, messages, self.temperature, n=1)
            raw = response["choices"][0]["message"]["content"]
            result = _extract_json(raw)
        except Exception as e:
            logging.error(f"[feedback_agent] contextualize_response failed: {e}")
            result = {}

        result.setdefault("feedback_summary", raw_feedback)
        result.setdefault("expected_direction", "improve task performance")
        result.setdefault("relates_to_previous", bool(query_history))
        result.setdefault("previous_feedback_context", None)
        result.setdefault("reward_update_instruction", raw_feedback)
        return result
