"""Multi-agent LLM feedback pipeline for Eureka.

Three-agent chain (Feedback Interpreter -> Prompt Architect -> Code Agent) used as
an opt-in alternative to Eureka's single-call feedback loop. Activated via
`cfg.feedback_mode = 'multi'` in config.yaml.
"""

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import openai


_REQUIRED_ARCHITECT_SUBSTRINGS = (
    "compute_reward",
    "@torch.jit.script",
    "Tuple[torch.Tensor, Dict[str, torch.Tensor]]",
    "temperature",
    "```python",
)


def load_multi_agent_prompts(prompt_dir: str) -> Dict[str, str]:
    """Read the five multi-agent prompt templates into memory."""
    names = [
        "interpreter_system",
        "interpreter_user",
        "architect_system",
        "architect_user",
        "multi_code_system",
    ]
    prompts = {}
    for name in names:
        path = os.path.join(prompt_dir, f"{name}.txt")
        with open(path, "r", encoding="utf-8") as f:
            prompts[name] = f.read()
    return prompts


def prompt_human_feedback(iter_idx: int, cfg, context: Optional[Dict[str, Any]] = None) -> str:
    """Collect human feedback between iterations.

    When running under SLURM or when feedback_scratch_path is configured, uses
    file-based polling so the job doesn't need an interactive TTY. Otherwise falls
    back to the original blocking stdin path (local interactive runs).
    """
    is_slurm = bool(os.getenv("SLURM_JOB_ID"))
    scratch = getattr(cfg, "feedback_scratch_path", "") or ""
    use_file_based = is_slurm or bool(scratch)

    if use_file_based:
        if not scratch:
            # No scratch path configured — default to a subfolder of the current Hydra output dir
            scratch = os.path.join(os.getcwd(), "feedback")
        from utils.feedback_io import write_feedback_request, poll_for_feedback
        write_feedback_request(iter_idx, scratch, context or {})
        timeout = float(getattr(cfg, "human_feedback_timeout", 0))
        return poll_for_feedback(iter_idx, scratch, timeout)

    # Original interactive stdin path (local runs without scratch configured)
    print("")
    print(f"=============== Human feedback for iteration {iter_idx} ===============")
    print("Enter natural-language feedback about the previous reward. Finish with a")
    print("blank line. Leave entirely empty to skip (runs as if no feedback).")
    print("")
    lines: List[str] = []
    try:
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
    except EOFError:
        pass
    text = "\n".join(lines).strip()
    if not text:
        return "<no human feedback provided>"
    return text


def _chat_call(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    n: int = 1,
    max_retries: int = 10,
):
    """Call openai.ChatCompletion with the same chunked-retry logic as eureka.py."""
    chunk_size = n
    total_samples = 0
    all_choices: List[Any] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    last_response: Optional[Dict[str, Any]] = None

    while total_samples < n:
        response = None
        for attempt in range(1000):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=chunk_size,
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= max_retries:
                    chunk_size = max(int(chunk_size / 2), 1)
                logging.info(f"[multi-agent] Attempt {attempt+1} failed: {e}")
                time.sleep(1)
        if response is None:
            raise RuntimeError("multi-agent chat call failed after repeated retries")
        all_choices.extend(response["choices"])
        total_prompt_tokens = response["usage"]["prompt_tokens"]
        total_completion_tokens += response["usage"]["completion_tokens"]
        total_tokens += response["usage"]["total_tokens"]
        last_response = response

    merged = dict(last_response) if last_response is not None else {}
    merged["choices"] = all_choices
    merged["usage"] = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }
    return merged


def _extract_json(raw_text: str) -> Dict[str, Any]:
    """Tolerant JSON extractor. Tries direct parse, then regex for first {...} block."""
    if not raw_text:
        return {}
    stripped = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence is not None:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match is not None:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def _write_trace(trace_dir: Optional[str], iter_idx: int, agent_name: str, payload: Dict[str, Any]) -> None:
    if trace_dir is None:
        return
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"iter{iter_idx}_{agent_name}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        logging.warning(f"[multi-agent] Failed to write trace {path}: {e}")


def run_feedback_interpreter(
    cfg,
    prompts: Dict[str, str],
    *,
    human_feedback: str,
    metrics_block: str,
    current_reward_code: str,
    task_obs_code_string: str,
    task_description: str,
    trace_dir: Optional[str],
    iter_idx: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call Agent 1. Returns (parsed_json, trace_metadata)."""
    system_msg = prompts["interpreter_system"]
    user_msg = prompts["interpreter_user"].format(
        task_description=task_description,
        task_obs_code_string=task_obs_code_string,
        current_reward_code=current_reward_code,
        metrics_block=metrics_block,
        human_feedback=human_feedback,
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    t0 = time.time()
    response = _chat_call(
        model=cfg.multi_agent_model,
        messages=messages,
        temperature=cfg.interpreter_temperature,
        n=1,
    )
    raw = response["choices"][0]["message"]["content"]
    parsed = _extract_json(raw)
    repair_retry_used = False

    if not parsed:
        repair_retry_used = True
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your previous response could not be parsed as JSON. "
                    "Re-emit a single JSON object matching the schema in your "
                    "system instructions. No prose, no fences, no commentary."
                ),
            },
        ]
        response2 = _chat_call(
            model=cfg.multi_agent_model,
            messages=repair_messages,
            temperature=cfg.interpreter_temperature,
            n=1,
        )
        raw = response2["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)
        response["usage"]["prompt_tokens"] += response2["usage"]["prompt_tokens"]
        response["usage"]["completion_tokens"] += response2["usage"]["completion_tokens"]
        response["usage"]["total_tokens"] += response2["usage"]["total_tokens"]

    wall = time.time() - t0
    parse_ok = bool(parsed)

    trace = {
        "iter": iter_idx,
        "agent": "interpreter",
        "model": cfg.multi_agent_model,
        "temperature": cfg.interpreter_temperature,
        "wall_clock_seconds": wall,
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "total_tokens": response["usage"]["total_tokens"],
        "input_messages": messages,
        "raw_response_text": raw,
        "parsed_output": parsed,
        "parse_ok": parse_ok,
        "repair_retry_used": repair_retry_used,
        "fallback_to_single": False,
    }
    _write_trace(trace_dir, iter_idx, "interpreter", trace)
    return parsed, trace


def run_prompt_architect(
    cfg,
    prompts: Dict[str, str],
    *,
    interpreter_json: Dict[str, Any],
    current_reward_code: str,
    task_obs_code_string: str,
    reward_signature: str,
    task_description: str,
    code_output_tip: str,
    trace_dir: Optional[str],
    iter_idx: int,
) -> Tuple[str, Dict[str, Any]]:
    """Call Agent 2. Returns (master_prompt_text, trace_metadata)."""
    system_msg = prompts["architect_system"]
    user_msg = prompts["architect_user"].format(
        interpreter_json=json.dumps(interpreter_json, indent=2),
        current_reward_code=current_reward_code,
        task_obs_code_string=task_obs_code_string,
        task_reward_signature_string=reward_signature,
        task_description=task_description,
        code_output_tip=code_output_tip,
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    t0 = time.time()
    response = _chat_call(
        model=cfg.multi_agent_model,
        messages=messages,
        temperature=cfg.architect_temperature,
        n=1,
    )
    raw = response["choices"][0]["message"]["content"]
    repair_retry_used = False
    missing = [s for s in _REQUIRED_ARCHITECT_SUBSTRINGS if s not in raw]

    if missing:
        repair_retry_used = True
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your previous output is missing required substrings: "
                    f"{missing}. Re-emit the master prompt following the "
                    "structure and rules in your system instructions. It must "
                    "contain all required literal substrings and use the "
                    "required section headings verbatim."
                ),
            },
        ]
        response2 = _chat_call(
            model=cfg.multi_agent_model,
            messages=repair_messages,
            temperature=cfg.architect_temperature,
            n=1,
        )
        raw = response2["choices"][0]["message"]["content"]
        missing = [s for s in _REQUIRED_ARCHITECT_SUBSTRINGS if s not in raw]
        response["usage"]["prompt_tokens"] += response2["usage"]["prompt_tokens"]
        response["usage"]["completion_tokens"] += response2["usage"]["completion_tokens"]
        response["usage"]["total_tokens"] += response2["usage"]["total_tokens"]

    wall = time.time() - t0
    validation_ok = len(missing) == 0

    trace = {
        "iter": iter_idx,
        "agent": "architect",
        "model": cfg.multi_agent_model,
        "temperature": cfg.architect_temperature,
        "wall_clock_seconds": wall,
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "total_tokens": response["usage"]["total_tokens"],
        "input_messages": messages,
        "raw_response_text": raw,
        "parsed_output": {"master_prompt": raw},
        "parse_ok": validation_ok,
        "missing_substrings": missing,
        "repair_retry_used": repair_retry_used,
        "fallback_to_single": False,
    }
    _write_trace(trace_dir, iter_idx, "architect", trace)
    return raw, trace


def run_code_agent(
    cfg,
    prompts: Dict[str, str],
    *,
    master_prompt: str,
    reward_signature: str,
    trace_dir: Optional[str],
    iter_idx: int,
    n_samples: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call Agent 3. Returns (response_dict_like_openai, trace_metadata)."""
    system_msg = prompts["multi_code_system"].format(
        task_reward_signature_string=reward_signature
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": master_prompt},
    ]

    t0 = time.time()
    response = _chat_call(
        model=cfg.multi_agent_model,
        messages=messages,
        temperature=cfg.code_agent_temperature,
        n=n_samples,
    )
    wall = time.time() - t0

    raw_samples = [c["message"]["content"] for c in response["choices"]]
    trace = {
        "iter": iter_idx,
        "agent": "code_agent",
        "model": cfg.multi_agent_model,
        "temperature": cfg.code_agent_temperature,
        "wall_clock_seconds": wall,
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "total_tokens": response["usage"]["total_tokens"],
        "input_messages": messages,
        "raw_response_text": raw_samples,
        "parsed_output": {"n_samples": len(raw_samples)},
        "parse_ok": True,
        "repair_retry_used": False,
        "fallback_to_single": False,
    }
    _write_trace(trace_dir, iter_idx, "code_agent", trace)
    return response, trace


def _fallback_single_call(
    cfg,
    *,
    model: str,
    messages: List[Dict[str, str]],
    n_samples: int,
) -> Dict[str, Any]:
    """Fallback path that mirrors eureka.py's single-call behavior."""
    return _chat_call(
        model=model,
        messages=messages,
        temperature=cfg.temperature,
        n=n_samples,
    )


def run_multi_agent_iteration(
    cfg,
    prompts: Dict[str, str],
    *,
    human_feedback: str,
    metrics_block: str,
    current_reward_code: str,
    task_obs_code_string: str,
    task_description: str,
    reward_signature: str,
    code_output_tip: str,
    iter_idx: int,
    trace_dir: Optional[str],
    n_samples: int,
    fallback_messages: Optional[List[Dict[str, str]]] = None,
    fallback_model: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the three-agent chain for one iteration.

    Returns a response dict shaped like openai.ChatCompletion.create output and
    a summary metadata dict suitable for appending to multi_agent_summary.json.
    """
    summary: Dict[str, Any] = {
        "iter": iter_idx,
        "fell_back_to_single": False,
        "fallback_reason": None,
    }

    try:
        interp_parsed, interp_trace = run_feedback_interpreter(
            cfg,
            prompts,
            human_feedback=human_feedback,
            metrics_block=metrics_block,
            current_reward_code=current_reward_code,
            task_obs_code_string=task_obs_code_string,
            task_description=task_description,
            trace_dir=trace_dir,
            iter_idx=iter_idx,
        )
        summary["interpreter"] = {
            "parse_ok": interp_trace["parse_ok"],
            "repair_retry_used": interp_trace["repair_retry_used"],
            "total_tokens": interp_trace["total_tokens"],
            "wall_clock_seconds": interp_trace["wall_clock_seconds"],
        }

        if not interp_trace["parse_ok"]:
            raise RuntimeError("interpreter output could not be parsed as JSON after repair retry")

        logging.info(
            f"Iteration {iter_idx}: Interpreter Tokens: {interp_trace['total_tokens']}"
        )

        architect_text, architect_trace = run_prompt_architect(
            cfg,
            prompts,
            interpreter_json=interp_parsed,
            current_reward_code=current_reward_code,
            task_obs_code_string=task_obs_code_string,
            reward_signature=reward_signature,
            task_description=task_description,
            code_output_tip=code_output_tip,
            trace_dir=trace_dir,
            iter_idx=iter_idx,
        )
        summary["architect"] = {
            "parse_ok": architect_trace["parse_ok"],
            "missing_substrings": architect_trace.get("missing_substrings", []),
            "repair_retry_used": architect_trace["repair_retry_used"],
            "total_tokens": architect_trace["total_tokens"],
            "wall_clock_seconds": architect_trace["wall_clock_seconds"],
        }

        if not architect_trace["parse_ok"]:
            raise RuntimeError(
                f"architect output missing required substrings: {architect_trace.get('missing_substrings')}"
            )

        logging.info(
            f"Iteration {iter_idx}: Architect Tokens: {architect_trace['total_tokens']}"
        )

        code_response, code_trace = run_code_agent(
            cfg,
            prompts,
            master_prompt=architect_text,
            reward_signature=reward_signature,
            trace_dir=trace_dir,
            iter_idx=iter_idx,
            n_samples=n_samples,
        )
        summary["code_agent"] = {
            "total_tokens": code_trace["total_tokens"],
            "wall_clock_seconds": code_trace["wall_clock_seconds"],
            "n_samples": len(code_response["choices"]),
        }
        logging.info(
            f"Iteration {iter_idx}: Code Agent Tokens: {code_trace['total_tokens']}"
        )

        summary["total_tokens"] = (
            summary["interpreter"]["total_tokens"]
            + summary["architect"]["total_tokens"]
            + summary["code_agent"]["total_tokens"]
        )
        summary["total_wall_clock_seconds"] = (
            summary["interpreter"]["wall_clock_seconds"]
            + summary["architect"]["wall_clock_seconds"]
            + summary["code_agent"]["wall_clock_seconds"]
        )
        return code_response, summary

    except Exception as e:
        logging.warning(f"[multi-agent] Pipeline failed at iter {iter_idx}: {e}")
        summary["fell_back_to_single"] = True
        summary["fallback_reason"] = str(e)
        if cfg.multi_agent_strict_json and fallback_messages is not None:
            fallback_response = _fallback_single_call(
                cfg,
                model=fallback_model or cfg.model,
                messages=fallback_messages,
                n_samples=n_samples,
            )
            _write_trace(
                trace_dir,
                iter_idx,
                "fallback_single",
                {
                    "iter": iter_idx,
                    "agent": "fallback_single",
                    "reason": str(e),
                    "model": fallback_model or cfg.model,
                    "total_tokens": fallback_response["usage"]["total_tokens"],
                    "input_messages": fallback_messages,
                },
            )
            return fallback_response, summary
        raise
