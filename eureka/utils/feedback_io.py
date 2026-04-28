"""File-based feedback I/O for SLURM-hosted LOVE Island training jobs.

Directory layout under feedback_scratch_path:
    iter{N}/
        request.json   - written by training job; describes what feedback is needed
        WAITING        - sentinel flag; training polls until this file disappears
        response.json  - written atomically by submit_feedback.py
        videos/        - placeholder for future rollout videos (render stub)
"""

import json
import logging
import os
import time
from typing import Any, Dict


def _iter_dir(scratch_path: str, iter_idx: int) -> str:
    return os.path.join(scratch_path, f"iter{iter_idx}")


def write_feedback_request(iter_idx: int, scratch_path: str, context: Dict[str, Any]) -> None:
    """Write a feedback request to scratch and set the WAITING sentinel.

    Called by the training job before entering the polling loop. The context dict
    should contain whatever information is useful for the reviewer (metrics, best
    code path, island id, gif_paths, etc.).
    """
    iter_dir = _iter_dir(scratch_path, iter_idx)
    os.makedirs(iter_dir, exist_ok=True)

    request = {
        "iter": iter_idx,
        **context,
        "submit_cmd": (
            f"python submit_feedback.py "
            f"--scratch {scratch_path} --iter {iter_idx} "
            "--response \"<your feedback here>\""
        ),
    }

    request_path = os.path.join(iter_dir, "request.json")
    with open(request_path, "w") as f:
        json.dump(request, f, indent=2)

    # Write WAITING sentinel last so reviewers see a complete request.json
    waiting_path = os.path.join(iter_dir, "WAITING")
    with open(waiting_path, "w") as f:
        f.write(f"Waiting for human feedback for iteration {iter_idx}\n")

    logging.info(
        f"[feedback] Feedback requested for iter {iter_idx}.\n"
        f"  Review : {request_path}\n"
        f"  Submit : python submit_feedback.py "
        f"--scratch {scratch_path} --iter {iter_idx} --response \"...\""
    )


def poll_for_feedback(iter_idx: int, scratch_path: str, timeout_minutes: float) -> str:
    """Poll until the WAITING sentinel disappears, then return the feedback string.

    If timeout_minutes > 0 and the deadline passes, removes the sentinel and
    returns the no-feedback sentinel string so training can continue.
    If timeout_minutes == 0, blocks indefinitely (preserves local-run semantics).
    """
    iter_dir = _iter_dir(scratch_path, iter_idx)
    waiting_path = os.path.join(iter_dir, "WAITING")
    response_path = os.path.join(iter_dir, "response.json")

    deadline = time.time() + timeout_minutes * 60 if timeout_minutes > 0 else None
    poll_interval = 30  # seconds

    timeout_str = "none (blocking)" if deadline is None else f"{timeout_minutes:.0f} min"
    logging.info(f"[feedback] Polling for iter {iter_idx} feedback (timeout: {timeout_str})")

    while os.path.exists(waiting_path):
        if deadline is not None and time.time() >= deadline:
            logging.warning(
                f"[feedback] No feedback received within {timeout_minutes:.0f} min — "
                "resuming training with no human feedback."
            )
            try:
                os.remove(waiting_path)
            except OSError:
                pass
            return "<no human feedback provided>"
        time.sleep(poll_interval)

    if os.path.exists(response_path):
        try:
            with open(response_path, "r") as f:
                data = json.load(f)
            feedback = data.get("response", "").strip()
            if feedback:
                logging.info(
                    f"[feedback] Received feedback for iter {iter_idx}: "
                    f"{feedback[:120]}{'...' if len(feedback) > 120 else ''}"
                )
                return feedback
        except Exception as e:
            logging.warning(f"[feedback] Could not read response.json: {e}")

    logging.info("[feedback] WAITING removed but no response found — treating as no feedback.")
    return "<no human feedback provided>"
