#!/usr/bin/env python3
"""Submit human feedback to a running LOVE Island SLURM training job.

Usage
-----
# List all pending feedback requests:
    python submit_feedback.py --scratch /scratch/general/vast/<user>/love_island --list

# Submit feedback for a specific iteration:
    python submit_feedback.py \\
        --scratch /scratch/general/vast/<user>/love_island \\
        --iter 2 \\
        --response "The hand is rotating too fast — penalize angular velocity more strongly"
"""

import argparse
import json
import os
import sys


def _iter_dir(scratch: str, iter_idx: int) -> str:
    return os.path.join(scratch, f"iter{iter_idx}")


def list_pending(scratch: str) -> None:
    if not os.path.isdir(scratch):
        print(f"Scratch directory not found: {scratch}")
        return
    found = False
    for entry in sorted(os.listdir(scratch)):
        iter_dir = os.path.join(scratch, entry)
        if not os.path.isdir(iter_dir):
            continue
        waiting = os.path.join(iter_dir, "WAITING")
        request_path = os.path.join(iter_dir, "request.json")
        if os.path.isfile(waiting):
            found = True
            print(f"\n[PENDING] {entry}")
            if os.path.isfile(request_path):
                try:
                    with open(request_path) as f:
                        req = json.load(f)
                    metrics = req.get("metrics") or "(none)"
                    videos = req.get("video_paths") or []
                    iter_idx = req.get("iter", "?")
                    print(f"  Metrics  : {metrics[:200] if metrics else '(none)'}")
                    print(f"  Videos   : {videos if videos else '(none yet)'}")
                    print(f"  Island   : {req.get('island_id', '?')}")
                    print(f"  Submit   : python submit_feedback.py --scratch {scratch} --iter {iter_idx} --response \"...\"")
                except Exception as e:
                    print(f"  (could not read request.json: {e})")
    if not found:
        print("No pending feedback requests found in:", scratch)


def submit(scratch: str, iter_idx: int, response_text: str) -> None:
    iter_dir = _iter_dir(scratch, iter_idx)
    waiting_path = os.path.join(iter_dir, "WAITING")
    response_path = os.path.join(iter_dir, "response.json")
    tmp_path = response_path + ".tmp"

    if not os.path.isdir(iter_dir):
        print(f"Error: iter directory not found: {iter_dir}", file=sys.stderr)
        print(f"  Is the scratch path correct? ({scratch})", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(waiting_path):
        print(
            "Warning: no WAITING file found — the training job may have already "
            "timed out and resumed, or feedback was already submitted.",
            file=sys.stderr,
        )

    # Print context from request.json so the submitter can confirm
    request_path = os.path.join(iter_dir, "request.json")
    if os.path.isfile(request_path):
        try:
            with open(request_path) as f:
                req = json.load(f)
            print(f"\nRequest context (iter {iter_idx}, island {req.get('island_id', '?')}):")
            metrics = req.get("metrics") or "(none)"
            print(f"  Metrics : {metrics[:300] if metrics else '(none)'}")
            videos = req.get("video_paths") or []
            print(f"  Videos  : {videos if videos else '(none yet)'}")
        except Exception:
            pass

    # Atomic write: write to .tmp then rename so the polling loop never sees a partial file
    with open(tmp_path, "w") as f:
        json.dump({"iter": iter_idx, "response": response_text}, f, indent=2)
    os.rename(tmp_path, response_path)

    # Remove the WAITING sentinel — this is the signal the training job is waiting for
    try:
        os.remove(waiting_path)
    except OSError:
        pass

    print(f"\nFeedback submitted for iter {iter_idx}:")
    print(f"  {response_text}")
    print("\nThe training job will resume shortly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit human feedback to a LOVE Island SLURM training job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scratch", required=True, help="Shared scratch feedback directory")
    parser.add_argument("--iter", type=int, help="Iteration index to submit feedback for")
    parser.add_argument("--response", type=str, help="Feedback text")
    parser.add_argument("--list", action="store_true", help="List all pending requests and exit")
    args = parser.parse_args()

    if args.list:
        list_pending(args.scratch)
        return

    if args.iter is None or args.response is None:
        parser.error("--iter and --response are required (or use --list)")

    submit(args.scratch, args.iter, args.response)


if __name__ == "__main__":
    main()
