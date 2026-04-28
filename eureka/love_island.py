"""LOVE Island: Human-in-the-loop reward generation with ensemble disagreement signals.

Drop-in replacement for eureka.py. Extends the standard Eureka training loop with:
  - Terminal state saving per island run
  - Ensemble σ computation across all 5 reward functions
  - Disagreement pattern detection with persistence gating
  - Human feedback cycle (binary / comparative / corrective / guidance / explanatory)
  - Signal 2: conditional retrain vs checkpoint-resume after reward update
  - Signal 3: reflection check with Code Agent retry up to max_reflection_retries

Usage:
  python love_island.py env=cartpole love_island.enabled=true
"""

import hydra
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
import sys
from pathlib import Path
import shutil
import time

from utils.misc import *
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *
from alerts import setup_alerts, send_alert, send_alert_with_attachments

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"


def build_initial_messages(initial_system, initial_user):
    return [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]


def _build_subprocess_env():
    sub_env = os.environ.copy()
    conda_prefix = sys.prefix
    extra_paths = [f"{conda_prefix}/lib"]
    existing = sub_env.get("LD_LIBRARY_PATH", "")
    if existing:
        sub_env["LD_LIBRARY_PATH"] = ":".join(extra_paths + [existing])
    else:
        sub_env["LD_LIBRARY_PATH"] = ":".join(extra_paths)
    return sub_env


_SUBPROCESS_ENV = _build_subprocess_env()


def _launch_island_training(
    cfg,
    *,
    iter: int,
    island_id: int,
    response_id: int,
    code_string: str,
    task_code_string: str,
    output_file: str,
    backend: str,
    REPO_ROOT: str,
    EUREKA_ROOT_DIR: str,
    ISAAC_ROOT_DIR: str,
    suffix: str,
    states_dir: str,
    trajectories_dir: str = "",
    load_checkpoint: bool = False,
    load_path: str = "",
) -> dict:
    """Write reward code, launch training subprocess, return run record dict."""
    from utils.extract_task_code import get_function_signature

    if "@torch.jit.script" not in code_string:
        code_string = "@torch.jit.script\n" + code_string

    try:
        gpt_reward_signature, input_lst = get_function_signature(code_string)
    except Exception:
        return None

    if backend == "isaaclab":
        with open(output_file, "w") as f:
            f.writelines(task_code_string + "\n")
            f.writelines("from typing import Tuple, Dict\n")
            f.writelines("import torch\n")
            f.writelines(code_string + "\n")
    else:
        reward_signature_lines = [
            f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
            f"self.extras['gpt_reward'] = self.rew_buf.mean()",
            f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
        ]
        indent = " " * 8
        reward_sig = "\n".join([indent + line for line in reward_signature_lines])
        if "def compute_reward(self)" in task_code_string:
            task_code_string_iter = task_code_string.replace(
                "def compute_reward(self):", "def compute_reward(self):\n" + reward_sig
            )
        elif "def compute_reward(self, actions)" in task_code_string:
            task_code_string_iter = task_code_string.replace(
                "def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_sig
            )
        else:
            raise NotImplementedError
        with open(output_file, "w") as f:
            f.writelines(task_code_string_iter + "\n")
            f.writelines("from typing import Tuple, Dict\n")
            f.writelines("import math\n")
            f.writelines("import torch\n")
            f.writelines("from torch import Tensor\n")
            f.writelines(code_string + "\n")

    reward_only_path = f"env_iter{iter}_response{response_id}_island{island_id}_rewardonly.py"
    env_code_path = f"env_iter{iter}_response{response_id}_island{island_id}.py"
    rl_filepath = f"env_iter{iter}_response{response_id}_island{island_id}.txt"

    with open(reward_only_path, "w") as f:
        f.write("import torch\nfrom typing import Tuple, Dict\n\n")
        f.writelines(code_string + "\n")
    shutil.copy(output_file, env_code_path)

    task = cfg.env.task

    if backend == "isaaclab":
        _container = os.getenv("ISAACLAB_CONTAINER", getattr(cfg, "container_path", ""))
        _scratch = os.getenv("SCRATCH_DIR", f"/scratch/general/vast/{os.getenv('USER', '')}")
        _user = os.getenv("USER", "")
        _local_assets = f"/scratch/general/vast/{_user}/isaac-assets"
        _runs_dir = getattr(cfg, "runs_dir", "") or f"{EUREKA_ROOT_DIR}/runs"
        _run_name = f"env_iter{iter}_response{response_id}_island{island_id}"

        cmd = [
            "apptainer", "exec", "--nv",
            "--bind", f"{REPO_ROOT}:{REPO_ROOT}",
            "--bind", f"{_scratch}/isaac-sim-cache:/isaac-sim/kit/cache",
            "--bind", f"{_scratch}/isaac-sim-data:/isaac-sim/kit/data",
            "--bind", f"{_local_assets}:/local-assets",
            "--env", "ACCEPT_EULA=Y",
            "--env", "PRIVACY_CONSENT=Y",
            _container,
            "/isaac-sim/python.sh",
            f"{REPO_ROOT}/isaaclab_train.py",
            f"--task={task}{suffix}",
            "--headless",
            f"--max_iterations={cfg.max_iterations}",
            f"--num_envs={getattr(cfg, 'num_envs', 512)}",
            f"--run_name={_run_name}",
            f"--runs_dir={_runs_dir}",
        ]
        if states_dir:
            cmd.append(f"--states_dir={states_dir}")
        if trajectories_dir:
            cmd.append(f"--trajectories_dir={trajectories_dir}")
        if load_checkpoint and load_path:
            cmd.append("--load_checkpoint")
            cmd.append(f"--load_path={load_path}")
        with open(rl_filepath, "w") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
    else:
        set_freest_gpu()
        with open(rl_filepath, "w") as f:
            process = subprocess.Popen(
                [
                    "python", "-u",
                    f"{ISAAC_ROOT_DIR}/train.py",
                    "hydra/output=subprocess",
                    f"task={task}{suffix}",
                    f"max_iterations={cfg.max_iterations}",
                ],
                stdout=f,
                stderr=f,
                env=_SUBPROCESS_ENV,
            )

    block_until_training(rl_filepath, log_status=False)
    return {
        "response_id": response_id,
        "process": process,
        "rl_filepath": rl_filepath,
        "code_path": env_code_path,
        "reward_only_path": reward_only_path,
        "code_string": code_string,
        "states_dir": states_dir,
        "trajectories_dir": trajectories_dir,
    }


def _parse_run_results(run_records, execution_error_feedback, policy_feedback, code_feedback, code_output_tip):
    """Wait for all subprocesses, parse TensorBoard logs, return metrics."""
    DUMMY_FAILURE = -10000.0
    contents, successes, reward_correlations, metrics_blocks = [], [], [], []
    valid_response_ids, code_paths = [], []
    exec_success = False

    for run_record in run_records:
        response_id = run_record["response_id"]
        run_record["process"].communicate()
        code_paths.append(run_record["code_path"])
        valid_response_ids.append(response_id)

        try:
            with open(run_record["rl_filepath"], "r") as f:
                stdout_str = f.read()
        except Exception:
            contents.append(execution_error_feedback.format(
                traceback_msg="Code Run cannot be executed due to function signature error!"
            ) + code_output_tip)
            metrics_blocks.append("")
            successes.append(DUMMY_FAILURE)
            reward_correlations.append(DUMMY_FAILURE)
            run_record["tensorboard_dir"] = ""
            run_record["checkpoint_dir"] = ""
            continue

        content = ""
        metrics_block_cur = ""
        traceback_msg = filter_traceback(stdout_str)

        tensorboard_logdir = ""
        checkpoint_dir = ""
        for line in stdout_str.split("\n"):
            if line.startswith("Network Directory:"):
                checkpoint_dir = line.split(":", 1)[1].strip()
            elif line.startswith("Tensorboard Directory:"):
                tensorboard_logdir = line.split(":", 1)[1].strip()
        run_record["checkpoint_dir"] = checkpoint_dir
        run_record["tensorboard_dir"] = tensorboard_logdir

        if traceback_msg == "":
            tensorboard_logs = load_tensorboard_logs(tensorboard_logdir) if tensorboard_logdir else {}
            if not tensorboard_logs.get("gt_reward"):
                traceback_msg = (
                    "Training completed without errors but no 'gt_reward' metrics were written. "
                    "The reward function may have caused a silent crash."
                )
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                contents.append(execution_error_feedback.format(traceback_msg=traceback_msg) + code_output_tip)
                metrics_blocks.append("")
                continue

            exec_success = True
            max_iters = np.array(tensorboard_logs["gt_reward"]).shape[0]
            epoch_freq = max(int(max_iters // 10), 1)
            content += policy_feedback.format(epoch_freq=epoch_freq)

            reward_correlation = DUMMY_FAILURE
            if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                gt = np.array(tensorboard_logs["gt_reward"])
                gpt = np.array(tensorboard_logs["gpt_reward"])
                reward_correlation = np.corrcoef(gt, gpt)[0, 1]
            reward_correlations.append(reward_correlation)

            success_score = DUMMY_FAILURE
            for metric in tensorboard_logs:
                if "/" not in metric:
                    vals = tensorboard_logs[metric]
                    metric_cur = ["{:.2f}".format(x) for x in vals[::epoch_freq]]
                    if metric == "consecutive_successes":
                        success_score = max(vals)
                    if metric not in ("gt_reward", "gpt_reward"):
                        mname = metric if metric != "consecutive_successes" else "task_score"
                        line = (
                            f"{mname}: {metric_cur}, Max: {max(vals):.2f}, "
                            f"Mean: {sum(vals)/len(vals):.2f}, Min: {min(vals):.2f}\n"
                        )
                        content += line
                        metrics_block_cur += line
                    else:
                        if "consecutive_successes" not in tensorboard_logs:
                            line = (
                                f"ground-truth score: {metric_cur}, Max: {max(vals):.2f}, "
                                f"Mean: {sum(vals)/len(vals):.2f}, Min: {min(vals):.2f}\n"
                            )
                            content += line
                            metrics_block_cur += line

            successes.append(success_score)
            content += code_feedback
        else:
            successes.append(DUMMY_FAILURE)
            reward_correlations.append(DUMMY_FAILURE)
            content += execution_error_feedback.format(traceback_msg=traceback_msg)

        content += code_output_tip
        contents.append(content)
        metrics_blocks.append(metrics_block_cur)

    return {
        "contents": contents,
        "successes": successes,
        "reward_correlations": reward_correlations,
        "metrics_blocks": metrics_blocks,
        "valid_response_ids": valid_response_ids,
        "code_paths": code_paths,
        "exec_success": exec_success,
    }


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    EUREKA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = str(Path(EUREKA_ROOT_DIR).parent)

    if cfg.alerts.enabled:
        setup_alerts(cfg.alerts)
    send_alert(cfg, "Job Started",
        f"LOVE Island started\nEnv: {cfg.env}\nModel: {cfg.model}\nIslands: {cfg.num_islands}"
    )

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")

    _openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    backend = getattr(cfg, "backend", "isaacgym")

    env_name = cfg.env.env_name.lower()
    if backend == "isaaclab":
        env_parent = "isaaclab"
    elif f"{env_name}.py" in os.listdir(f"{EUREKA_ROOT_DIR}/envs/isaac"):
        env_parent = "isaac"
    else:
        env_parent = "dexterity"

    task_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py"
    task_obs_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py"
    shutil.copy(task_obs_file, "env_init_obs.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    if backend == "isaaclab":
        output_file = f"{EUREKA_ROOT_DIR}/envs/isaaclab/{env_name}{suffix.lower()}.py"
    else:
        output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    prompt_dir = f"{EUREKA_ROOT_DIR}/utils/prompts"
    initial_system = file_to_string(f"{prompt_dir}/initial_system.txt")
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    initial_user = file_to_string(f"{prompt_dir}/initial_user.txt")
    reward_signature = file_to_string(f"{prompt_dir}/reward_signature.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    execution_error_feedback = file_to_string(f"{prompt_dir}/execution_error_feedback.txt")

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(
        task_obs_code_string=task_obs_code_string, task_description=task_description
    )

    num_islands = int(getattr(cfg, "num_islands", 1))
    island_messages = [build_initial_messages(initial_system, initial_user) for _ in range(num_islands)]

    if backend != "isaaclab":
        task_code_string = task_code_string.replace(task, task + suffix)
        create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    # ── LOVE Island components ──────────────────────────────────────────────
    li_cfg = cfg.love_island
    _runs_dir = getattr(cfg, "runs_dir", "") or f"{EUREKA_ROOT_DIR}/runs"

    from utils.love_island import (
        EvaluationBuffer,
        DisagreementMonitor,
        EpisodeSelector,
        FeedbackAgent,
        RewardUpdateManager,
    )

    eval_buffer = EvaluationBuffer(
        capacity=li_cfg.buffer_capacity,
        random_ratio=li_cfg.random_ratio,
        staleness_penalty=li_cfg.staleness_penalty,
        high_sigma_threshold=li_cfg.high_sigma_threshold,
    )
    monitor = DisagreementMonitor(eval_buffer=eval_buffer, cfg=li_cfg)
    selector = EpisodeSelector(persistence_window=li_cfg.persistence_threshold)
    feedback_agent = FeedbackAgent(
        prompts_dir=prompt_dir,
        model=getattr(cfg, "multi_agent_model", model),
        temperature=getattr(cfg, "code_agent_temperature", 0.3),
    )
    reward_manager = RewardUpdateManager(eval_buffer=eval_buffer, cfg=li_cfg)

    # ── Training state ──────────────────────────────────────────────────────
    DUMMY_FAILURE = -10000.0
    max_successes = [[] for _ in range(num_islands)]
    max_successes_reward_correlation = [[] for _ in range(num_islands)]
    execute_rates = [[] for _ in range(num_islands)]
    best_code_paths = [[] for _ in range(num_islands)]
    best_checkpoint_paths = [[] for _ in range(num_islands)]
    current_reward_code_for_next_iter = [None for _ in range(num_islands)]
    last_metrics_block = [None for _ in range(num_islands)]
    best_reward_fn_paths = [None for _ in range(num_islands)]

    # Track most recent checkpoint per island for Signal 2 checkpoint-resume
    best_checkpoint_per_island = [None for _ in range(num_islands)]

    max_success_overall = DUMMY_FAILURE
    max_reward_code_path = None

    # ── Main loop ───────────────────────────────────────────────────────────
    for mini_iteration in range(cfg.iteration):
        logging.info(f"\n{'='*60}")
        logging.info(f"LOVE Island mini-iteration {mini_iteration}")
        logging.info(f"{'='*60}")

        # ── PHASE 1: Generate + train all islands ───────────────────────────
        all_run_records = []  # flat list across all islands × samples
        all_tb_dirs_this_iter = []
        all_states_dirs_this_iter = []
        all_trajectories_dirs_this_iter = []

        for island_id in range(num_islands):
            responses = []
            total_samples = 0
            total_token = 0
            chunk_size = cfg.sample if "gpt-3.5" in model else min(4, cfg.sample)

            logging.info(f"Iter {mini_iteration}, Island {island_id}: generating {cfg.sample} samples")

            while True:
                if total_samples >= cfg.sample:
                    break
                n_samples = min(chunk_size, cfg.sample - total_samples)
                for attempt in range(1000):
                    try:
                        response_cur = _openai_client.chat.completions.create(
                            model=model,
                            messages=island_messages[island_id],
                            temperature=cfg.temperature,
                            n=n_samples,
                        )
                        total_samples += n_samples
                        break
                    except Exception as e:
                        if attempt >= 10:
                            chunk_size = max(int(chunk_size / 2), 1)
                        logging.info(f"Iter {mini_iteration}, Island {island_id}: attempt {attempt+1} failed: {e}")
                        time.sleep(1)
                responses.extend([
                    {"message": {"role": c.message.role, "content": c.message.content}}
                    for c in response_cur.choices
                ])
                total_token += response_cur.usage.total_tokens

            run_records = []
            for response_id in range(cfg.sample):
                response_content = responses[response_id]["message"]["content"]
                patterns = [
                    r'```python(.*?)```', r'```(.*?)```',
                    r'"""(.*?)"""', r'""(.*?)""', r'"(.*?)"',
                ]
                code_string = None
                for pat in patterns:
                    match = re.search(pat, response_content, re.DOTALL)
                    if match:
                        code_string = match.group(1).strip()
                        break
                code_string = response_content if not code_string else code_string
                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        code_string = "\n".join(lines[i:])
                        break

                _run_name = f"env_iter{mini_iteration}_response{response_id}_island{island_id}"
                _states_dir = os.path.join(_runs_dir, _run_name, "states")
                _trajectories_dir = os.path.join(_runs_dir, _run_name, "trajectories")
                all_states_dirs_this_iter.append(_states_dir)
                all_trajectories_dirs_this_iter.append(_trajectories_dir)

                # Resume from checkpoint if Signal 2 cleared it as safe to do so
                _prior_ckpt = best_checkpoint_per_island[island_id]
                record = _launch_island_training(
                    cfg,
                    iter=mini_iteration,
                    island_id=island_id,
                    response_id=response_id,
                    code_string=code_string,
                    task_code_string=task_code_string,
                    output_file=output_file,
                    backend=backend,
                    REPO_ROOT=REPO_ROOT,
                    EUREKA_ROOT_DIR=EUREKA_ROOT_DIR,
                    ISAAC_ROOT_DIR=ISAAC_ROOT_DIR,
                    suffix=suffix,
                    states_dir=_states_dir,
                    trajectories_dir=_trajectories_dir,
                    load_checkpoint=bool(_prior_ckpt),
                    load_path=_prior_ckpt or "",
                )
                if record is not None:
                    record["island_id"] = island_id
                    record["responses"] = responses
                    run_records.append(record)

            all_run_records.extend(run_records)

            # Parse results for this island
            parsed = _parse_run_results(
                run_records, execution_error_feedback, policy_feedback, code_feedback, code_output_tip
            )
            for rec in run_records:
                if rec.get("tensorboard_dir"):
                    all_tb_dirs_this_iter.append(rec["tensorboard_dir"])

            if not run_records or not parsed["exec_success"]:
                execute_rates[island_id].append(0.0)
                max_successes[island_id].append(DUMMY_FAILURE)
                max_successes_reward_correlation[island_id].append(DUMMY_FAILURE)
                best_code_paths[island_id].append(None)
                best_checkpoint_paths[island_id].append(None)
                logging.info(f"Iter {mini_iteration}, Island {island_id}: all runs failed")
                continue

            best_idx = int(np.argmax(np.array(parsed["successes"])))
            best_run = run_records[best_idx]

            execute_rate = np.sum(np.array(parsed["successes"]) >= 0.0) / max(len(parsed["successes"]), 1)
            execute_rates[island_id].append(execute_rate)
            max_successes[island_id].append(parsed["successes"][best_idx])
            max_successes_reward_correlation[island_id].append(parsed["reward_correlations"][best_idx])
            best_code_paths[island_id].append(parsed["code_paths"][best_idx])
            best_checkpoint_paths[island_id].append(best_run.get("checkpoint_dir"))
            _ckpt_dir = best_run.get("checkpoint_dir", "")
            # rl_games saves best checkpoint as {run_name}.pth inside the nn/ dir
            _ckpt_pth = os.path.join(_ckpt_dir, f"{_run_name}.pth") if _ckpt_dir else ""
            best_checkpoint_per_island[island_id] = _ckpt_pth if os.path.exists(_ckpt_pth) else None
            current_reward_code_for_next_iter[island_id] = best_run["code_string"]
            last_metrics_block[island_id] = parsed["metrics_blocks"][best_idx]
            best_reward_fn_paths[island_id] = best_run["reward_only_path"]

            if parsed["successes"][best_idx] > max_success_overall:
                max_success_overall = parsed["successes"][best_idx]
                max_reward_code_path = parsed["code_paths"][best_idx]

            # Update LLM context for next iteration
            best_response_id = parsed["valid_response_ids"][best_idx]
            best_content = parsed["contents"][best_idx]
            if len(island_messages[island_id]) == 2:
                island_messages[island_id] += [
                    {"role": "assistant", "content": responses[best_response_id]["message"]["content"]},
                    {"role": "user", "content": best_content},
                ]
            else:
                island_messages[island_id][-2] = {"role": "assistant", "content": responses[best_response_id]["message"]["content"]}
                island_messages[island_id][-1] = {"role": "user", "content": best_content}

            with open(f"messages_island{island_id}.json", "w") as f:
                json.dump(island_messages[island_id], f, indent=4)

        # ── PHASE 2: Compute σ across all island reward functions ───────────
        reward_fn_paths = [p for p in best_reward_fn_paths if p is not None]
        sigma_result = monitor.compute_sigma_for_iteration(
            reward_fn_paths=reward_fn_paths,
            states_dirs=all_states_dirs_this_iter,
            tb_dirs=all_tb_dirs_this_iter,
        )
        logging.info(f"LOVE Island σ result: {sigma_result}")

        # ── PHASE 3: Evaluate — query or continue ───────────────────────────
        should_query, feedback_type, pattern = monitor.evaluate_mini_iteration()

        if not should_query:
            eval_buffer.reset_for_new_mini_iteration(mini_iteration)
            _save_summary(cfg, mini_iteration, max_successes, execute_rates, best_code_paths, best_checkpoint_paths)
            continue

        # ── PHASE 4: Feedback cycle ─────────────────────────────────────────
        logging.info(f"LOVE Island: QUERY triggered — pattern={pattern} feedback_type={feedback_type}")

        # Load reward functions for comparative selection
        from utils.love_island.sigma import load_reward_fn
        reward_fns = [load_reward_fn(p) for p in reward_fn_paths]

        selected = selector.select(eval_buffer, feedback_type, pattern, reward_fns=reward_fns)
        if not selected:
            logging.warning("LOVE Island: no episodes selected — skipping feedback cycle")
            eval_buffer.reset_for_new_mini_iteration(mini_iteration)
            continue

        primary_episode = selected[0]
        similar_history = eval_buffer.get_similar_feedback_history(primary_episode)
        sigma_breakdown = monitor.get_sigma_breakdown()

        # Render episode GIFs for human review
        from utils.love_island.renderer import render_episode_gifs
        _scratch_dir = getattr(cfg, "feedback_scratch_path", "") or os.path.join(EUREKA_ROOT_DIR, "scratch")
        gif_dir = os.path.join(_scratch_dir, "gifs", f"iter_{mini_iteration}")
        gif_paths = render_episode_gifs(
            selected_records=selected,
            trajectories_dirs=all_trajectories_dirs_this_iter,
            output_dir=gif_dir,
            feedback_type=feedback_type,
            sigma_breakdown=sigma_breakdown,
            max_gifs=int(getattr(li_cfg, "max_gif_episodes", 2)),
        )
        logging.info(f"LOVE Island: rendered {len(gif_paths)} GIF(s) to {gif_dir}")

        schema = feedback_agent.populate_schema(
            feedback_type=feedback_type,
            episode_data=selected,
            sigma_breakdown=sigma_breakdown,
            similar_history=similar_history,
        )

        # Build and send email with GIFs attached
        _response_file = os.path.join(_scratch_dir, f"feedback_response_{mini_iteration}.txt")
        _email_body = (
            f"LOVE Island needs your feedback (iteration {mini_iteration})\n\n"
            f"Pattern detected: {pattern}\n"
            f"Feedback type: {feedback_type}\n\n"
            f"{schema.get('context_for_human', '')}\n\n"
            f"Question: {schema.get('question_text', '')}\n\n"
            f"To submit your feedback, write to:\n  {_response_file}\n"
            f"\nOr run:\n  echo 'YOUR FEEDBACK HERE' > {_response_file}\n"
        )
        send_alert_with_attachments(
            cfg,
            f"Feedback needed — iter {mini_iteration} ({pattern})",
            _email_body,
            gif_paths,
        )

        # Collect human response via existing feedback_io mechanism
        from utils.multi_agent import prompt_human_feedback
        raw_feedback = (
            prompt_human_feedback(mini_iteration, cfg, context={
                "feedback_type": feedback_type,
                "pattern": pattern,
                "schema": schema,
                "sigma_breakdown": sigma_breakdown,
                "gif_paths": gif_paths,
            })
            if getattr(cfg, "human_feedback_enabled", True)
            else "<no human feedback provided>"
        )

        sigma_before = eval_buffer.get_mean_sigma()
        mu_at_update = sigma_result.get("mean_reward", 0.0)

        context = feedback_agent.contextualize_response(
            raw_feedback=raw_feedback,
            feedback_type=feedback_type,
            episode_data=selected,
            sigma_breakdown=sigma_breakdown,
            query_history=eval_buffer.get_context_for_reflection(primary_episode.episode_id),
        )

        eval_buffer.mark_episode_queried(
            episode_id=primary_episode.episode_id,
            feedback_type=feedback_type,
            feedback_text=raw_feedback,
            iteration=mini_iteration,
            reward_update_summary=context.get("reward_update_instruction", ""),
            expected_direction=context.get("expected_direction", ""),
        )

        # ── PHASE 5: Signal 2 — update rewards + retrain check ─────────────
        reward_update_instruction = context.get("reward_update_instruction", raw_feedback)
        _apply_reward_update(
            cfg, island_messages, current_reward_code_for_next_iter,
            reward_update_instruction, mini_iteration, num_islands,
            task_obs_code_string, task_description, reward_signature,
            code_output_tip, prompt_dir, model,
        )

        new_reward_fn_paths = [p for p in best_reward_fn_paths if p is not None]
        should_retrain, sigma_after = reward_manager.check_retrain(
            sigma_before=sigma_before,
            new_reward_fn_paths=new_reward_fn_paths,
            states_dirs=all_states_dirs_this_iter,
        )

        if should_retrain:
            logging.info("LOVE Island Signal 2: σ spiked — full retrain next iteration")
            best_checkpoint_per_island = [None for _ in range(num_islands)]
        else:
            logging.info("LOVE Island Signal 2: σ stable — checkpoint-resume next iteration")

        # ── PHASE 6: Signal 3 — reflection check ───────────────────────────
        if mini_iteration > 0 and mini_iteration % li_cfg.reflection_window == 0:
            passed, reflection_prompt = reward_manager.check_reflection(
                episode_id=primary_episode.episode_id,
                mu_at_update=mu_at_update,
                tb_dirs=all_tb_dirs_this_iter,
            )

            if not passed:
                query_history = eval_buffer.get_context_for_reflection(primary_episode.episode_id)
                retry_count = len(query_history)

                if retry_count < li_cfg.max_reflection_retries:
                    logging.info(
                        f"LOVE Island Signal 3: reflection failed — "
                        f"Code Agent retry {retry_count + 1}/{li_cfg.max_reflection_retries}"
                    )
                    _apply_reward_update(
                        cfg, island_messages, current_reward_code_for_next_iter,
                        reflection_prompt, mini_iteration, num_islands,
                        task_obs_code_string, task_description, reward_signature,
                        code_output_tip, prompt_dir, model,
                    )
                else:
                    logging.info("LOVE Island Signal 3: MAX RETRIES reached — escalating to human (guidance query)")
                    # Force a guidance query next iteration by resetting persistence counter
                    monitor._current_pattern = "stagnation"
                    monitor._persistence_counter = li_cfg.persistence_threshold

        eval_buffer.reset_for_new_mini_iteration(mini_iteration)
        _save_summary(cfg, mini_iteration, max_successes, execute_rates, best_code_paths, best_checkpoint_paths)

    # ── Final summary ───────────────────────────────────────────────────────
    _save_summary(cfg, cfg.iteration - 1, max_successes, execute_rates, best_code_paths, best_checkpoint_paths)
    logging.info(f"LOVE Island complete. Best overall success: {max_success_overall:.2f}")
    logging.info(f"Best reward code: {max_reward_code_path}")


def _apply_reward_update(
    cfg, island_messages, current_reward_code_for_next_iter,
    update_instruction, mini_iteration, num_islands,
    task_obs_code_string, task_description, reward_signature,
    code_output_tip, prompt_dir, model,
):
    """Inject feedback into each island's message history so the next LLM call uses it."""
    from utils.multi_agent import load_multi_agent_prompts, run_multi_agent_iteration

    multi_prompts = load_multi_agent_prompts(prompt_dir)
    for island_id in range(num_islands):
        if current_reward_code_for_next_iter[island_id] is None:
            continue
        island_trace_dir = os.path.join(os.getcwd(), "love_island_trace", f"island_{island_id}")
        try:
            response_cur, _ = run_multi_agent_iteration(
                cfg,
                multi_prompts,
                human_feedback=update_instruction,
                metrics_block=f"Love Island feedback cycle at iteration {mini_iteration}",
                current_reward_code=current_reward_code_for_next_iter[island_id],
                task_obs_code_string=task_obs_code_string,
                task_description=task_description,
                reward_signature=reward_signature,
                code_output_tip=code_output_tip,
                iter_idx=mini_iteration,
                trace_dir=island_trace_dir,
                n_samples=cfg.sample,
                fallback_messages=island_messages[island_id],
                fallback_model=model,
            )
            if response_cur and response_cur.get("choices"):
                new_code = response_cur["choices"][0]["message"]["content"]
                # Extract code block
                match = re.search(r'```python(.*?)```', new_code, re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    current_reward_code_for_next_iter[island_id] = extracted
                    logging.info(f"LOVE Island: island {island_id} reward updated by Code Agent")
        except Exception as e:
            logging.warning(f"LOVE Island: reward update failed for island {island_id}: {e}")


def _save_summary(cfg, iteration, max_successes, execute_rates, best_code_paths, best_checkpoint_paths):
    np.savez(
        "summary.npz",
        max_successes=np.array(max_successes, dtype=object),
        execute_rates=np.array(execute_rates, dtype=object),
        best_code_paths=np.array(best_code_paths, dtype=object),
        best_checkpoint_paths=np.array(best_checkpoint_paths, dtype=object),
    )

    num_islands = len(max_successes)
    fig, axs = plt.subplots(2, figsize=(6, 6))
    fig.suptitle(f"{cfg.env.task} — LOVE Island")
    for island_id in range(num_islands):
        x = np.arange(len(max_successes[island_id]))
        axs[0].plot(x, np.array(max_successes[island_id]), label=f"Island {island_id}")
        axs[1].plot(x, np.array(execute_rates[island_id]), label=f"Island {island_id}")
    axs[0].set_title("Max Success")
    axs[1].set_title("Execute Rate")
    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.savefig("summary.png")
    plt.close()


if __name__ == "__main__":
    main()
