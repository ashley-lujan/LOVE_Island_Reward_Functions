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
from alerts import setup_alerts, send_alert

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"


def build_initial_messages(initial_system, initial_user):
    return [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]


def _build_subprocess_env():
    """Patch LD_LIBRARY_PATH so IsaacGym subprocesses can import their deps."""
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


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    EUREKA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    print("EUREKA_ROOT_DIR:", EUREKA_ROOT_DIR)
    if cfg.alerts.enabled:
        setup_alerts(cfg.alerts)

    send_alert(cfg, "Job Started", 
        f"Eureka started\nEnv: {cfg.env}\nModel: {cfg.model}\nIslands: {cfg.num_islands}"
    )
    
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, "env_init_obs.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    num_islands = int(getattr(cfg, "num_islands", 1))
    island_messages = [build_initial_messages(initial_system, initial_user) for _ in range(num_islands)]

    # Keep the raw template text available for the optional multi-agent path.
    reward_signature_text = reward_signature

    multi_prompts = None
    trace_dir = None
    multi_agent_summary = []
    current_reward_code_for_next_iter = [None for _ in range(num_islands)]
    last_metrics_block = [None for _ in range(num_islands)]
    if cfg.feedback_mode == 'multi':
        from utils.multi_agent import (
            load_multi_agent_prompts,
            run_multi_agent_iteration,
            prompt_human_feedback,
        )

        multi_prompts = load_multi_agent_prompts(prompt_dir)
        trace_dir = os.path.join(os.getcwd(), cfg.multi_agent_trace_dir)
        os.makedirs(trace_dir, exist_ok=True)
        logging.info(
            f"feedback_mode=multi; multi_agent_model={cfg.multi_agent_model}; "
            f"trace_dir={trace_dir}"
        )

    task_code_string = task_code_string.replace(task, task + suffix)
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.0
    max_successes = [[] for _ in range(num_islands)]
    max_successes_reward_correlation = [[] for _ in range(num_islands)]
    execute_rates = [[] for _ in range(num_islands)]
    best_code_paths = [[] for _ in range(num_islands)]
    best_checkpoint_paths = [[] for _ in range(num_islands)]
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None
    max_reward_code_island = None
    max_reward_checkpoint_path = None

    # Eureka generation loop
    for iter in range(cfg.iteration):
        for island_id in range(num_islands):
            responses = []
            response_cur = None
            total_samples = 0
            total_token = 0
            total_completion_token = 0
            prompt_tokens = 0
            chunk_size = cfg.sample if "gpt-3.5" in model else min(4, cfg.sample)

            logging.info(f"Iteration {iter}, Island {island_id}: Generating {cfg.sample} samples with {cfg.model}")

            use_multi_agent = (
                cfg.feedback_mode == 'multi'
                and iter >= 1
                and current_reward_code_for_next_iter[island_id] is not None
                and last_metrics_block[island_id] is not None
            )

            if use_multi_agent:
                human_fb = (
                    prompt_human_feedback(iter, cfg, context={
                        "island_id": island_id,
                        "best_code_path": best_code_paths[island_id][-1] if best_code_paths[island_id] else None,
                        "best_checkpoint_path": best_checkpoint_paths[island_id][-1] if best_checkpoint_paths[island_id] else None,
                        "metrics": last_metrics_block[island_id],
                    })
                    if cfg.human_feedback_enabled
                    else "<no human feedback provided>"
                )
                logging.info(f"Iteration {iter}, Island {island_id}: Running multi-agent feedback pipeline")
                island_trace_dir = os.path.join(trace_dir, f"island_{island_id}")
                response_cur, agent_summary = run_multi_agent_iteration(
                    cfg,
                    multi_prompts,
                    human_feedback=human_fb,
                    metrics_block=last_metrics_block[island_id],
                    current_reward_code=current_reward_code_for_next_iter[island_id],
                    task_obs_code_string=task_obs_code_string,
                    task_description=task_description,
                    reward_signature=reward_signature_text,
                    code_output_tip=code_output_tip,
                    iter_idx=iter,
                    trace_dir=island_trace_dir,
                    n_samples=cfg.sample,
                    fallback_messages=island_messages[island_id],
                    fallback_model=model,
                )
                multi_agent_summary.append(
                    {
                        "iter": iter,
                        "island_id": island_id,
                        **agent_summary,
                    }
                )
                with open("multi_agent_summary.json", "w") as f:
                    json.dump(multi_agent_summary, f, indent=2, default=str)
                responses.extend(response_cur["choices"])
                prompt_tokens = response_cur["usage"]["prompt_tokens"]
                total_completion_token += response_cur["usage"]["completion_tokens"]
                total_token += response_cur["usage"]["total_tokens"]
            else:
                while True:
                    if total_samples >= cfg.sample:
                        break
                    n_samples = min(chunk_size, cfg.sample - total_samples)
                    for attempt in range(1000):
                        try:
                            response_cur = openai.ChatCompletion.create(
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
                                print("Current Chunk Size", chunk_size)
                            logging.info(f"Iteration {iter}, Island {island_id}: Attempt {attempt+1} failed with error: {e}")
                            time.sleep(1)
                    if response_cur is None:
                        logging.info("Code terminated due to too many failed attempts!")
                        exit()

                    responses.extend(response_cur["choices"])
                    prompt_tokens = response_cur["usage"]["prompt_tokens"]
                    total_completion_token += response_cur["usage"]["completion_tokens"]
                    total_token += response_cur["usage"]["total_tokens"]

            if cfg.sample == 1:
                logging.info(f"Iteration {iter}, Island {island_id}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

            logging.info(
                f"Iteration {iter}, Island {island_id}: Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
            )

            run_records = []
            for response_id in range(cfg.sample):
                response_cur = responses[response_id]["message"]["content"]
                logging.info(f"Iteration {iter}, Island {island_id}: Processing Code Run {response_id}")

                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                code_string = None
                for pattern in patterns:
                    match = re.search(pattern, response_cur, re.DOTALL)
                    if match is not None:
                        code_string = match.group(1).strip()
                        break
                code_string = response_cur if not code_string else code_string

                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        code_string = "\n".join(lines[i:])
                        break

                try:
                    gpt_reward_signature, input_lst = get_function_signature(code_string)
                except Exception:
                    logging.info(f"Iteration {iter}, Island {island_id}: Code Run {response_id} cannot parse function signature!")
                    continue

                reward_signature = [
                    f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                    f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                    f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]
                indent = " " * 8
                reward_signature = "\n".join([indent + line for line in reward_signature])
                if "def compute_reward(self)" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
                elif "def compute_reward(self, actions)" in task_code_string:
                    task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
                else:
                    raise NotImplementedError

                if "@torch.jit.script" not in code_string:
                    code_string = "@torch.jit.script\n" + code_string

                with open(output_file, "w") as file:
                    file.writelines(task_code_string_iter + "\n")
                    file.writelines("from typing import Tuple, Dict\n")
                    file.writelines("import math\n")
                    file.writelines("import torch\n")
                    file.writelines("from torch import Tensor\n")
                    file.writelines(code_string + "\n")

                reward_only_path = f"env_iter{iter}_response{response_id}_island{island_id}_rewardonly.py"
                env_code_path = f"env_iter{iter}_response{response_id}_island{island_id}.py"
                rl_filepath = f"env_iter{iter}_response{response_id}_island{island_id}.txt"

                with open(reward_only_path, "w") as file:
                    file.writelines(code_string + "\n")

                shutil.copy(output_file, env_code_path)

                set_freest_gpu()
                with open(rl_filepath, "w") as f:
                    process = subprocess.Popen(
                        [
                            "python",
                            "-u",
                            f"{ISAAC_ROOT_DIR}/train.py",
                            "hydra/output=subprocess",
                            f"task={task}{suffix}",
                            f"wandb_activate={cfg.use_wandb}",
                            f"wandb_entity={cfg.wandb_username}",
                            f"wandb_project={cfg.wandb_project}",
                            f"headless={not cfg.capture_video}",
                            f"capture_video={cfg.capture_video}",
                            "force_render=False",
                            f"max_iterations={cfg.max_iterations}",
                        ],
                        stdout=f,
                        stderr=f,
                        env=_SUBPROCESS_ENV,
                    )
                block_until_training(rl_filepath, log_status=False)
                run_records.append(
                    {
                        "response_id": response_id,
                        "process": process,
                        "rl_filepath": rl_filepath,
                        "code_path": env_code_path,
                        "reward_only_path": reward_only_path,
                        "code_string": code_string,
                    }
                )

            contents = []
            successes = []
            reward_correlations = []
            metrics_blocks = []
            valid_response_ids = []
            code_paths = []

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
                    content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                    content += code_output_tip
                    contents.append(content)
                    metrics_blocks.append("")
                    successes.append(DUMMY_FAILURE)
                    reward_correlations.append(DUMMY_FAILURE)
                    continue

                content = ""
                metrics_block_cur = ""
                traceback_msg = filter_traceback(stdout_str)

                if traceback_msg == "":
                    exec_success = True
                    lines = stdout_str.split("\n")
                    tensorboard_logdir = ""
                    checkpoint_dir = ""
                    for line in lines:
                        if line.startswith("Network Directory:"):
                            checkpoint_dir = line.split(":", 1)[1].strip()
                        elif line.startswith("Tensorboard Directory:"):
                            tensorboard_logdir = line.split(":", 1)[1].strip()
                    run_record["checkpoint_dir"] = checkpoint_dir
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                    max_iterations = np.array(tensorboard_logs["gt_reward"]).shape[0]
                    epoch_freq = max(int(max_iterations // 10), 1)

                    content += policy_feedback.format(epoch_freq=epoch_freq)

                    reward_correlation = DUMMY_FAILURE
                    if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                        gt_reward = np.array(tensorboard_logs["gt_reward"])
                        gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                        reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                    success_score = DUMMY_FAILURE
                    for metric in tensorboard_logs:
                        if "/" not in metric:
                            metric_cur = ["{:.2f}".format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            if metric == "consecutive_successes":
                                success_score = metric_cur_max
                            metric_cur_min = min(tensorboard_logs[metric])
                            if metric != "gt_reward" and metric != "gpt_reward":
                                metric_name = metric if metric != "consecutive_successes" else "task_score"
                                metric_line = (
                                    f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, "
                                    f"Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                                )
                                content += metric_line
                                metrics_block_cur += metric_line
                            else:
                                if "consecutive_successes" not in tensorboard_logs:
                                    metric_line = (
                                        f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, "
                                        f"Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                                    )
                                    content += metric_line
                                    metrics_block_cur += metric_line
                    successes.append(success_score)
                    content += code_feedback
                else:
                    successes.append(DUMMY_FAILURE)
                    reward_correlations.append(DUMMY_FAILURE)
                    content += execution_error_feedback.format(traceback_msg=traceback_msg)

                content += code_output_tip
                contents.append(content)
                metrics_blocks.append(metrics_block_cur)

            if not run_records or (not exec_success and cfg.sample != 1):
                execute_rates[island_id].append(0.0)
                max_successes[island_id].append(DUMMY_FAILURE)
                max_successes_reward_correlation[island_id].append(DUMMY_FAILURE)
                best_code_paths[island_id].append(None)
                best_checkpoint_paths[island_id].append(None)
                logging.info(f"Iteration {iter}, Island {island_id}: All code generation failed. Keeping the prior island state.")
                continue

            best_sample_idx = int(np.argmax(np.array(successes)))
            best_content = contents[best_sample_idx]
            best_response_id = valid_response_ids[best_sample_idx]

            max_success = successes[best_sample_idx]
            max_success_reward_correlation = reward_correlations[best_sample_idx]
            execute_rate = np.sum(np.array(successes) >= 0.0) / max(len(successes), 1)

            if max_success > max_success_overall:
                max_success_overall = max_success
                max_success_reward_correlation_overall = max_success_reward_correlation
                max_reward_code_path = code_paths[best_sample_idx]
                max_reward_code_island = island_id
                max_reward_checkpoint_path = run_records[best_sample_idx].get("checkpoint_dir")

            execute_rates[island_id].append(execute_rate)
            max_successes[island_id].append(max_success)
            max_successes_reward_correlation[island_id].append(max_success_reward_correlation)
            best_code_paths[island_id].append(code_paths[best_sample_idx])
            best_checkpoint_paths[island_id].append(run_records[best_sample_idx].get("checkpoint_dir"))
            current_reward_code_for_next_iter[island_id] = run_records[best_sample_idx]["code_string"]
            last_metrics_block[island_id] = metrics_blocks[best_sample_idx]

            logging.info(
                f"Iteration {iter}, Island {island_id}: Max Success: {max_success}, "
                f"Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}"
            )
            logging.info(f"Iteration {iter}, Island {island_id}: Best Generation ID: {best_response_id}")
            logging.info(f"Iteration {iter}, Island {island_id}: GPT Output Content:\n" + responses[best_response_id]["message"]["content"] + "\n")
            logging.info(f"Iteration {iter}, Island {island_id}: User Content:\n" + best_content + "\n")

            if len(island_messages[island_id]) == 2:
                island_messages[island_id] += [{"role": "assistant", "content": responses[best_response_id]["message"]["content"]}]
                island_messages[island_id] += [{"role": "user", "content": best_content}]
            else:
                assert len(island_messages[island_id]) == 4
                island_messages[island_id][-2] = {"role": "assistant", "content": responses[best_response_id]["message"]["content"]}
                island_messages[island_id][-1] = {"role": "user", "content": best_content}

            with open(f"messages_island{island_id}.json", "w") as file:
                json.dump(island_messages[island_id], file, indent=4)

        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f"{cfg.env.task}")

        for island_id in range(num_islands):
            x_axis = np.arange(len(max_successes[island_id]))
            axs[0].plot(x_axis, np.array(max_successes[island_id]), label=f"Island {island_id}")
            axs[1].plot(x_axis, np.array(execute_rates[island_id]), label=f"Island {island_id}")

        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")
        axs[0].legend()

        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")
        axs[1].legend()

        fig.tight_layout(pad=3.0)
        plt.savefig("summary.png")
        np.savez(
            "summary.npz",
            max_successes=np.array(max_successes),
            execute_rates=np.array(execute_rates),
            best_code_paths=np.array(best_code_paths, dtype=object),
            best_checkpoint_paths=np.array(best_checkpoint_paths, dtype=object),
            max_successes_reward_correlation=np.array(max_successes_reward_correlation),
        )

        with open("messages.json", "w") as file:
            json.dump({f"island_{island_id}": island_messages[island_id] for island_id in range(num_islands)}, file, indent=4)

    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(
        f"Task: {task}, Max Training Success {max_success_overall}, "
        f"Correlation {max_success_reward_correlation_overall}, "
        f"Best Reward Code Path: {max_reward_code_path}, Best Island: {max_reward_code_island}"
    )
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)

    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()

        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, "w") as f:
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    f"{ISAAC_ROOT_DIR}/train.py",
                    "hydra/output=subprocess",
                    f"task={task}{suffix}",
                    f"wandb_activate={cfg.use_wandb}",
                    f"wandb_entity={cfg.wandb_username}",
                    f"wandb_project={cfg.wandb_project}",
                    f"headless={not cfg.capture_video}",
                    f"capture_video={cfg.capture_video}",
                    "force_render=False",
                    f"seed={i}",
                ],
                stdout=f,
                stderr=f,
                env=_SUBPROCESS_ENV,
            )

        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, "r") as f:
            stdout_str = f.read()
        lines = stdout_str.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Tensorboard Directory:"):
                break
        tensorboard_logdir = line.split(":")[-1].strip()
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs["consecutive_successes"])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez("final_eval.npz", reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()
