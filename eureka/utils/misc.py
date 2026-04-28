import subprocess
import os
import json
import logging
import time

from utils.extract_task_code import file_to_string

def set_freest_gpu():
    if os.getenv("SLURM_JOB_ID"):
        logging.info("Running under SLURM — skipping set_freest_gpu() (CUDA_VISIBLE_DEVICES already set by scheduler)")
        return
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        # Match both bare Python tracebacks and Isaac Lab's prefixed format:
        # [omni.kit.app._impl] [py stderr]: Traceback (most recent call last):
        is_traceback_start = line.startswith('Traceback') or (
            '[py stderr]:' in line and 'Traceback' in line
        )
        if is_traceback_start:
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                # Strip Isaac Lab's timestamp/prefix for readability
                clean = lines[j]
                if '[py stderr]:' in clean:
                    clean = clean.split('[py stderr]:', 1)[-1].strip()
                filtered_lines.append(clean)
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on.
    # "fps step:" = IsaacGym rl_games stdout signal.
    # "Tensorboard Directory:" = Isaac Lab isaaclab_train.py early print (flush=True),
    #   used because Kit Python intercepts print() and routes it through its own logger,
    #   so rl_games' fps output never reaches our captured file.
    while True:
        rl_log = file_to_string(rl_filepath)
        started = "fps step:" in rl_log or "Tensorboard Directory:" in rl_log
        if started or "Traceback" in rl_log:
            if log_status and started:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
        time.sleep(5)

if __name__ == "__main__":
    print(get_freest_gpu())