# LOVE Island — CHPC Handoff Guide

How to run the LOVE Island reward-generation pipeline on your own CHPC account.
This uses the **Isaac Lab** backend (Apptainer container), not the old IsaacGym backend.

---

## Prerequisites

- A CHPC account with access to the `soc-gpu-class-grn` partition and the `cs6966` allocation
  (if you're in the class). Otherwise, substitute your own `--account` and `--partition` values
  in the SLURM script.
- An OpenAI API key.
- A Gmail account (or other SMTP provider) for job alerts — optional but recommended.

---

## 1. Clone the repo

```bash
git clone <repo-url> ~/LOVE_Island_Reward_Functions
cd ~/LOVE_Island_Reward_Functions
```

---

## 2. Create the conda environment

The pipeline uses **miniforge3** (not miniconda3).

```bash
# Load miniforge — adjust path to where miniforge is installed on your account
module load miniforge3   # or source the conda.sh directly if not a module

conda create -n eureka python=3.10 -y
conda activate eureka

# Install the repo packages
pip install -e .
cd rl_games && pip install -e . && cd ..

# Extra dependencies
pip install matplotlib pillow torch tensorboard hydra-core omegaconf
```

> **Note**: the SLURM script activates conda via `source .../miniforge3/etc/profile.d/conda.sh`.
> Update the path in `job_scripts/slurm_love_island.sh` (see Section 4).

---

## 3. Pull the Isaac Lab container

The training loop runs inside a Singularity/Apptainer container. Pull it once to your scratch space:

```bash
bash job_scripts/pull_isaac_sim.sh
```

This saves the container to `/scratch/general/vast/${USER}/containers/isaac-lab-2.3.2.sif`.
The SLURM script expects it there — no changes needed since it uses `$USER`.

---

## 4. Edit the SLURM script — 2 lines to change

Open `job_scripts/slurm_love_island.sh` and update the two hardcoded paths near the top:

```bash
# Line 15 — change u1427573 to YOUR CHPC username:
source /uufs/chpc.utah.edu/common/home/YOUR_UID/software/pkg/miniforge3/etc/profile.d/conda.sh

# Line 19 — change u1427573 to YOUR CHPC username:
REPO=/uufs/chpc.utah.edu/common/home/YOUR_UID/LOVE_Island_Reward_Functions
```

Everything else in the script uses `$USER` and `$SLURM_JOB_ID` — fully portable.

If you're not on the `cs6966` allocation, also change:
```bash
#SBATCH --account=cs6966          # → your allocation
#SBATCH --partition=soc-gpu-class-grn  # → your partition
#SBATCH --qos=soc-gpu-class-grn        # → matching QOS
```

---

## 5. Set your OpenAI API key

The pipeline loads secrets from `~/.env_secrets`. Create that file:

```bash
cat >> ~/.env_secrets << 'EOF'
export OPENAI_API_KEY="sk-your-key-here"
EOF
chmod 600 ~/.env_secrets
```

This file is never committed to git (it's in `.gitignore`).

---

## 6. Configure email alerts

Alerts are currently configured with **Ash's email** (`lujan.Ash@gmail.com`).
Update `eureka/cfg/alerts/alerts.yaml` with your own address:

```yaml
enabled: true
gmail_user: "your.address@gmail.com"
gmail_password: "your-gmail-app-password"   # 16-char app password, not your login password
recipient: "your.address@gmail.com"

on_start: true
on_finish: true
on_error: true
on_iteration: true
iteration_interval: 50
```

> **Gmail app password**: go to myaccount.google.com → Security → 2-Step Verification →
> App passwords → generate one for "Mail / Other". Use that 16-character string, not your
> Gmail login password.

To disable alerts entirely (useful for testing), set `enabled: false`.

A blank template is at `eureka/cfg/alerts/alerts_template.yaml`.

---

## 7. Submit the job

From the repo root:

```bash
mkdir -p logs
sbatch job_scripts/slurm_love_island.sh
```

Watch the log:
```bash
tail -f logs/love-island-<JOBID>.out
```

---

## 8. Giving human feedback

When the pipeline needs your input it writes a file and waits. You'll get an email with GIF
attachments and a submit command. From any login node:

```bash
python submit_feedback.py \
    --scratch /scratch/general/vast/${USER}/love_island \
    --iter <N> \
    --response "your feedback text here"
```

The training job polls every 30 seconds and resumes automatically once feedback is received.
If you don't respond within the timeout window (default: 30 minutes), it continues without
human input.

---

## 9. Tuning the run

Key parameters in `slurm_love_island.sh` (or pass as extra args after `sbatch -- `):

| Parameter | Default | What it controls |
|---|---|---|
| `num_islands` | 2 | Number of parallel reward islands |
| `sample` | 1 | Reward candidates per LLM call |
| `iteration` | 5 | RL training iterations per island per cycle |
| `max_iterations` | 200 | Maximum RL iterations before stopping |
| `human_feedback_timeout` | 30 | Minutes to wait before skipping feedback |
| `human_feedback_enabled` | true | Set to `false` for fully autonomous run |
| `love_island.max_gif_episodes` | 2 | GIFs rendered per feedback query (0 = skip) |

For a quick smoke test (no Isaac Lab, no GPU required):

```bash
cd eureka
python smoke_test.py
```

---

## What you do NOT need to change

- All Python code — it uses `os.path.dirname(__file__)` and `$USER` throughout
- `submit_feedback.py` — reads scratch path from the same `$USER` variable
- The Isaac Lab container path — derived from `$USER` at runtime
- Hydra config files (except `alerts.yaml`)

---

## Current alert status (as of April 2026)

Alerts are sending **from and to `lujan.Ash@gmail.com`**. If you're Ashley, you're all set.
If you're someone else on the team, update `alerts.yaml` as described in Section 6 before
submitting your first job — otherwise your job events go to Ash's inbox.
