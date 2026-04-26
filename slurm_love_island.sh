#!/bin/bash
#SBATCH --job-name love-island
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/love-island-%j.out
#SBATCH -e logs/love-island-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate eureka

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/uufs/chpc.utah.edu/common/home/u1427573/LOVE_Island_Reward_Functions
SCRATCH=/scratch/general/vast/${USER}/love_island

mkdir -p "${SCRATCH}" "${REPO}/logs"

# ── OpenAI key (set in your environment or uncomment and fill in below) ───────
# export OPENAI_API_KEY="sk-..."

# ── Note: SLURM sets CUDA_VISIBLE_DEVICES — do not override with gpustat ──────
echo "Job ${SLURM_JOB_ID} running on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "Scratch feedback path: ${SCRATCH}"
echo "SLURM log: logs/love-island-${SLURM_JOB_ID}.out"

# ── Launch ────────────────────────────────────────────────────────────────────
cd "${REPO}/eureka"

python eureka.py \
    feedback_mode=multi \
    human_feedback_enabled=true \
    feedback_scratch_path="${SCRATCH}" \
    human_feedback_timeout=30 \
    "$@"

# ── Example overrides (append to sbatch command or add here) ──────────────────
# env=cartpole
# sample=2
# iteration=3
# max_iterations=500
# model=gpt-5.4-mini
# human_feedback_enabled=false   # disable feedback for fully autonomous runs
