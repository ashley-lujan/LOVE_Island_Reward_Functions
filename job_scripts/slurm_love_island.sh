#!/bin/bash
#SBATCH --job-name love-island
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=4:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/love-island-%j.out
#SBATCH -e logs/love-island-%j.err

# ── Environment
source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate eureka

# ── Paths
REPO=/uufs/chpc.utah.edu/common/home/u1427573/LOVE_Island_Reward_Functions
SCRATCH=/scratch/general/vast/${USER}/love_island
CONTAINER_DIR=/scratch/general/vast/${USER}/containers

mkdir -p "${SCRATCH}" "${SCRATCH}/runs" "${REPO}/logs" \
         "${SCRATCH}/gifs" "${SCRATCH}/isaac-sim-cache" "${SCRATCH}/isaac-sim-data"

# Isaac Lab container (pulled by job_scripts/pull_isaac_sim.sh)
export ISAACLAB_CONTAINER="${CONTAINER_DIR}/isaac-lab-2.3.2.sif"
# Scratch base for Apptainer cache dirs used inside the container
export SCRATCH_DIR="${SCRATCH}"

# Load API keys from ~/.env_secrets (never committed to git — add your keys there)
if [ -f "${HOME}/.env_secrets" ]; then
    source "${HOME}/.env_secrets"
fi

# ── Note: SLURM sets CUDA_VISIBLE_DEVICES — do not override with gpustat ──────
echo "Job ${SLURM_JOB_ID} running on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "Scratch path : ${SCRATCH}"
echo "GIF output   : ${SCRATCH}/gifs"
echo "Submit cmd   : python submit_feedback.py --scratch ${SCRATCH} --iter <N> --response \"...\""
echo "SLURM log    : logs/love-island-${SLURM_JOB_ID}.out"

# ── Launch
# ENV can be overridden: sbatch job_scripts/slurm_love_island.sh ENV=ant
ENV=${ENV:-cartpole}

cd "${REPO}/eureka"

module load apptainer/1.4.1

python love_island.py \
    env=${ENV} \
    backend=isaaclab \
    love_island.enabled=true \
    \
    num_islands=5 \
    sample=1 \
    iteration=10 \
    max_iterations=200 \
    \
    feedback_mode=multi \
    human_feedback_enabled=true \
    human_feedback_timeout=30 \
    feedback_scratch_path="${SCRATCH}" \
    runs_dir="${SCRATCH}/runs" \
    \
    love_island.warmup_iterations=1 \
    love_island.persistence_threshold=1 \
    love_island.stagnation_window=5 \
    love_island.max_gif_episodes=2 \
    "$@"

# ── Common overrides (pass after sbatch -- or append above) ──────────────────
#
# Fully autonomous (no human blocking):
#   human_feedback_enabled=false
#
# Longer / production run:
#   num_islands=5  sample=3  iteration=10  max_iterations=500
#   love_island.warmup_iterations=2  love_island.persistence_threshold=3
#
# Skip GIF rendering (faster, lower memory):
#   love_island.max_gif_episodes=0
