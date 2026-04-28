#!/bin/bash
#SBATCH --job-name=love-island-smoke
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH -o logs/smoke-test-%j.out
#SBATCH -e logs/smoke-test-%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate eureka

REPO=/uufs/chpc.utah.edu/common/home/u1427573/LOVE_Island_Reward_Functions

mkdir -p "${REPO}/logs"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job ${SLURM_JOB_ID} running on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES}"

# ---------------------------------------------------------------------------
# Run smoke test
# ---------------------------------------------------------------------------
python "${REPO}/eureka/smoke_test.py"
