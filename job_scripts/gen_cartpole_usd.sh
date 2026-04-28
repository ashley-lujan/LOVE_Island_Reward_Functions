#!/bin/bash
#SBATCH --job-name=gen-cartpole-usd
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:10:00
#SBATCH -o logs/gen-cartpole-usd-%j.out
#SBATCH -e logs/gen-cartpole-usd-%j.err

mkdir -p logs

CONTAINER=/scratch/general/vast/${USER}/containers/isaac-lab-2.3.2.sif
BINDS="--bind /scratch/general/vast/${USER}/isaac-sim-cache:/isaac-sim/kit/cache \
       --bind /scratch/general/vast/${USER}/isaac-sim-data:/isaac-sim/kit/data"

PROJECT_ROOT=/uufs/chpc.utah.edu/common/home/${USER}/LOVE_Island_Reward_Functions
OUTPUT_DIR=/scratch/general/vast/${USER}/isaac-assets/Isaac/IsaacLab/Robots/Classic/Cartpole
mkdir -p "${OUTPUT_DIR}"

module load apptainer/1.4.1

echo "Generating cartpole.usd ..."
# Unset conda env vars so /isaac-sim/python.sh doesn't abort
apptainer exec --nv $BINDS \
    --env CONDA_DEFAULT_ENV="" \
    --env CONDA_PREFIX="" \
    --env CONDA_EXE="" \
    --env CONDA_PYTHON_EXE="" \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "${CONTAINER}" \
    /isaac-sim/python.sh "${PROJECT_ROOT}/scripts/create_cartpole_usd.py" \
    --output "${OUTPUT_DIR}/cartpole.usd"

echo "Exit code: $?"
ls -la "${OUTPUT_DIR}/cartpole.usd" 2>/dev/null || echo "File not created."
