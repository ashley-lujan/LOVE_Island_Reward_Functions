#!/bin/bash
#SBATCH --job-name convert-ant
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=0:30:00
#SBATCH --mem=30GB
#SBATCH -o logs/convert-ant-%j.out
#SBATCH -e logs/convert-ant-%j.err

# One-shot job: converts isaacgymenvs/assets/mjcf/nv_ant.xml → ant.usd
# and places the result in the bind-mounted assets directory so ant.py can find it.
#
# Submit: sbatch job_scripts/slurm_convert_ant.sh
# Check:  ls /scratch/general/vast/${USER}/isaac-assets/Isaac/IsaacLab/Robots/Classic/Ant/

REPO=/uufs/chpc.utah.edu/common/home/u1427573/LOVE_Island_Reward_Functions
SCRATCH=/scratch/general/vast/${USER}
CONTAINER="${SCRATCH}/containers/isaac-lab-2.3.2.sif"

# Output path must match ANT_USD_PATH in eureka/envs/isaaclab/ant.py:
#   /local-assets/Isaac/IsaacLab/Robots/Classic/Ant/ant.usd
# Host side of that bind-mount is:
LOVE_ISLAND_SCRATCH=/scratch/general/vast/${USER}/love_island
OUTPUT_HOST="${SCRATCH}/isaac-assets/Isaac/IsaacLab/Robots/Classic/Ant"

mkdir -p "${OUTPUT_HOST}" "${REPO}/logs" \
         "${LOVE_ISLAND_SCRATCH}/isaac-sim-cache" \
         "${LOVE_ISLAND_SCRATCH}/isaac-sim-data"

module load apptainer/1.4.1

# Bind mounts and env vars must match love_island.py's _launch_island_training() exactly.
apptainer exec --nv \
    --bind "${REPO}:${REPO}" \
    --bind "${LOVE_ISLAND_SCRATCH}/isaac-sim-cache:/isaac-sim/kit/cache" \
    --bind "${LOVE_ISLAND_SCRATCH}/isaac-sim-data:/isaac-sim/kit/data" \
    --bind "${OUTPUT_HOST}:/ant-output" \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    "${CONTAINER}" \
    /isaac-sim/python.sh "${REPO}/scripts/convert_ant_mjcf.py" \
        --input "${REPO}/isaacgymenvs/assets/mjcf/nv_ant.xml" \
        --output-dir /ant-output \
        --output-name ant.usd

echo "---"
echo "Output: ${OUTPUT_HOST}/ant.usd"
ls -lh "${OUTPUT_HOST}/" 2>/dev/null || echo "(no files found — check the log above for errors)"
