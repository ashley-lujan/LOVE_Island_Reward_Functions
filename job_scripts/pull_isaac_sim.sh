#!/bin/bash
#SBATCH --job-name=pull-isaac-sim
#SBATCH --account=cs6966
#SBATCH --partition=granite-guest
#SBATCH --qos=granite-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH -o logs/pull-isaac-sim-%j.out
#SBATCH -e logs/pull-isaac-sim-%j.err

# ---------------------------------------------------------------------------
# Redirect Apptainer cache/tmp to scratch to avoid filling home quota
# ---------------------------------------------------------------------------
SCRATCH=/scratch/general/vast/${USER}
CONTAINER_DIR=${SCRATCH}/containers

mkdir -p "${CONTAINER_DIR}" "${SLURM_SUBMIT_DIR}/logs"

export APPTAINER_CACHEDIR=${SCRATCH}/.apptainer_cache
export APPTAINER_TMPDIR=${SCRATCH}/.apptainer_tmp
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# ---------------------------------------------------------------------------
# Load Apptainer
# ---------------------------------------------------------------------------
module load apptainer/1.4.1

echo "Pulling Isaac Lab 2.3.2 container to: ${CONTAINER_DIR}/isaac-lab-2.3.2.sif"
echo "This may take 20-60 minutes depending on network speed."

# ---------------------------------------------------------------------------
# NGC authentication — required for nvcr.io images.
# Set NGC_API_KEY in your environment before submitting, or paste it here:
# export NGC_API_KEY="<your-key>"   # do not hardcode — pass via environment instead
# Get a free key at: ngc.nvidia.com -> Setup -> Generate API Key
# ---------------------------------------------------------------------------
if [ -n "${NGC_API_KEY}" ]; then
    export APPTAINER_DOCKER_USERNAME='$oauthtoken'
    export APPTAINER_DOCKER_PASSWORD="${NGC_API_KEY}"
    echo "NGC credentials set."
else
    echo "WARNING: NGC_API_KEY not set. Pull may fail if auth is required."
fi

apptainer pull \
    --force \
    "${CONTAINER_DIR}/isaac-lab-2.3.2.sif" \
    docker://nvcr.io/nvidia/isaac-lab:2.3.2

echo "Pull complete. Container at: ${CONTAINER_DIR}/isaac-lab-2.3.2.sif"
ls -lh "${CONTAINER_DIR}/isaac-lab-2.3.2.sif"
