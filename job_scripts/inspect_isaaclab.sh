#!/bin/bash
#SBATCH --job-name=inspect-isaaclab
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=0:20:00
#SBATCH -o logs/inspect-isaaclab-%j.out
#SBATCH -e logs/inspect-isaaclab-%j.err

mkdir -p logs

CONTAINER=/scratch/general/vast/${USER}/containers/isaac-lab-2.3.2.sif
BINDS="--bind /scratch/general/vast/${USER}/isaac-sim-cache:/isaac-sim/kit/cache \
       --bind /scratch/general/vast/${USER}/isaac-sim-data:/isaac-sim/kit/data"

module load apptainer/1.4.1

echo "=== Isaac Lab version ==="
apptainer exec --nv $BINDS "${CONTAINER}" \
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
    cat /workspace/isaaclab/VERSION 2>/dev/null || echo "(no VERSION file)"

echo ""
echo "=== asset_base.py __init__ lines 70-120 ==="
apptainer exec $BINDS "${CONTAINER}" \
    sed -n '70,120p' /workspace/isaaclab/source/isaaclab/isaaclab/assets/asset_base.py 2>/dev/null || \
    apptainer exec $BINDS "${CONTAINER}" \
    find /workspace/isaaclab -name "asset_base.py" 2>/dev/null | head -3

echo ""
echo "=== Official cartpole DirectRLEnv example (prim_path) ==="
apptainer exec $BINDS "${CONTAINER}" \
    grep -r "prim_path" /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/ 2>/dev/null | head -20

echo ""
echo "=== prims.py validation check (lines 685-700) ==="
apptainer exec $BINDS "${CONTAINER}" \
    grep -n "is not global\|ENV_REGEX_NS\|startswith" \
    /workspace/isaaclab/source/isaaclab/isaaclab/sim/utils/prims.py 2>/dev/null | head -30

echo ""
echo "=== ENV_REGEX_NS occurrences in asset_base.py ==="
apptainer exec $BINDS "${CONTAINER}" \
    grep -n "ENV_REGEX_NS" \
    /workspace/isaaclab/source/isaaclab/isaaclab/assets/asset_base.py 2>/dev/null

echo ""
echo "=== Done ==="
