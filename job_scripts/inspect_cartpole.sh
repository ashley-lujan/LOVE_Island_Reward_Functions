#!/bin/bash
#SBATCH --job-name=inspect-cartpole
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:10:00
#SBATCH -o logs/inspect-cartpole-%j.out
#SBATCH -e logs/inspect-cartpole-%j.err

mkdir -p logs

CONTAINER=/scratch/general/vast/${USER}/containers/isaac-lab-2.3.2.sif
BINDS="--bind /scratch/general/vast/${USER}/isaac-sim-cache:/isaac-sim/kit/cache \
       --bind /scratch/general/vast/${USER}/isaac-sim-data:/isaac-sim/kit/data"

module load apptainer/1.4.1

echo "=== RlGamesGpuEnv class (lines 383+) ==="
apptainer exec $BINDS "${CONTAINER}" \
    sed -n '380,450p' /workspace/isaaclab/source/isaaclab_rl/isaaclab_rl/rl_games/rl_games.py 2>/dev/null

echo ""
echo "=== Official train.py runner section ==="
apptainer exec $BINDS "${CONTAINER}" \
    grep -n "Runner\|vecenv\|env_configurations\|RlGames\|rlgpu\|env_name\|train_dir\|config" \
    /workspace/isaaclab/scripts/reinforcement_learning/rl_games/train.py 2>/dev/null | head -60

echo ""
echo "=== Done ==="
