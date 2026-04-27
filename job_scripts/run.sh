#!/bin/bash
#SBATCH --job-name=eureka_test
#SBATCH --account=cs6966
#SBATCH --partition=coe-gpu-class-grn
#SBATCH --qos=coe-gpu-students-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Create log dir if it doesn't exist
mkdir -p logs


module load miniconda3/25.9.1
module load cuda/11.8       # verify version with: module spider cuda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate eureka



# ---------------------------------------------------------------------------
# Run Eureka
# ---------------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR/eureka
# IsaacGym needs this to find shared libs — adjust path to wherever you installed it
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python eureka.py \
    feedback_mode=multi \
    num_islands=4 \
    env=cartpole \
    sample=2 \
    iteration=1 \
    max_iterations=500 \
    model=gpt-4o-mini

# ---------------------------------------------------------------------------
# Notes:
#   - Check your account name with:         myallocation
#   - Check available GPU partitions with:  sinfo -o "%P %G" | grep gpu
#   - Check module names with:              module spider cuda
#   - Submit this script with:              sbatch run_eureka.sh
#   - Monitor your job with:               squeue -u $USER
#   - Cancel a job with:                   scancel <job-id>
# ---------------------------------------------------------------------------
