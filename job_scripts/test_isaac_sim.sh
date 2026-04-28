#!/bin/bash
#SBATCH --job-name=test-isaac-sim
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=0:30:00
#SBATCH -o logs/test-isaac-sim-%j.out
#SBATCH -e logs/test-isaac-sim-%j.err

mkdir -p logs

CONTAINER=/scratch/general/vast/${USER}/containers/isaac-lab-2.3.2.sif
PYTHON=/isaac-sim/python.sh   # Isaac Sim's bundled Python wrapper (present in Isaac Lab container)

# Writable directories for Isaac Sim cache/data (container filesystem is read-only)
ISAAC_CACHE=/scratch/general/vast/${USER}/isaac-sim-cache
ISAAC_DATA=/scratch/general/vast/${USER}/isaac-sim-data
mkdir -p "${ISAAC_CACHE}" "${ISAAC_DATA}"

APPTAINER_BINDS="--bind ${ISAAC_CACHE}:/isaac-sim/kit/cache --bind ${ISAAC_DATA}:/isaac-sim/kit/data"

module load apptainer/1.4.1

echo "========================================"
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# ── 0. Discover Python location inside container ──────────────────────────
echo ""
echo "[0] Locating Python inside container"
apptainer exec --nv ${APPTAINER_BINDS} "${CONTAINER}" find /isaac-sim -name "python*" -maxdepth 4 2>/dev/null | head -10

# ── 1. Host-side GPU check ────────────────────────────────────────────────
echo ""
echo "[1] nvidia-smi (host)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ── 2. PyTorch CUDA inside container ─────────────────────────────────────
echo ""
echo "[2] PyTorch CUDA check (inside container)"
apptainer exec --nv ${APPTAINER_BINDS} "${CONTAINER}" ${PYTHON} - <<'EOF'
import torch
print(f"  torch version : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU name      : {torch.cuda.get_device_name(0)}")
    t = torch.ones(3, device='cuda')
    print(f"  Tensor on GPU : {t}  [OK]")
EOF

# ── 3. Isaac Sim import inside container ─────────────────────────────────
echo ""
echo "[3] Isaac Sim import check (inside container)"
apptainer exec --nv ${APPTAINER_BINDS} \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    "${CONTAINER}" ${PYTHON} - <<'EOF'
try:
    import isaacsim
    print("  import isaacsim                     [OK]")
except ImportError as e:
    print(f"  import isaacsim                     [FAIL] {e}")

try:
    from isaacsim import SimulationApp
    print("  from isaacsim import SimulationApp  [OK]")
except ImportError as e:
    print(f"  from isaacsim import SimulationApp  [FAIL] {e}")
EOF

# ── 4. Headless SimulationApp startup ────────────────────────────────────
echo ""
echo "[4] Headless SimulationApp startup (may take ~60s on first run)"
apptainer exec --nv ${APPTAINER_BINDS} \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    "${CONTAINER}" ${PYTHON} - <<'EOF'
import time
start = time.time()
try:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracingLighting"})
    elapsed = time.time() - start
    print(f"  SimulationApp started in {elapsed:.1f}s  [OK]")
    app.close()
    print("  SimulationApp closed                [OK]")
except Exception as e:
    print(f"  SimulationApp startup               [FAIL] {e}")
EOF

echo ""
echo "========================================"
echo "Test complete."
echo "========================================"
