# Eureka Setup Instructions (Windows + CUDA)

Starting point: Windows laptop with an NVIDIA GPU, no conda, no Linux environment other than Docker Desktop's WSL2 distros.

---

## Step 1 — Install Ubuntu via WSL2

WSL2 is already present if you have Docker Desktop. You just need a real Ubuntu distro.

In PowerShell (as admin or normal user):
```powershell
wsl --install -d Ubuntu
```

Reboot if prompted. Afterwards, open **Ubuntu** from the Start menu. Create a username and password when asked.

Verify your GPU is visible inside Ubuntu:
```bash
nvidia-smi
```
You should see your GPU listed. CUDA passthrough from Windows is automatic with recent NVIDIA drivers.

---

## Step 2 — Install Miniconda inside Ubuntu

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

During the installer:
- Press `Enter` to scroll through the license, type `yes` to accept
- Keep the default install location (press `Enter`)
- Type `yes` when asked to initialize conda

Reload your shell and verify:
```bash
source ~/.bashrc
conda --version
```

---

## Step 3 — Create the Python 3.8 environment

```bash
conda create -n eureka python=3.8
conda activate eureka
```

Your prompt should now show `(eureka)` at the start.

---

## Step 4 — Download and install IsaacGym Preview 4

IsaacGym is not on pip — download it manually:

1. Go to [developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) in your Windows browser
2. Download `IsaacGym_Preview_4_Package.tar.gz` (free NVIDIA account required)

Copy it from Windows into WSL2 and install (your Windows `C:\` drive is at `/mnt/c/` in Ubuntu):
```bash
cd ~
cp /mnt/c/Users/<your-username>/Downloads/IsaacGym_Preview_4_Package.tar.gz .
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
```

Fix a missing shared library that IsaacGym needs:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

Test the installation (a viewer window should open with animated robot figures):
```bash
cd ~/isaacgym/python/examples
python joint_monkey.py
```

Close the window when satisfied.

---

## Step 5 — Install a C++ compiler

IsaacGym compiles a CUDA extension at first run and needs `g++`:
```bash
sudo apt update && sudo apt install -y build-essential
```

---

## Step 6 — Copy the repo into WSL2

The repo must live on the Linux filesystem — running from `/mnt/c/...` causes permission errors.

```bash
cp -r /mnt/c/Users/<your-username>/OneDrive/Desktop/Programming/EurekaFriends/LOVE_Island_Reward_Functions ~/eureka_project
```

---

## Step 7 — Install the repo packages

```bash
cd ~/eureka_project
pip install -e .
cd isaacgymenvs && pip install -e .
cd ../rl_games && pip install -e .
```

---

## Step 8 — Fix dependency version issues

These are compatibility fixes required because the repo was written for older library versions:

```bash
# OpenAI library must be the old API version
pip install openai==0.28

# Pillow 10+ breaks on Python 3.8
pip install "Pillow<10.0.0"

# gpustat is required for GPU selection
pip install gpustat
```

---

## Step 9 — Fix CUDA visibility for IsaacGym

WSL2 puts `libcuda.so` in a non-standard path that IsaacGym can't find by default. Without this fix, PhysX falls back to CPU while PyTorch stays on GPU, causing a crash.

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

**To avoid re-running these exports every session**, add them to your `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 10 — Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

To persist across sessions:
```bash
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
```

---

## Running Eureka

Navigate to the eureka directory and run:
```bash
conda activate eureka
cd ~/eureka_project/eureka
python eureka.py env=cartpole sample=2 iteration=1 model=gpt-4-0314
```

**Key parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `env` | `shadow_hand` | Robot task to run (see `eureka/cfg/env/` for full list) |
| `sample` | `3` | Reward candidates generated per iteration |
| `iteration` | `1` | Number of Eureka LLM→RL loops |
| `model` | `gpt-4-0314` | OpenAI model to use |
| `max_iterations` | `3000` | RL training steps per reward candidate |

**Recommended for fast testing:**
```bash
python eureka.py env=cartpole sample=2 iteration=1 max_iterations=500 model=gpt-4-0314
```

**Outputs** are saved to `eureka/outputs/eureka/<timestamp>/` — includes logs, all generated reward functions, and trained policies.

---

## Notes

- Training runs on **CPU** by default in this WSL2 setup due to IsaacGym's PhysX CUDA limitations. This is slow but functional. Lighter environments (`cartpole`, `ant`) are more manageable than heavy ones (`shadow_hand`).
- The repo must always be run from `~/eureka_project/`, not from the Windows-mounted path.
- Each new Ubuntu session requires `conda activate eureka` before running anything.
