"""
Smoke test for the LOVE Island / Eureka CHPC environment.

Checks (in order):
  1. PyTorch + CUDA GPU access
  2. Isaac Gym import and gym handle acquisition
  3. Isaac Gym headless sim creation (confirms GPU-backed simulation works)
  4. Eureka utility imports (hydra, omegaconf, openai, tensorboard)
  5. Multi-agent prompt file loading
  6. OpenAI API reachability (skipped if OPENAI_API_KEY is unset)

Run via:  sbatch job_scripts/smoke_test.sh
Or local: python eureka/smoke_test.py  (no GPU available — steps 1-3 will report failures)
"""

import os
import sys
import traceback

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []


def check(label, fn):
    try:
        msg = fn()
        results.append((PASS, label, msg or ""))
    except Exception as e:
        results.append((FAIL, label, str(e)))


# ---------------------------------------------------------------------------
# 1. PyTorch + CUDA
# ---------------------------------------------------------------------------
def _check_torch_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False — no GPU visible")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    t = torch.ones(3, device="cuda")
    assert t.sum().item() == 3
    return f"{n} GPU(s): {', '.join(names)}"

check("PyTorch CUDA access", _check_torch_cuda)


# ---------------------------------------------------------------------------
# 2. Isaac Gym import
# ---------------------------------------------------------------------------
def _check_isaacgym_import():
    import isaacgym  # noqa: F401 — side-effect: registers gym backend
    from isaacgym import gymapi
    gym = gymapi.acquire_gym()
    if gym is None:
        raise RuntimeError("acquire_gym() returned None")
    return "gym handle acquired"

check("Isaac Gym import + acquire_gym", _check_isaacgym_import)


# ---------------------------------------------------------------------------
# 3. Isaac Gym headless sim creation
# ---------------------------------------------------------------------------
def _check_isaacgym_sim():
    import isaacgym  # noqa: F401
    from isaacgym import gymapi
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = False  # minimal: physics on GPU, pipeline CPU-safe
    sim = gym.create_sim(
        compute_device=0,
        graphics_device=-1,   # headless — no display needed
        type=gymapi.SIM_PHYSX,
        params=sim_params,
    )
    if sim is None:
        raise RuntimeError("gym.create_sim() returned None — GPU sim creation failed")
    gym.destroy_sim(sim)
    return "headless sim created and destroyed successfully"

check("Isaac Gym headless sim (GPU)", _check_isaacgym_sim)


# ---------------------------------------------------------------------------
# 4. Eureka / framework imports
# ---------------------------------------------------------------------------
def _check_framework_imports():
    import hydra                           # noqa: F401
    from omegaconf import OmegaConf        # noqa: F401
    import openai                          # noqa: F401
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    import matplotlib                      # noqa: F401
    import numpy                           # noqa: F401
    return "hydra, omegaconf, openai, tensorboard, matplotlib, numpy"

check("Eureka framework imports", _check_framework_imports)


# ---------------------------------------------------------------------------
# 5. Multi-agent prompt files
# ---------------------------------------------------------------------------
def _check_prompts():
    EUREKA_ROOT = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(EUREKA_ROOT, "utils", "prompts")

    sys.path.insert(0, EUREKA_ROOT)
    from utils.multi_agent import load_multi_agent_prompts

    prompts = load_multi_agent_prompts(prompt_dir)
    if not prompts:
        raise RuntimeError("load_multi_agent_prompts returned empty dict")
    return f"{len(prompts)} prompt files loaded: {', '.join(sorted(prompts))}"

check("Multi-agent prompt loading", _check_prompts)


# ---------------------------------------------------------------------------
# 6. OpenAI API reachability
# ---------------------------------------------------------------------------
def _check_openai_api():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None  # will be marked SKIP below

    import openai as _openai
    _openai.api_key = api_key
    resp = _openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=5,
        temperature=0,
    )
    reply = resp["choices"][0]["message"]["content"].strip()
    return f"API reachable; model replied: '{reply}'"

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key:
    check("OpenAI API reachability", _check_openai_api)
else:
    results.append((SKIP, "OpenAI API reachability", "OPENAI_API_KEY not set"))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("LOVE Island / Eureka CHPC Smoke Test Results")
print("=" * 60)
for status, label, detail in results:
    detail_str = f"  →  {detail}" if detail else ""
    print(f"  {status}  {label}{detail_str}")
print("=" * 60)

n_fail = sum(1 for s, _, _ in results if s == FAIL)
if n_fail:
    print(f"\n{n_fail} check(s) FAILED — review the details above.")
    sys.exit(1)
else:
    print("\nAll checks passed.")
