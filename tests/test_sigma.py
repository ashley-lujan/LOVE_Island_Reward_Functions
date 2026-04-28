"""Unit tests for sigma.py — reward function loading and σ computation."""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eureka"))

from utils.love_island.sigma import (
    compute_sigma_batch,
    load_reward_fn,
    load_terminal_states,
)


def _write_reward_fn(path: str, scale: float) -> None:
    code = (
        f"import torch\n"
        f"from typing import Tuple, Dict\n"
        f"\n"
        f"@torch.jit.script\n"
        f"def compute_reward(\n"
        f"    joint_pos: torch.Tensor,\n"
        f"    joint_vel: torch.Tensor,\n"
        f") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:\n"
        f"    reward = joint_pos.mean(dim=1) * {scale}\n"
        f"    return reward, {{\"base\": reward}}\n"
    )
    with open(path, "w") as f:
        f.write(code)


def _write_terminal_state(path: str, n: int = 4, j: int = 2) -> dict:
    jp = torch.randn(n, j)
    jv = torch.randn(n, j)
    gt = torch.zeros(n)
    torch.save({"joint_pos": jp, "joint_vel": jv, "gt_reward": gt}, path)
    return {"joint_pos": jp, "joint_vel": jv, "gt_reward": gt}


class TestLoadRewardFn(unittest.TestCase):
    def test_loads_valid_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "reward.py")
            _write_reward_fn(p, scale=1.0)
            fn = load_reward_fn(p)
            self.assertIsNotNone(fn)

    def test_returns_none_for_missing_file(self):
        fn = load_reward_fn("/nonexistent/path.py")
        self.assertIsNone(fn)

    def test_returns_none_for_file_without_compute_reward(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "no_reward.py")
            with open(p, "w") as f:
                f.write("x = 1\n")
            fn = load_reward_fn(p)
            self.assertIsNone(fn)

    def test_loaded_function_is_callable(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "reward.py")
            _write_reward_fn(p, scale=2.0)
            fn = load_reward_fn(p)
            jp = torch.zeros(3, 2)
            jv = torch.zeros(3, 2)
            r, comp = fn(jp, jv)
            self.assertEqual(r.shape, (3,))
            self.assertIn("base", comp)


class TestLoadTerminalStates(unittest.TestCase):
    def test_loads_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_terminal_state(os.path.join(d, "terminal_0.pt"), n=5)
            states = load_terminal_states(d)
            self.assertIsNotNone(states)
            self.assertEqual(states["joint_pos"].shape[0], 5)

    def test_concatenates_multiple_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write_terminal_state(os.path.join(d, "terminal_0.pt"), n=3)
            _write_terminal_state(os.path.join(d, "terminal_1.pt"), n=4)
            states = load_terminal_states(d)
            self.assertEqual(states["joint_pos"].shape[0], 7)

    def test_returns_none_for_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            states = load_terminal_states(d)
            self.assertIsNone(states)


class TestComputeSigmaBatch(unittest.TestCase):
    def test_sigma_zero_for_identical_functions(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "r1.py")
            p2 = os.path.join(d, "r2.py")
            _write_reward_fn(p1, scale=1.0)
            _write_reward_fn(p2, scale=1.0)
            fns = [load_reward_fn(p1), load_reward_fn(p2)]
            jp = torch.ones(5, 2)
            jv = torch.zeros(5, 2)
            sigma, _ = compute_sigma_batch(fns, jp, jv)
            np.testing.assert_allclose(sigma, np.zeros(5), atol=1e-4)

    def test_sigma_positive_for_different_scales(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "r1.py")
            p2 = os.path.join(d, "r2.py")
            _write_reward_fn(p1, scale=1.0)
            _write_reward_fn(p2, scale=3.0)
            fns = [load_reward_fn(p1), load_reward_fn(p2)]
            jp = torch.ones(5, 2)
            jv = torch.zeros(5, 2)
            sigma, mean_r = compute_sigma_batch(fns, jp, jv)
            self.assertTrue(np.all(sigma > 0), f"Expected σ > 0, got {sigma}")
            np.testing.assert_allclose(mean_r, np.full(5, 2.0), atol=1e-4)

    def test_handles_none_fns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.py")
            _write_reward_fn(p, scale=1.0)
            fns = [load_reward_fn(p), None]
            sigma, _ = compute_sigma_batch(fns, torch.ones(3, 2), torch.zeros(3, 2))
            self.assertEqual(sigma.shape, (3,))

    def test_all_none_returns_zeros(self):
        sigma, mean_r = compute_sigma_batch([None, None], torch.ones(4, 2), torch.zeros(4, 2))
        np.testing.assert_array_equal(sigma, np.zeros(4))

    def test_output_shape_matches_input(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "r1.py")
            p2 = os.path.join(d, "r2.py")
            _write_reward_fn(p1, scale=1.0)
            _write_reward_fn(p2, scale=2.0)
            fns = [load_reward_fn(p1), load_reward_fn(p2)]
            n = 10
            sigma, mean_r = compute_sigma_batch(fns, torch.randn(n, 2), torch.randn(n, 2))
            self.assertEqual(sigma.shape, (n,))
            self.assertEqual(mean_r.shape, (n,))


if __name__ == "__main__":
    unittest.main()
