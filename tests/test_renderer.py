"""Unit tests for renderer.py — trajectory loading and GIF rendering."""

import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eureka"))

from utils.love_island.renderer import (
    load_recent_trajectories,
    load_trajectories,
    render_cartpole_gif,
    render_episode_gifs,
)
from utils.love_island.buffer import EpisodeRecord


def _write_traj(directory: str, step: int, env_idx: int, T: int = 30) -> str:
    fname = f"traj_{step}_{env_idx}.pt"
    fpath = os.path.join(directory, fname)
    torch.save(
        {
            "joint_pos_seq": torch.zeros(T, 2),
            "joint_vel_seq": torch.zeros(T, 2),
            "episode_length": T,
            "cart_dof_idx": 0,
            "pole_dof_idx": 1,
        },
        fpath,
    )
    idx_path = os.path.join(directory, "index.json")
    with open(idx_path, "a") as f:
        f.write(json.dumps({"step": step, "env_idx": env_idx, "filename": fname}) + "\n")
    return fname


class TestLoadTrajectories(unittest.TestCase):
    def test_loads_by_traj_key(self):
        with tempfile.TemporaryDirectory() as d:
            _write_traj(d, step=100, env_idx=3)
            result = load_trajectories(d, ["100_3"])
            self.assertIn("100_3", result)
            self.assertEqual(result["100_3"]["episode_length"], 30)

    def test_missing_key_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            _write_traj(d, step=100, env_idx=3)
            result = load_trajectories(d, ["999_0"])
            self.assertEqual(result, {})

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            result = load_trajectories(d, ["100_3"])
            self.assertEqual(result, {})

    def test_trajectory_shape(self):
        with tempfile.TemporaryDirectory() as d:
            _write_traj(d, step=50, env_idx=1, T=25)
            data = load_trajectories(d, ["50_1"])["50_1"]
            self.assertEqual(data["joint_pos_seq"].shape, (25, 2))
            self.assertEqual(data["joint_vel_seq"].shape, (25, 2))


class TestLoadRecentTrajectories(unittest.TestCase):
    def test_loads_last_n(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                _write_traj(d, step=i * 10, env_idx=0)
            result = load_recent_trajectories(d, n=2)
            self.assertEqual(len(result), 2)

    def test_returns_empty_for_no_index(self):
        with tempfile.TemporaryDirectory() as d:
            result = load_recent_trajectories(d, n=2)
            self.assertEqual(result, {})


class TestRenderCartpoleGif(unittest.TestCase):
    def test_gif_file_created(self):
        with tempfile.TemporaryDirectory() as d:
            traj_data = {
                "joint_pos_seq": torch.zeros(15, 2),
                "joint_vel_seq": torch.zeros(15, 2),
                "episode_length": 15,
                "cart_dof_idx": 0,
                "pole_dof_idx": 1,
            }
            out = os.path.join(d, "test.gif")
            render_cartpole_gif(traj_data, out, title="unit test", fps=10)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)

    def test_non_zero_pole_angle_renders(self):
        with tempfile.TemporaryDirectory() as d:
            import math
            jp = torch.zeros(10, 2)
            jp[:, 1] = torch.linspace(0.0, math.pi / 4, 10)
            traj_data = {
                "joint_pos_seq": jp,
                "joint_vel_seq": torch.zeros(10, 2),
                "episode_length": 10,
                "cart_dof_idx": 0,
                "pole_dof_idx": 1,
            }
            out = os.path.join(d, "pole.gif")
            render_cartpole_gif(traj_data, out, fps=5)
            self.assertTrue(os.path.exists(out))


class TestRenderEpisodeGifs(unittest.TestCase):
    def _make_record(self, traj_key=None):
        return EpisodeRecord(
            episode_id="test-id",
            joint_pos_snap=torch.zeros(1, 2),
            joint_vel_snap=torch.zeros(1, 2),
            sigma=0.5,
            iteration_cached=0,
            traj_key=traj_key,
        )

    def test_fallback_to_recent_when_no_traj_key(self):
        with tempfile.TemporaryDirectory() as d:
            traj_dir = os.path.join(d, "trajs")
            os.makedirs(traj_dir)
            _write_traj(traj_dir, step=1, env_idx=0, T=10)
            gif_dir = os.path.join(d, "gifs")
            record = self._make_record(traj_key=None)
            sigma_breakdown = {"mean_sigma_overall": 0.5}
            paths = render_episode_gifs(
                [record], [traj_dir], gif_dir, "binary", sigma_breakdown,
                fps=5, max_gifs=1,
            )
            self.assertEqual(len(paths), 1)
            self.assertTrue(os.path.exists(paths[0]))

    def test_exact_match_by_traj_key(self):
        with tempfile.TemporaryDirectory() as d:
            traj_dir = os.path.join(d, "trajs")
            os.makedirs(traj_dir)
            _write_traj(traj_dir, step=42, env_idx=7, T=10)
            _write_traj(traj_dir, step=99, env_idx=0, T=10)
            gif_dir = os.path.join(d, "gifs")
            record = self._make_record(traj_key="42_7")
            sigma_breakdown = {"mean_sigma_overall": 0.4}
            paths = render_episode_gifs(
                [record], [traj_dir], gif_dir, "corrective", sigma_breakdown,
                fps=5, max_gifs=1,
            )
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith("episode_0.gif"))

    def test_returns_empty_for_empty_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            gif_dir = os.path.join(d, "gifs")
            paths = render_episode_gifs(
                [], [], gif_dir, "binary", {}, fps=5, max_gifs=2,
            )
            self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
