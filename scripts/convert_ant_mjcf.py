#!/usr/bin/env python3
"""Convert nv_ant.xml → ant.usd using Isaac Lab's MjcfConverter.

Runs INSIDE the Isaac Lab container (requires GPU via --nv).
Invoked by job_scripts/slurm_convert_ant.sh — do not call directly.

Usage (inside container):
  /isaac-sim/python.sh scripts/convert_ant_mjcf.py \
      --input /repo/isaacgymenvs/assets/mjcf/nv_ant.xml \
      --output-dir /ant-output \
      --output-name ant.usd
"""

# SimulationApp must be created before any other Isaac Sim / Isaac Lab imports.
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import argparse
import os

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to nv_ant.xml inside container")
parser.add_argument("--output-dir", required=True, help="Directory for output USD")
parser.add_argument("--output-name", default="ant.usd")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Print available fields so we can see what this version of Isaac Lab accepts
import dataclasses
print("MjcfConverterCfg fields:", [f.name for f in dataclasses.fields(MjcfConverterCfg)])

cfg = MjcfConverterCfg(
    asset_path=os.path.abspath(args.input),
    usd_dir=args.output_dir,
    usd_file_name=args.output_name,
    fix_base=False,       # ant is free-floating (has freejoint)
    self_collision=False,
)
converter = MjcfConverter(cfg)
print(f"Ant USD written to: {converter.usd_path}")

sim_app.close()
