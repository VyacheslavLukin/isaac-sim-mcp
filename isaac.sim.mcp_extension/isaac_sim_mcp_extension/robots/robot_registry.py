"""Robot registry definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RobotSpec:
    robot_type: str
    asset_subpath: str
    prim_path: str


ROBOT_REGISTRY: Dict[str, RobotSpec] = {
    "franka": RobotSpec("franka", "/Isaac/Robots/Franka/franka_alt_fingers.usd", "/Franka"),
    "jetbot": RobotSpec("jetbot", "/Isaac/Robots/Jetbot/jetbot.usd", "/Jetbot"),
    "carter": RobotSpec("carter", "/Isaac/Robots/Carter/carter.usd", "/Carter"),
    "g1": RobotSpec("g1", "/Isaac/Robots/Unitree/G1/g1.usd", "/G1"),
    "g1_minimal": RobotSpec("g1_minimal", "/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd", "/G1"),
    "go1": RobotSpec("go1", "/Isaac/Robots/Unitree/Go1/go1.usd", "/Go1"),
}
