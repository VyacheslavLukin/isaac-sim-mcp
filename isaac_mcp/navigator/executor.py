"""Executor abstractions for planner-to-robot control."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from threading import Lock
from typing import Any

import numpy as np


class RobotExecutor(ABC):
    """Interface for sending velocity commands and reading robot pose."""

    @abstractmethod
    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        """Send velocity command in m/s and rad/s."""

    @abstractmethod
    def get_pose(self) -> tuple[np.ndarray, float]:
        """Return (position_xy, yaw_radians)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop robot motion."""


class IsaacSimExecutor(RobotExecutor):
    """Isaac Sim command adapter for the MCP planner."""

    def __init__(self, connection: Any, command_lock: Lock | None = None, robot_prim_path: str = "/G1"):
        self._connection = connection
        self._command_lock = command_lock
        self._robot_prim_path = robot_prim_path

    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        params = {"lin_vel_x": vx, "lin_vel_y": vy, "ang_vel_z": yaw_rate}
        with self._lock_ctx():
            self._connection.send_command("set_velocity_command", params)

    def get_pose(self) -> tuple[np.ndarray, float]:
        with self._lock_ctx():
            result = self._connection.send_command("get_robot_pose", {"prim_path": self._robot_prim_path})

        if result.get("status") == "error":
            raise RuntimeError(result.get("message", "Failed to get robot pose"))

        if "result" in result and isinstance(result["result"], dict):
            result = result["result"]

        pos = result.get("position")
        quat = result.get("orientation_quat")
        if not pos or not quat or len(pos) < 2 or len(quat) < 4:
            raise RuntimeError(f"Unexpected robot pose payload: {result}")

        return np.array([float(pos[0]), float(pos[1])], dtype=float), self._quat_to_yaw(quat)

    def stop(self) -> None:
        self.set_velocity_command(0.0, 0.0, 0.0)

    def _lock_ctx(self):
        return self._command_lock if self._command_lock is not None else nullcontext()

    @staticmethod
    def _quat_to_yaw(quat: list[float]) -> float:
        # Isaac returns quaternion as [w, x, y, z] in Z-up frame.
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
