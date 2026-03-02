"""Robot creation and pose tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import carb
import numpy as np
import omni.timeline
import omni.usd
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path

from isaac_sim_mcp_extension.core_state import ExtensionState
from isaac_sim_mcp_extension.robots.robot_registry import ROBOT_REGISTRY
from isaac_sim_mcp_extension.scene.scene_manager import SceneManager


class RobotFactory:
    """Creates robots from a data-driven registry."""

    G1_STANDING_BASE_HEIGHT = 0.74
    G1_NAV_VELOCITY_CMD_PATH = "/tmp/g1_nav_velocity_cmd.json"

    def __init__(self, state: ExtensionState, scene_manager: SceneManager) -> None:
        self._state = state
        self._scene_manager = scene_manager

    def create_robot(self, robot_type: str = "g1", position: List[float] = [0, 0, 0]) -> Dict[str, Any]:
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()

        spec = ROBOT_REGISTRY.get(robot_type.lower())
        if spec is None:
            return {"status": "error", "message": f"Unknown robot type '{robot_type}'"}

        x = 0.0
        y = 0.0
        if position is not None and len(position) >= 2:
            x, y = float(position[0]), float(position[1])
        terrain_z = self._scene_manager.get_terrain_height_at(x, y)
        default_z = terrain_z + self.G1_STANDING_BASE_HEIGHT
        if position is None or len(position) < 3:
            position = [x, y, default_z]
        elif len(position) >= 3 and float(position[2]) == 0.0:
            # Caller passed (x, y, 0) — use standing height so robot is above ground
            position = [x, y, default_z]
        else:
            position = [float(position[0]), float(position[1]), float(position[2])]

        assets_root_path = get_assets_root_path()
        asset_path = assets_root_path + spec.asset_subpath
        add_reference_to_stage(asset_path, spec.prim_path)
        robot_prim = XFormPrim(prim_path=spec.prim_path)
        robot_prim.set_world_pose(position=np.array(position))
        return {"status": "success", "message": f"{robot_type} robot created", "prim_path": spec.prim_path}

    def set_velocity_command(self, lin_vel_x: float = 0.5, lin_vel_y: float = 0.0, ang_vel_z: float = 0.0) -> Dict[str, Any]:
        lin_vel_x = max(0.0, min(1.0, float(lin_vel_x)))
        lin_vel_y = max(-0.5, min(0.5, float(lin_vel_y)))
        ang_vel_z = max(-1.0, min(1.0, float(ang_vel_z)))
        payload = {"lin_vel_x": lin_vel_x, "lin_vel_y": lin_vel_y, "ang_vel_z": ang_vel_z}
        self._state.vel_cmd_x = lin_vel_x
        self._state.vel_cmd_y = lin_vel_y
        self._state.vel_cmd_yaw = ang_vel_z
        try:
            with open(self.G1_NAV_VELOCITY_CMD_PATH, "w", encoding="utf-8") as file:
                json.dump(payload, file)
        except Exception as exc:
            carb.log_warn(f"Could not persist velocity command: {exc}")
        return {"status": "success", "message": "Velocity command set",
                "lin_vel_x": payload.get("lin_vel_x"), "lin_vel_y": payload.get("lin_vel_y"),
                "ang_vel_z": payload.get("ang_vel_z")}

    def get_robot_pose(self, prim_path: str = "/G1") -> Dict[str, Any]:
        try:
            art = self._state.policy.robot_articulation
            art_path = getattr(art, "prim_path", None) if art is not None else None
            if art is not None and art_path is not None and art_path.startswith(prim_path):
                pos, quat = art.get_world_pose()
                return {
                    "status": "success",
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "orientation_quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                }

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return {"status": "error", "message": f"Prim not found: {prim_path}"}
            xform = XFormPrim(prim_path)
            pos, quat = xform.get_world_pose()
            return {
                "status": "success",
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "orientation_quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
