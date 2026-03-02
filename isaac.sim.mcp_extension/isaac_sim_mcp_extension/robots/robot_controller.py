"""Robot articulation control helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from omni.isaac.core import World

from isaac_sim_mcp_extension.core_state import ExtensionState


class RobotController:
    """Encapsulates articulation state queries and action application."""

    def __init__(self, state: ExtensionState) -> None:
        self._state = state

    def get_robot_state(self, robot_prim_path: str = "/G1") -> Dict[str, Any]:
        try:
            from omni.isaac.core import SimulationContext
            from omni.isaac.core.articulations import Articulation

            if self._state.policy.robot_articulation is None:
                sim_ctx = SimulationContext.instance()
                if sim_ctx is None:
                    return {"status": "error", "message": "SimulationContext not initialized"}
                self._state.policy.robot_articulation = Articulation(prim_path=robot_prim_path)
                sim_view = getattr(sim_ctx, "physics_sim_view", None)
                if sim_view is not None:
                    self._state.policy.robot_articulation.initialize(sim_view)

            robot = self._state.policy.robot_articulation
            joint_positions = robot.get_joint_positions()
            joint_velocities = robot.get_joint_velocities()

            # Fail fast if tensor data is unavailable (stale view).
            if joint_positions is None or len(joint_positions) == 0:
                import carb as _carb
                _carb.log_warn("[ROBOT_CTRL] get_joint_positions returned None/empty — view may be stale, attempting re-init")
                sim_ctx = SimulationContext.instance()
                sim_view = getattr(sim_ctx, "physics_sim_view", None) if sim_ctx else None
                if sim_view is not None:
                    # Re-initialize with current view; never call step() here — we may be
                    # inside a physics callback and stepping would recurse into the renderer.
                    robot.initialize(sim_view)
                    joint_positions = robot.get_joint_positions()
                    joint_velocities = robot.get_joint_velocities()
                if joint_positions is None or len(joint_positions) == 0:
                    return {"status": "error", "message": "Joint positions unavailable (physics view stale). Restart simulation."}

            base_position, base_orientation = robot.get_world_pose()
            base_lin_vel = robot.get_linear_velocity()
            base_ang_vel = robot.get_angular_velocity()
            return {
                "status": "success",
                "state": {
                    "joint_positions": joint_positions.tolist(),
                    "joint_velocities": joint_velocities.tolist() if joint_velocities is not None else [0.0] * len(joint_positions),
                    "base_position": base_position.tolist() if base_position is not None else [0, 0, 0],
                    "base_orientation": base_orientation.tolist() if base_orientation is not None else [1, 0, 0, 0],
                    "base_linear_velocity": base_lin_vel.tolist() if base_lin_vel is not None else [0, 0, 0],
                    "base_angular_velocity": base_ang_vel.tolist() if base_ang_vel is not None else [0, 0, 0],
                    "num_joints": len(joint_positions),
                    "dof_names": list(robot.dof_names) if robot.dof_names is not None else [],
                },
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def apply_joint_actions(
        self,
        robot_prim_path: str = "/G1",
        joint_positions: Optional[List[float]] = None,
        joint_velocities: Optional[List[float]] = None,
        joint_efforts: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        try:
            from omni.isaac.core.utils.types import ArticulationAction

            if self._state.policy.robot_articulation is None or self._state.policy.controller is None:
                return {"status": "error", "message": "Robot not initialized. Call load_policy or get_robot_state first."}

            action = ArticulationAction(
                joint_positions=np.array(joint_positions) if joint_positions is not None else None,
                joint_velocities=np.array(joint_velocities) if joint_velocities is not None else None,
                joint_efforts=np.array(joint_efforts) if joint_efforts is not None else None,
            )
            self._state.policy.controller.apply_action(action)
            return {"status": "success", "message": "Joint actions applied"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def reset_robot_pose(
        self,
        robot_prim_path: str = "/G1",
        base_position: Optional[List[float]] = None,
        joint_positions: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        try:
            from omni.isaac.core.utils.types import ArticulationAction
            from pxr import Gf, UsdGeom
            import omni.usd

            if base_position is None:
                base_position = [0.0, 0.0, 0.8]

            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(robot_prim_path)
            if robot_prim.IsValid():
                xformable = UsdGeom.Xformable(robot_prim)
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(float(base_position[0]), float(base_position[1]), float(base_position[2])))

            if joint_positions is not None and self._state.policy.controller is not None:
                self._state.policy.controller.apply_action(ArticulationAction(joint_positions=np.array(joint_positions)))
            elif joint_positions is None and self._state.policy.robot_articulation is not None:
                num_joints = len(self._state.policy.robot_articulation.dof_names)
                standing_pose = np.zeros(num_joints)
                if num_joints >= 12:
                    standing_pose[0] = 0.0
                    standing_pose[1] = 0.2
                    standing_pose[2] = -0.4
                    standing_pose[3] = 0.8
                    standing_pose[4] = -0.4
                    standing_pose[5] = 0.0
                    standing_pose[6] = 0.0
                    standing_pose[7] = -0.2
                    standing_pose[8] = -0.4
                    standing_pose[9] = 0.8
                    standing_pose[10] = -0.4
                    standing_pose[11] = 0.0
                self._state.policy.controller.apply_action(ArticulationAction(joint_positions=standing_pose))

            world = World.instance()
            if world is not None and getattr(getattr(world, "_physics_context", None), "_step", None) is not None:
                for _ in range(10):
                    world.step(render=False)

            return {"status": "success", "message": f"Robot reset to position {base_position}"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
