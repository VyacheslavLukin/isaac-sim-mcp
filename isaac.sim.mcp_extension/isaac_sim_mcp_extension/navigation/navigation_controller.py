"""Navigation controller built on top of policy walking."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import omni.kit.app

from isaac_sim_mcp_extension.core_state import ExtensionState
from isaac_sim_mcp_extension.policy.policy_runner import PolicyRunner
from isaac_sim_mcp_extension.robots.robot_factory import RobotFactory


class NavigationController:
    """Point-to-point navigation using policy velocity commands."""

    def __init__(self, state: ExtensionState, policy_runner: PolicyRunner, robot_factory: RobotFactory) -> None:
        self._state = state
        self._policy_runner = policy_runner
        self._robot_factory = robot_factory

    def navigate_to(
        self,
        target_position: List[float],
        robot_prim_path: str = "/G1",
        policy_path: str = "",
        arrival_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        try:
            if len(target_position) < 2:
                return {"status": "error", "message": "target_position must have at least [x, y]"}

            target_x = float(target_position[0])
            target_y = float(target_position[1])
            if self._state.navigation.subscription is not None:
                try:
                    self._state.navigation.subscription.unsubscribe()
                except Exception:
                    pass
                self._state.navigation.subscription = None

            if self._state.policy.walk_subscription is None:
                if not policy_path:
                    return {"status": "error", "message": "policy_path is required to start walking."}
                walk_result = self._policy_runner.start_g1_policy_walk(policy_path=policy_path, robot_prim_path=robot_prim_path)
                if walk_result.get("status") != "success":
                    return walk_result

            self._state.navigation.active = True
            self._state.navigation.status = "navigating"
            self._state.navigation.target = [target_x, target_y]
            self._state.navigation.threshold = arrival_threshold
            self._state.navigation.robot_prim_path = robot_prim_path

            def callback(_event: Any) -> None:
                if not self._state.navigation.active:
                    return
                pose_result = self._robot_factory.get_robot_pose(robot_prim_path)
                if pose_result.get("status") != "success":
                    return
                pos = pose_result["position"]
                quat = pose_result["orientation_quat"]
                dx = target_x - pos[0]
                dy = target_y - pos[1]
                distance = float(np.sqrt(dx**2 + dy**2))
                if distance < arrival_threshold:
                    self._state.vel_cmd_x = 0.0
                    self._state.vel_cmd_y = 0.0
                    self._state.vel_cmd_yaw = 0.0
                    self._state.navigation.active = False
                    self._state.navigation.status = "arrived"
                    if self._state.navigation.subscription is not None:
                        try:
                            self._state.navigation.subscription.unsubscribe()
                        except Exception:
                            pass
                        self._state.navigation.subscription = None
                    return

                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                current_yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
                target_yaw_angle = float(np.arctan2(dy, dx))
                yaw_error = float((target_yaw_angle - current_yaw + np.pi) % (2.0 * np.pi) - np.pi)
                ang_vel = float(np.clip(yaw_error * 1.5, -1.0, 1.0))
                lin_vel_x = 0.1 if abs(yaw_error) > 0.3 else float(np.clip(distance * 0.5, 0.0, 1.0))
                self._state.vel_cmd_x = lin_vel_x
                self._state.vel_cmd_y = 0.0
                self._state.vel_cmd_yaw = ang_vel

            stream = omni.kit.app.get_app().get_pre_update_event_stream()
            self._state.navigation.subscription = stream.create_subscription_to_pop(callback, name="g1_nav_control_callback")
            return {
                "status": "success",
                "message": f"Navigation started toward [{target_x:.2f}, {target_y:.2f}]. Call get_navigation_status() to monitor progress.",
                "target_position": [target_x, target_y],
                "arrival_threshold": arrival_threshold,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def stop_navigation(self) -> Dict[str, Any]:
        try:
            prev_status = self._state.navigation.status
            if self._state.navigation.subscription is not None:
                self._state.navigation.subscription.unsubscribe()
                self._state.navigation.subscription = None
            self._state.navigation.active = False
            self._state.navigation.status = "idle"
            self._state.vel_cmd_x = 0.0
            self._state.vel_cmd_y = 0.0
            self._state.vel_cmd_yaw = 0.0
            return {"status": "success", "message": f"Navigation stopped (was: {prev_status}). Policy walk is still running."}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def get_navigation_status(self) -> Dict[str, Any]:
        try:
            info: Dict[str, Any] = {
                "nav_active": self._state.navigation.active,
                "nav_status": self._state.navigation.status,
                "target_position": self._state.navigation.target,
                "arrival_threshold": self._state.navigation.threshold,
            }
            if self._state.navigation.active or self._state.navigation.status in ("navigating", "arrived"):
                robot_prim = self._state.navigation.robot_prim_path or "/G1"
                pose_result = self._robot_factory.get_robot_pose(robot_prim)
                if pose_result.get("status") == "success":
                    pos = pose_result["position"]
                    info["current_position"] = pos
                    if self._state.navigation.target is not None:
                        dx = self._state.navigation.target[0] - pos[0]
                        dy = self._state.navigation.target[1] - pos[1]
                        info["distance_to_target"] = float(np.sqrt(dx**2 + dy**2))
            return {"status": "success", "navigation": info}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
