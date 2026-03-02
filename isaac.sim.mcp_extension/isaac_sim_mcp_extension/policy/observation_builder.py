"""Policy observation construction helpers."""

from __future__ import annotations

from typing import Any, Optional

import carb
import numpy as np


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse quaternion q=[w,x,y,z]."""
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class ObservationBuilder:
    """Builds 123-dim and 310-dim observations for G1 policy execution.

    All joint arrays arrive in robot DOF order. The policy_to_robot index array
    reorders them to Isaac Lab policy order before concatenation.  The caller
    must still supply last_action already in policy order.
    """

    def compute_height_scan(self, base_pos: np.ndarray, base_orient: np.ndarray) -> np.ndarray:
        """Compute 187-element (17x11) height scan matching Isaac Lab GridPatternCfg."""
        try:
            from omni.physx import get_physx_scene_query_interface

            query = get_physx_scene_query_interface()
            robot_x = float(base_pos[0])
            robot_y = float(base_pos[1])
            robot_z = float(base_pos[2])
            qw, qx, qy, qz = [float(v) for v in base_orient]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            # Grid matches GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
            xs = np.linspace(-0.8, 0.8, 17)
            ys = np.linspace(-0.5, 0.5, 11)
            heights = []
            for x_loc in xs:
                for y_loc in ys:
                    wx = robot_x + cos_yaw * x_loc - sin_yaw * y_loc
                    wy = robot_y + sin_yaw * x_loc + cos_yaw * y_loc
                    hit = query.raycast_closest(carb.Float3(wx, wy, robot_z + 20.0), carb.Float3(0.0, 0.0, -1.0), 30.0)
                    terrain_z = float(hit["position"][2]) if (hit and hit.get("hit")) else 0.0
                    # Match Isaac Lab: obs = (sensor_z - hit_z) - 0.5
                    heights.append(np.clip((robot_z - terrain_z) - 0.5, -1.0, 1.0))
            return np.array(heights, dtype=np.float32)
        except Exception:
            return np.zeros(187, dtype=np.float32)

    def build(
        self,
        state: dict[str, Any],
        velocity_commands: np.ndarray,
        # default_pos and last_action must be in robot DOF order
        default_pos_robot_order: np.ndarray,
        last_action_policy_order: np.ndarray,
        add_height_scan: bool,
        # Index array: policy_to_robot[pi] = robot_dof_index for policy index pi.
        # None means robot DOF order already matches policy order (identity mapping).
        policy_to_robot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        joint_pos_robot = np.array(state["joint_positions"])
        joint_vel_robot = np.array(state["joint_velocities"])
        base_pos = np.array(state["base_position"])
        base_orient = np.array(state["base_orientation"])
        base_lin_vel_world = np.array(state["base_linear_velocity"])
        base_ang_vel_world = np.array(state["base_angular_velocity"])

        base_lin_vel = quat_rotate_inverse(base_orient, base_lin_vel_world)
        base_ang_vel = quat_rotate_inverse(base_orient, base_ang_vel_world)
        projected_gravity = quat_rotate_inverse(base_orient, np.array([0.0, 0.0, -1.0]))

        # Reorder robot DOF arrays to Isaac Lab policy order before building obs.
        if policy_to_robot is not None:
            joint_pos_policy = joint_pos_robot[policy_to_robot]
            joint_vel_policy = joint_vel_robot[policy_to_robot]
            default_policy = default_pos_robot_order[policy_to_robot]
        else:
            joint_pos_policy = joint_pos_robot
            joint_vel_policy = joint_vel_robot
            default_policy = default_pos_robot_order

        joint_pos_rel = joint_pos_policy - default_policy

        obs = np.concatenate(
            [
                base_lin_vel,           # (3,)  body-frame linear velocity
                base_ang_vel,           # (3,)  body-frame angular velocity
                projected_gravity,      # (3,)  gravity in body frame
                velocity_commands,      # (3,)  [vx, vy, yaw_rate] targets
                joint_pos_rel,          # (37,) relative to default — policy order
                joint_vel_policy,       # (37,) joint velocities — policy order
                last_action_policy_order,  # (37,) previous action — policy order
            ]
        ).astype(np.float32)

        if add_height_scan:
            obs = np.concatenate([obs, self.compute_height_scan(base_pos, base_orient)]).astype(np.float32)
        return obs
