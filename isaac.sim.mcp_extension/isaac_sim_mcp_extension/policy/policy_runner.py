"""Policy execution runtime (callback-based walking)."""

from __future__ import annotations

from typing import Any, Dict, List

import carb
import numpy as np
import omni.kit.app
import omni.timeline
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction

from isaac_sim_mcp_extension.core_state import ExtensionState
from isaac_sim_mcp_extension.policy.observation_builder import ObservationBuilder, quat_rotate_inverse
from isaac_sim_mcp_extension.policy.policy_loader import PolicyLoader
from isaac_sim_mcp_extension.robots.robot_controller import RobotController
from isaac_sim_mcp_extension.scene.scene_manager import SceneManager


# G1 standing base height above ground (m).
G1_STANDING_BASE_HEIGHT = 0.74

# Action scale matching Isaac Lab G1 JointPositionActionCfg(scale=0.5).
# Source: velocity_env_cfg.py line 112:
#   joint_pos = mdp.JointPositionActionCfg(joint_names=[".*"], scale=0.5, use_default_offset=True)
# The G1 rough/flat env configs do NOT override this value.
ACTION_SCALE = 0.5

# NOTE on joint ordering: Isaac Lab trains with joint_names=[".*"] on the articulation,
# which resolves to the robot's natural USD DOF order — identical to what Isaac Sim's
# Articulation.dof_names reports.  Therefore NO reordering is needed between policy
# output and robot command; we always use the robot's DOF order as-is.


def _build_default_joint_pos(dof_names: List[str]) -> np.ndarray:
    """Build G1_CFG.init_state.joint_pos defaults in robot DOF order.

    Values from isaaclab_assets/robots/unitree.py: G1_CFG.init_state.
    These are the standing-pose offsets used both for the initial pose and
    as the zero-point for relative joint position observations.
    """
    defaults: dict[str, float] = {
        "left_hip_pitch_joint":    -0.20,
        "right_hip_pitch_joint":   -0.20,
        "left_knee_joint":          0.42,
        "right_knee_joint":         0.42,
        "left_ankle_pitch_joint":  -0.23,
        "right_ankle_pitch_joint": -0.23,
        "left_elbow_pitch_joint":   0.87,
        "right_elbow_pitch_joint":  0.87,
        "left_shoulder_pitch_joint":  0.35,
        "right_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint":   0.16,
        "right_shoulder_roll_joint": -0.16,
        "left_one_joint":   1.00,
        "right_one_joint": -1.00,
        "left_two_joint":   0.52,
        "right_two_joint": -0.52,
    }
    pos = np.zeros(len(dof_names))
    for i, name in enumerate(dof_names):
        pos[i] = defaults.get(name, 0.0)
    return pos



class PolicyRunner:
    """Starts/stops policy loops and policy callbacks."""

    def __init__(
        self,
        state: ExtensionState,
        loader: PolicyLoader,
        robot_controller: RobotController,
        scene_manager: SceneManager,
    ) -> None:
        self._state = state
        self._loader = loader
        self._robot_controller = robot_controller
        self._scene_manager = scene_manager
        self._obs_builder = ObservationBuilder()

    # ------------------------------------------------------------------
    # Simulation lifecycle
    # ------------------------------------------------------------------

    def start_simulation(self) -> Dict[str, Any]:
        try:
            omni.timeline.get_timeline_interface().play()
            return {"status": "success", "message": "Simulation started"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def stop_simulation(self) -> Dict[str, Any]:
        try:
            omni.timeline.get_timeline_interface().stop()
            self._state.policy.running = False
            return {"status": "success", "message": "Simulation stopped"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def step_simulation(self, num_steps: int = 1, render: bool = True) -> Dict[str, Any]:
        try:
            world = World.instance()
            if world is None:
                return {"status": "error", "message": "World not initialized"}
            for _ in range(num_steps):
                world.step(render=render)
            return {"status": "success", "message": f"Stepped simulation {num_steps} times"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Policy walk
    # ------------------------------------------------------------------

    def run_policy_loop(self, robot_prim_path: str = "/G1", num_steps: int = 100, deterministic: bool = True) -> Dict[str, Any]:
        if self._state.policy.loaded_policy is None:
            return {"status": "error", "message": "No policy loaded. Call load_policy first."}
        result = self.start_g1_policy_walk(
            policy_path=self._state.policy.loaded_policy.get("path", ""),
            robot_prim_path=robot_prim_path,
            target_velocity=0.5,
            deterministic=deterministic,
        )
        if result.get("status") != "success":
            return result
        return {
            "status": "success",
            "message": "run_policy_loop started callback-based policy walk. Use stop_g1_policy_walk to stop.",
            "steps_requested": num_steps,
        }

    def start_g1_policy_walk(
        self,
        policy_path: str,
        robot_prim_path: str = "/G1",
        target_velocity: float = 0.5,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        try:
            import torch

            # Stop any running walk and clear stale articulation so we re-init fresh.
            if self._state.policy.walk_subscription is not None:
                self.stop_g1_policy_walk()
            self._state.policy.robot_articulation = None
            self._state.policy.controller = None
            # Reset navigation status so a stale "arrived" doesn't suppress default velocity.
            self._state.navigation.status = "idle"
            self._state.navigation.active = False

            # Load policy and initialise articulation.
            load_result = self._loader.load_policy(policy_path=policy_path, robot_prim_path=robot_prim_path)
            if load_result.get("status") != "success":
                return load_result
            self._state.policy.loaded_policy["path"] = policy_path

            # Detect obs dimension: 123 = flat terrain, 310 = rough terrain + height scan.
            expected_obs_dim = self._loader.detect_obs_dim(
                self._state.policy.loaded_policy.get("checkpoint"),
                self._state.policy.loaded_policy.get("type", "pytorch"),
            )
            self._state.policy.walk_use_height_scan = (expected_obs_dim == 310)
            self._state.policy.walk_robot_prim_path = robot_prim_path
            self._state.policy.walk_target_velocity = target_velocity
            self._state.policy.walk_deterministic = deterministic
            self._state.policy.walk_step_count = 0
            self._state.policy.walk_initialized = False

            robot = self._state.policy.robot_articulation
            carb.log_info("[WALK] Got robot articulation, reading dof_names...")
            try:
                dof_names = list(robot.dof_names)
            except Exception as e:
                import traceback
                carb.log_error(f"[WALK] robot.dof_names failed: {type(e).__name__}: {e}")
                carb.log_error(traceback.format_exc())
                return {"status": "error", "message": str(e)}

            # Build default joint positions in robot DOF order.
            # Policy order == robot DOF order (Isaac Lab trains with joint_names=[".*"]).
            default_pos_robot = _build_default_joint_pos(dof_names)
            self._state.policy.walk_default_joint_pos = default_pos_robot

            # Last action starts as zeros (same length as DOF count).
            self._state.policy.walk_last_action = np.zeros(len(dof_names))

            # Physics dt (60 Hz) is set by scene_manager for responsive UI.
            # Do NOT call set_simulation_dt here — it invalidates the sim view.

            # Decimation: policy runs every DECIMATION physics steps.
            # At 60 Hz physics, decimation=4 → 15 Hz control; decimation=2 → 30 Hz.
            DECIMATION = 2
            physics_step_counter = [0]  # mutable counter captured by closure

            carb.log_info(
                f"[WALK] DOF order: {dof_names[:6]}... "
                f"(identity mapping, action_scale={ACTION_SCALE}, decimation={DECIMATION})"
            )

            # Terrain-aware spawn reset: use physics raycast for Z, not hardcoded value.
            carb.log_info("[WALK] Getting robot get_world_pose() for spawn reset...")
            try:
                spawn_pos, _ = robot.get_world_pose()
            except Exception as e:
                import traceback
                carb.log_error(f"[WALK] robot.get_world_pose() failed: {type(e).__name__}: {e}")
                carb.log_error(traceback.format_exc())
                return {"status": "error", "message": str(e)}
            reset_x, reset_y = float(spawn_pos[0]), float(spawn_pos[1])
            terrain_z = self._scene_manager.get_terrain_height_at(reset_x, reset_y)
            reset_z = terrain_z + G1_STANDING_BASE_HEIGHT
            robot.set_world_pose(
                position=np.array([reset_x, reset_y, reset_z]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
            robot.set_joint_positions(default_pos_robot)
            robot.set_joint_velocities(np.zeros(len(dof_names)))
            robot.set_linear_velocity(np.zeros(3))
            robot.set_angular_velocity(np.zeros(3))

            # ----------------------------------------------------------
            # PRE_UPDATE callback — runs before each physics step.
            # ----------------------------------------------------------
            def policy_walk_callback(_event: Any) -> None:
                try:
                    # First frame: hold standing pose so physics settles.
                    if not self._state.policy.walk_initialized:
                        self._state.policy.walk_initialized = True
                        return

                    # Decimation: only run policy every DECIMATION physics steps.
                    # At 60 Hz physics, decimation=2 → 30 Hz control.
                    physics_step_counter[0] += 1
                    if physics_step_counter[0] % DECIMATION != 0:
                        return

                    self._state.policy.walk_step_count += 1

                    # --- Robot state (robot DOF order) ---
                    state_result = self._robot_controller.get_robot_state(
                        self._state.policy.walk_robot_prim_path or "/G1"
                    )
                    if state_result.get("status") != "success":
                        carb.log_error(f"Failed to get robot state: {state_result.get('message')}")
                        return
                    # Guard: joint arrays must match expected DOF count (37 for G1 minimal).
                    jp_len = len(state_result["state"]["joint_positions"])
                    expected = len(self._state.policy.walk_default_joint_pos)
                    if jp_len != expected:
                        if self._state.policy.walk_step_count <= 3:
                            carb.log_error(f"[WALK] joint_positions len={jp_len} != expected {expected}, skipping step")
                        return

                    # --- Velocity commands ---
                    # Use nav/external command when navigation has taken control (even if zero — arrived).
                    # Fall back to target_velocity only when nothing has overridden the command.
                    nav_in_control = self._state.navigation.status in ("navigating", "arrived")
                    vel_x = self._state.vel_cmd_x if nav_in_control else (
                        self._state.vel_cmd_x if self._state.vel_cmd_x != 0.0 else target_velocity
                    )
                    velocity_commands = np.array([vel_x, self._state.vel_cmd_y, self._state.vel_cmd_yaw])

                    # --- Build observation (identity joint order: DOF order == policy order) ---
                    obs = self._obs_builder.build(
                        state=state_result["state"],
                        velocity_commands=velocity_commands,
                        default_pos_robot_order=self._state.policy.walk_default_joint_pos,
                        last_action_policy_order=self._state.policy.walk_last_action,
                        add_height_scan=self._state.policy.walk_use_height_scan,
                        policy_to_robot=None,
                    )

                    # Align observation length with policy input dimension.
                    # The loaded policy was traced with expected_obs_dim inputs (e.g. 123 or 310).
                    # If the runtime observation is longer (e.g. extra features added on MCP side),
                    # slice to the first expected_obs_dim entries so matmul shapes match.
                    if obs.shape[-1] != expected_obs_dim:
                        if self._state.policy.walk_step_count <= 3:
                            carb.log_warn(
                                f"[WALK] obs dim {obs.shape[-1]} != expected {expected_obs_dim}; "
                                f"adapting to policy input size"
                            )
                        if obs.shape[-1] >= expected_obs_dim:
                            obs = obs[:expected_obs_dim]
                        else:
                            # Pad with zeros if the observation is unexpectedly short.
                            obs = np.pad(obs, (0, expected_obs_dim - obs.shape[-1]))

                    # --- Policy inference ---
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(
                            self._state.policy.loaded_policy["device"]
                        )
                        policy_type = self._state.policy.loaded_policy["type"]
                        if policy_type == "sb3_ppo":
                            network = self._state.policy.loaded_policy.get("network")
                            if network is None:
                                carb.log_error("SB3 policy network not loaded")
                                return
                            action = network(obs_tensor)
                        else:
                            checkpoint = self._state.policy.loaded_policy["checkpoint"]
                            if callable(checkpoint) and not isinstance(checkpoint, dict):
                                out = checkpoint(obs_tensor)
                                action = out[0] if isinstance(out, (list, tuple)) else out
                            else:
                                return
                        action_np = action.cpu().numpy().flatten()

                    # Store action for next obs (policy order == DOF order).
                    self._state.policy.walk_last_action = action_np.copy()

                    # --- Apply: target = default + action * scale (DOF order, no reordering) ---
                    target_positions = (
                        self._state.policy.walk_default_joint_pos + action_np * ACTION_SCALE
                    )

                    # Diagnostic print on first 3 policy steps to verify obs/action sanity.
                    if self._state.policy.walk_step_count <= 3:
                        s = self._state.policy.walk_step_count
                        base_z = float(np.array(state_result["state"]["base_position"])[2])
                        jp = np.array(state_result["state"]["joint_positions"])
                        jv = np.array(state_result["state"]["joint_velocities"])
                        pg = quat_rotate_inverse(
                            np.array(state_result["state"]["base_orientation"]),
                            np.array([0.0, 0.0, -1.0])
                        )
                        print(f"[DIAG step={s}] base_z={base_z:.3f} proj_g={pg.round(3).tolist()}")
                        print(f"[DIAG step={s}] jp[:6]={jp[:6].round(3).tolist()} jv[:6]={jv[:6].round(3).tolist()}")
                        print(f"[DIAG step={s}] action[:6]={action_np[:6].round(3).tolist()} target[:6]={target_positions[:6].round(3).tolist()}")
                        print(f"[DIAG step={s}] default[:6]={self._state.policy.walk_default_joint_pos[:6].round(3).tolist()}")
                    self._state.policy.controller.apply_action(
                        ArticulationAction(joint_positions=target_positions)
                    )

                    # --- Fall detection and reset ---
                    base_pos = np.array(state_result["state"]["base_position"])
                    if float(base_pos[2]) < 0.3:
                        carb.log_warn(
                            f"[FALL] step={self._state.policy.walk_step_count} "
                            f"z={float(base_pos[2]):.3f}m — resetting"
                        )
                        fall_terrain_z = self._scene_manager.get_terrain_height_at(reset_x, reset_y)
                        fall_z = fall_terrain_z + G1_STANDING_BASE_HEIGHT
                        robot.set_world_pose(
                            position=np.array([reset_x, reset_y, fall_z]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                        )
                        robot.set_joint_positions(self._state.policy.walk_default_joint_pos)
                        robot.set_joint_velocities(np.zeros(len(self._state.policy.walk_default_joint_pos)))
                        robot.set_linear_velocity(np.zeros(3))
                        robot.set_angular_velocity(np.zeros(3))
                        self._state.policy.walk_step_count = 0
                        self._state.policy.walk_last_action = np.zeros(
                            len(self._state.policy.walk_default_joint_pos)
                        )

                    # --- Periodic progress log ---
                    if self._state.policy.walk_step_count % 50 == 0:
                        fwd_vel = float(np.array(state_result["state"]["base_linear_velocity"])[0])
                        carb.log_info(
                            f"[WALK] step={self._state.policy.walk_step_count} "
                            f"z={float(base_pos[2]):.3f}m fwd_vel={fwd_vel:.2f}m/s"
                        )

                except Exception as exc:
                    carb.log_error(f"Error in policy walk callback: {exc}")

            # Register callback on PRE_UPDATE so control is applied before physics step.
            stream = omni.kit.app.get_app().get_pre_update_event_stream()
            self._state.policy.walk_subscription = stream.create_subscription_to_pop(
                policy_walk_callback, name="g1_policy_walk_callback"
            )

            return {
                "status": "success",
                "message": "Policy walk started with callback registered. Robot will walk continuously until stop_g1_policy_walk is called.",
                "policy_path": policy_path,
                "robot_prim_path": robot_prim_path,
                "target_velocity": target_velocity,
                "obs_dim": expected_obs_dim,
                "use_height_scan": self._state.policy.walk_use_height_scan,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def stop_g1_policy_walk(self) -> Dict[str, Any]:
        try:
            if self._state.policy.walk_subscription is not None:
                self._state.policy.walk_subscription.unsubscribe()
                self._state.policy.walk_subscription = None
            total_steps = self._state.policy.walk_step_count
            self._state.policy.walk_initialized = False
            self._state.policy.walk_step_count = 0
            return {
                "status": "success",
                "message": f"Policy walk stopped after {total_steps} steps",
                "total_steps": total_steps,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
