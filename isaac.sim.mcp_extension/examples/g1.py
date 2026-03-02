"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# G1 policy-driven walking example for Isaac Sim 5.x via MCP execute_script.
#
# For the recommended Isaac Sim 5.x flow (single execute_script, no separate start_simulation),
# use workspace/scripts/g1_policy_walk_mcp.py instead. This extension example uses the
# start_simulation()-after-execute_script pattern and is kept for extension demo parity.
#
# Prerequisites (run via MCP tools before execute_script):
#   1. get_scene_info()               - verify connection
#   2. create_physics_scene(floor=True) - physics scene + ground plane
#   3. create_robot("g1", [0,0,0])    - G1 robot at /G1 (43 DOF full or 37 DOF minimal both work)
#
# After execute_script, call start_simulation() to begin walking.
#
# Velocity control: use MCP set_velocity_command(lin_vel_x, lin_vel_y, ang_vel_z) to steer.

import os
import re
import json
import math
import numpy as np
import torch
import omni.kit.app
import omni.timeline
import omni.usd
from omni.isaac.core import SimulationContext, PhysicsContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Usd, UsdPhysics, Sdf

TAG = "[G1-policy]"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Policy file path inside Isaac Sim container (workspace mounted at /home/workspace)
POLICY_PATH = os.environ.get("G1_POLICY_PATH", "").strip()
if not POLICY_PATH or not os.path.isfile(POLICY_PATH):
    for p in [
        "/home/workspace/exported/g1_flat_policy.pt",
        "/workspace/exported/g1_flat_policy.pt",
    ]:
        if os.path.isfile(p):
            POLICY_PATH = p
            break

if not POLICY_PATH or not os.path.isfile(POLICY_PATH):
    raise FileNotFoundError(
        f"{TAG} Policy file not found. Export first with: "
        "python3 workspace/scripts/export_rslrl_to_jit.py"
    )

# Policy runs every DECIMATION sim ticks (~50 Hz with sim at 200 Hz)
DECIMATION = 4
# Hold default standing pose for this many policy ticks before enabling policy
STABILIZE_TICKS = 150
# Action scale from Isaac Lab env.yaml (scale 0.5)
ACTION_SCALE = 0.5
# Velocity command file (written by MCP set_velocity_command tool)
VEL_CMD_PATH = os.environ.get("G1_NAV_VELOCITY_CMD_PATH", "/tmp/g1_nav_velocity_cmd.json")
# Default velocity command: forward 0.5 m/s
DEFAULT_VEL_CMD = np.array([0.5, 0.0, 0.0], dtype=np.float32)
# Velocity clamp ranges (G1 flat training ranges)
VEL_RANGES = {"x": (0.0, 1.0), "y": (-0.5, 0.5), "yaw": (-1.0, 1.0)}
# Robot prim path (create_robot("g1") places it at /G1)
ROBOT_PRIM = "/G1"


# ---------------------------------------------------------------------------
# Helper functions (from g1_policy_walk_mcp.py, simplified)
# ---------------------------------------------------------------------------

def _get_velocity_command():
    """Read velocity command from file written by MCP set_velocity_command."""
    try:
        if not os.path.isfile(VEL_CMD_PATH):
            return DEFAULT_VEL_CMD.copy()
        with open(VEL_CMD_PATH, "r") as f:
            data = json.load(f)
        vx = float(data.get("lin_vel_x", DEFAULT_VEL_CMD[0]))
        vy = float(data.get("lin_vel_y", DEFAULT_VEL_CMD[1]))
        vz = float(data.get("ang_vel_z", DEFAULT_VEL_CMD[2]))
        # Defensive clamp
        vx = max(VEL_RANGES["x"][0], min(VEL_RANGES["x"][1], vx))
        vy = max(VEL_RANGES["y"][0], min(VEL_RANGES["y"][1], vy))
        vz = max(VEL_RANGES["yaw"][0], min(VEL_RANGES["yaw"][1], vz))
        return np.array([vx, vy, vz], dtype=np.float32)
    except (json.JSONDecodeError, TypeError, KeyError, OSError):
        return DEFAULT_VEL_CMD.copy()


def _quat_apply_inverse(quat, vec):
    """Apply inverse rotation of quaternion (w,x,y,z) to vector. Returns vec in body frame."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    vx, vy, vz = vec[0], vec[1], vec[2]
    return np.array([
        (1 - 2*(y*y + z*z)) * vx + 2*(x*y - z*w) * vy + 2*(x*z + y*w) * vz,
        2*(x*y + z*w) * vx + (1 - 2*(x*x + z*z)) * vy + 2*(y*z - x*w) * vz,
        2*(x*z - y*w) * vx + 2*(y*z + x*w) * vy + (1 - 2*(x*x + y*y)) * vz,
    ], dtype=np.float32)


def _default_joint_pos(dof_names):
    """Default standing joint positions from Isaac Lab G1 env.yaml init_state.joint_pos."""
    out = np.zeros(len(dof_names), dtype=np.float32)
    for i, name in enumerate(dof_names):
        nl = name.lower()
        if "hip_pitch" in nl:
            out[i] = -0.2
        elif "knee" in nl:
            out[i] = 0.42
        elif "ankle_pitch" in nl:
            out[i] = -0.23
        elif "elbow_pitch" in nl:
            out[i] = 0.87
        elif nl == "left_shoulder_roll_joint":
            out[i] = 0.16
        elif nl == "left_shoulder_pitch_joint":
            out[i] = 0.35
        elif nl == "right_shoulder_roll_joint":
            out[i] = -0.16
        elif nl == "right_shoulder_pitch_joint":
            out[i] = 0.35
        elif "one_joint" in nl and "left" in nl:
            out[i] = 1.0
        elif "one_joint" in nl and "right" in nl:
            out[i] = -1.0
        elif "two_joint" in nl and "left" in nl:
            out[i] = 0.52
        elif "two_joint" in nl and "right" in nl:
            out[i] = -0.52
    return out


def _pd_gains(dof_names):
    """PD gains from Isaac Lab env.yaml (different gains per joint group)."""
    kps = np.zeros(len(dof_names), dtype=np.float64)
    kds = np.zeros(len(dof_names), dtype=np.float64)
    for i, name in enumerate(dof_names):
        nl = name.lower()
        if "hip_yaw" in nl or "hip_roll" in nl:
            kps[i], kds[i] = 150.0, 5.0
        elif "hip_pitch" in nl or "knee" in nl or "torso" in nl:
            kps[i], kds[i] = 200.0, 5.0
        elif "ankle" in nl:
            kps[i], kds[i] = 20.0, 2.0
        else:
            kps[i], kds[i] = 40.0, 10.0
    return kps, kds


def _set_drive_targets_on_stage(stage, prim_path, dof_names, joint_pos_rad):
    """Write standing pose into USD joint drive targets so PhysX starts from standing."""
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return False
    joint_by_name = {}
    def visit(prim):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            joint_by_name[prim.GetName()] = prim
        for child in prim.GetChildren():
            visit(child)
    visit(root)
    for i, dof_name in enumerate(dof_names):
        if dof_name not in joint_by_name:
            continue
        jp = joint_by_name[dof_name]
        is_angular = jp.IsA(UsdPhysics.RevoluteJoint)
        drive_api = UsdPhysics.DriveAPI.Get(jp, "angular" if is_angular else "linear")
        if not drive_api:
            drive_api = UsdPhysics.DriveAPI.Apply(jp, "angular" if is_angular else "linear")
        attr = drive_api.GetTargetPositionAttr()
        if not attr:
            attr = drive_api.CreateTargetPositionAttr()
        val = float(joint_pos_rad[i])
        if is_angular:
            val = math.degrees(val)
        attr.Set(val)
    return True


def _build_obs(art, default_pos, last_action, velocity_command, dof_indices=None):
    """Build Isaac Lab observation: base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
    velocity_commands(3), joint_pos_rel(n), joint_vel(n), last_actions(n). All in body frame."""
    pos_w, quat_w = art.get_world_pose()
    quat_w = np.array(quat_w, dtype=np.float32)
    v_w = np.array(art.get_linear_velocity(), dtype=np.float32)
    omega_w = np.array(art.get_angular_velocity(), dtype=np.float32)
    joint_pos = np.array(art.get_joint_positions(), dtype=np.float32)
    joint_vel = np.array(art.get_joint_velocities(), dtype=np.float32)

    # Subset to policy DOFs if robot has more joints than policy expects
    dp = default_pos
    if dof_indices is not None:
        joint_pos = joint_pos[dof_indices]
        joint_vel = joint_vel[dof_indices]
        dp = default_pos[dof_indices]

    # Transform to body frame
    base_lin_vel = _quat_apply_inverse(quat_w, v_w)
    base_ang_vel = _quat_apply_inverse(quat_w, omega_w)
    projected_gravity = _quat_apply_inverse(quat_w, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    obs = np.concatenate([
        base_lin_vel,           # 3
        base_ang_vel,           # 3
        projected_gravity,      # 3
        velocity_command,       # 3
        joint_pos - dp,         # n (relative to default)
        joint_vel,              # n
        last_action,            # n
    ]).astype(np.float32)
    return obs


def _fix_physics_context(sim_ctx):
    """Fix _physics_context=None bug when Isaac Sim is launched from terminal (Docker).
    In terminal mode, SimulationContext.__init__ skips _init_stage() which creates PhysicsContext."""
    if sim_ctx._physics_context is not None:
        return
    import builtins
    if not getattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", False):
        return
    print(f"{TAG} Fixing _physics_context (None due to ISAAC_LAUNCHED_FROM_TERMINAL=True)")
    # Find the physics scene prim path on stage
    stage = omni.usd.get_context().get_stage()
    physics_prim = "/World/PhysicsScene"
    if not stage.GetPrimAtPath(physics_prim).IsValid():
        physics_prim = "/physicsScene"
    sim_ctx._physics_context = PhysicsContext(
        physics_dt=1.0 / 60.0,
        prim_path=physics_prim,
    )
    print(f"{TAG} PhysicsContext set, dt={sim_ctx.get_physics_dt()}")


# ---------------------------------------------------------------------------
# Main setup (runs once when execute_script is called)
# ---------------------------------------------------------------------------

print(f"{TAG} Starting setup...")

# Stop timeline during setup so robot doesn't fall before we're ready
timeline = omni.timeline.get_timeline_interface()
timeline.stop()
print(f"{TAG} Timeline stopped for setup")

# Get or create SimulationContext, fix the physics_context bug
sim_ctx = SimulationContext.instance()
if sim_ctx is None:
    sim_ctx = SimulationContext()
_fix_physics_context(sim_ctx)

# Initialize physics (needed for articulation)
sim_ctx.initialize_physics()

# Initialize robot articulation
art = Articulation(ROBOT_PRIM)
art.initialize(sim_ctx.physics_sim_view)

# Reset robot to standing pose
art.set_world_pose(
    position=np.array([0.0, 0.0, 0.8]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
)
try:
    art.set_linear_velocity(np.zeros(3))
    art.set_angular_velocity(np.zeros(3))
except Exception:
    pass

dof_names = list(art.dof_names)
n_dofs = art.num_dof
default_pos = _default_joint_pos(dof_names)
print(f"{TAG} Robot has {n_dofs} DOFs: {dof_names[:6]}...")

# Set PD gains (match Isaac Lab training config)
kps, kds = _pd_gains(dof_names)
ctrl = art.get_articulation_controller()
ctrl.set_gains(kps=kps.tolist(), kds=kds.tolist())

# Apply standing pose to joints and USD drive targets
try:
    ctrl.apply_action(ArticulationAction(joint_positions=default_pos))
    art.set_joint_positions(default_pos)
except Exception:
    pass
stage = omni.usd.get_context().get_stage()
_set_drive_targets_on_stage(stage, ROBOT_PRIM, dof_names, default_pos)
print(f"{TAG} Standing pose applied")

# Load TorchScript policy
policy = torch.jit.load(POLICY_PATH, map_location="cpu")
policy.eval()

# Infer policy DOF count from first Linear layer input dimension
obs_dim_expected = None
for _name, param in policy.named_parameters():
    if param.dim() == 2:
        obs_dim_expected = int(param.shape[1])
        break
if obs_dim_expected is None:
    raise RuntimeError(f"{TAG} Could not infer policy obs dim from model weights")

n_policy = (obs_dim_expected - 12) // 3
if 12 + 3 * n_policy != obs_dim_expected:
    raise RuntimeError(f"{TAG} Policy obs dim {obs_dim_expected} doesn't match 12+3*n format")

# If robot has more DOFs than policy expects, use only first n_policy DOFs
dof_indices = list(range(n_policy)) if n_policy < n_dofs else None
obs_dim = 12 + 3 * n_policy
print(f"{TAG} Policy loaded: obs_dim={obs_dim}, n_policy_dofs={n_policy}, robot_dofs={n_dofs}")
print(f"{TAG} Policy file: {POLICY_PATH}")

# ---------------------------------------------------------------------------
# Callback state
# ---------------------------------------------------------------------------

step_count = [0]
last_action = np.zeros(n_policy, dtype=np.float32)
last_joint_target = default_pos.copy()
policy_enabled = [False]
physics_ready = [False]
logged_states = {"waiting": False, "started": False, "policy_active": False}


def _safe_apply(joint_positions):
    """Apply joint positions without crashing if physics isn't ready yet."""
    try:
        ctrl.apply_action(ArticulationAction(joint_positions=joint_positions))
        return True
    except Exception:
        return False


def on_update(_event):
    """PRE_UPDATE callback: applies control before each physics step."""
    tl = omni.timeline.get_timeline_interface()

    # While timeline is stopped, keep applying standing pose
    if not tl.is_playing():
        _safe_apply(default_pos)
        try:
            art.set_joint_positions(default_pos)
        except Exception:
            pass
        if not logged_states["waiting"]:
            logged_states["waiting"] = True
            print(f"{TAG} Waiting for start_simulation(). Applying standing pose each frame.")
        return

    # Timeline is playing
    if not logged_states["started"]:
        logged_states["started"] = True
        print(f"{TAG} Timeline playing. Control active.")

    # Wait until physics sim view is ready
    if not physics_ready[0]:
        try:
            art.get_joint_positions()
            physics_ready[0] = True
        except Exception:
            pass

    # Determine which pose to apply
    tick = step_count[0] // DECIMATION
    pose = default_pos if tick < STABILIZE_TICKS else last_joint_target
    if not _safe_apply(pose):
        return

    step_count[0] += 1

    # Log progress every 200 sim steps (~1 sec at 200 Hz)
    if step_count[0] % 200 == 0:
        try:
            pos = art.get_world_pose()[0]
            print(f"{TAG} step={step_count[0]}, tick={tick}, "
                  f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                  f"stabilizing={tick < STABILIZE_TICKS}")
        except Exception:
            pass

    # Only run policy inference every DECIMATION steps
    if step_count[0] % DECIMATION != 0:
        return

    # Build observation and run policy
    velocity_command = _get_velocity_command()
    obs = _build_obs(art, default_pos, last_action, velocity_command, dof_indices)

    with torch.no_grad():
        act = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

    # During stabilization: run policy but don't apply output (warm up the action buffer)
    if tick < STABILIZE_TICKS:
        last_action[:] = act
        last_joint_target[:] = default_pos
        return

    if not logged_states["policy_active"]:
        logged_states["policy_active"] = True
        print(f"{TAG} Stabilization complete at tick {tick}. Policy control active!")

    # Safety: reject NaN/inf outputs
    if not np.isfinite(act).all():
        last_joint_target[:] = default_pos
        return

    last_action[:] = act

    # Apply policy output: default_pos + ACTION_SCALE * action
    if dof_indices is not None:
        # Policy controls subset of DOFs; rest stay at default
        last_joint_target[:] = default_pos
        last_joint_target[dof_indices] = default_pos[dof_indices] + ACTION_SCALE * act
    else:
        last_joint_target[:] = default_pos + ACTION_SCALE * act

    _safe_apply(last_joint_target)


# Register callback on PRE_UPDATE (runs before each physics step)
_update_stream = omni.kit.app.get_app().get_pre_update_event_stream()
_update_sub = _update_stream.create_subscription_to_pop(on_update, name="G1_policy_walk")

print(f"{TAG} Callback registered on PRE_UPDATE.")
print(f"{TAG} Setup complete. Call start_simulation() (MCP) to begin walking.")
print(f"{TAG} Use set_velocity_command(vx, vy, yaw) to steer the robot.")
