# Pre-trained Policy Execution Guide

## Overview

This guide explains how to use the Isaac Sim MCP extension tools to load and execute pre-trained reinforcement learning policies on robots, enabling autonomous walking and other learned behaviors.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         MCP Client                          │
│                    (Claude, Python, etc.)                   │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server                             │
│                  (isaac_mcp/server.py)                      │
│  Tools: load_policy, get_robot_state, apply_joint_actions, │
│         run_policy_loop, reset_robot_pose, etc.            │
└────────────────────────┬────────────────────────────────────┘
                         │ Socket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Isaac Sim MCP Extension                      │
│       (isaac.sim.mcp_extension/extension.py)               │
│  - Robot articulation control                               │
│  - Physics simulation                                       │
│  - Policy inference                                         │
└─────────────────────────────────────────────────────────────┘
```

## New MCP Tools

### 1. Policy Management

#### `load_policy(policy_path, robot_prim_path)`

Load a pre-trained RL policy from disk.

**Supported Formats:**
- `.zip` - Stable Baselines3 (PPO, SAC, TD3, etc.)
- `.pt`, `.pth` - PyTorch checkpoints (RSL-RL, direct PyTorch)

**Parameters:**
- `policy_path` (str): Absolute path to policy file
- `robot_prim_path` (str): USD path to robot (default: "/G1")

**Returns:**
- Success: Policy type and metadata
- Error: Loading error message

**Example:**
```python
load_policy("/isaac-sim/g1_walking_trained.zip", "/G1")
```

**Implementation Details:**
- Initializes robot articulation with physics view
- Sets PD gains (200/5 for G1, matching Isaac Lab defaults)
- Loads policy weights into memory
- Supports SB3 and PyTorch checkpoint formats

---

### 2. Robot State Observation

#### `get_robot_state(robot_prim_path)`

Get comprehensive robot state for policy observations.

**Parameters:**
- `robot_prim_path` (str): USD path to robot (default: "/G1")

**Returns (JSON):**
```json
{
  "joint_positions": [0.0, 0.2, -0.4, ...],    // radians
  "joint_velocities": [0.0, 0.01, ...],         // rad/s
  "base_position": [0.0, 0.0, 0.78],            // meters (x, y, z)
  "base_orientation": [1.0, 0.0, 0.0, 0.0],     // quaternion (w, x, y, z)
  "base_linear_velocity": [0.5, 0.0, -0.02],    // m/s
  "base_angular_velocity": [0.01, -0.02, 0.0],  // rad/s
  "num_joints": 43,
  "dof_names": ["left_hip_pitch_joint", ...]
}
```

**Example:**
```python
state = get_robot_state("/G1")
# Parse state and use for policy observation
```

---

### 3. Joint Control

#### `apply_joint_actions(robot_prim_path, joint_positions, joint_velocities, joint_efforts)`

Apply control commands to robot joints.

**Parameters:**
- `robot_prim_path` (str): USD path to robot
- `joint_positions` (List[float]): Target positions (radians)
- `joint_velocities` (List[float]): Target velocities (rad/s)
- `joint_efforts` (List[float]): Target torques (N·m)

**Control Modes:**
- **Position control** (most common): Specify only `joint_positions`
- **Velocity control**: Specify only `joint_velocities`
- **Torque control**: Specify only `joint_efforts`
- **Hybrid**: Combine multiple modes

**Example:**
```python
# Position control (PD controller with gains set by load_policy)
apply_joint_actions(
    robot_prim_path="/G1",
    joint_positions=[0.0, 0.2, -0.4, 0.8, -0.4, 0.0, ...]  # 43 joints for G1
)
```

---

### 4. Robot Reset

#### `reset_robot_pose(robot_prim_path, base_position, joint_positions)`

Reset robot to standing pose.

**Parameters:**
- `robot_prim_path` (str): USD path to robot
- `base_position` (List[float]): Base [x, y, z] (default: [0, 0, 0.8])
- `joint_positions` (List[float]): Joint positions (default: G1 standing pose)

**Default Standing Poses:**
- **G1 humanoid**: Left/right leg bent configuration (height 0.8m)

**Example:**
```python
# Reset to default standing
reset_robot_pose("/G1")

# Reset to custom pose
reset_robot_pose("/G1", base_position=[0, 0, 1.0], joint_positions=[0.0]*43)
```

---

### 5. Simulation Control

#### `start_simulation()`

Start the physics simulation (timeline play).

**Example:**
```python
start_simulation()
```

#### `stop_simulation()`

Stop the physics simulation (timeline stop).

**Example:**
```python
stop_simulation()
```

#### `step_simulation(num_steps, render)`

Step simulation forward.

**Parameters:**
- `num_steps` (int): Number of steps (default: 1)
- `render` (bool): Render viewport (default: True)

**Example:**
```python
# Single step with rendering
step_simulation(1)

# 100 steps without rendering (faster)
step_simulation(100, render=False)
```

---

### 6. Policy Execution

#### `run_policy_loop(robot_prim_path, num_steps, deterministic)`

Run loaded policy in an automated inference loop.

**Parameters:**
- `robot_prim_path` (str): USD path to robot
- `num_steps` (int): Maximum steps to run (default: 100)
- `deterministic` (bool): Use deterministic actions (default: True)

**Automated Steps:**
1. Reset robot to standing pose
2. Get robot state (observations)
3. Policy forward pass (inference)
4. Apply actions to joints
5. Step simulation
6. Check termination (falling detection)
7. Compute rewards

**Returns (JSON):**
```json
{
  "message": "Policy loop completed 487 steps",
  "total_reward": 243.5,
  "mean_reward": 0.5,
  "steps_completed": 487
}
```

**Example:**
```python
result = run_policy_loop(
    robot_prim_path="/G1",
    num_steps=500,
    deterministic=True
)
```

---

## Complete Workflow

### Basic Policy Execution

```python
# 1. Setup scene
create_physics_scene(objects=[], floor=True)
create_robot(robot_type="g1", position=[0, 0, 0])

# 2. Load policy
load_policy("/path/to/g1_walking_trained.zip", "/G1")

# 3. Reset robot
reset_robot_pose("/G1")

# 4. Start simulation
start_simulation()

# 5. Run policy
result = run_policy_loop("/G1", num_steps=500)

# 6. Stop simulation
stop_simulation()
```

### Manual Control Loop

```python
# Setup (same as above)
load_policy("/path/to/policy.zip", "/G1")
reset_robot_pose("/G1")
start_simulation()

# Manual loop
for step in range(100):
    # Get observation
    state_json = get_robot_state("/G1")
    state = json.loads(state_json)

    # Prepare observation
    obs = prepare_observation(state)  # Your normalization

    # Policy inference
    action = policy.predict(obs)  # Your policy

    # Apply action
    apply_joint_actions("/G1", joint_positions=action.tolist())

    # Step
    step_simulation(1)

    # Check termination
    if state["base_position"][2] < 0.3:  # Fell
        break

stop_simulation()
```

---

## Observation Space Format

The observation format matches the G1WalkingEnv training environment:

```python
obs = [
    # Joint states (normalized)
    joint_positions / pi,        # 43 values, range [-1, 1]
    joint_velocities / 10.0,     # 43 values, range [-1, 1]

    # Base states (world frame)
    base_position,               # 3 values (x, y, z) in meters
    base_orientation,            # 4 values (w, x, y, z) quaternion
    base_linear_velocity,        # 3 values (vx, vy, vz) in m/s
    base_angular_velocity,       # 3 values (wx, wy, wz) in rad/s

    # Command
    target_velocity              # 1 value (desired forward speed)
]
# Total: 43 + 43 + 3 + 4 + 3 + 3 + 1 = 100 dimensions
```

---

## Action Space Format

Actions are joint position targets:

```python
action = policy.predict(obs)  # Range [-1, 1] (normalized)
scaled_action = action * pi   # Scale to radians
apply_joint_actions("/G1", joint_positions=scaled_action)
```

**G1 Joint Order (43 DOF):**
1. Left leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
2. Right leg (6): same as left
3. Torso (3): waist joints
4. Left arm (14): shoulder, elbow, wrist, hand
5. Right arm (14): same as left

---

## Termination Conditions

The policy loop automatically detects falling:

```python
# Height check
if base_position[2] < 0.3:  # 30cm threshold
    terminate = True

# Orientation check (roll/pitch)
if abs(roll) > 0.7 or abs(pitch) > 0.7:  # ~40 degrees
    terminate = True
```

---

## PD Gains Configuration

Default gains match Isaac Lab G1 training:

```python
# Set by load_policy() and reset_robot_pose()
kp = [200.0] * 43  # Stiffness (N·m/rad)
kd = [5.0] * 43    # Damping (N·m·s/rad)

# Isaac Lab uses different gains per joint group:
# - Legs: kp=200, kd=5
# - Feet: kp=20, kd=2
# - Arms: kp=40, kd=10
# Our implementation uses uniform leg-like gains for simplicity
```

---

## Error Handling

All tools return status and error messages:

```python
result = load_policy("/path/to/policy.zip", "/G1")
# On error:
# {
#   "status": "error",
#   "message": "Policy file not found",
#   "traceback": "..."
# }
```

**Common Errors:**
- "Physics simulation view not available" → Call `start_simulation()` first
- "Robot not initialized" → Call `load_policy()` or `get_robot_state()` first
- "Policy file not found" → Check file path
- "PyTorch not available" → Install torch: `pip install torch`

---

## Performance Tips

1. **Disable rendering for faster training:**
   ```python
   step_simulation(num_steps=100, render=False)
   ```

2. **Use GPU for policy inference:**
   - Ensure CUDA is available
   - Policy loads to `cuda:0` automatically if available

3. **Batch observations:**
   - For vectorized environments, get state for multiple robots
   - Apply actions in batches

4. **Tune simulation frequency:**
   - Default: 60 Hz (1/60s per step)
   - For control: Match training frequency (e.g., 50 Hz for Isaac Lab)

---

## Troubleshooting

### Robot Doesn't Move

**Issue:** Actions applied but robot doesn't respond.

**Solutions:**
1. Check physics view is available:
   ```python
   start_simulation()
   # Wait for physics initialization
   step_simulation(100, render=False)
   ```

2. Verify gains are set:
   ```python
   load_policy(...)  # Sets gains automatically
   ```

3. Check action scaling:
   ```python
   # Actions should be in radians
   scaled_action = action * np.pi
   ```

### Robot Falls Immediately

**Issue:** Robot collapses on reset.

**Solutions:**
1. Use proper standing pose:
   ```python
   reset_robot_pose("/G1")  # Uses default standing pose
   ```

2. Wait for physics to stabilize:
   ```python
   reset_robot_pose("/G1")
   step_simulation(50, render=False)  # Let it settle
   ```

3. Check ground plane exists:
   ```python
   create_physics_scene(floor=True)
   ```

### Policy Inference Errors

**Issue:** Policy loading or inference fails.

**Solutions:**
1. Check file format:
   - SB3: Must be `.zip`
   - PyTorch: `.pt` or `.pth`

2. Verify dependencies:
   ```bash
   pip install torch stable-baselines3
   ```

3. Match observation dimensions:
   - G1: 100 dimensions (43 joints × 2 + 13 base states + 1 command)

---

## Files Modified

### Extension (Isaac Sim side)
- `/home/ubuntu/isaac-sim-mcp/isaac.sim.mcp_extension/isaac_sim_mcp_extension/extension.py`
  - Added policy state variables
  - Added methods: `load_policy`, `get_robot_state`, `apply_joint_actions`, `reset_robot_pose`, `start_simulation`, `stop_simulation`, `step_simulation`, `run_policy_loop`

### MCP Server
- `/home/ubuntu/isaac-sim-mcp/isaac_mcp/server.py`
  - Added 8 new MCP tools for policy execution
  - Comprehensive documentation and examples

### Examples
- `/home/ubuntu/isaac-sim-mcp/examples/run_pretrained_policy.py`
  - Complete usage examples
  - Manual control loop demonstration
  - State monitoring utilities

---

## Next Steps

1. **Train a policy:**
   ```bash
   python /home/ubuntu/workspace/train_g1_walking_fixed.py
   ```

2. **Load and test:**
   ```python
   load_policy("/isaac-sim/g1_walking_trained.zip", "/G1")
   run_policy_loop("/G1", num_steps=1000)
   ```

3. **Integrate with your application:**
   - Use MCP tools in Claude conversations
   - Build autonomous robot behaviors
   - Combine with navigation and perception

---

## References

- **Isaac Sim Documentation:** https://docs.omniverse.nvidia.com/isaacsim/
- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/
- **Isaac Lab (for training):** https://isaac-sim.github.io/IsaacLab/

---

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review example scripts in `/home/ubuntu/isaac-sim-mcp/examples/`
3. Inspect training environment: `/home/ubuntu/workspace/train_g1_walking_fixed.py`
4. Check MCP server logs for detailed error messages
