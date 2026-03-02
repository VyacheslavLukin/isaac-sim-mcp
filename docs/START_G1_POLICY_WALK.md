# start_g1_policy_walk Tool Documentation

## Overview

The `start_g1_policy_walk` MCP tool enables continuous policy-driven robot walking using a persistent physics callback. Unlike `run_policy_loop` which runs for a fixed number of steps, this tool registers a callback that executes policy inference on every simulation frame, allowing the robot to walk indefinitely until explicitly stopped.

## Key Features

- **Persistent Callback**: Registers on Isaac Sim's update event stream, runs automatically each frame
- **Non-blocking**: Returns immediately after setup; robot walks in background
- **State Persistence**: Policy, articulation, and controller remain in extension memory across frames
- **Auto-reset**: Automatically resets robot to standing pose if it falls (height < 0.3m)
- **Flexible Control**: Can adjust target velocity or restart without reloading policy

## Tool Signature

### start_g1_policy_walk

```python
start_g1_policy_walk(
    policy_path: str,
    robot_prim_path: str = "/G1",
    target_velocity: float = 0.5,
    deterministic: bool = True
) -> str
```

**Parameters:**
- `policy_path` (str, required): Absolute path to trained policy file
  - `.zip` for Stable Baselines3 (PPO, SAC, etc.)
  - `.pt` or `.pth` for PyTorch/RSL-RL checkpoints
- `robot_prim_path` (str, default="/G1"): USD path to the robot articulation
- `target_velocity` (float, default=0.5): Target forward velocity in m/s
- `deterministic` (bool, default=True): Use deterministic actions vs stochastic

**Returns:**
- String with callback registration status and configuration

### stop_g1_policy_walk

```python
stop_g1_policy_walk() -> str
```

**Parameters:** None

**Returns:**
- String with stop status including total steps executed

## Implementation Details

### Callback Lifecycle

1. **Registration** (`start_g1_policy_walk` called):
   - Loads policy checkpoint into extension memory
   - Initializes robot articulation and controller
   - Registers callback on `omni.kit.app.get_update_event_stream()`
   - Returns immediately (non-blocking)

2. **First Frame** (callback initialized):
   - Resets robot to standing pose (height 0.8m)
   - Starts Isaac Sim timeline (physics simulation)
   - Waits 50 frames for physics to stabilize
   - Sets `_policy_walk_initialized = True`

3. **Subsequent Frames** (policy inference loop):
   - Gets robot state (joint positions/velocities, base pose/velocities)
   - Normalizes observations (clip joint angles to [-1, 1], velocities to [-1, 1])
   - Runs policy forward pass (PyTorch inference)
   - Scales output actions (multiply by π for radians)
   - Applies joint position commands via articulation controller
   - Checks termination (resets if height < 0.3m)
   - Logs progress every 100 steps

4. **Unsubscription** (`stop_g1_policy_walk` called):
   - Unsubscribes callback from update stream
   - Resets state flags and counters
   - Policy and articulation remain loaded for reuse

### State Management

The extension maintains the following persistent state:

```python
# In MCPExtension class:
self._loaded_policy = {
    'type': 'sb3_ppo' | 'pytorch',
    'state_dict': ...,  # For SB3
    'checkpoint': ...,  # For PyTorch
    'device': 'cuda:0' | 'cpu',
    'path': str  # Policy file path
}
self._policy_robot_articulation = Articulation(...)
self._policy_controller = ArticulationController(...)
self._policy_walk_subscription = EventSubscription(...)
self._policy_walk_step_count = int
self._policy_walk_initialized = bool
self._policy_walk_robot_prim_path = str
self._policy_walk_target_velocity = float
```

### Observation Space

The policy observation matches G1 walking environment format:

```python
obs = [
    normalized_joint_positions,    # (37,) for g1_minimal, clipped to [-1, 1]
    normalized_joint_velocities,   # (37,) clipped to [-1, 1]
    base_position,                 # (3,) [x, y, z] in meters
    base_orientation,              # (4,) [w, x, y, z] quaternion
    base_linear_velocity,          # (3,) [vx, vy, vz] in m/s
    base_angular_velocity,         # (3,) [wx, wy, wz] in rad/s
    target_velocity                # (1,) desired forward speed
]
```

### Action Space

The policy outputs normalized joint position commands:

```python
action = policy(obs)               # Network output in [-1, 1]
scaled_action = action * π         # Scale to radians
articulation.set_joint_positions(scaled_action)
```

## Usage Examples

### Basic Usage

```python
# 1. Setup scene
get_scene_info()
create_physics_scene(objects=[], floor=True)
create_robot(robot_type="g1_minimal", position=[0, 0, 0])

# 2. Start continuous walking
start_g1_policy_walk(
    policy_path="/workspace/checkpoints_terminal/best_model.zip",
    robot_prim_path="/G1",
    target_velocity=0.5,
    deterministic=True
)

# 3. Robot walks indefinitely...
# (MCP server is free to handle other requests)

# 4. Stop when done
stop_g1_policy_walk()
```

### Variable Speed Control

```python
# Start slow
start_g1_policy_walk(
    policy_path="/path/to/policy.zip",
    target_velocity=0.3
)
time.sleep(10)

# Speed up (stop and restart with new velocity)
stop_g1_policy_walk()
start_g1_policy_walk(
    policy_path="/path/to/policy.zip",
    target_velocity=0.8
)
```

### Navigation Integration

```python
# Start policy walking
start_g1_policy_walk(policy_path="...", target_velocity=0.5)

# Periodically check robot pose for navigation
while True:
    pose = get_robot_pose(prim_path="/G1")
    position = pose["position"]
    orientation = pose["orientation_quat"]

    # Compute navigation command based on pose
    # (e.g., adjust target velocity or stop at waypoint)

    if reached_waypoint(position):
        stop_g1_policy_walk()
        break

    time.sleep(0.1)
```

## Comparison with run_policy_loop

| Feature | start_g1_policy_walk | run_policy_loop |
|---------|---------------------|-----------------|
| **Duration** | Continuous until stopped | Fixed number of steps |
| **Blocking** | Non-blocking (callback) | Blocking |
| **State** | Persistent callback | One-shot execution |
| **Auto-reset** | Yes (on falls) | No (terminates) |
| **Metrics** | Step count only | Episode reward, mean reward |
| **Use case** | Demos, navigation, long-running | Policy evaluation, testing |

**When to use start_g1_policy_walk:**
- Interactive demonstrations
- Navigation tasks requiring continuous walking
- Long-running simulations
- When you need to query robot state during walking

**When to use run_policy_loop:**
- Policy evaluation with metrics
- Fixed-length episode testing
- Benchmark comparisons
- When you want episode statistics

## Error Handling

### Common Errors

**Policy Loading Errors:**
```
Error: "PyTorch not available"
Solution: pip install torch
```

**Robot Not Found:**
```
Error: "Prim not found: /G1"
Solution: Call create_robot() before start_g1_policy_walk()
```

**Physics Not Initialized:**
```
Error: "Physics simulation view not available"
Solution: Ensure physics scene is created and timeline is started
```

### Debugging

Enable verbose logging in the extension:

```python
import carb
carb.log_info("Policy walk step {}: height={}, velocity={}".format(...))
```

Check callback subscription status:
```python
# In extension.py
if self._policy_walk_subscription is not None:
    print("Callback is active")
```

## Performance Considerations

- **GPU Acceleration**: Policy inference runs on CUDA if available
- **Physics Step Rate**: Callback executes at simulation frequency (~60 Hz)
- **Overhead**: Minimal (<1ms per step for policy inference on GPU)
- **Memory**: Policy and articulation state persist in extension (~50-100 MB)

## Limitations

1. **Single Policy**: Only one policy walk callback can be active at a time
2. **Policy Format**: Currently supports SB3 .zip and PyTorch .pt/.pth formats
3. **Observation Matching**: Observation space must match training environment
4. **Auto-reset**: Uses fixed standing pose; may not work for all robot types

## Future Enhancements

Potential improvements for future versions:

- [ ] Support multiple robots with independent policies
- [ ] Configurable auto-reset poses
- [ ] Real-time velocity command updates (without restart)
- [ ] Callback metrics (FPS, inference time, success rate)
- [ ] Support for other RL frameworks (TensorFlow, JAX)
- [ ] Navigation waypoint following integration

## Related Tools

- **Navigation (point-to-point):** `navigate_to`, `get_navigation_status`, `stop_navigation` — see [NAVIGATION_TOOLS.md](NAVIGATION_TOOLS.md)
- `load_policy`: Load policy without starting callback
- `run_policy_loop`: Run fixed-length policy episode
- `get_robot_state`: Query robot state during walking
- `get_robot_pose`: Get base pose for navigation
- `set_velocity_command`: Set target velocity (legacy; navigation overwrites while active)
- `stop_simulation`: Stop timeline (also stops callback)

## References

- Isaac Sim Update Events: https://docs.omniverse.nvidia.com/kit/docs/omni.kit.app/latest/omni.kit.app.html
- Articulation API: https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_adding_manipulator.html
- Isaac Lab G1 Training: /home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/

## Example Output

```
[start_g1_policy_walk]
Policy walk started: Policy walk started with callback registered. Robot will walk continuously until stop_g1_policy_walk is called.

[Extension logs during walking]
Policy walk initialized - robot standing and timeline playing
Policy walk step 100: height=0.782m, forward_vel=0.48m/s
Policy walk step 200: height=0.779m, forward_vel=0.52m/s
Policy walk step 300: height=0.775m, forward_vel=0.51m/s
...

[stop_g1_policy_walk]
Policy walk stopped after 1523 steps
```
