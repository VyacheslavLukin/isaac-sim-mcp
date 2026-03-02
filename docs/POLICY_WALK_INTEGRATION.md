# Policy Walk Integration Guide

## Overview

This guide explains how to integrate the new `start_g1_policy_walk` tool with trained policies from IsaacLab or Stable Baselines3.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ MCP Server (isaac_mcp/server.py)                          │
│  - start_g1_policy_walk() tool                             │
│  - stop_g1_policy_walk() tool                              │
│  - Sends commands via socket to extension                  │
└───────────────────┬─────────────────────────────────────────┘
                    │ Socket (localhost:8766)
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Isaac Sim MCP Extension (extension.py)                     │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Persistent State                                      │ │
│  │  - _loaded_policy (checkpoint data)                  │ │
│  │  - _policy_robot_articulation (Articulation)         │ │
│  │  - _policy_controller (ArticulationController)       │ │
│  │  - _policy_walk_subscription (EventSubscription)     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Policy Walk Callback (registered on update stream)   │ │
│  │                                                       │ │
│  │  1. First frame:                                      │ │
│  │     - Reset robot to standing pose                    │ │
│  │     - Start timeline                                  │ │
│  │                                                       │ │
│  │  2. Subsequent frames:                                │ │
│  │     - Get robot state                                 │ │
│  │     - Normalize observations                          │ │
│  │     - Run policy inference                            │ │
│  │     - Apply joint commands                            │ │
│  │     - Check termination / auto-reset                  │ │
│  └───────────────────────────────────────────────────────┘ │
│                    │                                         │
│                    ▼                                         │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ omni.kit.app.update_event_stream                     │ │
│  │  - Callback executes on every frame (~60 Hz)         │ │
│  └───────────────────────────────────────────────────────┘ │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Isaac Sim Physics / Rendering                               │
│  - Articulation API                                         │
│  - Timeline                                                 │
│  - PhysX simulation                                         │
└─────────────────────────────────────────────────────────────┘
```

## Training to Deployment Workflow

### 1. Train Policy with IsaacLab

```bash
# Train G1 walking policy
cd /home/ubuntu/IsaacLab
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 \
  --headless \
  --num_envs 4096

# Exports checkpoint to:
# logs/rsl_rl/g1_flat/*/model_*.pt
```

### 2. Convert Policy (if needed)

For IsaacLab RSL-RL policies, the `.pt` checkpoint can be used directly.

For Stable Baselines3 policies from `/home/ubuntu/workspace/`:
```bash
# Already in .zip format, ready to use
ls /home/ubuntu/workspace/checkpoints_terminal/
# best_model.zip, g1_walking_3000000_steps.zip, etc.
```

### 3. Deploy with MCP

```python
# In Cursor/Claude or via MCP client:

# Setup scene
get_scene_info()
create_physics_scene(objects=[], floor=True)
create_robot(robot_type="g1_minimal", position=[0, 0, 0])

# Start policy walking
start_g1_policy_walk(
    policy_path="/home/ubuntu/workspace/checkpoints_terminal/best_model.zip",
    robot_prim_path="/G1",
    target_velocity=0.5
)

# Robot walks continuously...

# Stop when done
stop_g1_policy_walk()
```

## Policy Compatibility

### Supported Formats

| Format | Framework | Extension | Loading Method |
|--------|-----------|-----------|----------------|
| **Stable Baselines3** | PPO, SAC, TD3 | `.zip` | `zipfile` + `torch.load('policy.pth')` |
| **RSL-RL** | PPO | `.pt`, `.pth` | `torch.load()` with `model_state_dict` |
| **PyTorch Direct** | Custom | `.pt`, `.pth` | `torch.load()` raw checkpoint |

### Observation Space Requirements

The policy must be trained on an observation space matching:

```python
obs_dim = num_joints * 2 + 3 + 4 + 3 + 3 + 1
#         joint_pos/vel + base_pos + base_quat + base_lin_vel + base_ang_vel + target_vel

# For G1 minimal (37 DOF):
obs_dim = 37*2 + 3 + 4 + 3 + 3 + 1 = 88
```

Normalization:
- Joint positions: divided by π, clipped to [-1, 1]
- Joint velocities: divided by 10, clipped to [-1, 1]
- Base state: unnormalized (raw values)

### Action Space Requirements

The policy must output joint position commands:

```python
action_dim = num_joints  # 37 for g1_minimal, 43 for g1 full

# Expected range: [-1, 1] (normalized)
# Scaled by π before applying to robot
```

## Integration with Existing Training Scripts

### From workspace/train_g1_terminal.py

The SB3 training script in `/home/ubuntu/workspace/train_g1_terminal.py` outputs `.zip` checkpoints that are directly compatible:

```python
# Training creates checkpoints at:
checkpoints_terminal/
├── best_model.zip          # Best performing model
├── g1_walking_1000000_steps.zip
├── g1_walking_2000000_steps.zip
└── ...

# Use directly with:
start_g1_policy_walk(
    policy_path="/home/ubuntu/workspace/checkpoints_terminal/best_model.zip",
    robot_prim_path="/G1",
    target_velocity=0.5
)
```

### From IsaacLab RSL-RL Training

IsaacLab training outputs PyTorch checkpoints:

```python
# Training creates:
IsaacLab/logs/rsl_rl/g1_flat/2024-01-15_10-30-45/
├── model_1000.pt
├── model_2000.pt
└── ...

# Convert path to absolute:
policy_path = "/home/ubuntu/IsaacLab/logs/rsl_rl/g1_flat/2024-01-15_10-30-45/model_2000.pt"

start_g1_policy_walk(
    policy_path=policy_path,
    robot_prim_path="/G1",
    target_velocity=0.5
)
```

## Callback vs Loop Comparison

### start_g1_policy_walk (Callback-based)

**Pros:**
- Non-blocking: MCP server free for other commands
- Continuous: Runs indefinitely until stopped
- Efficient: No polling overhead
- Persistent: State maintained in extension memory
- Robust: Auto-resets on falls

**Cons:**
- No episode metrics (reward, success rate)
- Requires explicit stop command
- Single policy at a time
- Harder to debug (callback executes asynchronously)

**Best for:**
- Interactive demonstrations
- Navigation tasks
- Long-running simulations
- Real-time control adjustments

### run_policy_loop (Blocking-based)

**Pros:**
- Episode metrics: Returns total/mean reward
- Automatic termination: Stops after N steps or on fall
- Synchronous: Easier to debug
- Evaluation: Good for benchmarking

**Cons:**
- Blocking: MCP server busy during execution
- Fixed duration: Must specify num_steps
- Overhead: Each step requires MCP round-trip

**Best for:**
- Policy evaluation
- Benchmarking
- Testing trained policies
- Collecting episode statistics

## Example: Navigation with Policy Walk

```python
import time
import math

# Setup
get_scene_info()
create_physics_scene(objects=[], floor=True)
create_robot(robot_type="g1_minimal", position=[0, 0, 0])

# Start walking
start_g1_policy_walk(
    policy_path="/workspace/checkpoints_terminal/best_model.zip",
    robot_prim_path="/G1",
    target_velocity=0.5
)

# Navigation loop
waypoints = [[5, 0], [5, 5], [0, 5], [0, 0]]
for waypoint in waypoints:
    print(f"Walking to waypoint: {waypoint}")

    while True:
        # Get current pose
        pose = get_robot_pose(prim_path="/G1")
        pos = pose["position"]
        current_xy = [pos[0], pos[1]]

        # Check if reached waypoint
        distance = math.sqrt(
            (current_xy[0] - waypoint[0])**2 +
            (current_xy[1] - waypoint[1])**2
        )

        if distance < 0.5:  # Within 0.5m
            print(f"Reached waypoint {waypoint}")
            break

        # Continue walking (callback keeps robot moving)
        time.sleep(0.1)

# Stop at final waypoint
stop_g1_policy_walk()
print("Navigation complete!")
```

## Debugging Tips

### Enable Verbose Logging

In the extension, add logging:

```python
# In policy_walk_callback():
carb.log_info(f"Step {self._policy_walk_step_count}: "
              f"obs={obs[:5]}, action={action[:5]}, "
              f"height={height:.3f}, vel={forward_vel:.2f}")
```

### Check Callback Status

```python
# In extension console (Isaac Sim Script Editor):
import omni.kit.app
app = omni.kit.app.get_app()
stream = app.get_update_event_stream()
print(f"Active subscriptions: {stream.get_num_subscribers()}")
```

### Visualize Policy Actions

```python
# Add visualization in callback:
from omni.isaac.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

# Draw target joint angles as lines
for i, target_pos in enumerate(scaled_action[:6]):
    # Visualize leg joint targets
    draw.draw_line(joint_pos, target_pos, color=(1, 0, 0))
```

## Performance Tuning

### Reduce Inference Latency

```python
# Use half precision (FP16) for faster inference
self._loaded_policy['checkpoint'] = checkpoint.half()

# Or use TorchScript compilation
traced_policy = torch.jit.trace(policy_network, example_obs)
```

### Adjust Callback Frequency

If policy inference is too slow, run callback every N frames:

```python
# In policy_walk_callback():
if self._policy_walk_step_count % 2 == 0:
    return  # Skip this frame, run at 30 Hz instead of 60 Hz
```

### Optimize Observation Collection

Cache robot state that doesn't change:

```python
# Cache DOF names (don't query every frame)
if not hasattr(self, '_cached_dof_names'):
    self._cached_dof_names = self._policy_robot_articulation.dof_names
```

## Troubleshooting

### Callback Not Executing

**Symptom**: Robot doesn't move after `start_g1_policy_walk`

**Solutions:**
1. Check timeline is playing: `timeline.is_playing()`
2. Verify callback subscription: `self._policy_walk_subscription is not None`
3. Check for exceptions in callback (won't crash, but logged as errors)

### Robot Falls Immediately

**Symptom**: Auto-reset loop (falls, resets, falls again)

**Solutions:**
1. Verify policy matches robot DOF count (37 for g1_minimal, 43 for g1)
2. Check observation normalization matches training
3. Test with `run_policy_loop` first (easier to debug)
4. Verify standing pose is stable (tune PD gains if needed)

### Poor Walking Performance

**Symptom**: Robot walks but unstable or slow

**Solutions:**
1. Check target_velocity matches training range (G1 trained on [0, 1] m/s)
2. Verify action scaling (multiply by π for radians)
3. Tune PD gains: `controller.set_gains(kps=[200]*num_joints, kds=[5]*num_joints)`
4. Check physics timestep matches training (1/60 = 16.67ms)

## Related Files

- Extension: `/home/ubuntu/isaac-sim-mcp/isaac.sim.mcp_extension/isaac_sim_mcp_extension/extension.py`
- Server: `/home/ubuntu/isaac-sim-mcp/isaac_mcp/server.py`
- Documentation: `/home/ubuntu/isaac-sim-mcp/docs/START_G1_POLICY_WALK.md`
- Test Script: `/home/ubuntu/isaac-sim-mcp/examples/test_g1_policy_walk.py`
- Training Scripts: `/home/ubuntu/workspace/train_g1_terminal.py`
- IsaacLab Tasks: `/home/ubuntu/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`

## Next Steps

After implementing `start_g1_policy_walk`, consider:

1. **Multi-robot support**: Extend to handle multiple robots with independent policies
2. **Real-time velocity updates**: Allow changing `target_velocity` without restart
3. **Waypoint navigation**: Built-in waypoint following with automatic path planning
4. **Metric collection**: Track success rate, average velocity, distance traveled
5. **Policy switching**: Hot-swap policies during runtime (e.g., walk → climb stairs)

## References

- Isaac Sim Events API: https://docs.omniverse.nvidia.com/kit/docs/omni.kit.app/latest/
- Articulation API: https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_adding_manipulator.html
- IsaacLab Training: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html
- MCP Protocol: https://modelcontextprotocol.io/introduction
