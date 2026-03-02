# Quick Start: Robot Walking with Pre-trained Policies

## 5-Minute Setup

### 1. Prerequisites
- Isaac Sim running with MCP extension enabled
- MCP server running: `python /home/ubuntu/isaac-sim-mcp/isaac_mcp/server.py`
- Trained policy file (e.g., `g1_walking_trained.zip`)

### 2. Minimal Working Example

```python
# Setup scene
create_physics_scene(objects=[], floor=True)
create_robot(robot_type="g1", position=[0, 0, 0])

# Load and run policy
load_policy("/path/to/g1_walking_trained.zip", "/G1")
reset_robot_pose("/G1")
start_simulation()
result = run_policy_loop("/G1", num_steps=500)

# Result: {"total_reward": 243.5, "steps_completed": 487, ...}
```

### 3. That's It!

Watch your robot walk in the Isaac Sim viewport.

---

## Essential Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `load_policy()` | Load trained model | `load_policy("/path/to/policy.zip", "/G1")` |
| `reset_robot_pose()` | Stand up robot | `reset_robot_pose("/G1")` |
| `start_simulation()` | Start physics | `start_simulation()` |
| `run_policy_loop()` | Run inference | `run_policy_loop("/G1", num_steps=500)` |
| `stop_simulation()` | Stop physics | `stop_simulation()` |

---

## Common Workflows

### Train and Test

```python
# 1. Train policy (outside Isaac Sim)
# python train_g1_walking_fixed.py

# 2. Test in Isaac Sim
create_physics_scene(floor=True)
create_robot("g1", [0, 0, 0])
load_policy("/isaac-sim/g1_walking_trained.zip", "/G1")
reset_robot_pose("/G1")
start_simulation()
run_policy_loop("/G1", num_steps=1000)
stop_simulation()
```

### Manual Control

```python
# Setup
load_policy("/path/to/policy.zip", "/G1")
start_simulation()

# Control loop
for step in range(100):
    state_json = get_robot_state("/G1")
    # ... process state, run policy ...
    apply_joint_actions("/G1", joint_positions=actions)
    step_simulation(1)
```

### Monitor Robot

```python
state_json = get_robot_state("/G1")
state = json.loads(state_json)

print(f"Height: {state['base_position'][2]:.3f}m")
print(f"Forward velocity: {state['base_linear_velocity'][0]:.2f}m/s")
print(f"Joints: {state['num_joints']}")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Robot doesn't move | Call `start_simulation()` first |
| Robot falls immediately | Use `reset_robot_pose("/G1")` before running |
| "Physics view not available" | Wait 100 steps: `step_simulation(100, render=False)` |
| Policy loading fails | Check file path and format (.zip or .pt) |
| "Robot not initialized" | Call `load_policy()` before using robot tools |

---

## File Paths

- **Extension:** `/home/ubuntu/isaac-sim-mcp/isaac.sim.mcp_extension/isaac_sim_mcp_extension/extension.py`
- **MCP Server:** `/home/ubuntu/isaac-sim-mcp/isaac_mcp/server.py`
- **Full Guide:** `/home/ubuntu/isaac-sim-mcp/POLICY_EXECUTION_GUIDE.md`
- **Examples:** `/home/ubuntu/isaac-sim-mcp/examples/run_pretrained_policy.py`
- **Training Env:** `/home/ubuntu/workspace/train_g1_walking_fixed.py`

---

## Next Steps

1. Read the full guide: `POLICY_EXECUTION_GUIDE.md`
2. Try the examples: `examples/run_pretrained_policy.py`
3. Train your own policy: `workspace/train_g1_walking_fixed.py`
4. Integrate with your application!

---

## Support

For detailed information, see:
- **User Guide:** `POLICY_EXECUTION_GUIDE.md` (comprehensive reference)
- **Implementation:** `IMPLEMENTATION_SUMMARY.md` (technical details)
- **Examples:** `examples/run_pretrained_policy.py` (working code)
