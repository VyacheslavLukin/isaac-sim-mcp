# Debugging Notes

## 2026-02-22: Rough Terrain Policy Investigation

### Session Goal
Deploy g1_rough_policy_1450.pt (obs=310, act=37) via MCP.

### Root Cause: Observation Dimension Mismatch

**Extension behavior** (`extension.py` lines 1805-1816):
The `_policy_walk_callback` builds a fixed 123-dim obs:
```python
obs = np.concatenate([
    base_lin_vel,           # (3,) body frame
    base_ang_vel,           # (3,) body frame
    projected_gravity,      # (3,)
    velocity_commands,      # (3,)
    joint_pos_rel,          # (37,)
    joint_vel,              # (37,)
    last_action             # (37,)
])  # total = 123
```

**Rough terrain policy obs breakdown** (from `velocity_env_cfg.py`):
```
base_lin_vel:        3
base_ang_vel:        3
projected_gravity:   3
velocity_commands:   3
joint_pos_rel:      37
joint_vel_rel:      37
last_action:        37
height_scan:       187  <- MISSING from extension
                  ----
Total:             310
```

**height_scan** comes from `GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`:
- x points: round(1.6/0.1) + 1 = 17
- y points: round(1.0/0.1) + 1 = 11
- total rays: 17 * 11 = 187

**Network architecture** (confirmed by probing):
```
Layer 0: Linear(310 -> 512) + ELU
Layer 2: Linear(512 -> 256) + ELU
Layer 4: Linear(256 -> 128) + ELU
Layer 6: Linear(128 -> 37)
```
(flat policy is 123->256->128->128->37, different hidden sizes too)

### Observed Failure Mode

1. `start_g1_policy_walk` returns `{"status": "success"}` - callback is registered
2. Every physics step: callback builds 123-dim obs, calls `model(obs_tensor)` where obs_tensor is shape [1,123]
3. PyTorch raises: `mat1 and mat2 shapes cannot be multiplied (1x123 and 310x512)`
4. `carb.log_error(traceback.format_exc())` logs it to Isaac Sim log but does NOT propagate to MCP caller
5. Robot receives no joint commands, collapses under gravity
6. Auto-reset (height < 0.3m) triggers, puts robot back to [0,0,0.74], but next step fails again -> infinite fall-reset loop
7. `stop_g1_policy_walk` reports N steps (all failed inference, not walking steps)

### Evidence

- After 10 seconds, `get_robot_state` showed robot at [0.86, -0.001, 0.054] (fallen, ~0.05m height)
- `get_robot_pose` (which uses cached reset position) showed [0,0,0.74] - misleading
- max joint velocity = 0.0 rad/s confirmed no active control
- After stop, reported "806 steps" - callback ran 806 times, all failed

### Fix Required in extension.py

To support rough terrain policies, the observation builder needs:
1. A RayCaster sensor initialized on the G1 robot (torso_link) with same pattern as training
2. Height scan data read per step and appended to obs
3. Possibly: auto-detect obs_dim from loaded policy and zero-pad or error-out gracefully

### Docker Volume Issue

The rough policy file existed on host but NOT in the Docker named volume.
- Host: `/home/ubuntu/workspace/exported/g1_rough_policy_1450.pt` (owner: ubuntu:ubuntu, mode 664)
- Named volume: `g1-exported-policy` mounted at `/home/workspace/exported` in container
- Container user: 1234 (not ubuntu group member)
- Fix: `sudo cp <host> /var/lib/docker/volumes/g1-exported-policy/_data/ && sudo chmod 644 <file>`
- Container path used: `/home/workspace/exported/g1_rough_policy_1450.pt`

## 2026-02-22: Stale .pyc Bytecode Issue

### Root Cause
The extension source (`extension.py`) was updated on 2026-02-22 to add `_detect_policy_obs_dim()`,
`_compute_height_scan()`, and `_policy_walk_use_height_scan` support. But the Kit Python runtime
loaded the OLD compiled `.pyc` from 2026-02-08 (42KB vs 102KB source) ignoring the mtime mismatch.

**Why CPython's normal invalidation didn't fire:** Omniverse Kit uses a custom Python import
mechanism for extensions. When the `.pyc` timestamp doesn't match `.py`, normal CPython would
recompile - but Kit's loader apparently caches the compiled module across restarts by using the
existing `.pyc` unconditionally (likely a Kit-specific import hook).

### Evidence
- `source_mtime_in_pyc=1770562911` (Feb 8) != `actual_py_mtime=1771799566` (Feb 22)
- `source_size_in_pyc=83379` != `actual_py_size=102507`
- Strings in .pyc: no `_detect_policy_obs_dim` or `_compute_height_scan`
- Running session: still produced `mat1=[1x123] mat2=[310x512]` errors every step

### Fix
Delete the stale `.pyc` from the HOST (which is the bind-mount source, read-only in container):
```bash
rm /home/ubuntu/isaac-sim-mcp/isaac.sim.mcp_extension/isaac_sim_mcp_extension/__pycache__/extension.cpython-310.pyc
```
Then **restart Isaac Sim** so Kit recompiles from the updated source.

### Prevention
After EVERY update to `extension.py`, run:
```bash
rm -f /home/ubuntu/isaac-sim-mcp/isaac.sim.mcp_extension/isaac_sim_mcp_extension/__pycache__/*.pyc
```
before (re)starting Isaac Sim, to guarantee fresh bytecode.

### Docker Volume & Mount Map (confirmed 2026-02-22)
- Extension: `/home/ubuntu/isaac-sim-mcp/isaac.sim.mcp_extension` -> `/isaac-sim/exts/isaac.sim.mcp_extension` (RO)
- Workspace: `/home/ubuntu/workspace` -> `/home/workspace` (RW)
- Exported policies: `g1-exported-policy` volume -> `/home/workspace/exported` (RW)
- Isaac Sim logs: `/home/ubuntu/docker/isaac-sim/logs` -> `/isaac-sim/.nvidia-omniverse/logs` (RW)
