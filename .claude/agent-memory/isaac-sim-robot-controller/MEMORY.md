# Isaac Sim Robot Controller - Agent Memory

## Critical: Docker Path Mapping

Isaac Sim runs in Docker as user 1234. Policy files must be in the Docker named volume:
- Host volume mountpoint: `/var/lib/docker/volumes/g1-exported-policy/_data/`
- Container path: `/home/workspace/exported/<filename>.pt`
- Host workspace path `/home/ubuntu/workspace/exported/` is NOT directly bind-mounted; it is SHADOWED by the named volume
- Always copy new policy files: `sudo cp <host_path> /var/lib/docker/volumes/g1-exported-policy/_data/`
- Set permissions: `sudo chmod 644 /var/lib/docker/volumes/g1-exported-policy/_data/<file>.pt`
- Passing the host path `/home/ubuntu/workspace/exported/...` to MCP tools will produce "Permission denied"

## MCP Connection

- Socket: `localhost:8766` (TCP)
- Message format: `{"type": "<command_type>", "params": {...}}` (NOT `{"command": ...}`)
- Always verify connection first with `get_scene_info` before any other call
- `get_scene_info` returns `{"status": "success", "message": "pong", "assets_root_path": "..."}`

## Policy Files (Confirmed Working)

| File | Container Path | obs_dim | act_dim | Status |
|------|---------------|---------|---------|--------|
| g1_flat_policy_4498.pt | `/home/workspace/exported/g1_flat_policy_4498.pt` | 123 | 37 | Works with extension |
| g1_flat_policy_12100.pt | `/home/workspace/exported/g1_flat_policy_12100.pt` | 123 | 37 | Available |
| g1_rough_policy_1450.pt | `/home/workspace/exported/g1_rough_policy_1450.pt` | **310** | 37 | **Incompatible** (see below) |

## Rough Terrain Policy (g1_rough_policy_1450.pt) - Status: PATCHED BUT NEEDS RESTART

The extension SOURCE (`extension.py`) has been fully patched with:
- `_detect_policy_obs_dim()`: dummy forward-pass with dims [123, 310] to auto-detect
- `_compute_height_scan()`: PhysX raycasting 17x11 grid matching GridPatternCfg(resolution=0.1, size=[1.6,1.0])
- `_policy_walk_use_height_scan` flag: set True when obs_dim==310, appends 187-dim scan in callback

**CRITICAL: Stale `.pyc` bytecode issue (2026-02-22)**
The Kit/Omniverse Python runtime loaded the OLD compiled `.pyc` (from Feb 8, 42KB) instead of
recompiling from the updated `extension.py` (Feb 22, 102KB). This caused the old 123-dim-only
behavior to run despite the source being correct.

Fix applied: deleted `/home/ubuntu/isaac-sim-mcp/isaac.sim.mcp_extension/isaac_sim_mcp_extension/__pycache__/extension.cpython-310.pyc`
Effect: Isaac Sim MUST BE RESTARTED for the new code to load. After restart, the stale pyc is gone
and Kit will recompile from source. Look for "[Info] Policy obs_dim auto-detected: 310" in kit log.

**After restart, confirm by looking for these log messages:**
1. `Policy obs_dim auto-detected: 310`
2. `Rough terrain policy detected (obs_dim=310): height scan will be computed via PhysX raycasting`
If neither appears, the old code is still running.

## Robot Architecture

- Always use `robot_type="g1_minimal"` (37 DOF) for Isaac Lab policies
- Spawn position: `[0, 0, 0.74]` (standing height)
- The `g1` type has 43 DOF and will mismatch all 37-DOF policies

## Correct Operation Order

1. `get_scene_info` (verify connection)
2. `create_physics_scene(floor=True, objects=[], gravity=[0,0,-9.81])`
3. `create_robot(robot_type="g1_minimal", position=[0,0,0.74])`
4. `start_g1_policy_walk(policy_path=<CONTAINER path>, ...)` OR `navigate_to(...)`
5. `start_simulation()`
6. Poll / monitor
7. `stop_navigation()` (if nav active) -> `stop_g1_policy_walk()` -> `stop_simulation()`

Note: simulation was started before `start_g1_policy_walk` in the 2026-02-22 session and it still worked.

## Detailed Notes

See `debugging.md` for the full rough terrain policy incompatibility investigation.
