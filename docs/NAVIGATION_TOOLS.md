# Navigation Tools Documentation

## Overview

The Isaac Sim MCP extension provides three tools for **point-to-point navigation** of the G1 robot using the locomotion policy:

- **`navigate_to`** — Start navigation toward a target XY position (non-blocking).
- **`get_navigation_status`** — Report current navigation state and distance to target.
- **`stop_navigation`** — Cancel navigation; policy walk keeps running.

Navigation runs as a **PRE_UPDATE callback** that computes a P-controller each physics step: rotate in place when |yaw_error| > 0.3 rad, then drive forward while correcting yaw. The callback **unsubscribes itself on arrival**, so no cleanup is required from the caller.

---

## Tool Signatures

### navigate_to

```python
navigate_to(
    target_position: list[float],   # [x, y] world coordinates in metres (required)
    robot_prim_path: str = "/G1",
    policy_path: str = "",          # Required if start_g1_policy_walk was not already called
    arrival_threshold: float = 0.5  # Distance in metres to consider "arrived"
) -> str
```

**Behaviour:**

- If the policy walk is **not** already running and `policy_path` is provided, calls `start_g1_policy_walk` internally (policy walk reuse).
- Registers a PRE_UPDATE callback that each frame:
  - Reads current robot pose and target.
  - If |yaw_error| > 0.3 rad: rotate in place.
  - Else: drive forward while correcting yaw via `_vel_cmd_x`, `_vel_cmd_y`, `_vel_cmd_yaw` (same as `set_velocity_command`).
- On arrival (distance ≤ `arrival_threshold`), the callback unsubscribes itself and sets `nav_status` to `"arrived"`.
- Returns immediately (non-blocking).

**Example:**

```python
navigate_to(target_position=[5.0, 3.0], policy_path="/path/to/policy.pt")
# Returns: "Navigation started toward [5.00, 3.00]. Call get_navigation_status() to monitor."
```

---

### get_navigation_status

```python
get_navigation_status() -> str  # JSON with nav state
```

**Returns** (as stringified JSON):

- `nav_status`: `"idle"` | `"navigating"` | `"arrived"` | `"failed"`
- `target_position`: `[x, y]`
- `current_position`: `[x, y, z]` (robot base)
- `distance_to_target`: float (metres)
- `nav_active`: bool

**Example:**

```python
get_navigation_status()
# → '{"nav_status": "navigating", "distance_to_target": 3.21, ...}'
get_navigation_status()
# → '{"nav_status": "arrived", "distance_to_target": 0.42, ...}'
```

---

### stop_navigation

```python
stop_navigation() -> str
```

**Behaviour:**

- Unsubscribes the navigation callback (if active).
- Zeros velocity commands (`_vel_cmd_x`, `_vel_cmd_y`, `_vel_cmd_yaw`).
- The **policy walk continues** (robot remains balanced); use `stop_g1_policy_walk` to stop walking entirely.

**Example:**

```python
stop_navigation()
# Returns: "Navigation stopped (was: navigating). Policy walk is still running."
```

---

## Usage flow for an AI agent

1. **Setup (once):**  
   `get_scene_info()` → `create_physics_scene()` → `create_robot("g1_minimal", position)`  
   Optionally start the policy walk first:  
   `start_g1_policy_walk(policy_path=..., robot_prim_path="/G1")`

2. **Start navigation:**  
   `navigate_to(target_position=[5.0, 3.0], policy_path="/path/to/policy.pt")`  
   (Omit `policy_path` if `start_g1_policy_walk` was already called.)

3. **Start simulation:**  
   `start_simulation()`

4. **Poll until arrived:**  
   Loop: `get_navigation_status()` until `nav_status == "arrived"` (or use `stop_navigation()` to cancel).

5. **Shutdown:**  
   `stop_g1_policy_walk()` then `stop_simulation()`.

---

## Design notes

| Aspect | Detail |
|--------|--------|
| **Non-blocking** | `navigate_to` returns immediately; navigation runs in the PRE_UPDATE callback. |
| **Policy walk reuse** | If `start_g1_policy_walk` was already called, `navigate_to` skips re-initialization and only registers the nav callback. |
| **1-frame lag** | Nav callback is subscribed after the policy callback, so velocity commands apply on the next step — negligible at navigation timescales. |
| **Self-cleaning** | Nav callback unsubscribes itself on arrival; caller does not need to clean up. |
| **No obstacle avoidance** | Current implementation is point-to-point in open space; no path planning or obstacle avoidance. |

---

## Related tools

- **`start_g1_policy_walk`** — Starts the locomotion policy; required (or pass `policy_path` to `navigate_to`).
- **`stop_g1_policy_walk`** — Stops policy walking entirely.
- **`set_velocity_command`** — Manual velocity control; navigation overwrites these each frame while active.
- **`get_robot_pose`** — Raw pose query; use `get_navigation_status` for nav-specific state.
