"""
Full G1 5m square navigation:
- Steps 1-6: scene setup, robot spawn, policy walk, simulation start
- Steps 5+: navigate_waypoints for 5m square + return to origin
- Step 8: cleanup

Run: python3 /home/ubuntu/isaac-sim-mcp/run_square_nav.py
"""
import sys
import time
import json

sys.path.insert(0, '/home/ubuntu/isaac-sim-mcp')

from isaac_mcp.server import (
    get_isaac_connection,
    navigate_waypoints,
    get_navigation_status,
    stop_navigation,
    stop_g1_policy_walk,
)

POLICY_PATH = '/home/workspace/exported/g1_nav_flat_from_rough_1450_jit.pt'
ROBOT_PRIM = '/G1'
ARRIVAL_THRESHOLD = 0.5
POLL_INTERVAL = 3.0
NAV_TIMEOUT = 300  # seconds total

# 5m square: 4 corners + return to origin
WAYPOINTS = [
    [2.5,  2.5],
    [2.5, -2.5],
    [-2.5, -2.5],
    [-2.5,  2.5],
    [0.0,   0.0],
]

SEP = "=" * 60

def send(conn, cmd_type, params=None):
    """Send a raw command to Isaac Sim extension and return parsed result."""
    result = conn.send_command(cmd_type, params or {})
    return result

def step(label, result):
    print(f"\n{SEP}\n  {label}\n{SEP}")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)
    if isinstance(result, dict) and result.get("status") == "error":
        raise RuntimeError(f"Step failed: {result.get('message', result)}")
    return result

def main():
    print(f"\n{SEP}")
    print("  G1 ROBOT - 5m SQUARE NAVIGATION")
    print(f"{SEP}")
    print(f"  Policy:    {POLICY_PATH}")
    print(f"  Robot:     {ROBOT_PRIM}")
    print(f"  Waypoints: {WAYPOINTS}")
    print()

    # --- Step 1: Verify MCP connection ---
    print(f"\n{SEP}\n  Step 1: get_scene_info (verify MCP connection)\n{SEP}")
    conn = get_isaac_connection()
    r = send(conn, "get_scene_info")
    print(json.dumps(r, indent=2))
    if r.get("message") != "pong" and r.get("status") != "success":
        raise RuntimeError(f"MCP connection check failed: {r}")
    print("  -> MCP connection verified.")

    # --- Step 2: Create physics scene ---
    step("Step 2: create_physics_scene",
         send(conn, "create_physics_scene", {"floor": True, "objects": [], "gravity": [0, 0, -9.81]}))
    time.sleep(0.5)

    # --- Step 3: Spawn G1 minimal robot ---
    step("Step 3: create_robot (g1_minimal)",
         send(conn, "create_robot", {"robot_type": "g1_minimal", "position": [0, 0, 0.74]}))
    time.sleep(0.5)

    # --- Step 4: Start policy walk ---
    step("Step 4: start_g1_policy_walk",
         send(conn, "start_g1_policy_walk", {
             "policy_path": POLICY_PATH,
             "robot_prim_path": ROBOT_PRIM,
             "target_velocity": 0.5,
             "deterministic": True,
         }))
    time.sleep(0.5)

    # --- Step 5: Launch navigate_waypoints (non-blocking, MCP-server-side A*) ---
    print(f"\n{SEP}\n  Step 5: navigate_waypoints (5m square + origin)\n{SEP}")
    nav_result = navigate_waypoints(
        positions=WAYPOINTS,
        robot_prim_path=ROBOT_PRIM,
        arrival_threshold=ARRIVAL_THRESHOLD,
        visualize_corners=True,
    )
    print(nav_result)
    if "Error" in str(nav_result):
        raise RuntimeError(f"navigate_waypoints failed: {nav_result}")

    # --- Step 6: Start simulation ---
    step("Step 6: start_simulation",
         send(conn, "start_simulation", {}))
    print("  -> Physics timeline started. Navigation running in background.")
    time.sleep(2.0)  # let simulation settle

    # --- Step 7: Poll navigation status ---
    print(f"\n{SEP}\n  Step 7: Polling navigation status\n{SEP}")
    print(f"  Timeout: {NAV_TIMEOUT}s | Poll interval: {POLL_INTERVAL}s")
    start_time = time.time()
    last_seq_index = -1
    poll_count = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > NAV_TIMEOUT:
            print(f"\n  TIMEOUT after {NAV_TIMEOUT}s")
            break

        status_str = get_navigation_status()
        try:
            status = json.loads(status_str)
        except Exception:
            status = {"raw": status_str}

        nav_status = status.get("nav_status", "unknown")
        seq_index  = status.get("seq_index", 0)
        seq_total  = status.get("seq_total", 0)
        seq_active = status.get("seq_active", False)
        dist       = status.get("distance_to_target")
        current_pos = status.get("current_position", "?")
        poll_count += 1

        if seq_index != last_seq_index:
            wp_label = f"[{WAYPOINTS[seq_index][0]}, {WAYPOINTS[seq_index][1]}]" \
                if seq_index < len(WAYPOINTS) else "(beyond list)"
            dist_str = f"{dist:.2f}m" if isinstance(dist, float) else str(dist)
            print(f"\n  [WP {seq_index}/{seq_total}] target={wp_label} status={nav_status} "
                  f"pos={current_pos} dist={dist_str}")
            last_seq_index = seq_index
        elif poll_count % 4 == 0:
            dist_str = f"{dist:.2f}m" if isinstance(dist, float) else str(dist)
            print(f"  ... seq={seq_index}/{seq_total} status={nav_status} dist={dist_str} "
                  f"pos={current_pos} t={elapsed:.0f}s")

        if nav_status == "failed":
            print(f"\n  Navigation FAILED: {status.get('error', 'unknown')}")
            break

        if nav_status == "arrived" and not seq_active:
            elapsed_total = time.time() - start_time
            print(f"\n  All {seq_total} waypoints complete! Arrived at origin.")
            print(f"  Final position: {current_pos}")
            print(f"  Total navigation time: {elapsed_total:.1f}s")
            break

        time.sleep(POLL_INTERVAL)

    # --- Step 8: Cleanup ---
    print(f"\n{SEP}\n  Step 8: Cleanup\n{SEP}")
    print(f"  stop_navigation:      {stop_navigation()}")
    print(f"  stop_g1_policy_walk:  {stop_g1_policy_walk()}")
    r = send(conn, "stop_simulation", {})
    print(f"  stop_simulation:      {json.dumps(r)}")
    print(f"\n  Done.")

if __name__ == "__main__":
    main()
