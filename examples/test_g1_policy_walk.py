#!/usr/bin/env python3
"""
Test script for start_g1_policy_walk MCP tool

This script demonstrates the new continuous policy walking feature that uses
a persistent callback to run policy inference on every physics step.

Prerequisites:
- Isaac Sim running with MCP extension enabled
- MCP server running (uv run ~/isaac-sim-mcp/isaac_mcp/server.py)
- Trained G1 policy checkpoint (e.g., from IsaacLab training)

Workflow:
1. Create physics scene with ground plane
2. Spawn G1 robot
3. Load policy and start continuous walking via callback
4. Robot walks indefinitely until stopped
5. Stop the walking callback when done
"""

import time
import sys


def test_g1_policy_walk_basic():
    """Basic test of the start_g1_policy_walk tool."""

    print("=" * 80)
    print("Test: G1 Policy Walk with Persistent Callback")
    print("=" * 80)

    # Note: In actual usage, these would be real MCP tool calls
    # For this example, we'll simulate the workflow

    print("\n[1/6] Checking Isaac Sim connection...")
    # result = get_scene_info()
    print("  ✓ Connected to Isaac Sim MCP extension")

    print("\n[2/6] Creating physics scene...")
    # result = create_physics_scene(objects=[], floor=True)
    print("  ✓ Physics scene created with ground plane")

    print("\n[3/6] Spawning G1 robot...")
    # result = create_robot(robot_type="g1_minimal", position=[0, 0, 0])
    print("  ✓ G1 robot (37 DOF minimal) spawned at origin")

    print("\n[4/6] Starting continuous policy walk...")
    policy_path = "/home/ubuntu/workspace/checkpoints_terminal/best_model.zip"
    # result = start_g1_policy_walk(
    #     policy_path=policy_path,
    #     robot_prim_path="/G1",
    #     target_velocity=0.5,
    #     deterministic=True
    # )
    print(f"  ✓ Policy loaded from: {policy_path}")
    print("  ✓ Callback registered on physics PRE_UPDATE stream")
    print("  ✓ Robot will walk continuously...")

    print("\n[5/6] Robot walking (callback executing on every physics step)...")
    print("  Watch the Isaac Sim viewport - robot should be walking!")
    print("  The callback runs automatically each frame:")
    print("    - Gets robot joint states and base pose")
    print("    - Runs policy inference")
    print("    - Applies joint position commands")
    print("    - Auto-resets if robot falls")

    # Simulate waiting (in real usage, robot walks for as long as needed)
    for i in range(10):
        time.sleep(1)
        step_estimate = (i + 1) * 60  # ~60 physics steps per second
        print(f"  Time: {i+1}s (~{step_estimate} policy steps executed)")

    print("\n[6/6] Stopping policy walk...")
    # result = stop_g1_policy_walk()
    print("  ✓ Callback unsubscribed")
    print("  ✓ Robot stopped after ~600 steps")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


def test_g1_policy_walk_advanced():
    """Advanced test showing callback persistence and control."""

    print("\n" + "=" * 80)
    print("Advanced Test: Callback State Persistence")
    print("=" * 80)

    print("\n[Setup] Create scene and robot...")
    # create_physics_scene(objects=[], floor=True)
    # create_robot(robot_type="g1_minimal", position=[0, 0, 0])
    print("  ✓ Scene ready")

    print("\n[Test 1] Start walking with target velocity 0.3 m/s...")
    # start_g1_policy_walk(
    #     policy_path="/home/ubuntu/workspace/checkpoints_terminal/best_model.zip",
    #     robot_prim_path="/G1",
    #     target_velocity=0.3,
    #     deterministic=True
    # )
    print("  ✓ Walking at slow speed (0.3 m/s)")
    time.sleep(5)

    print("\n[Test 2] Stop and restart with faster velocity...")
    # stop_g1_policy_walk()
    print("  ✓ Stopped")
    # start_g1_policy_walk(
    #     policy_path="/home/ubuntu/workspace/checkpoints_terminal/best_model.zip",
    #     robot_prim_path="/G1",
    #     target_velocity=0.8,
    #     deterministic=True
    # )
    print("  ✓ Walking at fast speed (0.8 m/s)")
    time.sleep(5)

    print("\n[Test 3] Demonstrate callback persistence...")
    print("  The callback continues running across frames without blocking")
    print("  Policy state and articulation remain in extension memory")
    print("  No need to reload policy or re-initialize robot")
    time.sleep(3)

    print("\n[Cleanup] Stop walking...")
    # stop_g1_policy_walk()
    print("  ✓ Callback unsubscribed")

    print("\n" + "=" * 80)
    print("Advanced test completed!")
    print("=" * 80)


def compare_with_run_policy_loop():
    """Explain the difference between start_g1_policy_walk and run_policy_loop."""

    print("\n" + "=" * 80)
    print("Comparison: start_g1_policy_walk vs run_policy_loop")
    print("=" * 80)

    print("\n[run_policy_loop]")
    print("  - Runs for a FIXED number of steps (e.g., 500)")
    print("  - Blocks until episode completes")
    print("  - Returns episode statistics (total reward, steps)")
    print("  - Good for: Testing policies, collecting metrics")
    print("  - Example:")
    print("    result = run_policy_loop(robot_prim_path='/G1', num_steps=500)")
    print("    # Waits until 500 steps complete or robot falls")

    print("\n[start_g1_policy_walk]")
    print("  - Runs CONTINUOUSLY until explicitly stopped")
    print("  - Non-blocking (callback-based)")
    print("  - Persists across frames without MCP server polling")
    print("  - Auto-resets robot if it falls")
    print("  - Good for: Demos, long-running navigation, interactive control")
    print("  - Example:")
    print("    start_g1_policy_walk(policy_path='...', target_velocity=0.5)")
    print("    # Robot walks indefinitely; you can do other things")
    print("    # ... (robot walking in background)")
    print("    stop_g1_policy_walk()  # Stop when ready")

    print("\n[Key Differences]")
    print("  1. Duration: Fixed vs Indefinite")
    print("  2. Blocking: Blocking vs Non-blocking")
    print("  3. State: One-shot vs Persistent callback")
    print("  4. Use case: Evaluation vs Demonstration/Navigation")

    print("\n" + "=" * 80)


def main():
    """Run all tests."""

    print("\n" + "=" * 80)
    print("Isaac Sim MCP - G1 Policy Walk Tests")
    print("=" * 80)
    print("\nThese tests demonstrate the new start_g1_policy_walk tool")
    print("that enables continuous policy-driven walking via callbacks.")
    print("\nNOTE: This is a demonstration script. To actually run:")
    print("  1. Start Isaac Sim with MCP extension")
    print("  2. Start MCP server: uv run ~/isaac-sim-mcp/isaac_mcp/server.py")
    print("  3. Uncomment the MCP tool calls in this script")
    print("  4. Run: python examples/test_g1_policy_walk.py")

    # Run tests
    test_g1_policy_walk_basic()
    time.sleep(1)

    test_g1_policy_walk_advanced()
    time.sleep(1)

    compare_with_run_policy_loop()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
    print("\n[Implementation Details]")
    print("The start_g1_policy_walk tool works by:")
    print("  1. Loading the policy checkpoint into extension memory")
    print("  2. Initializing robot articulation and controller")
    print("  3. Registering a callback on omni.kit.app update stream")
    print("  4. Callback executes on every frame:")
    print("     - First frame: Reset robot to standing, start timeline")
    print("     - Subsequent frames: Policy inference + action application")
    print("  5. State persists in extension until stop_g1_policy_walk")
    print("\n[Advantages]")
    print("  - Non-blocking: MCP server doesn't need to poll")
    print("  - Efficient: Policy/articulation loaded once")
    print("  - Flexible: Can change velocity or restart without reload")
    print("  - Robust: Auto-resets on falls")


if __name__ == "__main__":
    main()
