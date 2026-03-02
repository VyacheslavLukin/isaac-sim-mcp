#!/usr/bin/env python3
"""
Example: Running Pre-trained Policy via MCP Tools

This script demonstrates how to use the new MCP tools to:
1. Set up a physics scene with a G1 robot
2. Load a pre-trained RL policy
3. Execute the policy to make the robot walk
4. Monitor robot state and control simulation

Prerequisites:
- Isaac Sim MCP extension running
- MCP server running (isaac_mcp/server.py)
- Trained policy file (e.g., g1_walking_trained.zip)
"""

import json
import time


def example_policy_execution():
    """Complete example of loading and running a pre-trained policy."""

    print("=" * 70)
    print("Pre-trained Policy Execution via MCP Tools")
    print("=" * 70)

    # Step 1: Check scene info
    print("\n[1/8] Checking Isaac Sim connection...")
    # In real usage: result = get_scene_info()
    print("  Scene info retrieved successfully")

    # Step 2: Create physics scene with ground plane
    print("\n[2/8] Creating physics scene...")
    # create_physics_scene(objects=[], floor=True)
    print("  Physics scene created with ground plane")

    # Step 3: Create G1 robot
    print("\n[3/8] Creating G1 humanoid robot...")
    # create_robot(robot_type="g1", position=[0, 0, 0])
    print("  G1 robot spawned at origin")

    # Step 4: Reset robot to standing pose
    print("\n[4/8] Resetting robot to standing pose...")
    # reset_robot_pose(robot_prim_path="/G1", base_position=[0, 0, 0.8])
    print("  Robot positioned at height 0.8m in standing configuration")

    # Step 5: Load pre-trained policy
    print("\n[5/8] Loading pre-trained policy...")
    policy_path = "/isaac-sim/g1_walking_trained.zip"
    # result = load_policy(policy_path=policy_path, robot_prim_path="/G1")
    print(f"  Loaded SB3 PPO policy from {policy_path}")
    print("  Policy architecture: MlpPolicy (256x256 hidden layers)")
    print("  Training: 1M steps, walking task")

    # Step 6: Start simulation
    print("\n[6/8] Starting physics simulation...")
    # start_simulation()
    print("  Timeline playing, physics active")

    # Step 7: Run policy loop
    print("\n[7/8] Running policy inference loop...")
    print("  Watch the robot walk in Isaac Sim viewport!")
    # result = run_policy_loop(
    #     robot_prim_path="/G1",
    #     num_steps=500,
    #     deterministic=True
    # )
    # Simulated result
    result = {
        "message": "Policy loop completed 487 steps",
        "total_reward": 243.5,
        "mean_reward": 0.5,
        "steps_completed": 487
    }
    print(f"  Total reward: {result['total_reward']:.2f}")
    print(f"  Mean reward: {result['mean_reward']:.3f}")
    print(f"  Steps completed: {result['steps_completed']} (fell at 487)")

    # Step 8: Stop simulation
    print("\n[8/8] Stopping simulation...")
    # stop_simulation()
    print("  Simulation stopped")

    print("\n" + "=" * 70)
    print("Policy execution completed successfully!")
    print("=" * 70)


def example_manual_control_loop():
    """Example of manual policy control using low-level tools."""

    print("\n" + "=" * 70)
    print("Manual Policy Control Loop (Advanced)")
    print("=" * 70)

    # Setup (same as above)
    print("\n[Setup] Creating scene and loading policy...")
    # create_physics_scene(objects=[], floor=True)
    # create_robot(robot_type="g1", position=[0, 0, 0])
    # load_policy("/isaac-sim/g1_walking_trained.zip", "/G1")
    # reset_robot_pose("/G1")
    # start_simulation()
    print("  Scene ready, policy loaded, simulation started")

    # Manual control loop
    print("\n[Control Loop] Running manual inference...")
    num_steps = 100

    for step in range(num_steps):
        # 1. Get robot state
        # state_json = get_robot_state("/G1")
        # state = json.loads(state_json)

        # Simulated state
        state = {
            "joint_positions": [0.0] * 43,
            "joint_velocities": [0.0] * 43,
            "base_position": [0.1 * step / 100, 0, 0.8],
            "base_orientation": [1, 0, 0, 0],
            "base_linear_velocity": [0.5, 0, 0],
            "base_angular_velocity": [0, 0, 0],
            "num_joints": 43,
            "dof_names": ["joint_" + str(i) for i in range(43)]
        }

        # 2. Prepare observation (normalize as needed)
        import numpy as np
        joint_pos = np.array(state["joint_positions"])
        joint_vel = np.array(state["joint_velocities"])
        base_lin_vel = np.array(state["base_linear_velocity"])

        # Normalize
        norm_pos = np.clip(joint_pos / np.pi, -1.0, 1.0)
        norm_vel = np.clip(joint_vel / 10.0, -1.0, 1.0)

        obs = np.concatenate([
            norm_pos,
            norm_vel,
            state["base_position"],
            state["base_orientation"],
            base_lin_vel,
            state["base_angular_velocity"],
            [0.5]  # target velocity
        ])

        # 3. Policy inference (placeholder - would use loaded model)
        # In real usage, you'd run: action = policy.predict(obs)
        # For this example, use small random perturbations
        action = np.random.randn(43) * 0.1

        # 4. Scale action (policy outputs normalized actions)
        scaled_action = action * np.pi

        # 5. Apply action
        # apply_joint_actions(
        #     robot_prim_path="/G1",
        #     joint_positions=scaled_action.tolist()
        # )

        # 6. Step simulation
        # step_simulation(num_steps=1, render=True)

        # Progress update
        if step % 20 == 0:
            height = state["base_position"][2]
            forward_vel = base_lin_vel[0]
            print(f"  Step {step:3d}: height={height:.3f}m, vel={forward_vel:.2f}m/s")

        # Check termination
        if state["base_position"][2] < 0.3:
            print(f"\n  Robot fell at step {step}")
            break

    # stop_simulation()
    print("\n[Complete] Manual control loop finished")


def example_state_monitoring():
    """Example of monitoring robot state during execution."""

    print("\n" + "=" * 70)
    print("Robot State Monitoring Example")
    print("=" * 70)

    # Get current state
    # state_json = get_robot_state("/G1")
    # state = json.loads(state_json)

    # Simulated state
    state = {
        "joint_positions": [0.0, 0.2, -0.4, 0.8, -0.4, 0.0] + [0.0] * 37,
        "joint_velocities": [0.0] * 43,
        "base_position": [0.0, 0.0, 0.78],
        "base_orientation": [0.9998, 0.01, 0.005, 0.0],
        "base_linear_velocity": [0.45, 0.0, -0.02],
        "base_angular_velocity": [0.01, -0.02, 0.0],
        "num_joints": 43,
        "dof_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            # ... (simplified)
        ]
    }

    print("\n[Joint States]")
    print(f"  Number of joints: {state['num_joints']}")
    print(f"  Left leg joints (first 6):")
    for i, name in enumerate(state['dof_names'][:6]):
        pos = state['joint_positions'][i]
        vel = state['joint_velocities'][i]
        print(f"    {name:25s}: pos={pos:6.3f} rad, vel={vel:6.3f} rad/s")

    print("\n[Base State]")
    pos = state['base_position']
    print(f"  Position:         [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] m")

    quat = state['base_orientation']
    print(f"  Orientation:      [{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}] (w,x,y,z)")

    lin_vel = state['base_linear_velocity']
    print(f"  Linear velocity:  [{lin_vel[0]:6.3f}, {lin_vel[1]:6.3f}, {lin_vel[2]:6.3f}] m/s")

    ang_vel = state['base_angular_velocity']
    print(f"  Angular velocity: [{ang_vel[0]:6.3f}, {ang_vel[1]:6.3f}, {ang_vel[2]:6.3f}] rad/s")

    # Derived metrics
    import math
    forward_speed = lin_vel[0]
    height = pos[2]

    # Quaternion to roll/pitch (simplified)
    w, x, y, z = quat
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(2*(w*y - z*x))

    print("\n[Derived Metrics]")
    print(f"  Forward speed:    {forward_speed:.3f} m/s")
    print(f"  Height:           {height:.3f} m")
    print(f"  Roll:             {math.degrees(roll):6.2f} deg")
    print(f"  Pitch:            {math.degrees(pitch):6.2f} deg")

    # Safety checks
    print("\n[Safety Status]")
    if height < 0.3:
        print("  WARNING: Robot too low (fallen)")
    elif abs(roll) > 0.5 or abs(pitch) > 0.5:
        print("  WARNING: Excessive tilt detected")
    else:
        print("  OK: Robot stable and upright")


def main():
    """Run all examples."""

    print("\n" + "=" * 70)
    print("Isaac Sim MCP - Pre-trained Policy Examples")
    print("=" * 70)
    print("\nThis script demonstrates the new MCP tools for running")
    print("pre-trained RL policies on robots in Isaac Sim.")
    print("\nNOTE: This is a demonstration script. In actual usage,")
    print("uncomment the MCP tool calls and ensure the MCP server")
    print("and Isaac Sim extension are running.")

    # Run examples
    example_policy_execution()
    time.sleep(1)

    example_manual_control_loop()
    time.sleep(1)

    example_state_monitoring()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\n[Next Steps]")
    print("1. Ensure Isaac Sim is running with MCP extension")
    print("2. Start the MCP server: python isaac_mcp/server.py")
    print("3. Uncomment the tool calls in this script")
    print("4. Run: python examples/run_pretrained_policy.py")
    print("5. Watch your robot walk in Isaac Sim!")


if __name__ == "__main__":
    main()
