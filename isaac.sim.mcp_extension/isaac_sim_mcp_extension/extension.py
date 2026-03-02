"""Extension module for Isaac Sim MCP."""

from __future__ import annotations

import gc
import sys as _sys
from typing import Any, Dict

import carb
import omni
import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

from isaac_sim_mcp_extension.assets.asset_generator import AssetGenerator
from isaac_sim_mcp_extension.assets.asset_loader import AssetLoader
from isaac_sim_mcp_extension.core_state import ExtensionState
from isaac_sim_mcp_extension.navigation.navigation_controller import NavigationController
from isaac_sim_mcp_extension.policy.policy_loader import PolicyLoader
from isaac_sim_mcp_extension.policy.policy_runner import PolicyRunner
from isaac_sim_mcp_extension.robots.robot_controller import RobotController
from isaac_sim_mcp_extension.robots.robot_factory import RobotFactory
from isaac_sim_mcp_extension.scene.scene_manager import SceneManager
from isaac_sim_mcp_extension.server.command_dispatcher import CommandDispatcher
from isaac_sim_mcp_extension.server.socket_server import SocketServer

if _sys.getrecursionlimit() < 10000:
    _sys.setrecursionlimit(10000)


class MCPExtension(omni.ext.IExt):
    """Thin orchestrator that wires services and command routing."""

    def __init__(self) -> None:
        super().__init__()
        self.ext_id = None
        self._state = ExtensionState(settings=carb.settings.get_settings())
        self._dispatcher = CommandDispatcher()
        self._socket_server: SocketServer | None = None

        self._scene_manager = SceneManager()
        self._robot_factory = RobotFactory(self._state, self._scene_manager)
        self._robot_controller = RobotController(self._state)
        self._policy_loader = PolicyLoader(self._state)
        self._policy_runner = PolicyRunner(self._state, self._policy_loader, self._robot_controller, self._scene_manager)
        self._navigation = NavigationController(self._state, self._policy_runner, self._robot_factory)
        self._asset_generator = AssetGenerator(self._state)
        self._asset_loader = AssetLoader()

    def on_startup(self, ext_id: str) -> None:
        self.ext_id = ext_id
        self._state.usd_context = omni.usd.get_context()
        self._state.port = self._state.settings.get("/exts/isaac.sim.mcp/server, port") or 8766
        self._state.host = self._state.settings.get("/exts/isaac.sim.mcp/server.host") or "localhost"
        self._register_handlers()
        self._socket_server = SocketServer(self._state.host, self._state.port, self.execute_command)
        self._socket_server.start()
        print(f"Isaac Sim MCP server started on {self._state.host}:{self._state.port}")

    def on_shutdown(self) -> None:
        gc.collect()
        if self._socket_server is not None:
            self._socket_server.stop()
            self._socket_server = None
        print("Isaac Sim MCP server stopped")

    def _register_handlers(self) -> None:
        # Core scene/commands
        self._dispatcher.register("execute_script", self.execute_script)
        self._dispatcher.register("get_scene_info", self._scene_manager.get_scene_info)
        self._dispatcher.register("omini_kit_command", self.omini_kit_command)
        self._dispatcher.register("create_physics_scene", self._scene_manager.create_physics_scene)

        # Robot APIs
        self._dispatcher.register("create_robot", self._robot_factory.create_robot)
        self._dispatcher.register("set_velocity_command", self._robot_factory.set_velocity_command)
        self._dispatcher.register("get_robot_pose", self._robot_factory.get_robot_pose)
        self._dispatcher.register("get_robot_state", self._robot_controller.get_robot_state)
        self._dispatcher.register("apply_joint_actions", self._robot_controller.apply_joint_actions)
        self._dispatcher.register("reset_robot_pose", self._robot_controller.reset_robot_pose)

        # Asset APIs
        self._dispatcher.register("generate_3d_from_text_or_image", self._asset_generator.generate_3d_from_text_or_image)
        self._dispatcher.register("search_3d_usd_by_text", self._asset_loader.search_3d_usd_by_text)
        self._dispatcher.register("transform", self._asset_loader.transform)

        # Policy & simulation APIs
        self._dispatcher.register("load_policy", self._policy_loader.load_policy)
        self._dispatcher.register("start_simulation", self._policy_runner.start_simulation)
        self._dispatcher.register("stop_simulation", self._policy_runner.stop_simulation)
        self._dispatcher.register("step_simulation", self._policy_runner.step_simulation)
        self._dispatcher.register("run_policy_loop", self._policy_runner.run_policy_loop)
        self._dispatcher.register("start_g1_policy_walk", self._policy_runner.start_g1_policy_walk)
        self._dispatcher.register("stop_g1_policy_walk", self._policy_runner.stop_g1_policy_walk)

        # Navigation APIs
        self._dispatcher.register("navigate_to", self._navigation.navigate_to)
        self._dispatcher.register("stop_navigation", self._navigation.stop_navigation)
        self._dispatcher.register("get_navigation_status", self._navigation.get_navigation_status)

    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch command through validated command dispatcher.

        This always returns a response dict and fixes the prior return-value bug.
        """
        cmd_type = command.get("type")
        params = command.get("params", {})
        self._state.usd_context = omni.usd.get_context()
        return self._dispatcher.dispatch(cmd_type, params)

    def execute_script(self, code: str) -> Dict[str, Any]:
        """Execute a Python script within the Isaac Sim context."""
        try:
            local_ns = {
                "omni": omni,
                "carb": carb,
                "Usd": Usd,
                "UsdGeom": UsdGeom,
                "Sdf": Sdf,
                "Gf": Gf,
            }
            exec(code, local_ns)
            return {"status": "success", "message": "Script executed successfully"}
        except Exception as exc:
            carb.log_error(f"Error executing script: {exc}")
            return {"status": "error", "message": str(exc)}

    def omini_kit_command(self, command: str, prim_type: str) -> Dict[str, Any]:
        omni.kit.commands.execute(command, prim_type=prim_type)
        return {"status": "success", "message": "command executed"}
"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Extension module for Isaac Sim MCP."""

# G1 robot's deep USD hierarchy causes recursion in Articulation.__init__
# (is_prim_non_root_articulation_link -> get_prim_at_path -> pxr import chain).
# Raise limit once at module load so all code has enough stack headroom.
import sys as _sys
if _sys.getrecursionlimit() < 10000:
    _sys.setrecursionlimit(10000)

import asyncio
import carb
import omni.usd
import threading
import time
import socket
import json
import traceback
import gc
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, PhysicsSchemaTools, UsdPhysics, PhysxSchema
import omni
import omni.kit.commands
import omni.physx as _physx
import omni.timeline
from typing import Dict, Any, List, Optional, Union
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
import numpy as np
from omni.isaac.core import World
from isaac_sim_mcp_extension.gen3d import Beaver3d
from isaac_sim_mcp_extension.usd import USDLoader
from isaac_sim_mcp_extension.usd import USDSearch3d
import requests
# import omni.ext
# import omni.ui as ui


# Import Beaver3d and USDLoader

# G1 minimal (37 DOF) joint names in the order the Isaac Lab policy expects.
# Isaac Lab training uses this order (left leg, right leg, torso, left arm, right arm, left hand, right hand).
# Isaac Sim's USD can expose a different DOF order (e.g. alternating left/right); we reorder obs/action by name.
G1_MINIMAL_POLICY_JOINT_ORDER = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    "left_five_joint", "left_three_joint", "left_six_joint", "left_four_joint",
    "left_zero_joint", "left_one_joint", "left_two_joint",
    "right_five_joint", "right_three_joint", "right_six_joint", "right_four_joint",
    "right_zero_joint", "right_one_joint", "right_two_joint",
]

# Extension Methods required by Omniverse Kit
# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.

