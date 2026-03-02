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

# isaac_sim_mcp_server.py
import time
import math
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
import threading
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse

try:
    from isaac_mcp.navigator import AStarPlanner, IsaacSimExecutor, OccupancyGrid, WaypointFollower
except ModuleNotFoundError:
    # Support running server.py directly as a script.
    from navigator import AStarPlanner, IsaacSimExecutor, OccupancyGrid, WaypointFollower

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsaacMCPServer")


@dataclass
class IsaacConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict

    def connect(self) -> bool:
        """Connect to the Isaac addon socket server"""
        if self.sock:
            return True

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Isaac at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Isaac: {str(e)}")
            self.sock = None
            return False

    def disconnect(self):
        """Disconnect from the Isaac addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Isaac: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=16384):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(300.0)  # Match the extension's timeout

        try:
            while True:
                try:
                    logger.info("Waiting for data from Isaac")
                    # time.sleep(0.5)
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception(
                                "Connection closed before receiving any data")
                        break

                    chunks.append(chunk)

                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(
                            f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(
                        f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise

        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(
                f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Isaac and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac")

        command = {
            "type": command_type,
            "params": params or {}
        }

        try:
            # Log the command being sent
            logger.info(
                f"Sending command: {command_type} with params: {params}")

            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")

            # Set a timeout for receiving - use the same timeout as in receive_full_response
            self.sock.settimeout(300.0)  # Match the extension's timeout

            # Receive the response using the improved receive_full_response method
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")

            response = json.loads(response_data.decode('utf-8'))
            logger.info(
                f"Response parsed, status: {response.get('status', 'unknown')}")

            if response.get("status") == "error":
                logger.error(f"Isaac error: {response.get('message')}")
                raise Exception(response.get(
                    "message", "Unknown error from Isaac"))

            # Old extension wrapped responses as {"status":"success","result":{...}}.
            # Refactored extension returns flat dicts {"status":"success","message":"..."}.
            # Support both: use "result" only when it is a dict; otherwise return the
            # full response so callers can always do result.get("status"/"message").
            inner = response.get("result")
            return inner if isinstance(inner, dict) else response
        except socket.timeout:
            logger.error(
                "Socket timeout while waiting for response from Isaac")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception(
                "Timeout waiting for Isaac response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Isaac lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Isaac: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(
                    f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Isaac: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Isaac: {str(e)}")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Isaac: {str(e)}")


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools

    try:
        # Just log that we're starting up
        logger.info("IsaacMCP server starting up")

        # Try to connect to Isaac on startup to verify it's available
        try:
            # This will initialize the global connection if needed
            isaac = get_isaac_connection()
            logger.info("Successfully connected to Isaac on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Isaac on startup: {str(e)}")
            logger.warning(
                "Make sure the Isaac addon is running before using Isaac resources or tools")

        # Return an empty context - we're using the global connection
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _isaac_connection
        if _isaac_connection:
            logger.info("Disconnecting from Isaac Sim on shutdown")
            _isaac_connection.disconnect()
            _isaac_connection = None
        logger.info("Isaac SimMCP server shut down")

# Create the MCP server with lifespan support
mcp = FastMCP(
    "IsaacSimMCP",
    instructions="Isaac Sim integration through the Model Context Protocol",
    lifespan=server_lifespan
)

# Resource endpoints

# Global connection for resources (since resources can't access context)
_isaac_connection = None
# _polyhaven_enabled = False  # Add this global variable
_nav_lock = threading.Lock()
_nav_state_lock = threading.Lock()
_nav_follower: WaypointFollower | None = None
_nav_target: list[float] = []
_nav_arrival_threshold: float = 0.5
_nav_robot_prim_path: str = "/G1"
_nav_status: str = "idle"
_nav_last_error: str | None = None

_DEFAULT_OBSTACLE_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (-2.5, -2.0, 1.5, 1.0),
    (1.8, -0.2, 1.2, 1.8),
    (0.0, 2.4, 2.0, 1.0),
    (3.2, 2.5, 1.0, 1.0),
)


def get_isaac_connection():
    """Get or create a persistent Isaac connection"""
    global _isaac_connection, _polyhaven_enabled  # Add _polyhaven_enabled to globals

    # If we have an existing connection, check if it's still valid
    if _isaac_connection is not None:
        try:

            return _isaac_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _isaac_connection.disconnect()
            except:
                pass
            _isaac_connection = None

    # Create a new connection if needed
    if _isaac_connection is None:
        _isaac_connection = IsaacConnection(host="localhost", port=8766)
        if not _isaac_connection.connect():
            logger.error("Failed to connect to Isaac")
            _isaac_connection = None
            raise Exception(
                "Could not connect to Isaac. Make sure the Isaac addon is running.")
        logger.info("Created new persistent connection to Isaac")

    return _isaac_connection


@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Ping status of Isaac Sim Extension Server"""
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_scene_info")
        print("result: ", result)

        # Just return the JSON representation of what Isaac sent us
        return json.dumps(result, indent=2)
        # return json.dumps(result)
        # return result
    except Exception as e:
        logger.error(f"Error getting scene info from Isaac: {str(e)}")
        # return f"Error getting scene info: {str(e)}"
        return {"status": "error", "error": str(e), "message": "Error getting scene info"}

# @mcp.tool()
# def get_object_info(ctx: Context, object_name: str) -> str:
#     """
#     Get detailed information about a specific object in the Isaac scene.

#     Parameters:
#     - object_name: The name of the object to get information about
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("get_object_info", {"name": object_name})

#         # Just return the JSON representation of what Isaac sent us
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error getting object info from Isaac: {str(e)}")
#         return f"Error getting object info: {str(e)}"


@mcp.tool("create_physics_scene")
def create_physics_scene(
    objects: List[Dict[str, Any]] = [],
    floor: bool = True,
    gravity: List[float] = [0,  -0.981, 0],
    scene_name: str = "physics_scene",
    floor_type: str = "flat",
    roughness: float = 0.03,
    terrain_resolution: float = 0.25,
    terrain_seed: int = 42,
) -> Dict[str, Any]:
    """Create a physics scene with multiple objects. Before create physics scene, you need to call get_scene_info() first to verify availability of connection.

    Args:
        objects: List of objects to create. Each object should have at least 'type' and 'position'. 
        floor: Whether to create a floor. default is True
        floor_type: "flat" = ground plane; "rough" = heightfield mesh with bumps. default is "flat"
        roughness: When floor_type="rough", height amplitude in meters (±roughness). Larger = rougher. default 0.03
        terrain_resolution: When floor_type="rough", grid cell size in meters. Smaller = finer bumps. default 0.25
        terrain_seed: When floor_type="rough", random seed for terrain pattern. default 42
        gravity: The gravity vector. Default is [0, 0, -981.0] (cm/s^2).
        scene_name: The name of the scene. default is "physics_scene"

    Returns:
        Dictionary with result information.
    """
    _ft = (floor_type or "flat").strip().lower()
    params = {
        "objects": objects,
        "floor": floor,
        "floor_type": _ft,
        "roughness": roughness,
        "terrain_resolution": terrain_resolution,
        "terrain_seed": terrain_seed,
        "rough_floor": _ft == "rough",  # fallback so extension gets rough even if floor_type is dropped
    }

    if gravity is not None:
        params["gravity"] = gravity
    if scene_name is not None:
        params["scene_name"] = scene_name
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        result = isaac.send_command("create_physics_scene", params)
        # MCP outputSchema expects result to be an object (dict), not a string
        msg = result.get(
            "message", "") or f"Created physics scene with {len(objects)} objects"
        return {"result": {"message": msg, "status": "success", "scene_name": scene_name}}
    except Exception as e:
        logger.error(f"Error create_physics_scene: {str(e)}")
        return {"result": {"message": f"Error create_physics_scene: {str(e)}", "status": "error"}}


@mcp.tool("create_rough_floor_scene")
def create_rough_floor_scene(
    objects: List[Dict[str, Any]] = [],
    roughness: float = 0.25,
    terrain_resolution: float = 1,
    terrain_seed: int = 42,
) -> Dict[str, Any]:
    """Create a physics scene with a ROUGH floor (heightfield with bumps/pits). Always uses rough terrain; no floor_type param.

    Use this when you want a rough floor. Optional: roughness (m), terrain_resolution (m), terrain_seed.
    """
    return create_physics_scene(
        objects=objects,
        floor=True,
        floor_type="rough",
        roughness=roughness,
        terrain_resolution=terrain_resolution,
        terrain_seed=terrain_seed,
        gravity=[0, 0, -981],
        scene_name="physics_scene",
    )


@mcp.tool("create_robot")
def create_robot(robot_type: str = "g1", position: List[float] = [0, 0, 0]) -> str:
    """Create a robot in the Isaac scene. Directly create robot prim in stage at the right position. For any creation of robot, you need to call create_physics_scene() first. call create_robot() as first attmpt beofre call execute_script().

    Args:
        robot_type: The type of robot to create. Available options:
            - "franka": Franka Emika Panda robot
            - "jetbot": NVIDIA JetBot robot
            - "carter": Carter delivery robot
            - "g1": Unitree G1 full USD (43 DOF)
            - "g1_minimal": Unitree G1 minimal USD (37 DOF; use for Isaac Lab-trained policy)
            - "go1": Unitree Go1 quadruped robot

    Returns:
        String with result information.
    """
    isaac = get_isaac_connection()
    result = isaac.send_command(
        "create_robot", {"robot_type": robot_type, "position": position})
    return f"create_robot successfully: {result.get('result', '')}, {result.get('message', '')}"


@mcp.tool("set_velocity_command")
def set_velocity_command(
    lin_vel_x: float = 0.5,
    lin_vel_y: float = 0.0,
    ang_vel_z: float = 0.0,
) -> str:
    """Set the velocity command used by the G1 policy callback (when running the policy script via execute_script).
    Units: m/s for linear, rad/s for yaw. Values are clamped to G1 flat ranges (lin_vel_x [0,1], lin_vel_y [-0.5,0.5], ang_vel_z [-1,1]).
    To stop the robot, use set_velocity_command(0, 0, 0)."""
    isaac = get_isaac_connection()
    result = isaac.send_command(
        "set_velocity_command",
        {"lin_vel_x": lin_vel_x, "lin_vel_y": lin_vel_y, "ang_vel_z": ang_vel_z},
    )
    payload = result.get("result", result)
    return f"Velocity command set: lin_vel_x={payload.get('lin_vel_x', lin_vel_x)}, lin_vel_y={payload.get('lin_vel_y', lin_vel_y)}, ang_vel_z={payload.get('ang_vel_z', ang_vel_z)}"


@mcp.tool("get_robot_pose")
def get_robot_pose(prim_path: str = "/G1") -> str:
    """Return world position and orientation (quaternion w,x,y,z) for the prim (e.g. G1 robot root).
    Used for navigation so the AI can compute the next velocity command. For 2D navigation, yaw can be derived from the quaternion."""
    isaac = get_isaac_connection()
    result = isaac.send_command("get_robot_pose", {"prim_path": prim_path})
    return json.dumps(result, indent=2)


@mcp.tool("omni_kit_command")
def omni_kit_command(command: str = "CreatePrim", prim_type: str = "Sphere") -> str:
    """Execute an Omni Kit command.

    Args:
        command: The Omni Kit command to execute.
        prim_type: The primitive type for the command.

    Returns:
        String with result information.
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()

        result = isaac.send_command("omini_kit_command", {
            "command": command,
            "prim_type": prim_type
        })
        return f"Omni Kit command executed successfully: {result.get('message', '')}"
    except Exception as e:
        logger.error(f"Error executing Omni Kit command: {str(e)}")
        # return f"Error executing Omni Kit command: {str(e)}"
        return {"status": "error", "error": str(e), "message": "Error executing Omni Kit command"}


@mcp.tool()
def execute_script(ctx: Context, code: str) -> str:
    """
    Before execute script pls check prompt from asset_creation_strategy() to ensure the scene is properly initialized.
    Execute arbitrary Python code in Isaac Sim. Before executing any code, first verify if get_scene_info() has been called to ensure the scene is properly initialized. Always print the formatted code into chat to confirm before execution to confirm its correctness. 
    Before execute script pls check if create_physics_scene() has been called to ensure the physics scene is properly initialized.
    When working with robots, always try using the create_robot() function first before resorting to execute_script(). The create_robot() function provides a simpler, more reliable way to add robots to your scene with proper initialization and positioning. Only use execute_script() for robot creation when you need custom configurations or behaviors not supported by create_robot().

    For physics simulation, avoid using simulation_context to run simulations in the main thread as this can cause blocking. Instead, use the World class with async methods for initializing physics and running simulations. For example, use my_world = World(physics_dt=1.0/60.0) and my_world.step_async() in a loop, which allows for better performance and responsiveness. If you need to wait for physics to stabilize, consider using my_world.play() followed by multiple step_async() calls.
    To create an simulation of Franka robot, the code should be like this:
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
simulation_context = SimulationContext()
add_reference_to_stage(asset_path, "/Franka")
#create_prim("/DistantLight", "DistantLight")




    To control the Franka robot, the code should be like this:

from omni.isaac.core import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path

my_world = World(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

simulation_context = SimulationContext()
add_reference_to_stage(asset_path, "/Franka")

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
art = Articulation("/Franka")
art.initialize(my_world.physics_sim_view)
dof_ptr = art.get_dof_index("panda_joint2")

simulation_context.play()
# NOTE: before interacting with dc directly you need to step physics for one step at least
# simulation_context.step(render=True) which happens inside .play()
for i in range(1000):
    art.set_joint_positions([-1.5], [dof_ptr])
    simulation_context.step(render=True)

simulation_context.stop()



    Parameters:
    - code: The Python code to execute, e.g. "omni.kit.commands.execute("CreatePrim", prim_type="Sphere")"
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        print("code: ", code)

        result = isaac.send_command("execute_script", {"code": code})
        print("result: ", result)
        # MCP schema expects result to be a string; extension returns a dict
        if isinstance(result, dict):
            if result.get("status") == "error":
                return f"Error executing code: {result.get('message', result.get('error', str(result)))}"
            return result.get("message", "Script executed successfully")
        return str(result) if result is not None else "Script executed successfully"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"


@mcp.prompt()
def asset_creation_strategy() -> str:
    """Defines the preferred strategy for creating assets in Isaac Sim"""
    return """
    0. Before anything, always check the scene from get_scene_info(), retrive rool path of assset through return value of assets_root_path.
    1. If the scene is empty, create a physics scene with create_physics_scene()
    2. if execute script due to communication error, then retry 3 times at most

    3. For Franka robot simulation, the code should be like this:
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
add_reference_to_stage(asset_path, "/Franka")
#create_prim("/DistantLight", "DistantLight")


# need to initialize physics getting any articulation..etc
simulation_context = SimulationContext()
simulation_context.initialize_physics()
simulation_context.play()

for i in range(1000):
    simulation_context.step(render=True)

simulation_context.stop()

    4. For Franka robot control, the code should be like this:
    
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path
from pxr import UsdPhysics

def create_physics_scene(stage, scene_path="/World/PhysicsScene"):
    if not stage.GetPrimAtPath(scene_path):
        UsdPhysics.Scene.Define(stage, scene_path)
    
    return stage.GetPrimAtPath(scene_path)

stage = omni.usd.get_context().get_stage()
physics_scene = create_physics_scene(stage)
if not physics_scene:
    raise RuntimeError("Failed to create or find physics scene")
import omni.physics.tensors as physx

def create_simulation_view(stage):
    sim_view = physx.create_simulation_view(stage)
    if not sim_view:
        carb.log_error("Failed to create simulation view")
        return None
    
    return sim_view

sim_view = create_simulation_view(stage)
if not sim_view:
    raise RuntimeError("Failed to create simulation view")

simulation_context = SimulationContext()
assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
add_reference_to_stage(asset_path, "/Franka")
#create_prim("/DistantLight", "DistantLight")

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
art = Articulation("/Franka")
art.initialize()
dof_ptr = art.get_dof_index("panda_joint2")

simulation_context.play()
# NOTE: before interacting with dc directly you need to step physics for one step at least
# simulation_context.step(render=True) which happens inside .play()
for i in range(1000):
    art.set_joint_positions([-1.5], [dof_ptr])
    simulation_context.step(render=True)

simulation_context.stop()

    5. For Jetbot simulation, the code should be like this:
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot

simulation_context = SimulationContext()
simulation_context.initialize_physics()

my_world = World(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0, 0.0, 2.0]),
    )
)


create_prim("/DistantLight", "DistantLight")
# need to initialize physics getting any articulation..etc


my_world.scene.add_default_ground_plane()
my_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)
my_world.reset()

simulation_context.play()
for i in range(10):
    simulation_context.step(render=True) 

i = 0
reset_needed = False
while i < 2000:
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        if i >= 0 and i < 1000:
            # forward
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
            print(my_jetbot.get_linear_velocity())
        elif i >= 1000 and i < 1300:
            # rotate
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, np.pi / 12]))
            print(my_jetbot.get_angular_velocity())
        elif i >= 1300 and i < 2000:
            # forward
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
        elif i == 2000:
            i = 0
        i += 1
simulation_context.stop()

6. For G1 simulation, the code should be like this see g1_ok.py


    """


def _process_bbox(original_bbox: list[float] | list[int] | None) -> list[int] | None:
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox
    if any(i <= 0 for i in original_bbox):
        raise ValueError(
            "Incorrect number range: bbox must be bigger than zero!")
    return [int(float(i) / max(original_bbox) * 100) for i in original_bbox] if original_bbox else None


# @mcp.tool()
def get_beaver3d_status(ctx: Context) -> str:
    """
    TODO: Get the status of Beaver3D.
    """
    return "Beaver3D service is Available"


@mcp.tool("generate_3d_from_text_or_image")
def generate_3d_from_text_or_image(
    ctx: Context,
    text_prompt: str = None,
    image_url: str = None,
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Generate a 3D model from text or image, load it into the scene and transform it.

    Args:
        text_prompt (str, optional): Text prompt for 3D generation
        image_url (str, optional): URL of image for 3D generation
        position (list, optional): Position to place the model [x, y, z]
        scale (list, optional): Scale of the model [x, y, z]

    Returns:
        String with the task_id and prim_path information
    """
    if not (text_prompt or image_url):
        return "Error: Either text_prompt or image_url must be provided"

    try:
        # Get the global connection
        isaac = get_isaac_connection()

        result = isaac.send_command("generate_3d_from_text_or_image", {
            "text_prompt": text_prompt,
            "image_url": image_url,
            "position": position,
            "scale": scale
        })

        if result.get("status") == "success":
            task_id = result.get("task_id")
            prim_path = result.get("prim_path")
            return f"Successfully generated 3D model with task ID: {task_id}, loaded at prim path: {prim_path}"
        else:
            return f"Error generating 3D model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        return f"Error generating 3D model: {str(e)}"


@mcp.tool("search_3d_usd_by_text")
def search_3d_usd_by_text(
    ctx: Context,
    text_prompt: str = None,
    target_path: str = "/World/my_usd",
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Search for a 3D model using text prompt in USD libraries, then load and position it in the scene.

    Args:
        text_prompt (str): Text description to search for matching 3D models
        target_path (str, optional): Path where the USD model will be placed in the scene
        position (list, optional): Position coordinates [x, y, z] for placing the model
        scale (list, optional): Scale factors [x, y, z] to resize the model

    Returns:
        String with search results including task_id and prim_path of the loaded model
    """
    if not text_prompt:
        return "Error: Either text_prompt or image_url must be provided"

    try:
        # Get the global connection
        isaac = get_isaac_connection()
        params = {"text_prompt": text_prompt,
                  "target_path": target_path}

        result = isaac.send_command("search_3d_usd_by_text", params)
        if result.get("status") == "success":
            task_id = result.get("task_id")
            prim_path = result.get("prim_path")
            return f"Successfully generated 3D model with task ID: {task_id}, loaded at prim path: {prim_path}"
        else:
            return f"Error generating 3D model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        return f"Error generating 3D model: {str(e)}"


@mcp.tool("transform")
def transform(
    ctx: Context,
    prim_path: str,
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Transform a USD model by applying position and scale.

    Args:
        prim_path (str): Path to the USD prim to transform
        position (list, optional): The position to set [x, y, z]
        scale (list, optional): The scale to set [x, y, z]

    Returns:
        String with transformation result
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()

        result = isaac.send_command("transform", {
            "prim_path": prim_path,
            "position": position,
            "scale": scale
        })

        if result.get("status") == "success":
            return f"Successfully transformed model at {prim_path} to position {position} and scale {scale}"
        else:
            return f"Error transforming model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error transforming model: {str(e)}")
        return f"Error transforming model: {str(e)}"


# ============================================================================
# Pre-trained Policy Execution Tools
# ============================================================================

@mcp.tool("load_policy")
def load_policy(
    policy_path: str,
    robot_prim_path: str = "/G1"
) -> str:
    """Load a pre-trained RL policy from .zip or .pt file for robot control.

    This tool loads trained policies from Stable Baselines3 (.zip) or PyTorch (.pt/.pth) formats.
    After loading, use run_policy_loop to execute the policy or use get_robot_state and
    apply_joint_actions for manual control loops.

    Args:
        policy_path: Absolute path to the trained policy file
                    - .zip for Stable Baselines3 (PPO, SAC, etc.)
                    - .pt or .pth for PyTorch/RSL-RL checkpoints
        robot_prim_path: USD path to the robot articulation (default: /G1)

    Returns:
        String with loading status and policy information

    Example:
        load_policy("/isaac-sim/g1_walking_trained.zip", "/G1")
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("load_policy", {
            "policy_path": policy_path,
            "robot_prim_path": robot_prim_path
        })

        if result.get("status") == "success":
            policy_type = result.get("policy_type", "unknown")
            return f"Successfully loaded {policy_type} policy from {policy_path}"
        else:
            return f"Error loading policy: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error loading policy: {str(e)}")
        return f"Error loading policy: {str(e)}"


@mcp.tool("get_robot_state")
def get_robot_state(robot_prim_path: str = "/G1") -> str:
    """Get comprehensive robot state for policy observations.

    Returns joint positions, velocities, base pose, base velocities, and other
    sensor-like data needed for policy inference. The state format matches typical
    RL observation spaces (joint states + base states + IMU-like data).

    Args:
        robot_prim_path: USD path to the robot articulation (default: /G1)

    Returns:
        JSON string with robot state including:
        - joint_positions: List of joint angles (radians)
        - joint_velocities: List of joint velocities (rad/s)
        - base_position: [x, y, z] position in world frame (meters)
        - base_orientation: [w, x, y, z] quaternion orientation
        - base_linear_velocity: [vx, vy, vz] linear velocity (m/s)
        - base_angular_velocity: [wx, wy, wz] angular velocity (rad/s)
        - num_joints: Number of controllable joints
        - dof_names: List of joint names

    Example:
        state = get_robot_state("/G1")
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_robot_state", {
            "robot_prim_path": robot_prim_path
        })

        if result.get("status") == "success":
            return json.dumps(result["state"], indent=2)
        else:
            return f"Error getting robot state: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error getting robot state: {str(e)}")
        return f"Error getting robot state: {str(e)}"


@mcp.tool("apply_joint_actions")
def apply_joint_actions(
    robot_prim_path: str = "/G1",
    joint_positions: List[float] = None,
    joint_velocities: List[float] = None,
    joint_efforts: List[float] = None
) -> str:
    """Apply joint control commands to the robot.

    Send position, velocity, or torque commands to robot joints. At least one control
    mode must be specified. For position control with PD gains, only joint_positions
    is needed (gains are set by load_policy or reset_robot_pose).

    Args:
        robot_prim_path: USD path to the robot articulation (default: /G1)
        joint_positions: Target joint positions in radians (optional)
        joint_velocities: Target joint velocities in rad/s (optional)
        joint_efforts: Target joint torques in N·m (optional)

    Returns:
        String with action application status

    Example:
        # Position control (most common for walking policies)
        apply_joint_actions("/G1", joint_positions=[0.0, 0.2, -0.4, ...])

        # Combined position + velocity
        apply_joint_actions("/G1", joint_positions=[...], joint_velocities=[...])
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("apply_joint_actions", {
            "robot_prim_path": robot_prim_path,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_efforts": joint_efforts
        })

        if result.get("status") == "success":
            return "Joint actions applied successfully"
        else:
            return f"Error applying joint actions: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error applying joint actions: {str(e)}")
        return f"Error applying joint actions: {str(e)}"


@mcp.tool("reset_robot_pose")
def reset_robot_pose(
    robot_prim_path: str = "/G1",
    base_position: List[float] = None,
    joint_positions: List[float] = None
) -> str:
    """Reset robot to a specific pose (typically standing configuration).

    Resets the robot's base position and joint configuration. If joint_positions is not
    provided, uses the default standing pose for the robot type. This is essential before
    running a policy to ensure the robot starts in a valid configuration.

    Args:
        robot_prim_path: USD path to the robot articulation (default: /G1)
        base_position: Base position [x, y, z] in meters (default: [0, 0, 0.8] for G1)
        joint_positions: Joint positions in radians (default: G1 standing pose)

    Returns:
        String with reset status

    Example:
        # Reset to default standing pose
        reset_robot_pose("/G1")

        # Reset to custom pose
        reset_robot_pose("/G1", base_position=[0, 0, 1.0], joint_positions=[0]*43)
    """
    try:
        isaac = get_isaac_connection()
        params = {"robot_prim_path": robot_prim_path}
        if base_position is not None:
            params["base_position"] = base_position
        if joint_positions is not None:
            params["joint_positions"] = joint_positions

        result = isaac.send_command("reset_robot_pose", params)

        if result.get("status") == "success":
            return result.get("message", "Robot pose reset successfully")
        else:
            return f"Error resetting robot pose: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error resetting robot pose: {str(e)}")
        return f"Error resetting robot pose: {str(e)}"


@mcp.tool("start_simulation")
def start_simulation() -> str:
    """Start the physics simulation (timeline play).

    Starts the Isaac Sim timeline, enabling physics simulation. Must be called before
    running policies or stepping the simulation. The simulation continues running until
    stop_simulation is called.

    Returns:
        String with simulation start status

    Example:
        start_simulation()
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("start_simulation", {})

        if result.get("status") == "success":
            return "Simulation started"
        else:
            return f"Error starting simulation: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        return f"Error starting simulation: {str(e)}"


@mcp.tool("stop_simulation")
def stop_simulation() -> str:
    """Stop the physics simulation (timeline stop).

    Stops the Isaac Sim timeline, pausing physics simulation. Also stops any running
    policy loops. Use this before modifying the scene or resetting robot poses.

    Returns:
        String with simulation stop status

    Example:
        stop_simulation()
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("stop_simulation", {})

        if result.get("status") == "success":
            return "Simulation stopped"
        else:
            return f"Error stopping simulation: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error stopping simulation: {str(e)}")
        return f"Error stopping simulation: {str(e)}"


@mcp.tool("step_simulation")
def step_simulation(num_steps: int = 1, render: bool = True) -> str:
    """Step the simulation forward by a specified number of steps.

    Advances the physics simulation by num_steps. Each step is typically 1/60 second
    (16.67ms) by default. Use this for fine-grained control or when implementing
    custom control loops.

    Args:
        num_steps: Number of simulation steps to execute (default: 1)
        render: Whether to render the viewport during stepping (default: True)

    Returns:
        String with step completion status

    Example:
        # Single step
        step_simulation(1)

        # Run 100 steps without rendering (faster)
        step_simulation(100, render=False)
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("step_simulation", {
            "num_steps": num_steps,
            "render": render
        })

        if result.get("status") == "success":
            return f"Stepped simulation {num_steps} times"
        else:
            return f"Error stepping simulation: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error stepping simulation: {str(e)}")
        return f"Error stepping simulation: {str(e)}"


@mcp.tool("run_policy_loop")
def run_policy_loop(
    robot_prim_path: str = "/G1",
    num_steps: int = 100,
    deterministic: bool = True
) -> str:
    """Run the loaded policy in an inference loop.

    Executes the pre-trained policy for num_steps, automatically handling:
    1. Robot pose reset to standing
    2. Observation collection (get_robot_state)
    3. Policy forward pass (action inference)
    4. Action application (apply_joint_actions)
    5. Simulation stepping
    6. Termination detection (falling)

    This is the main tool for executing trained walking policies without manual control.

    Args:
        robot_prim_path: USD path to the robot articulation (default: /G1)
        num_steps: Maximum number of steps to run (default: 100)
        deterministic: Use deterministic policy actions vs stochastic (default: True)

    Returns:
        JSON string with episode results including:
        - total_reward: Sum of rewards over the episode
        - mean_reward: Average reward per step
        - steps_completed: Number of steps before termination or completion
        - message: Status message

    Example:
        # Run policy for 500 steps
        result = run_policy_loop("/G1", num_steps=500, deterministic=True)

    Note:
        Requires a policy to be loaded first using load_policy().
        The simulation must be started (start_simulation) before calling this.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("run_policy_loop", {
            "robot_prim_path": robot_prim_path,
            "num_steps": num_steps,
            "deterministic": deterministic
        })

        if result.get("status") == "success":
            return json.dumps({
                "message": result.get("message"),
                "total_reward": result.get("total_reward", 0.0),
                "mean_reward": result.get("mean_reward", 0.0),
                "steps_completed": result.get("steps_completed", 0)
            }, indent=2)
        else:
            return f"Error running policy loop: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error running policy loop: {str(e)}")
        return f"Error running policy loop: {str(e)}"


@mcp.tool("start_g1_policy_walk")
def start_g1_policy_walk(
    policy_path: str,
    robot_prim_path: str = "/G1",
    target_velocity: float = 0.5,
    deterministic: bool = True
) -> str:
    """Start continuous policy-driven walking using a persistent callback.

    This tool loads a trained RL policy and registers a physics callback that runs
    inference on every simulation step. The robot will continuously walk based on
    the policy until stop_g1_policy_walk is called.

    Unlike run_policy_loop which runs for a fixed number of steps, this creates a
    persistent callback that keeps the robot walking indefinitely. The articulation
    and policy state are kept in the extension memory across frames.

    Args:
        policy_path: Absolute path to trained policy file (.zip for SB3, .pt/.pth for PyTorch)
        robot_prim_path: USD path to the robot articulation (default: /G1)
        target_velocity: Target forward velocity in m/s (default: 0.5)
        deterministic: Use deterministic policy actions vs stochastic (default: True)

    Returns:
        String with callback registration status

    Workflow:
        1. First call: Loads policy, registers callback that will:
           - On first frame: Reset robot to standing pose and start timeline
           - On subsequent frames: Run policy inference and apply joint commands
        2. The callback persists across frames until stop_g1_policy_walk is called
        3. Robot automatically resets if it falls (height < 0.3m)

    Example:
        # Start the robot walking
        start_g1_policy_walk(
            policy_path="/workspace/checkpoints_terminal/best_model.zip",
            robot_prim_path="/G1",
            target_velocity=0.5
        )

        # Robot walks continuously in Isaac Sim...

        # Stop when done
        stop_g1_policy_walk()

    Note:
        - Requires a physics scene and G1 robot to be created first
        - The policy and articulation remain loaded for efficiency
        - Use get_scene_info and create_physics_scene before calling this
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("start_g1_policy_walk", {
            "policy_path": policy_path,
            "robot_prim_path": robot_prim_path,
            "target_velocity": target_velocity,
            "deterministic": deterministic
        })

        if result.get("status") == "success":
            return f"Policy walk started: {result.get('message')}"
        else:
            return f"Error starting policy walk: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error starting policy walk: {str(e)}")
        return f"Error starting policy walk: {str(e)}"


@mcp.tool("stop_g1_policy_walk")
def stop_g1_policy_walk() -> str:
    """Stop the continuous policy walking callback.

    Unsubscribes the policy inference callback, stopping the robot's walking motion.
    The loaded policy and robot articulation remain in memory for potential reuse.

    Returns:
        String with stop status including total steps executed

    Example:
        # After starting policy walk
        stop_g1_policy_walk()
        # Returns: "Policy walk stopped after 1523 steps"
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("stop_g1_policy_walk", {})

        if result.get("status") == "success":
            return result.get("message", "Policy walk stopped")
        else:
            return f"Error stopping policy walk: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error stopping policy walk: {str(e)}")
        return f"Error stopping policy walk: {str(e)}"


# -------------------------------------------------------------------------
# Stage loading tools
# -------------------------------------------------------------------------


@mcp.tool("load_stage_from_path")
def load_stage_from_path(usd_path: str) -> Dict[str, Any]:
    """Open a USD file as the current Isaac Sim stage, replacing whatever is open now.

    The path is resolved inside the Isaac Sim process. If Isaac Sim runs in Docker,
    the path must be visible inside the container (e.g. bind mount sim_worlds/).
    Relative references inside the USD file (e.g. to a .usdz) are resolved relative
    to the directory of usd_path; keep referenced assets co-located in the same directory.

    Use this tool to open a full USD scene (with physics, materials, and objects) without
    using execute_script. Prefer this over load_usd_reference_from_path when the asset
    carries its own physics scene.

    Args:
        usd_path: Absolute path to the USD/USDA/USDZ file visible to the Isaac Sim process.

    Returns:
        Dictionary with status and message.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("load_stage_from_path", {"usd_path": usd_path})
        return {"status": result.get("status", "success"), "message": result.get("message", f"Opened stage from {usd_path}")}
    except Exception as e:
        logger.error(f"Error in load_stage_from_path: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool("load_usd_reference_from_path")
def load_usd_reference_from_path(usd_path: str, prim_path: str) -> Dict[str, Any]:
    """Add a USD reference at prim_path in the current Isaac Sim stage.

    The path is resolved inside the Isaac Sim process. If Isaac Sim runs in Docker,
    the path must be visible inside the container (e.g. bind mount sim_worlds/).

    Note: if the referenced USD uses a different upAxis (e.g. Y-up) than the current
    stage (e.g. Z-up), the referenced content may appear rotated. When the asset carries
    its own physics scene, prefer load_stage_from_path instead.

    Args:
        usd_path: Absolute path to the USD/USDA/USDZ file visible to the Isaac Sim process.
        prim_path: Prim path in the current stage where the reference will be anchored
                   (must start with '/'), e.g. '/World/MySplat'.

    Returns:
        Dictionary with status, message, and prim_path on success.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("load_usd_reference_from_path", {"usd_path": usd_path, "prim_path": prim_path})
        out: Dict[str, Any] = {"status": result.get("status", "success"), "message": result.get("message", f"Referenced {usd_path} at {prim_path}")}
        if result.get("prim_path"):
            out["prim_path"] = result["prim_path"]
        return out
    except Exception as e:
        logger.error(f"Error in load_usd_reference_from_path: {str(e)}")
        return {"status": "error", "message": str(e)}


# -------------------------------------------------------------------------
# Navigation tools
# -------------------------------------------------------------------------


@mcp.tool("navigate_to")
def navigate_to(
    target_position: List[float],
    robot_prim_path: str = "/G1",
    policy_path: str = "",
    arrival_threshold: float = 0.5,
    obstacle_boxes: List[List[float]] | None = None,
) -> str:
    """Start MCP-side A* navigation toward a target XY position.

    The planner runs in the MCP server and sends velocity commands through
    set_velocity_command in a background thread. Returns immediately (non-blocking).
    """
    global _nav_follower, _nav_target, _nav_arrival_threshold, _nav_robot_prim_path, _nav_status, _nav_last_error
    try:
        if len(target_position) < 2:
            return "Error starting navigation: target_position must contain at least [x, y]"

        target_x = float(target_position[0])
        target_y = float(target_position[1])

        if policy_path:
            # Start locomotion callback if the caller provided a policy.
            start_result = start_g1_policy_walk(policy_path=policy_path, robot_prim_path=robot_prim_path)
            if isinstance(start_result, str) and start_result.startswith("Error"):
                return f"Error starting navigation: {start_result}"

        isaac = get_isaac_connection()
        executor = IsaacSimExecutor(isaac, _nav_lock, robot_prim_path=robot_prim_path)
        current_xy, _ = executor.get_pose()

        boxes = obstacle_boxes if obstacle_boxes else [list(box) for box in _DEFAULT_OBSTACLE_BOXES]
        if any(len(box) < 4 for box in boxes):
            return "Error starting navigation: each obstacle box must be [cx, cy, sx, sy]"

        grid = OccupancyGrid.from_scene_boxes(boxes=boxes, map_size_m=20.0, resolution_m=0.1)
        grid.inflate(radius_m=0.5)
        planner = AStarPlanner(grid)
        waypoints = planner.plan((float(current_xy[0]), float(current_xy[1])), (target_x, target_y))
        if not waypoints:
            return "Error starting navigation: no path found to target"

        new_follower = WaypointFollower(executor=executor, arrival_dist_m=arrival_threshold)

        def _on_status_change(status: str) -> None:
            global _nav_follower, _nav_status, _nav_last_error
            with _nav_state_lock:
                _nav_status = status
                _nav_last_error = new_follower.last_error
                if status in ("arrived", "failed", "idle"):
                    _nav_follower = None

        with _nav_state_lock:
            if _nav_follower is not None:
                _nav_follower.stop()
            _nav_follower = new_follower
            _nav_target = [target_x, target_y]
            _nav_arrival_threshold = float(arrival_threshold)
            _nav_robot_prim_path = robot_prim_path
            _nav_status = "navigating"
            _nav_last_error = None

        new_follower.follow(waypoints, on_status_change=_on_status_change)
        return (
            f"Navigation started toward [{target_x:.2f}, {target_y:.2f}] with "
            f"{len(waypoints)} waypoints. Call get_navigation_status() to monitor."
        )
    except Exception as e:
        logger.error(f"Error in navigate_to: {str(e)}")
        return f"Error starting navigation: {str(e)}"


@mcp.tool("stop_navigation")
def stop_navigation() -> str:
    """Stop active MCP-side navigation and zero velocity commands."""
    global _nav_follower, _nav_status, _nav_last_error
    try:
        with _nav_state_lock:
            follower = _nav_follower
            prev_status = _nav_status
            _nav_follower = None
            _nav_status = "idle"
            _nav_last_error = None

        if follower is not None:
            follower.stop()
            return f"Navigation stopped (was: {prev_status}). Policy walk is still running."
        return "Navigation is not active."
    except Exception as e:
        logger.error(f"Error in stop_navigation: {str(e)}")
        return f"Error stopping navigation: {str(e)}"


@mcp.tool("get_navigation_status")
def get_navigation_status() -> str:
    """Return current MCP-side navigation state."""
    try:
        with _nav_state_lock:
            nav_status = _nav_status
            nav_target = list(_nav_target)
            nav_threshold = _nav_arrival_threshold
            robot_prim_path = _nav_robot_prim_path
            nav_error = _nav_last_error

        nav_info: Dict[str, Any] = {
            "nav_active": nav_status == "navigating",
            "nav_status": nav_status,
            "target_position": nav_target,
            "arrival_threshold": nav_threshold,
            "robot_prim_path": robot_prim_path,
        }
        if nav_error:
            nav_info["last_error"] = nav_error

        try:
            isaac = get_isaac_connection()
            executor = IsaacSimExecutor(isaac, _nav_lock, robot_prim_path=robot_prim_path)
            current_xy, current_yaw = executor.get_pose()
            nav_info["current_position"] = [float(current_xy[0]), float(current_xy[1])]
            nav_info["current_yaw"] = float(current_yaw)
            if len(nav_target) >= 2:
                nav_info["distance_to_target"] = math.hypot(nav_target[0] - current_xy[0], nav_target[1] - current_xy[1])
        except Exception as pose_err:
            nav_info["pose_error"] = str(pose_err)

        return json.dumps(nav_info, indent=2)
    except Exception as e:
        logger.error(f"Error in get_navigation_status: {str(e)}")
        return f"Error getting navigation status: {str(e)}"


# Main execution


def main():
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
