"""Policy loading service."""

from __future__ import annotations

import json
import tempfile
import traceback
import zipfile
from typing import Any, Dict, Optional

import carb
from isaac_sim_mcp_extension.core_state import ExtensionState

_LOG_PREFIX = "[POLICY_LOADER]"


class PolicyLoader:
    """Loads SB3 and PyTorch policies and prepares articulation/controller."""

    def __init__(self, state: ExtensionState) -> None:
        self._state = state

    def _ensure_robot_initialized(self, robot_prim_path: str) -> Optional[Dict[str, Any]]:
        """Initialize the robot articulation for policy execution.

        If the simulation is already playing (start_simulation was called), we just grab
        the existing physics_sim_view and initialize the Articulation with it.
        We do NOT create PhysicsContext or call set_simulation_dt — doing so invalidates
        the sim view ("Simulation view object is invalidated").
        Physics dt is set by scene_manager (e.g. 60 Hz for responsive UI).
        """
        from omni.isaac.core import SimulationContext
        from omni.isaac.core.articulations import Articulation

        carb.log_info(f"{_LOG_PREFIX} _ensure_robot_initialized start robot_prim_path={robot_prim_path!r}")

        if self._state.policy.robot_articulation is not None:
            carb.log_info(f"{_LOG_PREFIX} robot already initialized, skipping")
            return None

        sim_ctx = SimulationContext.instance()
        if sim_ctx is None:
            sim_ctx = SimulationContext(stage_units_in_meters=1.0)
            carb.log_info(f"{_LOG_PREFIX} created SimulationContext (instance was None)")
        if getattr(sim_ctx, "_physics_context", None) is None:
            from omni.isaac.core.physics_context import PhysicsContext
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            physics_prim = "/World/PhysicsScene" if (stage and stage.GetPrimAtPath("/World/PhysicsScene").IsValid()) else "/physicsScene"
            # 60 Hz for responsive UI; match scene_manager PHYSICS_HZ
            physics_dt = 1.0 / 60.0
            sim_ctx._physics_context = PhysicsContext(physics_dt=physics_dt, prim_path=physics_prim)
            carb.log_info(f"{_LOG_PREFIX} created PhysicsContext prim_path={physics_prim!r} dt={physics_dt:.4f} (60 Hz)")
        carb.log_info(f"{_LOG_PREFIX} SimulationContext ready, is_playing={sim_ctx.is_playing()}")

        if not sim_ctx.is_playing():
            sim_ctx.play()
            carb.log_info(f"{_LOG_PREFIX} play() done")

        # Step physics only (render=False). Calling render=True from the extension thread
        # while Isaac Sim's rendering loop is active causes recursive viewport updates.
        for _ in range(10):
            sim_ctx.step(render=False)
        carb.log_info(f"{_LOG_PREFIX} stepped 10x (render=False)")

        try:
            actual_dt = sim_ctx.get_physics_dt()
            carb.log_info(f"{_LOG_PREFIX} physics_dt = {actual_dt}")
        except Exception:
            pass

        sim_view = getattr(sim_ctx, "physics_sim_view", None)
        carb.log_info(f"{_LOG_PREFIX} physics_sim_view exists={sim_view is not None}")

        if sim_view is None:
            carb.log_error(f"{_LOG_PREFIX} physics_sim_view is None")
            return {"status": "error", "message": "Physics simulation view not available. Ensure physics scene exists and start_simulation was called."}

        self._state.policy.robot_articulation = Articulation(prim_path=robot_prim_path)
        carb.log_info(f"{_LOG_PREFIX} Articulation created, calling initialize...")
        try:
            self._state.policy.robot_articulation.initialize(sim_view)
            carb.log_info(f"{_LOG_PREFIX} articulation.initialize() done")
        except Exception as e:
            carb.log_error(f"{_LOG_PREFIX} articulation.initialize() failed: {type(e).__name__}: {e}")
            carb.log_error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

        try:
            self._state.policy.controller = self._state.policy.robot_articulation.get_articulation_controller()
            carb.log_info(f"{_LOG_PREFIX} controller obtained")
        except Exception as e:
            carb.log_error(f"{_LOG_PREFIX} get_articulation_controller() failed: {e}")
            carb.log_error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

        try:
            dof_names = self._state.policy.robot_articulation.dof_names
            carb.log_info(f"{_LOG_PREFIX} dof_names len={len(dof_names)} first={dof_names[:4]}")
        except Exception as e:
            carb.log_error(f"{_LOG_PREFIX} dof_names failed: {e}")
            carb.log_error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

        # Verify tensor data is accessible (not just USD metadata).
        # Never call step(render=True) here — this runs in the extension handler thread,
        # not inside Isaac Sim's render loop, so render=True would recurse into viewport code.
        jp = self._state.policy.robot_articulation.get_joint_positions()
        if jp is None or len(jp) == 0:
            carb.log_warn(f"{_LOG_PREFIX} get_joint_positions() returned None/empty after init — stepping more (render=False)")
            for _ in range(10):
                sim_ctx.step(render=False)
            jp = self._state.policy.robot_articulation.get_joint_positions()
        if jp is None or len(jp) == 0:
            carb.log_error(f"{_LOG_PREFIX} get_joint_positions() still empty after extra steps")
            return {"status": "error", "message": "Articulation tensor data unavailable. Physics view may be corrupted."}
        carb.log_info(f"{_LOG_PREFIX} get_joint_positions() verified: len={len(jp)}")

        self._setup_pd_gains()
        carb.log_info(f"{_LOG_PREFIX} _ensure_robot_initialized success")
        return None

    def _setup_pd_gains(self) -> None:
        dof_names = self._state.policy.robot_articulation.dof_names
        kps = []
        kds = []
        for name in dof_names:
            if "hip_yaw" in name or "hip_roll" in name:
                kps.append(150.0)
                kds.append(5.0)
            elif "hip_pitch" in name or "knee" in name or "torso" in name:
                kps.append(200.0)
                kds.append(5.0)
            elif "ankle" in name:
                kps.append(20.0)
                kds.append(2.0)
            else:
                kps.append(40.0)
                kds.append(10.0)
        self._state.policy.controller.set_gains(kps=kps, kds=kds)

    def load_policy(self, policy_path: str, robot_prim_path: str = "/G1") -> Dict[str, Any]:
        try:
            import torch
            init_error = self._ensure_robot_initialized(robot_prim_path)
            if init_error is not None:
                return init_error
            if policy_path.endswith(".zip"):
                with zipfile.ZipFile(policy_path, "r") as archive:
                    model_data = json.loads(archive.read("data").decode("utf-8"))
                    policy_data = archive.read("policy.pth")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
                        tmp.write(policy_data)
                        tmp_path = tmp.name
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    state_dict = torch.load(tmp_path, map_location=device, weights_only=False)
                    import torch.nn as nn

                    class SB3PolicyNetwork(nn.Module):
                        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
                            super().__init__()
                            self.policy_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
                            self.action_net = nn.Linear(hidden_dim, action_dim)

                        def forward(self, obs: Any) -> Any:
                            return self.action_net(self.policy_net(obs))

                    obs_dim = state_dict["mlp_extractor.policy_net.0.weight"].shape[1]
                    action_dim = state_dict["action_net.weight"].shape[0]
                    hidden_dim = state_dict["mlp_extractor.policy_net.0.weight"].shape[0]
                    network = SB3PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
                    network.policy_net[0].weight.data = state_dict["mlp_extractor.policy_net.0.weight"]
                    network.policy_net[0].bias.data = state_dict["mlp_extractor.policy_net.0.bias"]
                    network.policy_net[2].weight.data = state_dict["mlp_extractor.policy_net.2.weight"]
                    network.policy_net[2].bias.data = state_dict["mlp_extractor.policy_net.2.bias"]
                    network.action_net.weight.data = state_dict["action_net.weight"]
                    network.action_net.bias.data = state_dict["action_net.bias"]
                    network.eval()
                    self._state.policy.loaded_policy = {
                        "type": "sb3_ppo",
                        "state_dict": state_dict,
                        "network": network,
                        "metadata": model_data,
                        "device": device,
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "path": policy_path,
                    }
                return {"status": "success", "message": f"Loaded SB3 PPO policy from {policy_path}", "policy_type": "sb3_ppo", "obs_dim": obs_dim, "action_dim": action_dim, "metadata": model_data}
            if policy_path.endswith(".pt") or policy_path.endswith(".pth"):
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                try:
                    checkpoint = torch.jit.load(policy_path, map_location=device)
                except Exception:
                    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
                self._state.policy.loaded_policy = {"type": "pytorch", "checkpoint": checkpoint, "device": device, "path": policy_path}
                return {"status": "success", "message": f"Loaded PyTorch policy from {policy_path}", "policy_type": "pytorch"}
            return {"status": "error", "message": "Unsupported policy file format. Use .zip (SB3) or .pt/.pth (PyTorch)"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @staticmethod
    def detect_obs_dim(checkpoint: Any, policy_type: str) -> int:
        if checkpoint is None:
            return 123
        try:
            if policy_type == "pytorch" and callable(checkpoint) and not isinstance(checkpoint, dict):
                for _name, param in checkpoint.named_parameters():
                    if len(param.shape) == 2:
                        return int(param.shape[1])
        except Exception:
            pass
        return 123
