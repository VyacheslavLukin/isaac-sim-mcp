"""Shared extension state for Isaac Sim MCP services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolicyState:
    """Mutable policy runtime state."""

    loaded_policy: Optional[Dict[str, Any]] = None
    robot_articulation: Any = None
    controller: Any = None
    running: bool = False
    walk_subscription: Any = None
    walk_step_count: int = 0
    walk_initialized: bool = False
    walk_robot_prim_path: Optional[str] = None
    walk_use_height_scan: bool = False
    walk_target_velocity: float = 0.5
    walk_deterministic: bool = True
    walk_last_action: Any = None
    walk_default_joint_pos: Any = None
    walk_policy_to_robot: Any = None
    walk_robot_to_policy: Any = None


@dataclass
class NavigationState:
    """Mutable navigation runtime state."""

    subscription: Any = None
    active: bool = False
    status: str = "idle"
    target: Optional[List[float]] = None
    threshold: Optional[float] = None
    robot_prim_path: Optional[str] = None


@dataclass
class ExtensionState:
    """Shared state object injected into domain services."""

    host: str = "localhost"
    port: int = 8766
    settings: Any = None
    usd_context: Any = None
    vel_cmd_x: float = 0.0
    vel_cmd_y: float = 0.0
    vel_cmd_yaw: float = 0.0
    image_url_cache: Dict[str, str] = field(default_factory=dict)
    text_prompt_cache: Dict[str, str] = field(default_factory=dict)
    policy: PolicyState = field(default_factory=PolicyState)
    navigation: NavigationState = field(default_factory=NavigationState)
