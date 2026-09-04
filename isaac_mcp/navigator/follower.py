"""Waypoint following loop for command-based locomotion policies."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

from .executor import RobotExecutor


class WaypointFollower:
    """Background waypoint follower that emits velocity commands."""

    def __init__(
        self,
        executor: RobotExecutor,
        *,
        arrival_dist_m: float = 0.4,
        waypoint_threshold_m: float = 0.5,
        k_lin: float = 1.2,
        k_lat: float = 1.0,
        k_ang: float = 1.5,
        max_vx: float = 1.0,
        min_vx: float = 0.0,
        max_vy: float = 0.35,
        max_yaw: float = 0.8,
        control_period_s: float = 0.1,
        heading_gate_deg: float = 55.0,
    ):
        self._executor = executor
        self._arrival_dist_m = arrival_dist_m
        self._waypoint_threshold_m = waypoint_threshold_m
        self._k_lin = k_lin
        self._k_lat = k_lat
        self._k_ang = k_ang
        self._max_vx = max_vx
        self._min_vx = min_vx
        self._max_vy = max_vy
        self._max_yaw = max_yaw
        # When far from goal, command at least this forward velocity so the robot walks while turning
        # instead of only turning in place (which can look like "not moving").
        self._min_vx_when_far = 0.35
        self._far_threshold_m = 1.0
        self._control_period_s = control_period_s
        # H.2: Heading gate. Translation (vx, vy) is ramped down as the heading error grows and
        # fully suppressed beyond this angle, so the robot turns to face the target BEFORE walking
        # instead of strafing sideways while yawing at full rate — the combination that tips the
        # flat command-tracking policy at tight detours.
        self._heading_gate_rad = math.radians(heading_gate_deg)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = "idle"
        self._status_lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def status(self) -> str:
        with self._status_lock:
            return self._status

    @property
    def last_error(self) -> str | None:
        with self._status_lock:
            return self._last_error

    def follow(
        self,
        waypoints: list[tuple[float, float]],
        *,
        on_arrive: Callable[[], None] | None = None,
        on_status_change: Callable[[str], None] | None = None,
    ) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("WaypointFollower is already running")
        if not waypoints:
            raise ValueError("waypoints must be non-empty")

        self._stop_event.clear()
        self._set_status("navigating", on_status_change)
        self._thread = threading.Thread(
            target=self._run,
            args=(waypoints, on_arrive, on_status_change),
            daemon=True,
            name="mcp-waypoint-follower",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._executor.stop()
        self._set_status("idle", None)

    def _run(
        self,
        waypoints: list[tuple[float, float]],
        on_arrive: Callable[[], None] | None,
        on_status_change: Callable[[str], None] | None,
    ) -> None:
        try:
            waypoint_idx = 0
            while not self._stop_event.is_set() and waypoint_idx < len(waypoints):
                pos_xy, yaw = self._executor.get_pose()
                target_xy = waypoints[waypoint_idx]

                dx = target_xy[0] - pos_xy[0]
                dy = target_xy[1] - pos_xy[1]
                dist = math.hypot(dx, dy)
                if dist <= self._waypoint_threshold_m and waypoint_idx < len(waypoints) - 1:
                    waypoint_idx += 1
                    continue

                goal_xy = waypoints[-1]
                goal_dist = math.hypot(goal_xy[0] - pos_xy[0], goal_xy[1] - pos_xy[1])
                if goal_dist <= self._arrival_dist_m:
                    self._executor.stop()
                    self._set_status("arrived", on_status_change)
                    if on_arrive:
                        on_arrive()
                    return

                desired_yaw = math.atan2(dy, dx)
                heading_error = self._wrap_to_pi(desired_yaw - yaw)

                cy = math.cos(yaw)
                sy = math.sin(yaw)
                fwd = cy * dx + sy * dy
                lat = -sy * dx + cy * dy

                # Yaw always turns toward the target waypoint.
                wz = self._clamp(self._k_ang * heading_error, -self._max_yaw, self._max_yaw)

                # Heading gate: 1.0 when facing the target, ramping linearly to 0.0 at the gate
                # angle (turn in place beyond it). Suppresses simultaneous strafe + yaw.
                gate = self._clamp(1.0 - abs(heading_error) / self._heading_gate_rad, 0.0, 1.0)

                vx = self._clamp(self._k_lin * fwd, self._min_vx, self._max_vx) * gate
                # When far from goal AND roughly aligned, enforce a minimum forward speed so the
                # robot walks while making small heading corrections (not just turning in place).
                if goal_dist > self._far_threshold_m and gate > 0.35 and vx < self._min_vx_when_far:
                    vx = self._min_vx_when_far
                vy = self._clamp(self._k_lat * lat, -self._max_vy, self._max_vy) * gate
                self._executor.set_velocity_command(vx, vy, wz)
                time.sleep(self._control_period_s)

            if self._stop_event.is_set():
                return
            self._executor.stop()
            self._set_status("arrived", on_status_change)
        except Exception as exc:
            self._executor.stop()
            self._set_status("failed", on_status_change, error=str(exc))

    def _set_status(
        self,
        status: str,
        on_status_change: Callable[[str], None] | None,
        *,
        error: str | None = None,
    ) -> None:
        with self._status_lock:
            self._status = status
            self._last_error = error
        if on_status_change:
            on_status_change(status)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
