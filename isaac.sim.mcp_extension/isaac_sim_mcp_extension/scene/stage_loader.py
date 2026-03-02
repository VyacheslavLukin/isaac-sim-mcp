"""Stage loading application service."""

from __future__ import annotations

import os
from typing import Any, Dict

import carb
import omni.usd
from omni.isaac.core.utils.stage import add_reference_to_stage


class StageLoader:
    """Open a USD file as the current stage or add a USD reference into the current stage.

    This service is the single point of responsibility for the *stage loading* bounded
    context.  It does not depend on SceneManager or AssetLoader and has no awareness of
    robots, physics, or assets beyond the USD file path it is asked to load.
    """

    def open_stage_from_path(self, usd_path: str) -> Dict[str, Any]:
        """Open a USD file as the current stage, replacing whatever is open now.

        The path is resolved inside the Isaac Sim process.  If Isaac Sim runs in Docker
        the path must be visible inside the container (e.g. a bind-mounted directory).
        Any relative references inside the USD file (e.g. ``@asset.usdz@``) are resolved
        relative to the directory of ``usd_path``; ensure referenced assets are
        co-located in the same directory.

        Args:
            usd_path: Absolute path to the USD/USDA/USDZ file visible to the Isaac Sim
                process.

        Returns:
            ``{"status": "success", "message": ...}`` on success or
            ``{"status": "error", "message": ...}`` on failure.
        """
        if not usd_path or not usd_path.strip():
            return {"status": "error", "message": "usd_path must not be empty"}

        usd_path = os.path.normpath(usd_path.strip())

        if not os.path.isfile(usd_path):
            return {
                "status": "error",
                "message": (
                    f"File not found: {usd_path}. If Isaac Sim runs in Docker, ensure "
                    "the path is visible inside the container (e.g. bind mount sim_worlds/)."
                ),
            }

        try:
            # open_stage() is synchronous and may block for large stages.
            # open_stage_async() exists as a fallback if timeouts occur in future.
            ctx = omni.usd.get_context()
            success = ctx.open_stage(usd_path)
            if not success:
                return {"status": "error", "message": f"open_stage returned False for {usd_path}"}
            return {"status": "success", "message": f"Opened stage from {usd_path}"}
        except Exception as exc:
            carb.log_error(f"[StageLoader] open_stage_from_path failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def load_reference_from_path(self, usd_path: str, prim_path: str) -> Dict[str, Any]:
        """Add a USD reference at ``prim_path`` in the current stage.

        The path is resolved inside the Isaac Sim process.  If Isaac Sim runs in Docker
        the path must be visible inside the container (e.g. a bind-mounted directory).

        Note: if the referenced USD uses a different upAxis (e.g. Y-up) than the current
        stage (e.g. Z-up), the referenced content may appear rotated.  When the asset
        carries its own physics scene, prefer opening it as a full stage with
        ``open_stage_from_path`` instead.

        Args:
            usd_path: Absolute path to the USD/USDA/USDZ file visible to the Isaac Sim
                process.
            prim_path: Prim path in the current stage where the reference will be
                anchored (must start with ``/``).

        Returns:
            ``{"status": "success", "message": ..., "prim_path": prim_path}`` on success
            or ``{"status": "error", "message": ...}`` on failure.
        """
        if not usd_path or not usd_path.strip():
            return {"status": "error", "message": "usd_path must not be empty"}
        if not prim_path or not prim_path.strip():
            return {"status": "error", "message": "prim_path must not be empty"}
        if not prim_path.startswith("/"):
            return {"status": "error", "message": f"prim_path must start with '/': {prim_path}"}

        usd_path = os.path.normpath(usd_path.strip())

        if not os.path.isfile(usd_path):
            return {
                "status": "error",
                "message": (
                    f"File not found: {usd_path}. If Isaac Sim runs in Docker, ensure "
                    "the path is visible inside the container (e.g. bind mount sim_worlds/)."
                ),
            }

        try:
            add_reference_to_stage(usd_path, prim_path)
            return {
                "status": "success",
                "message": f"Referenced USD at {usd_path} under prim {prim_path}",
                "prim_path": prim_path,
            }
        except Exception as exc:
            carb.log_error(f"[StageLoader] load_reference_from_path failed: {exc}")
            return {"status": "error", "message": str(exc)}
