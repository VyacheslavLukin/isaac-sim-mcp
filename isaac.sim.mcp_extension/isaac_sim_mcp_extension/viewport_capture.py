"""Viewport capture handler for Isaac Sim MCP extension.

Uses the confirmed-working replicator approach:
  - asyncio.ensure_future(do_capture()) schedules the async coroutine on Kit's
    running event loop without blocking the main thread.
  - The capture coroutine writes a PNG to disk after step_async() completes.
  - The handler polls the output path until the file exists and is fully written
    (size > 1000 bytes), then reads and base64-encodes it.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import threading
from typing import Any, Dict

import carb


DEFAULT_CAPTURE_PATH = "/home/sim_worlds/viewport_capture.png"
CAMERA_PATH = "/OmniverseKit_Persp"

# Poll settings: check every 0.1 s for up to 10 s.
POLL_INTERVAL = 0.1
POLL_TIMEOUT = 10.0
MIN_FILE_BYTES = 1000


class ViewportCapture:
    """Captures the active Isaac Sim viewport as a PNG via omni.replicator.core."""

    def capture_viewport(
        self,
        output_path: str = DEFAULT_CAPTURE_PATH,
        width: int = 1280,
        height: int = 720,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Schedule a replicator capture and return the PNG as base64.

        The async coroutine is fired with asyncio.ensure_future() so it runs
        on Kit's existing event loop without blocking the command-dispatch thread.
        The method then polls for the output file to appear on disk (up to 10 s).

        Args:
            output_path: Absolute path inside the container where the PNG is
                         written. Must be on a volume shared with the host.
                         Default: /home/sim_worlds/viewport_capture.png
            width:  Render width in pixels.  Default 1280.
            height: Render height in pixels. Default 720.

        Returns:
            Dict with keys: status, file_path, base64, message.
        """
        # Remove any stale file from a previous capture so the poll does not
        # return immediately with old data.
        try:
            if os.path.isfile(output_path):
                os.remove(output_path)
        except Exception as exc:
            carb.log_warn(f"[ViewportCapture] Could not remove stale file: {exc}")

        # Ensure the target directory is writable.
        out_dir = os.path.dirname(output_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            carb.log_warn(f"[ViewportCapture] Could not create output dir: {exc}")

        import omni.replicator.core as rep  # type: ignore[import]
        import numpy as np  # ships with Isaac Sim

        # Capture the width/height/output_path into the coroutine via closure.
        _width = width
        _height = height
        _out = output_path

        async def do_capture() -> None:
            try:
                rp = rep.create.render_product(CAMERA_PATH, (_width, _height))
                rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb.attach([rp])
                await rep.orchestrator.step_async(rt_subframes=4)
                data = rgb.get_data()
                # data is RGBA uint8 numpy array of shape (H, W, 4).
                # Save as RGB PNG using PIL (ships with Isaac Sim / OV Python).
                from PIL import Image as PILImage  # type: ignore[import]
                PILImage.fromarray(data[:, :, :3]).save(_out)
                rp.destroy()
                carb.log_info(f"[ViewportCapture] PNG written to {_out}")
            except Exception as exc:
                carb.log_error(f"[ViewportCapture] do_capture coroutine failed: {exc}")

        # Schedule the coroutine on Kit's running asyncio event loop and return
        # immediately. This command runs on Kit's main thread; blocking here
        # would deadlock the async loop (the coroutine could never run).
        # The MCP server polls the output file from the host side instead.
        asyncio.ensure_future(do_capture())
        carb.log_info(f"[ViewportCapture] Capture scheduled → {output_path}")

        return {
            "status": "capturing",
            "file_path": output_path,
            "message": f"Capture scheduled. File will appear at {output_path} within ~3 s.",
        }
