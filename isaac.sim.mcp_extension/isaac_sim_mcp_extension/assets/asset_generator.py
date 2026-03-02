"""3D generation service wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from isaac_sim_mcp_extension.core_state import ExtensionState
from isaac_sim_mcp_extension.gen3d import Beaver3d
from isaac_sim_mcp_extension.usd import USDLoader


class AssetGenerator:
    """Wraps Beaver3d generation and caches task ids."""

    def __init__(self, state: ExtensionState) -> None:
        self._state = state

    def generate_3d_from_text_or_image(
        self,
        text_prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        position: Tuple[float, float, float] = (0, 0, 50),
        scale: Tuple[float, float, float] = (10, 10, 10),
    ) -> Dict[str, Any]:
        try:
            beaver = Beaver3d()
            task_id = None
            if image_url and image_url in self._state.image_url_cache:
                task_id = self._state.image_url_cache[image_url]
            elif text_prompt and text_prompt in self._state.text_prompt_cache:
                task_id = self._state.text_prompt_cache[text_prompt]

            if task_id is None and image_url:
                task_id = beaver.generate_3d_from_image(image_url)
            elif task_id is None and text_prompt:
                task_id = beaver.generate_3d_from_text(text_prompt)
            elif task_id is None:
                return {"status": "error", "message": "Either text_prompt or image_url must be provided"}

            def load_model_into_scene(resolved_task_id: str, status: str, result_path: str) -> Dict[str, Any]:
                if image_url and image_url not in self._state.image_url_cache:
                    self._state.image_url_cache[image_url] = resolved_task_id
                elif text_prompt and text_prompt not in self._state.text_prompt_cache:
                    self._state.text_prompt_cache[text_prompt] = resolved_task_id

                loader = USDLoader()
                prim_path = loader.load_usd_model(task_id=resolved_task_id)
                try:
                    loader.load_texture_and_create_material(task_id=resolved_task_id)
                    loader.bind_texture_to_model()
                except Exception:
                    pass
                loader.transform(position=position, scale=scale)
                return {"status": "success", "task_id": resolved_task_id, "prim_path": prim_path}

            from omni.kit.async_engine import run_coroutine

            run_coroutine(beaver.monitor_task_status_async(task_id, on_complete_callback=load_model_into_scene))
            return {"status": "success", "task_id": task_id, "message": f"3D model generation started with task ID: {task_id}"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
