"""USD search and transform wrappers."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import omni.usd

from isaac_sim_mcp_extension.usd import USDLoader, USDSearch3d


class AssetLoader:
    """Loads USD assets from search and applies transforms."""

    def search_3d_usd_by_text(
        self,
        text_prompt: str,
        target_path: str,
        position: Tuple[float, float, float] = (0, 0, 50),
        scale: Tuple[float, float, float] = (10, 10, 10),
    ) -> Dict[str, Any]:
        try:
            if not text_prompt:
                return {"status": "error", "message": "text_prompt must be provided"}
            searcher3d = USDSearch3d()
            url = searcher3d.search(text_prompt)
            loader = USDLoader()
            prim_path = loader.load_usd_from_url(url, target_path)
            return {
                "status": "success",
                "prim_path": prim_path,
                "message": f"3D model searching with prompt: {text_prompt}, return url: {url}, prim path in current scene: {prim_path}",
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def transform(self, prim_path: str, position: Tuple[float, float, float] = (0, 0, 50), scale: Tuple[float, float, float] = (10, 10, 10)) -> Dict[str, Any]:
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                return {"status": "error", "message": f"Prim not found at path: {prim_path}"}
            loader = USDLoader()
            loader.transform(prim=prim, position=position, scale=scale)
            return {
                "status": "success",
                "message": f"Model at {prim_path} transformed successfully",
                "position": position,
                "scale": scale,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
