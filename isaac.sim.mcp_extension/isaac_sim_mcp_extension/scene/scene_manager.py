"""Scene and physics orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import carb
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from isaac_sim_mcp_extension.scene.terrain_builder import FlatTerrainBuilder, RoughTerrainBuilder


class SceneManager:
    """Handles scene creation, floor generation, and primitive object setup."""

    def __init__(self) -> None:
        self._flat = FlatTerrainBuilder()
        self._rough = RoughTerrainBuilder()

    def get_scene_info(self) -> Dict[str, Any]:
        from omni.isaac.nucleus import get_assets_root_path

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return {"status": "error", "message": "No USD stage"}
        return {"status": "success", "message": "pong", "assets_root_path": get_assets_root_path()}

    def create_physics_scene(
        self,
        objects: List[Dict[str, Any]] = [],
        floor: bool = True,
        gravity: List[float] = (0.0, -9.81, 0.0),
        scene_name: str = "None",
        floor_type: str = "flat",
        roughness: float = 0.03,
        terrain_resolution: float = 0.25,
        terrain_seed: Optional[int] = 42,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return {"status": "error", "message": "USD stage is not available"}

            scene_path = "/World/PhysicsScene"
            if not stage.GetPrimAtPath(scene_path).IsValid():
                scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path(scene_path))
            else:
                scene_prim = UsdPhysics.Scene.Get(stage, Sdf.Path(scene_path))
            scene_prim.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene_prim.CreateGravityMagnitudeAttr().Set(9.81)

            # Set 60 Hz physics for responsive UI (was 200 Hz; user reported robot moving too slowly).
            # Policy control = physics_steps_per_sec / decimation (e.g. 60/4 = 15 Hz).
            PHYSICS_HZ = 60
            try:
                from pxr import PhysxSchema
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim.GetPrim())
                physx_scene.CreateTimeStepsPerSecondAttr().Set(PHYSICS_HZ)
                carb.log_info(f"[SCENE] PhysxScene timeStepsPerSecond set to {PHYSICS_HZ} (dt={1.0/PHYSICS_HZ:.3f} s)")
            except Exception as dt_exc:
                carb.log_warn(f"[SCENE] Could not set PhysxScene timeStepsPerSecond: {dt_exc}")
                try:
                    stage.SetMetadataByDictKey("customLayerData", "physicsSettings:timeStepsPerSecond", PHYSICS_HZ)
                    carb.log_info(f"[SCENE] Fallback: set stage physicsSettings:timeStepsPerSecond={PHYSICS_HZ}")
                except Exception:
                    pass

            omni.kit.commands.execute("CreatePrim", prim_path="/World", prim_type="Xform")

            rough_flag = kwargs.get("rough_floor") or kwargs.get("rough_terrain")
            normalized_floor_type = (floor_type or ("rough" if rough_flag else "flat")).lower().strip()
            if floor:
                floor_path = "/World/groundPlane"
                if normalized_floor_type == "rough":
                    self._rough.build(
                        stage,
                        floor_path=floor_path,
                        size_xy=20.0,
                        resolution=float(terrain_resolution) if terrain_resolution is not None else 0.25,
                        height_scale=float(roughness) if roughness is not None else 0.03,
                        seed=int(terrain_seed) if terrain_seed is not None else 42,
                    )
                else:
                    self._flat.build(stage, floor_path=floor_path)

            objects_created = 0
            for i, obj in enumerate(objects):
                obj_name = obj.get("name", f"object_{i}")
                obj_type = obj.get("type", "Cube")
                obj_position = obj.get("position", [0, 0, 0])
                obj_scale = obj.get("scale", [1, 1, 1])
                obj_color = obj.get("color", [0.5, 0.5, 0.5, 1.0])
                obj_physics = obj.get("physics_enabled", True)
                obj_mass = obj.get("mass", 1.0)
                obj_kinematic = obj.get("is_kinematic", False)
                obj_path = obj.get("path", f"/World/{obj_name}")

                if not stage.GetPrimAtPath(obj_path):
                    if obj_type not in ["Cube", "Sphere", "Cylinder", "Cone", "Plane"]:
                        return {"status": "error", "message": f"Invalid object type: {obj_type}"}
                    omni.kit.commands.execute(
                        "CreatePrim",
                        prim_path=obj_path,
                        prim_type=obj_type,
                        attributes={
                            "size": obj.get("size", 100.0),
                            "position": obj_position,
                            "scale": obj_scale,
                            "color": obj_color,
                            "physics_enabled": obj_physics,
                            "mass": obj_mass,
                            "is_kinematic": obj_kinematic,
                        }
                        if obj_type in ["Sphere", "Plane"]
                        else {},
                    )
                    omni.kit.commands.execute(
                        "TransformPrimSRT",
                        path=obj_path,
                        new_translation=obj_position,
                        new_rotation_euler=[0, 0, 0],
                        new_scale=obj_scale,
                    )

                    obj_prim = stage.GetPrimAtPath(obj_path)
                    if obj_physics and obj_prim.IsValid():
                        UsdPhysics.CollisionAPI.Apply(obj_prim)
                        rbody_api = UsdPhysics.RigidBodyAPI.Apply(obj_prim)
                        rbody_api.CreateKinematicEnabledAttr(obj_kinematic)
                        if not obj_kinematic:
                            # Mass is on MassAPI, not RigidBodyAPI (USD / Isaac Sim schema).
                            mass_api = UsdPhysics.MassAPI.Apply(obj_prim)
                            mass_api.CreateMassAttr().Set(obj_mass)

                    rgb = obj_color[:3] if obj_color and len(obj_color) >= 3 else (0.5, 0.5, 0.5)
                    opacity = obj_color[3] if obj_color and len(obj_color) > 3 else 1.0
                    material_path = f"{obj_path}/Looks/Material"
                    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
                    pbr = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/Shader"))
                    pbr.CreateIdAttr("UsdPreviewSurface")
                    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(rgb[0], rgb[1], rgb[2]))
                    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                    pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
                    material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
                    if obj_prim.IsValid():
                        UsdShade.MaterialBindingAPI.Apply(obj_prim).Bind(material)
                    objects_created += 1

            timeline.stop()
            return {
                "status": "success",
                "message": f"Created physics scene with {objects_created} objects",
                "scene_name": scene_name or "physics_scene",
            }
        except Exception as exc:
            carb.log_error(f"create_physics_scene failed: {exc}")
            return {"status": "error", "message": str(exc)}

    def get_terrain_height_at(self, x: float, y: float, robot_prim_prefix: str = "/G1") -> float:
        """Return terrain height for terrain-aware robot spawning/resets."""
        try:
            from omni.physx import get_physx_scene_query_interface

            physx_query = get_physx_scene_query_interface()
            candidates = [(x, y), (x + 0.3, y), (x - 0.3, y), (x, y + 0.3), (x, y - 0.3)]
            for cx, cy in candidates:
                hit = physx_query.raycast_closest(carb.Float3(float(cx), float(cy), 10.0), carb.Float3(0.0, 0.0, -1.0), 20.0)
                if not (hit and hit.get("hit")):
                    continue
                hit_body = hit.get("rigidBody", "")
                hit_z = float(hit["position"][2])
                if hit_body.startswith(robot_prim_prefix):
                    continue
                if hit_body and "/World/groundPlane" not in hit_body and "/World/Ground" not in hit_body:
                    continue
                if -5.0 < hit_z < 5.0:
                    return hit_z
        except Exception:
            pass
        return 0.0
