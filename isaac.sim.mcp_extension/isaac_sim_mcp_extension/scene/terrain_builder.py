"""Terrain building strategies."""

from __future__ import annotations

from typing import Optional, Protocol

import carb
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


class TerrainBuilder(Protocol):
    def build(self, stage: any, floor_path: str, **kwargs: any) -> bool:
        ...


class FlatTerrainBuilder:
    """Build a regular Isaac Sim ground plane."""

    def build(self, stage: any, floor_path: str, **kwargs: any) -> bool:
        from pxr import PhysicsSchemaTools

        existing = stage.GetPrimAtPath(floor_path)
        if existing.IsValid():
            stage.RemovePrim(Sdf.Path(floor_path))
        PhysicsSchemaTools.addGroundPlane(
            stage,
            floor_path,
            "Z",
            1000.0,
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.5, 0.5, 0.5),
        )
        ground_prim = stage.GetPrimAtPath(floor_path)
        if ground_prim.IsValid():
            if not ground_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(ground_prim)
            for child in ground_prim.GetChildren():
                if not child.IsValid():
                    continue
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(child)
                if child.IsA(UsdGeom.Mesh) and not child.HasAPI(UsdPhysics.MeshCollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(child)
        return True


class RoughTerrainBuilder:
    """Build random rough terrain mesh with collision."""

    def build(
        self,
        stage: any,
        floor_path: str,
        size_xy: float = 20.0,
        resolution: float = 0.25,
        height_scale: float = 0.03,
        seed: Optional[int] = 42,
        **kwargs: any,
    ) -> bool:
        try:
            max_cells = 40
            resolution = max(float(resolution), size_xy / max_cells)
            rng = np.random.default_rng(seed)
            half = size_xy * 0.5
            nx = min(int(size_xy / resolution), max_cells)
            ny = min(int(size_xy / resolution), max_cells)
            nvx = nx + 1
            nvy = ny + 1

            xs = np.linspace(-half, half, nvx)
            ys = np.linspace(-half, half, nvy)
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            zz = rng.uniform(-height_scale, height_scale, size=(nvx, nvy)).astype(np.float32)

            try:
                from scipy.ndimage import gaussian_filter

                zz = gaussian_filter(zz.astype(np.float64), sigma=2.0).astype(np.float32)
            except ImportError:
                zz_pad = np.pad(zz, 1, mode="edge")
                zz = (
                    zz_pad[:-2, :-2]
                    + zz_pad[1:-1, :-2]
                    + zz_pad[2:, :-2]
                    + zz_pad[:-2, 1:-1]
                    + zz_pad[1:-1, 1:-1]
                    + zz_pad[2:, 1:-1]
                    + zz_pad[:-2, 2:]
                    + zz_pad[1:-1, 2:]
                    + zz_pad[2:, 2:]
                ) / 9.0
                zz = zz.astype(np.float32)

            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

            faces = []
            for i in range(nx):
                for j in range(ny):
                    a = i * nvy + j
                    b = i * nvy + (j + 1)
                    c = (i + 1) * nvy + (j + 1)
                    d = (i + 1) * nvy + j
                    faces.extend([a, d, c, a, c, b])
            face_vertex_indices = np.array(faces, dtype=np.int32)
            face_vertex_counts = np.full(len(faces) // 3, 3, dtype=np.int32)

            existing = stage.GetPrimAtPath(floor_path)
            if existing.IsValid():
                stage.RemovePrim(Sdf.Path(floor_path))

            mesh_prim = UsdGeom.Mesh.Define(stage, Sdf.Path(floor_path))
            mesh_prim.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in points])
            mesh_prim.CreateFaceVertexIndicesAttr().Set(face_vertex_indices.tolist())
            mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts.tolist())
            mesh_prim.CreateDoubleSidedAttr(True)

            prim = mesh_prim.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.none)
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)

            color_attr = prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
            color_attr.Set([Gf.Vec3f(0.5, 0.5, 0.5)])
            mat_path = f"{floor_path}/roughTerrainMaterial"
            material = UsdShade.Material.Define(stage, Sdf.Path(mat_path))
            pbr = UsdShade.Shader.Define(stage, Sdf.Path(f"{mat_path}/Shader"))
            pbr.CreateIdAttr("UsdPreviewSurface")
            pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
            pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            return True
        except Exception as exc:
            carb.log_error(f"Failed to create rough terrain: {exc}")
            return False
