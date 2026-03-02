"""Occupancy grid utilities used by the MCP-side A* planner."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class OccupancyGrid:
    """Boolean occupancy grid with world/grid conversion helpers."""

    def __init__(self, origin_xy: tuple[float, float], resolution_m: float, size_cells: int):
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        self.resolution_m = float(resolution_m)
        self.size_cells = int(size_cells)
        self.grid = np.zeros((self.size_cells, self.size_cells), dtype=bool)

    @classmethod
    def from_scene_boxes(
        cls,
        boxes: Iterable[Iterable[float]],
        map_size_m: float = 20.0,
        resolution_m: float = 0.1,
    ) -> "OccupancyGrid":
        size_cells = max(1, int(round(map_size_m / resolution_m)))
        half = map_size_m * 0.5
        occupancy = cls(origin_xy=(-half, -half), resolution_m=resolution_m, size_cells=size_cells)
        for box in boxes:
            cx, cy, sx, sy = box
            occupancy.mark_box(float(cx), float(cy), float(sx), float(sy))
        return occupancy

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x - self.origin_xy[0]) / self.resolution_m)
        gy = int((y - self.origin_xy[1]) / self.resolution_m)
        gx = max(0, min(self.size_cells - 1, gx))
        gy = max(0, min(self.size_cells - 1, gy))
        return gx, gy

    def grid_to_world(self, i: int, j: int) -> tuple[float, float]:
        x = self.origin_xy[0] + (i + 0.5) * self.resolution_m
        y = self.origin_xy[1] + (j + 0.5) * self.resolution_m
        return x, y

    def mark_box(self, cx: float, cy: float, sx: float, sy: float) -> None:
        min_x = cx - sx * 0.5
        max_x = cx + sx * 0.5
        min_y = cy - sy * 0.5
        max_y = cy + sy * 0.5
        gx0, gy0 = self.world_to_grid(min_x, min_y)
        gx1, gy1 = self.world_to_grid(max_x, max_y)
        self.grid[min(gx0, gx1):max(gx0, gx1) + 1,
                  min(gy0, gy1):max(gy0, gy1) + 1] = True

    def inflate(self, radius_m: float) -> None:
        """Dilate obstacles by the robot radius using a circular structuring element.

        Uses numpy convolution for O(N^2) performance instead of per-cell Python loops.
        """
        radius_cells = int(math.ceil(max(0.0, radius_m) / self.resolution_m))
        if radius_cells <= 0:
            return

        d = 2 * radius_cells + 1
        yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        kernel = (xx * xx + yy * yy <= radius_cells * radius_cells).astype(np.uint8)

        try:
            from scipy.ndimage import binary_dilation
            self.grid = binary_dilation(self.grid, structure=kernel).astype(bool)
        except ImportError:
            # Fallback: manual convolution via numpy (still much faster than Python loops)
            padded = np.pad(self.grid.astype(np.uint8), radius_cells, mode="constant")
            inflated = np.zeros_like(self.grid, dtype=bool)
            for di in range(d):
                for dj in range(d):
                    if kernel[di, dj]:
                        inflated |= padded[di:di + self.size_cells, dj:dj + self.size_cells].astype(bool)
            self.grid = inflated

    def is_occupied(self, i: int, j: int) -> bool:
        if i < 0 or i >= self.size_cells or j < 0 or j >= self.size_cells:
            return True
        return bool(self.grid[i, j])
