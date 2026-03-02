"""A* planning with path simplification for occupancy grids."""

from __future__ import annotations

import heapq
import math

from .occupancy_grid import OccupancyGrid

_SQRT2 = math.sqrt(2.0)

# 8-connected neighbour offsets and their movement costs
_NEIGHBORS_8 = [
    ((1, 0), 1.0),
    ((-1, 0), 1.0),
    ((0, 1), 1.0),
    ((0, -1), 1.0),
    ((1, 1), _SQRT2),
    ((1, -1), _SQRT2),
    ((-1, 1), _SQRT2),
    ((-1, -1), _SQRT2),
]


class AStarPlanner:
    """Grid-based A* planner with 8-connected movement and path simplification."""

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def plan(self, start_world: tuple[float, float], goal_world: tuple[float, float]) -> list[tuple[float, float]]:
        start = self.grid.world_to_grid(start_world[0], start_world[1])
        goal = self.grid.world_to_grid(goal_world[0], goal_world[1])
        raw_path = self._astar(start, goal)
        if not raw_path:
            return []
        simplified = self._simplify(raw_path)
        return [self.grid.grid_to_world(cell[0], cell[1]) for cell in simplified]

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Octile distance -- admissible and consistent for 8-connected grids."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)

    def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        if self.grid.is_occupied(start[0], start[1]) or self.grid.is_occupied(goal[0], goal[1]):
            return None
        if start == goal:
            return [start]

        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        closed_set: set[tuple[int, int]] = set()
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed_set.add(current)
            cx, cy = current
            for (dx, dy), step_cost in _NEIGHBORS_8:
                nx, ny = cx + dx, cy + dy
                if self.grid.is_occupied(nx, ny):
                    continue
                # For diagonal moves, also check both adjacent cardinal cells
                # to prevent corner-cutting through obstacles.
                if dx != 0 and dy != 0:
                    if self.grid.is_occupied(cx + dx, cy) or self.grid.is_occupied(cx, cy + dy):
                        continue
                neighbor = (nx, ny)
                tentative = current_g + step_cost
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, tentative, neighbor))
        return None

    def _simplify(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Remove waypoints that can be skipped via straight line-of-sight.

        Greedy algorithm: keep the first point, then find the farthest point
        still reachable via a clear line (Bresenham), advance, repeat.
        Always keeps start and goal.
        """
        if len(path) <= 2:
            return path

        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            farthest = i + 1
            for j in range(len(path) - 1, i, -1):
                if self._line_of_sight(path[i], path[j]):
                    farthest = j
                    break
            result.append(path[farthest])
            i = farthest
        return result

    def _line_of_sight(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Bresenham line check: True if all cells between a and b are free."""
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if self.grid.is_occupied(x0, y0):
                return False
            if x0 == x1 and y0 == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
