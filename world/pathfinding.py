"""栅格 A* 寻路。

运行在 `map_loader.rasterize()` 产出的 Grid 上：8 方向、禁止穿角、
终点落在障碍里时自动吸附到最近的可行走格。

返回的路径以**米制坐标**表示（格子中心），可直接喂给智能体运行时。
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Iterable

from .map_loader import Grid

SQRT2 = math.sqrt(2)

# (dx, dy, cost) —— dy 为 +1 表示向南
_NEIGHBORS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
)


def to_tile(grid: Grid, x: float, y: float) -> tuple[int, int]:
    c = min(grid.cols - 1, max(0, int(x / grid.tile_size)))
    r = min(grid.rows - 1, max(0, int(y / grid.tile_size)))
    return r, c


def to_world(grid: Grid, r: int, c: int) -> tuple[float, float]:
    t = grid.tile_size
    return ((c + 0.5) * t, (r + 0.5) * t)


def nearest_walkable(grid: Grid, x: float, y: float, max_rings: int = 24) -> tuple[float, float] | None:
    """从 (x, y) 出发向外一圈圈找最近的可行走格，返回其中心坐标。

    房间里随机选点很容易落到桌椅或墙上，这个函数负责把目标"拔"出来。
    """
    walk = grid.walkable()
    r0, c0 = to_tile(grid, x, y)
    if walk[r0, c0]:
        return to_world(grid, r0, c0)

    dq: deque[tuple[int, int]] = deque([(r0, c0)])
    seen = {(r0, c0)}
    while dq:
        r, c = dq.popleft()
        if walk[r, c]:
            return to_world(grid, r, c)
        if abs(r - r0) > max_rings or abs(c - c0) > max_rings:
            continue
        for dr, dc, _ in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.rows and 0 <= nc < grid.cols and (nr, nc) not in seen:
                seen.add((nr, nc))
                dq.append((nr, nc))
    return None


def has_line_of_sight(grid: Grid, a: tuple[float, float], b: tuple[float, float], step: float = 0.2) -> bool:
    """沿线段采样，判断两点之间是否全程可行走（用于路径抽稀）。"""
    walk = grid.walkable()
    dx, dy = b[0] - a[0], b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return True
    n = max(2, int(dist / step) + 1)
    for i in range(n + 1):
        k = i / n
        r, c = to_tile(grid, a[0] + dx * k, a[1] + dy * k)
        if not walk[r, c]:
            return False
    return True


def find_path(
    grid: Grid,
    start: tuple[float, float],
    goal: tuple[float, float],
) -> list[tuple[float, float]] | None:
    """A* 寻路。返回途经点列表（不含起点，含终点）；无解返回 None。"""
    walk = grid.walkable()
    s = nearest_walkable(grid, *start)
    g = nearest_walkable(grid, *goal)
    if s is None or g is None:
        return None

    sr, sc = to_tile(grid, *s)
    gr, gc = to_tile(grid, *g)
    if (sr, sc) == (gr, gc):
        return [g]

    def h(r: int, c: int) -> float:
        dr, dc = abs(r - gr), abs(c - gc)
        # 八方向启发式（octile）
        return (dr + dc) + (SQRT2 - 2) * min(dr, dc)

    open_heap: list[tuple[float, int, int]] = [(h(sr, sc), sr, sc)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore: dict[tuple[int, int], float] = {(sr, sc): 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, r, c = heapq.heappop(open_heap)
        if (r, c) == (gr, gc):
            break
        if (r, c) in closed:
            continue
        closed.add((r, c))

        for dr, dc, cost in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.rows and 0 <= nc < grid.cols) or not walk[nr, nc]:
                continue
            # 禁止穿角：斜向移动要求两个正交邻格也可走
            if dr and dc and not (walk[r + dr, c] and walk[r, c + dc]):
                continue
            ng = gscore[(r, c)] + cost
            if ng < gscore.get((nr, nc), math.inf):
                gscore[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (ng + h(nr, nc), nr, nc))

    if (gr, gc) not in came and (gr, gc) != (sr, sc):
        return None

    cells = [(gr, gc)]
    while cells[-1] != (sr, sc):
        cells.append(came[cells[-1]])
    cells.reverse()

    pts = [to_world(grid, r, c) for r, c in cells]
    pts[-1] = g                      # 用吸附后的目标点收尾
    return _smooth(grid, s, pts)


def _smooth(grid: Grid, start: tuple[float, float], pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """路径抽稀：能直达就跳过中间的格子中心，减少锯齿。"""
    if len(pts) <= 2:
        return pts
    out: list[tuple[float, float]] = []
    anchor = start
    i = 0
    while i < len(pts):
        j = len(pts) - 1
        # 从最远处往回找第一个能直达的点
        while j > i and not has_line_of_sight(grid, anchor, pts[j]):
            j -= 1
        out.append(pts[j])
        anchor = pts[j]
        i = j + 1
    return out


def path_length(points: Iterable[tuple[float, float]], start: tuple[float, float] | None = None) -> float:
    pts = list(points)
    if start is not None:
        pts = [start] + pts
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
