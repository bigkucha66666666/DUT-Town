"""地图加载、校验与栅格化。

地图数据是**纯语义**的（矩形 + 类别 + 语义字段），与渲染无关；
本模块负责把语义地图变成引擎需要的栅格代价图。

设计约定
--------
* 地图 JSON 是唯一数据源，本模块不改写它。
* 栅格化优先级（后者覆盖前者）：
      1. 全图填 BLOCKED
      2. ground  （道路/广场/绿地/停车场/步道/车棚）→ 可走
      3. buildings（建筑外轮廓）→ 建筑实体，不可走
      4. rooms   （房间/走廊）→ 室内可走（在建筑轮廓内"挖"出来）
  中间的空隙自然成为墙体，因此不需要手工绘制每一道墙。
* 走不通的地方 = 墙体 / 建筑实体 / 图外。

用法
----
    python -m world.map_loader --map world/maps/dut_sem.json --check
    python -m world.map_loader --map world/maps/dut_sem.json --ascii
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy 是声明依赖，缺失时给出明确提示
    np = None

# ---------------------------------------------------------------- 图层编码

BLOCKED = 0
WALK = 1
ROAD = 2
GREEN = 3
PLAZA = 4
PARKING = 5
PATH = 6
SHELTER = 7
ROOM = 8
CORRIDOR = 9
BUILDING = 10
FURNITURE = 11

LAYER_NAMES = {
    BLOCKED: "blocked",
    WALK: "walk",
    ROAD: "road",
    GREEN: "green",
    PLAZA: "plaza",
    PARKING: "parking",
    PATH: "path",
    SHELTER: "shelter",
    ROOM: "room",
    CORRIDOR: "corridor",
    BUILDING: "building",
    FURNITURE: "furniture",
}

WALKABLE_LAYERS = frozenset({WALK, ROAD, GREEN, PLAZA, PARKING, PATH, SHELTER, ROOM, CORRIDOR})

_GROUND_CATEGORY_TO_LAYER = {
    "road": ROAD,
    "green": GREEN,
    "plaza": PLAZA,
    "parking": PARKING,
    "path": PATH,
    "shelter": SHELTER,
}

_ROOM_CATEGORY_TO_LAYER = {
    "corridor": CORRIDOR,
}

# ---------------------------------------------------------------- 数据结构


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Rect":
        return cls(float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"]))

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def overlaps(self, other: "Rect") -> bool:
        return not (self.x2 <= other.x or other.x2 <= self.x or self.y2 <= other.y or other.y2 <= self.y)

    def contains(self, other: "Rect", tol: float = 1e-6) -> bool:
        return (
            other.x >= self.x - tol
            and other.y >= self.y - tol
            and other.x2 <= self.x2 + tol
            and other.y2 <= self.y2 + tol
        )

    def touches(self, other: "Rect", tol: float = 1e-6) -> bool:
        """两个矩形是否相接或相交（用于判断房间之间是否连通）。

        注意不能写成"某条边落进对方区间"——那会漏掉包含关系
        （例如走廊横跨整层、把房间完全包在 x 区间里）。这里用标准的
        区间相交判定：x 区间相交且 y 区间相交。
        """
        return (
            self.x - tol <= other.x2
            and other.x - tol <= self.x2
            and self.y - tol <= other.y2
            and other.y - tol <= self.y2
        )


@dataclass
class Feature:
    id: str
    name: str
    category: str
    rect: Rect
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


@dataclass
class MapData:
    meta: dict[str, Any]
    floors: list[dict[str, Any]]
    ground: list[Feature]
    buildings: list[Feature]
    rooms: list[Feature]
    furniture: list[Feature]
    spawns: list[dict[str, Any]]
    path: Path | None = None

    @property
    def world(self) -> dict[str, float]:
        return self.meta["world"]

    @property
    def tile_size(self) -> float:
        return float(self.meta.get("tile_size", 1))

    def building(self, bid: str) -> Feature | None:
        return next((b for b in self.buildings if b.id == bid), None)

    def rooms_on(self, floor: str) -> list[Feature]:
        return [r for r in self.rooms if r.get("floor") == floor]


@dataclass
class Grid:
    """栅格化结果。grid[y][x] 为图层编码，索引与地图米制坐标一致（1 格 = tile_size 米）。"""

    layer: Any  # numpy.ndarray
    tile_size: float
    cols: int
    rows: int
    _walk: Any = field(default=None, repr=False, compare=False)

    def walkable(self):
        """可行走掩码。缓存起来——A* 里每次邻居检查都会用到。"""
        if self._walk is None:
            self._walk = np.isin(self.layer, list(WALKABLE_LAYERS))
        return self._walk

    def at(self, x: float, y: float) -> int:
        c = min(self.cols - 1, max(0, int(x / self.tile_size)))
        r = min(self.rows - 1, max(0, int(y / self.tile_size)))
        return int(self.layer[r, c])

    def is_walkable(self, x: float, y: float) -> bool:
        return self.at(x, y) in WALKABLE_LAYERS

    def stats(self) -> dict[str, Any]:
        total = self.cols * self.rows
        counts: dict[str, int] = {}
        for code, name in LAYER_NAMES.items():
            n = int((self.layer == code).sum())
            if n:
                counts[name] = n
        walk = int(self.walkable().sum())
        return {
            "tiles_total": total,
            "tiles_walkable": walk,
            "walkable_ratio": round(walk / total, 4),
            "by_layer": counts,
        }


# ---------------------------------------------------------------- 载入与校验

REQUIRED_META = ("id", "name", "world", "tile_size")


def expand_furniture(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把家具的 grid 重复声明展开成一个个具体家具。"""
    out: list[dict[str, Any]] = []
    for item in items:
        grid = item.get("grid")
        if not grid:
            out.append(item)
            continue
        for r in range(int(grid["rows"])):
            for c in range(int(grid["cols"])):
                clone = {k: v for k, v in item.items() if k != "grid"}
                clone["id"] = f'{item["id"]}_r{r + 1}c{c + 1}'
                clone["rect"] = {
                    "x": grid["x"] + c * grid["dx"],
                    "y": grid["y"] + r * grid["dy"],
                    "w": grid["w"],
                    "h": grid["h"],
                }
                out.append(clone)
    return out


def load_map(path: str | Path) -> MapData:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    meta = raw.get("meta", {})
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"meta 缺少字段: {', '.join(missing)}")

    def feats(key: str, expand: bool = False) -> list[Feature]:
        items = raw.get(key, [])
        if expand:
            items = expand_furniture(items)
        out = []
        for item in items:
            out.append(
                Feature(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    category=item.get("category", item.get("type", "unknown")),
                    rect=Rect.from_dict(item["rect"]),
                    raw=item,
                )
            )
        return out

    return MapData(
        meta=meta,
        floors=raw.get("floors", []),
        ground=feats("ground"),
        buildings=feats("buildings"),
        rooms=feats("rooms"),
        furniture=feats("furniture", expand=True),
        spawns=raw.get("spawns", []),
        path=path,
    )


def validate(m: MapData) -> list[str]:
    """返回问题列表，空列表表示通过。每条以 'ERROR' 或 'WARN' 开头。"""
    issues: list[str] = []
    W = float(m.world["width"])
    H = float(m.world["height"])

    # --- id 唯一性
    seen: dict[str, str] = {}
    for group, items in (("ground", m.ground), ("buildings", m.buildings), ("rooms", m.rooms)):
        for f in items:
            if f.id in seen:
                issues.append(f"ERROR 重复 id: {f.id}（{seen[f.id]} 与 {group}）")
            seen[f.id] = group

    # --- 边界
    for group, items in (("ground", m.ground), ("buildings", m.buildings),
                         ("rooms", m.rooms), ("furniture", m.furniture)):
        for f in items:
            r = f.rect
            if r.w <= 0 or r.h <= 0:
                issues.append(f"ERROR {group}/{f.id} 尺寸非正: {r.w}x{r.h}")
            if r.x < 0 or r.y < 0 or r.x2 > W or r.y2 > H:
                issues.append(
                    f"ERROR {group}/{f.id}「{f.name}」超出地图边界 "
                    f"({r.x},{r.y})-({r.x2},{r.y2})，地图为 {W}x{H}"
                )

    # --- 建筑之间不应重叠
    for i, a in enumerate(m.buildings):
        for b in m.buildings[i + 1 :]:
            if a.rect.overlaps(b.rect):
                issues.append(f"ERROR 建筑重叠: {a.name} 与 {b.name}")

    # --- 房间：同一楼层不重叠、必须落在母建筑内
    by_floor: dict[tuple[str, str], list[Feature]] = {}
    for r in m.rooms:
        by_floor.setdefault((r.get("parent", ""), r.get("floor", "")), []).append(r)

    for (parent, floor), items in by_floor.items():
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if a.rect.overlaps(b.rect):
                    issues.append(f"ERROR 房间重叠（{floor}）: {a.name} 与 {b.name}")

    for r in m.rooms:
        parent = m.building(r.get("parent", ""))
        if parent is None:
            issues.append(f"ERROR 房间 {r.name} 的 parent 建筑不存在: {r.get('parent')}")
            continue
        if not parent.rect.contains(r.rect):
            issues.append(f"ERROR 房间「{r.name}」超出建筑「{parent.name}」轮廓")

    # --- 房间应能连到走廊（或经其它房间间接连通）
    # 并查集：矩形相接即视为连通，每个连通块里至少要有一条走廊。
    # 这只是快速预警；权威判定是后面基于栅格的"从入口出发是否可达"。
    for (parent, floor), items in by_floor.items():
        corridors = [r for r in items if r.category == "corridor"]
        if not corridors:
            issues.append(f"WARN {parent} {floor} 没有走廊，室内连通性未验证")
            continue

        idx = {r.id: i for i, r in enumerate(items)}
        par = list(range(len(items)))

        def find(a: int) -> int:
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a

        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if a.rect.touches(b.rect):
                    ra, rb = find(idx[a.id]), find(idx[b.id])
                    if ra != rb:
                        par[ra] = rb

        with_corridor = {find(idx[c.id]) for c in corridors}
        for r in items:
            if find(idx[r.id]) not in with_corridor:
                issues.append(f"WARN 房间「{r.name}」({floor}) 与走廊不连通")

    # --- 家具：必须落在自己声明的房间里
    room_by_id = {r.id: r for r in m.rooms}
    for f in m.furniture:
        rid = f.get("room")
        if not rid:
            issues.append(f"WARN 家具「{f.name}」({f.id}) 未声明 room")
            continue
        room = room_by_id.get(rid)
        if room is None:
            issues.append(f"ERROR 家具「{f.name}」({f.id}) 的 room 不存在: {rid}")
        elif not room.rect.contains(f.rect):
            issues.append(f"ERROR 家具「{f.name}」({f.id}) 超出房间「{room.name}」范围")

    solids = [f for f in m.furniture if f.get("solid")]
    for i, a in enumerate(solids):
        for b in solids[i + 1 :]:
            if a.rect.overlaps(b.rect):
                issues.append(f"WARN 实体家具重叠: {a.name}({a.id}) 与 {b.name}({b.id})")

    # --- 楼层登记一致性
    declared = {f["id"] for f in m.floors}
    used = {r.get("floor") for r in m.rooms}
    for fl in used - declared:
        issues.append(f"WARN 房间使用了未在 floors 中登记的楼层: {fl}")

    # --- 出生点必须可走
    grid = rasterize(m)
    for sp in m.spawns:
        x, y = sp["pos"]
        if not grid.is_walkable(x, y):
            issues.append(
                f"ERROR 出生点「{sp.get('name', sp['id'])}」({x},{y}) 落在不可通行格上"
                f"（图层 {LAYER_NAMES[grid.at(x, y)]}）"
            )

    # --- 连通性：从南门出发能到达哪些房间
    entrance = next((s for s in m.spawns if s.get("kind") == "entrance"), None)
    if entrance is None:
        issues.append("WARN 没有 kind=entrance 的出生点，跳过连通性检查")
    else:
        reach = _reachable_from(grid, *entrance["pos"])
        for r in m.rooms:
            cx, cy = r.rect.center()
            if not _reach_contains(reach, cx, cy, grid):
                issues.append(f"WARN 房间「{r.name}」从入口出发不可达")

    return issues


def _reachable_from(grid: Grid, x: float, y: float):
    """BFS 洪水填充，返回可达布尔矩阵。"""
    walk = grid.walkable()
    rows, cols = walk.shape
    start = (min(rows - 1, max(0, int(y / grid.tile_size))), min(cols - 1, max(0, int(x / grid.tile_size))))
    reach = np.zeros_like(walk)
    if not walk[start]:
        return reach
    dq = deque([start])
    reach[start] = True
    while dq:
        r, c = dq.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and walk[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                dq.append((nr, nc))
    return reach


def _reach_contains(reach, x: float, y: float, grid: Grid) -> bool:
    rows, cols = reach.shape
    r = min(rows - 1, max(0, int(y / grid.tile_size)))
    c = min(cols - 1, max(0, int(x / grid.tile_size)))
    return bool(reach[r, c])


# ---------------------------------------------------------------- 栅格化


def rasterize(m: MapData, tile_size: float | None = None) -> Grid:
    if np is None:
        raise RuntimeError("栅格化需要 numpy，请先安装：pip install numpy")

    t = float(tile_size or m.tile_size)
    cols = int(math.ceil(float(m.world["width"]) / t))
    rows = int(math.ceil(float(m.world["height"]) / t))
    layer = np.full((rows, cols), BLOCKED, dtype=np.int8)

    def paint(rect: Rect, code: int) -> None:
        c0 = max(0, int(math.floor(rect.x / t)))
        r0 = max(0, int(math.floor(rect.y / t)))
        c1 = min(cols, int(math.ceil(rect.x2 / t)))
        r1 = min(rows, int(math.ceil(rect.y2 / t)))
        if c1 > c0 and r1 > r0:
            layer[r0:r1, c0:c1] = code

    for f in m.ground:
        paint(f.rect, _GROUND_CATEGORY_TO_LAYER.get(f.category, WALK))
    for b in m.buildings:
        paint(b.rect, BUILDING)
    for r in m.rooms:
        paint(r.rect, _ROOM_CATEGORY_TO_LAYER.get(r.category, ROOM))
    # 实体家具最后覆盖：贴着墙放的书架/吧台/前台等参与碰撞
    for f in m.furniture:
        if f.get("solid"):
            paint(f.rect, FURNITURE)

    return Grid(layer=layer, tile_size=t, cols=cols, rows=rows)


# ---------------------------------------------------------------- CLI


def _ascii(grid: Grid, step: int = 4, max_cols: int = 150) -> str:
    """把栅格画成终端字符图，便于快速人眼校验。"""
    glyph = {
        BLOCKED: " ",
        WALK: ".",
        ROAD: "=",
        GREEN: ",",
        PLAZA: "+",
        PARKING: "P",
        PATH: "-",
        SHELTER: "b",
        ROOM: "#",
        CORRIDOR: ":",
        BUILDING: "X",
        FURNITURE: "o",
    }
    out = []
    for r in range(0, grid.rows, step):
        line = []
        for c in range(0, min(grid.cols, max_cols * step), step):
            line.append(glyph.get(int(grid.layer[r, c]), "?"))
        out.append("".join(line))
    return "\n".join(out)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="地图校验与栅格化")
    ap.add_argument("--map", default="world/maps/dut_sem.json", help="地图 JSON 路径")
    ap.add_argument("--check", action="store_true", help="只做校验并打印问题")
    ap.add_argument("--ascii", action="store_true", help="打印终端字符版地图")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出统计信息")
    args = ap.parse_args(list(argv) if argv is not None else None)

    m = load_map(args.map)
    issues = validate(m)
    errors = [i for i in issues if i.startswith("ERROR")]
    warns = [i for i in issues if i.startswith("WARN")]

    if args.json:
        print(json.dumps({"map": m.meta["id"], "stats": rasterize(m).stats()}, ensure_ascii=False, indent=2))
        return 1 if errors else 0

    print(f"地图：{m.meta['name']}  ({m.meta['id']})")
    print(f"尺寸：{m.world['width']}m x {m.world['height']}m   格子：{m.tile_size}m")
    print(
        f"要素：建筑 {len(m.buildings)} / 房间 {len(m.rooms)} / 家具 {len(m.furniture)} / 地面 {len(m.ground)} / "
        f"楼层 {len(m.floors)} / 出生点 {len(m.spawns)}"
    )

    grid = rasterize(m)
    st = grid.stats()
    print(
        f"栅格：{grid.cols} x {grid.rows} = {st['tiles_total']} 格，"
        f"可通行 {st['tiles_walkable']} 格（{st['walkable_ratio'] * 100:.1f}%）"
    )
    print("图层分布：" + "，".join(f"{k} {v}" for k, v in sorted(st["by_layer"].items())))

    if args.ascii:
        print()
        print(_ascii(grid))

    if issues:
        print()
        for line in errors + warns:
            print("  " + line)
    else:
        print("\n校验通过：未发现问题。")

    print(f"\n结论：{len(errors)} 个错误，{len(warns)} 个警告")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
