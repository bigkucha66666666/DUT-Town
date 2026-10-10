"""智能体运行时：位置、路径、朝向、走路动画相位。

这里只有"身体"，没有"脑子"。决策交给 brain（当前是规则占位，
将来换成 LLM 认知层），引擎负责把决策变成路径再交给这里执行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

FACINGS = ("down", "up", "left", "right")
FACING_INDEX = {f: i for i, f in enumerate(FACINGS)}


@dataclass
class Agent:
    id: str
    name: str
    role: str
    persona: dict[str, Any]
    x: float
    y: float
    speed: float = 1.3
    path: list[tuple[float, float]] = field(default_factory=list)
    goal: dict[str, Any] | None = None
    state: str = "idle"            # idle | walking
    facing: str = "down"
    walk_phase: float = 0.0        # 走路动画相位，前端也用得到
    done_routine: set[int] = field(default_factory=set)
    arrivals: list[dict[str, Any]] = field(default_factory=list)
    walked: float = 0.0
    # 认知层：这两个就是"人格"与"记忆"的载体
    memory: Any = None
    personality: Any = None

    @property
    def look(self) -> dict[str, str]:
        return self.persona.get("look", {})

    @property
    def heading(self) -> str:
        return self.goal.get("why", "") if self.goal else ""

    def assign(self, path: list[tuple[float, float]], goal: dict[str, Any]) -> None:
        self.path = list(path)
        self.goal = goal
        self.state = "walking" if self.path else "idle"

    def _face(self, dx: float, dy: float) -> None:
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        if abs(dx) >= abs(dy):
            self.facing = "right" if dx > 0 else "left"
        else:
            self.facing = "down" if dy > 0 else "up"

    def step(self, dt: float) -> dict[str, Any] | None:
        """推进 dt 游戏秒。返回"到达目的地"事件，或 None。"""
        if not self.path:
            self.state = "idle"
            return None

        budget = self.speed * dt
        while budget > 1e-9 and self.path:
            tx, ty = self.path[0]
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)
            if d <= budget:
                self.walked += d
                self.x, self.y = tx, ty
                self.path.pop(0)
                budget -= d
            else:
                self._face(dx, dy)
                self.x += dx / d * budget
                self.y += dy / d * budget
                self.walked += budget
                self.walk_phase += dt * 2.2
                budget = 0.0

        if not self.path:
            self.state = "idle"
            self.walk_phase = 0.0
            ev = {
                "type": "arrive",
                "agent": self.id,
                "name": self.name,
                "room": (self.goal or {}).get("room"),
                "why": (self.goal or {}).get("why"),
            }
            self.arrivals.append(ev)
            return ev

        nxt = self.path[0]
        self._face(nxt[0] - self.x, nxt[1] - self.y)
        self.walk_phase += dt * 2.2
        return None

    def snapshot(self) -> list[float]:
        """给回放轨迹的紧凑表示：[x, y, 朝向索引, 是否在走]。"""
        return [
            round(self.x, 2),
            round(self.y, 2),
            FACING_INDEX[self.facing],
            1 if self.state == "walking" else 0,
        ]
