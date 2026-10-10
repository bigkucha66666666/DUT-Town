"""记忆流。

每条记忆 = 一段文字 + 创建时间 + 最近访问时间 + 重要度（1–10）+ 相关对象。

检索得分沿用 Stanford Generative Agents 的三因子加权：

    得分 = w_rel · 相关性 + w_rec · 时效性 + w_imp · 重要度

**相关性不用 embedding**：中文按字符二元组算 Dice 系数，够用、零依赖、可离线。
将来要换成向量检索时，只要替换 `_relevance()` 一个函数。

重要度优先问 LLM（它知道"被老师当众表扬"比"走过走廊"重要），
模型不可用时退回关键词启发式——**记忆不会因为模型挂了就停止积累**。
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# 时效衰减半衰期（游戏分钟）：4 游戏小时前的记忆权重减半
HALF_LIFE_MIN = 240.0

WEIGHTS = {"relevance": 0.45, "recency": 0.30, "importance": 0.25}

# 关键词启发式打分——LLM 不可用时的兜底
_KEYWORD_BONUS = [
    ("老师", 2), ("答疑", 2), ("争执", 3), ("道歉", 3), ("表扬", 3),
    ("帮忙", 2), ("第一次", 2), ("组会", 2), ("咖啡", 1), ("挂科", 4),
]

def _bigrams(text: str) -> set[str]:
    t = "".join(ch for ch in text if not ch.isspace())
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def relevance(query: str, text: str) -> float:
    """字符二元组 Dice 系数，落在 [0, 1]。"""
    a, b = _bigrams(query), _bigrams(text)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


@dataclass
class Memory:
    id: str
    agent_id: str
    kind: str                      # observation | dialogue | reflection | event
    text: str
    importance: int
    created: float                 # 游戏分钟（当日，如 07:50 → 470.0）
    last_access: float
    subject: str | None = None     # 相关对象：另一个人、房间……
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> tuple:
        import json
        return (self.id, self.agent_id, self.kind, self.text, self.importance,
                self.created, self.last_access, self.subject,
                json.dumps(self.meta, ensure_ascii=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "text": self.text,
            "importance": self.importance, "created": round(self.created, 1),
            "clock": fmt_clock(self.created), "subject": self.subject,
            "meta": self.meta,
        }


def fmt_clock(minutes: float) -> str:
    m = int(minutes) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


class MemoryStream:
    def __init__(self, agent_id: str, gateway: Any = None, store: Any = None) -> None:
        self.agent_id = agent_id
        self.gateway = gateway
        self.store = store
        self.items: list[Memory] = []
        self._unflushed: list[Memory] = []
        self.last_reflection_importance = 0.0   # 上次反思时的累计重要度
        self.total_importance = 0.0

    # ---------------------------------------------------------- 写入

    def importance_of(self, text: str) -> int:
        if self.gateway is not None:
            res = self.gateway.chat(
                [{"role": "user", "content": f"给这段校园记忆的意外程度/重要性打分（1-10）：{text}"}],
                schema={"type": "object", "properties": {"importance": {"type": "integer"}},
                        "required": ["importance"]},
                purpose="memory.importance",
                meta={"agent": self.agent_id, "text": text},
            )
            if res.ok and isinstance(res.get("importance"), int):
                return max(1, min(10, res.get("importance")))
        score = 3
        for kw, bonus in _KEYWORD_BONUS:
            if kw in text:
                score += bonus
        return max(1, min(10, score))

    def add(
        self,
        text: str,
        kind: str = "observation",
        importance: int | None = None,
        clock: float = 0.0,
        subject: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Memory:
        # 一次真实对话比"走过走廊"重要，但重要度打分的启发式看不出来，
        # 所以给对话类别一个下限——否则聊了半天也攒不出一次反思。
        base = 5 if kind == "dialogue" else 1
        imp = importance if importance is not None else max(base, self.importance_of(text))
        mem = Memory(
            # id 必须**全局唯一**。原来用进程内自增（m000001…），重启后又从 1 开始，
            # INSERT OR REPLACE 会把上一轮的记忆静默覆盖掉——看起来在累积，其实在覆写。
            id=f"m{uuid.uuid4().hex[:12]}",
            agent_id=self.agent_id, kind=kind, text=text,
            importance=imp, created=clock, last_access=clock,
            subject=subject, meta=meta or {},
        )
        self.items.append(mem)
        self._unflushed.append(mem)
        self.total_importance += imp
        return mem

    # ---------------------------------------------------------- 检索

    def retrieve(self, query: str = "", k: int = 6, now: float = 0.0) -> list[Memory]:
        if not self.items:
            return []
        scored = [(self._score(m, query, now), m) for m in self.items]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in scored[:k]]
        for m in top:
            m.last_access = now
        return top

    def _score(self, mem: Memory, query: str, now: float) -> float:
        rel = relevance(query, mem.text) if query else 0.0
        age = max(0.0, now - mem.created)
        rec = 0.5 ** (age / HALF_LIFE_MIN)
        imp = mem.importance / 10.0
        return (WEIGHTS["relevance"] * rel + WEIGHTS["recency"] * rec + WEIGHTS["importance"] * imp)

    # ---------------------------------------------------------- 反思触发

    def importance_since_reflection(self) -> float:
        return self.total_importance - self.last_reflection_importance

    def mark_reflected(self) -> None:
        self.last_reflection_importance = self.total_importance

    def recent(self, n: int = 8) -> list[Memory]:
        return self.items[-n:]

    def about(self, subject: str, k: int = 5) -> list[Memory]:
        hits = [m for m in self.items if m.subject == subject]
        return hits[-k:]

    # ---------------------------------------------------------- 落盘

    def flush(self, day: int = 0) -> int:
        if self.store is None or not self._unflushed:
            return 0
        n = self.store.save_memories(self._unflushed, day)
        self._unflushed.clear()
        return n

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for m in self.items:
            by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
        return {
            "agent_id": self.agent_id,
            "count": len(self.items),
            "by_kind": by_kind,
            "avg_importance": round(self.total_importance / len(self.items), 2) if self.items else 0,
            "since_reflection": round(self.importance_since_reflection(), 1),
        }
