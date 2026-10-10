"""人格特质与关系状态：会漂移的状态，而且每次漂移都必须有出处。

为什么单独一个模块
------------------
人设 JSON 里的 `traits: ["内向","爱读书"]` 是**静态标签**——它只能"扮演"人格，
不可能"发生"人格变化。要看见变化，人格必须变成带时间序列的连续状态。

一条铁律
--------
**人格只能通过经历改变。** 每次漂移都必须带 `reason`（为什么变）和
`source`（哪次经历：dialogue / reflection），否则直接拒绝。这不是洁癖：
没有出处的漂移会累积成噪声，最后谁也说不清她为什么变成这样。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 四个维度都是 0–1 的连续值。挑这四个是因为它们**能从行为里看出来**，
# 而不是抽象的心理量表名词。
TRAITS: dict[str, tuple[str, str]] = {
    "social":     ("社交活跃", "愿不愿意主动开口、往人堆里凑"),
    "diligence":  ("学业投入", "自习、答疑、按时交作业的倾向"),
    "initiative": ("主动性",   "先迈出一步，还是等着别人来"),
    "stability":  ("情绪稳定", "遇事慌不慌、情绪起落大不大"),
}

# 一次经历最多能改变多少——防止"聊一次天就换了个人"
MAX_STEP = 0.06
# 硬下限/上限：性格不能被推到极端，那就不再是这个人了
FLOOR, CEIL = 0.05, 0.95

VALID_SOURCES = {"dialogue", "reflection"}


@dataclass
class TraitChange:
    clock: float
    trait: str
    old: float
    new: float
    reason: str
    source: str
    counterpart: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            # 注意别叫 clock：事件流里的 clock 是"07:42"这种显示串，
            # 被这个浮点分钟数覆盖了，前端就会显示成 462.1
            "clock_minutes": round(self.clock, 1), "at": _fmt(self.clock), "trait": self.trait,
            "label": TRAITS.get(self.trait, (self.trait, ""))[0],
            "old": round(self.old, 3), "new": round(self.new, 3),
            "delta": round(self.new - self.old, 3),
            "reason": self.reason, "source": self.source, "counterpart": self.counterpart,
        }


def _fmt(minutes: float) -> str:
    m = int(minutes) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass
class Personality:
    """一个人的四个特质 + 全部变更历史。"""

    agent_id: str
    values: dict[str, float]
    history: list[TraitChange] = field(default_factory=list)

    @classmethod
    def from_persona(cls, persona: dict[str, Any]) -> "Personality":
        base = persona.get("traits_base") or {}
        out: dict[str, float] = {}
        for key in TRAITS:
            v = base.get(key)
            if v is None:
                v = _infer_from_labels(persona.get("traits", []), key)
            out[key] = max(FLOOR, min(CEIL, float(v)))
        return cls(agent_id=persona["id"], values=out)

    def nudge(
        self,
        trait: str,
        delta: float,
        reason: str,
        source: str,
        clock: float,
        counterpart: str | None = None,
    ) -> TraitChange | None:
        """按一次经历微调某个特质。返回变化记录，被拒绝则返回 None。"""
        if trait not in TRAITS:
            raise KeyError(f"未知特质：{trait}")
        if source not in VALID_SOURCES:
            raise ValueError(f"人格只能由经历改变，不接受 source={source!r}")
        if not reason:
            raise ValueError("人格变化必须给出理由（reason），否则会漂成噪声")
        if not delta:
            return None

        step = max(-MAX_STEP, min(MAX_STEP, float(delta)))
        old = self.values[trait]
        new = max(FLOOR, min(CEIL, old + step))
        if abs(new - old) < 1e-6:
            return None
        self.values[trait] = new
        ch = TraitChange(clock=clock, trait=trait, old=old, new=new,
                         reason=reason, source=source, counterpart=counterpart)
        self.history.append(ch)
        return ch

    def snapshot(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.values.items()}

    def describe(self) -> str:
        parts = []
        for key, (label, _) in TRAITS.items():
            v = self.values[key]
            level = "高" if v >= 0.66 else ("低" if v <= 0.34 else "中")
            parts.append(f"{label}{level}")
        return "、".join(parts)

    def drift_since(self, clock: float) -> list[TraitChange]:
        return [c for c in self.history if c.clock >= clock]


def _infer_from_labels(labels: list[str], trait: str) -> float:
    """把人设里的中文标签粗略映射成初始值——只是起点，后面全靠经历漂移。"""
    text = "".join(labels)
    hints = {
        "social": [("外向", .78), ("内向", .22), ("社团", .72), ("安静", .28)],
        "diligence": [("爱读书", .8), ("拖延", .3), ("实验狂", .85), ("作息规律", .78)],
        "initiative": [("主动", .78), ("热心", .7), ("随和", .45), ("拖延", .32)],
        "stability": [("严谨", .7), ("情绪", .4), ("嗓门大", .42), ("随和", .72)],
    }
    for kw, v in hints.get(trait, []):
        if kw in text:
            return v
    return 0.5


# ---------------------------------------------------------------- 关系状态


@dataclass
class Relation:
    familiarity: float = 0.0    # 熟悉度：见过几次、聊过多久
    affinity: float = 0.0       # 好感
    trust: float = 0.0          # 信任

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in
                (("familiarity", self.familiarity), ("affinity", self.affinity), ("trust", self.trust))}


@dataclass
class RelationChange:
    clock: float
    a: str
    b: str
    before: dict[str, float]
    after: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock": round(self.clock, 1), "at": _fmt(self.clock),
            "a": self.a, "b": self.b, "before": self.before, "after": self.after,
            "reason": self.reason,
        }


class RelationBook:
    """所有人的两两关系。键是无序对，所以 A→B 与 B→A 是同一份状态。"""

    def __init__(self) -> None:
        self._rel: dict[tuple[str, str], Relation] = {}
        self.history: list[RelationChange] = []

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def get(self, a: str, b: str) -> Relation:
        return self._rel.setdefault(self._key(a, b), Relation())

    def update(
        self,
        a: str,
        b: str,
        clock: float,
        reason: str,
        familiarity: float = 0.0,
        affinity: float = 0.0,
        trust: float = 0.0,
    ) -> RelationChange | None:
        if a == b:
            return None
        rel = self.get(a, b)
        before = rel.to_dict()
        rel.familiarity = _clamp(rel.familiarity + familiarity)
        rel.affinity = _clamp(rel.affinity + affinity)
        rel.trust = _clamp(rel.trust + trust)
        after = rel.to_dict()
        if before == after:
            return None
        ch = RelationChange(clock=clock, a=a, b=b, before=before, after=after, reason=reason)
        self.history.append(ch)
        return ch

    def neighbors(self, agent_id: str, floor: float = 0.05) -> list[tuple[str, Relation]]:
        out = []
        for (a, b), rel in self._rel.items():
            if a == agent_id or b == agent_id:
                other = b if a == agent_id else a
                if rel.familiarity >= floor or rel.affinity >= floor:
                    out.append((other, rel))
        out.sort(key=lambda x: x[1].affinity, reverse=True)
        return out

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {f"{a}|{b}": rel.to_dict() for (a, b), rel in self._rel.items()}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))
