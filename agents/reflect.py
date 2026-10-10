"""反思：记忆攒够了，人就该有点想法。

为什么必须有这一环
------------------
人格漂移目前只有"对话"一个来源——被人说了一句话就变一点。但真实的人格变化
更多来自**事后回味**：连着三天在办公室门口折返，才意识到自己"怕麻烦老师"。

机制
----
* 触发：记忆的**累计重要度**（不是条数）超过阈值。这样"被表扬"比"走过走廊"
  更快攒够，符合直觉。
* 输入：近期 + 高重要度的记忆（复用记忆流的三因子检索）。
* 输出：若干条高层洞察（写成高重要度的 `reflection` 记忆）+ 对人格的微调。
* 约束：洞察本身也进记忆流，所以"反思过的反思"会自然形成层级；
  人格微调仍然走 `Personality.nudge()`，照样强制带出处。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {"type": "array", "items": {"type": "string"}},
        "trait_deltas": {"type": "object"},
    },
    "required": ["insights"],
    "additionalProperties": False,
}

# 反思本身的重要度：它比普通观察重要，但低于"被当众表扬"这种事
INSIGHT_IMPORTANCE = 8


@dataclass
class Reflection:
    agent_id: str
    clock: float
    insights: list[str]
    changes: list[Any]
    based_on: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_id,
            "insights": self.insights,
            "based_on": self.based_on,
            "changes": [c.to_dict() for c in self.changes],
        }


class Reflector:
    def __init__(self, gateway: Any, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.gateway = gateway
        self.threshold = float(cfg.get("reflect_importance_threshold", 26))
        self.max_memories = int(cfg.get("reflect_max_memories", 10))
        self.min_interval = float(cfg.get("reflect_min_interval_minutes", 30))
        self.max_insights = int(cfg.get("reflect_max_insights", 2))
        self._last: dict[str, float] = {}
        self.log: list[Reflection] = []

    # ---------------------------------------------------------- 触发判断

    def due(self, agent: Any, clock: float) -> bool:
        """这个智能体现在该反思了吗。"""
        if agent.memory is None or agent.personality is None:
            return False
        if agent.memory.importance_since_reflection() < self.threshold:
            return False
        last = self._last.get(agent.id)
        if last is not None and clock - last < self.min_interval:
            return False
        return True

    def candidates(self, agents: list[Any], clock: float) -> list[Any]:
        return [a for a in agents if self.due(a, clock)]

    # ---------------------------------------------------------- 执行

    async def reflect(self, agent: Any, ctx: dict[str, Any]) -> Reflection | None:
        import asyncio

        clock = ctx["clock"]
        # 拿近期 + 高重要度的记忆当素材
        query = " ".join(m.text for m in agent.memory.recent(4))
        memories = agent.memory.retrieve(query=query, k=self.max_memories, now=clock)
        if not memories:
            agent.memory.mark_reflected()      # 没素材就别反复触发
            return None

        messages = build_reflect_messages(agent, memories)
        res = await asyncio.to_thread(
            self.gateway.chat, messages, REFLECT_SCHEMA, "reflect",
            {"agent": agent.name, "memories": [m.text for m in memories],
             "subjects": [m.subject for m in memories if m.subject]},
        )
        self._last[agent.id] = clock

        if not res.ok:
            ctx["on_event"]({"type": "reflect_error", "agent": agent.id, "name": agent.name,
                             "message": res.error or "模型无响应"})
            return None

        raw = res.get("insights") or []
        insights = [str(x).strip() for x in raw if str(x).strip()][: self.max_insights]
        if not insights:
            agent.memory.mark_reflected()
            return None

        # 1) 洞察写成高重要度的反思记忆——它自己也会被后续检索到，
        #    于是"反思的反思"能自然形成层级
        for text in insights:
            agent.memory.add(
                f"我意识到：{text}", kind="reflection", importance=INSIGHT_IMPORTANCE,
                clock=clock, meta={"based_on": [m.id for m in memories[:6]]},
            )

        # 2) 洞察可以改人格——但照样走 nudge()，照样强制带出处
        deltas = res.get("trait_deltas") or {}
        changes = []
        reason = f"反思：{insights[0][:30]}"
        for trait, delta in list(deltas.items())[:4]:
            if trait not in agent.personality.values:
                continue
            try:
                ch = agent.personality.nudge(trait, float(delta), reason, "reflection", clock)
            except (TypeError, ValueError):
                continue
            if ch is not None:
                changes.append((agent.id, ch))

        agent.memory.mark_reflected()

        ref = Reflection(agent_id=agent.id, clock=clock, insights=insights,
                         changes=[c for _, c in changes], based_on=[m.id for m in memories[:6]])
        self.log.append(ref)

        ctx["on_event"]({
            "type": "reflection", "agent": agent.id, "name": agent.name,
            "insights": insights,
            "x": round(agent.x, 2), "y": round(agent.y, 2),
        })
        for agent_id, ch in changes:
            ctx["on_event"]({"type": "trait_change", "agent": agent_id, "name": agent.name,
                             **ch.to_dict()})
        return ref


# ---------------------------------------------------------------- prompt

def build_reflect_messages(agent: Any, memories: list[Any]) -> list[dict[str, str]]:
    """给真实模型用的 prompt。MockProvider 只读 meta。"""
    mem_txt = "\n".join(f"- [{m.kind}] {m.text}" for m in memories)
    sys_prompt = (
        "你在扮演一个中国大学校园里的人，正在独处时回想最近发生的事。\n"
        "请从这些记忆里提炼 1–2 条**高层洞察**（不是复述，而是「我意识到……」这样的句子），"
        "并给出这些洞察对性格的细微影响。\n"
        "只输出 JSON：insights（字符串数组）、trait_deltas（对象，键为 "
        "social/diligence/initiative/stability，值为 -0.06~0.06）。\n"
        "洞察要具体、克制，不要鸡汤。"
    )
    user = (
        f"你是{agent.name}，{agent.persona.get('identity','')}。\n"
        f"你当前的性格：{agent.personality.describe()}。\n"
        f"你的目标：{agent.persona.get('goal','')}。\n\n"
        f"最近这些事一直在你脑子里：\n{mem_txt}\n\n只输出 JSON。"
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
