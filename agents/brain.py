"""占位大脑：照本宣科地执行 persona.routine。

**这不是最终的大脑。** 它的存在只是为了让「地图 → 寻路 → 精灵」这条
链路先跑通，好在确定性代码上调试渲染与寻路，而不用同时面对 LLM 的
不确定性。

它的接口刻意与将来的 LLM 认知层保持一致：

    decide(agent, world, clock_minutes) -> goal | None

到时把 RuleBrain 换成 LLM 大脑即可，引擎与渲染一行都不用改。
"""

from __future__ import annotations

from typing import Any, Protocol


def parse_hhmm(s: str) -> int:
    """'07:50' -> 470（当日第几分钟）"""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


class Brain(Protocol):
    def decide(self, agent: Any, world: Any, clock_minutes: float) -> dict[str, Any] | None: ...


class RuleBrain:
    """按 routine 的时间表逐条执行，每条只触发一次。"""

    name = "rule"

    def decide(self, agent: Any, world: Any, clock_minutes: float) -> dict[str, Any] | None:
        for i, item in enumerate(agent.persona.get("routine", [])):
            if i in agent.done_routine:
                continue
            if clock_minutes >= parse_hhmm(item["at"]):
                agent.done_routine.add(i)
                return {"room": item["room"], "why": item["why"], "index": i}
        return None
