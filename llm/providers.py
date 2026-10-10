"""provider 实现：Mock（占位，用来跑通管道）与 OpenAI 兼容接口。

MockProvider 不是"随机文本"——它按 purpose 和 meta 里的角色/关系/话题，
从模板池里挑一句像样的中文。它明显是个提线木偶，但足够把
「相遇 → 逐句对话 → 摘要 → 写回记忆 → 关系与人格变化」这条链路跑通，
而且完全确定、零成本、可离线跑测试。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any

from llm.templates import (ACADEMIC_TOPICS, FOLLOWUPS, GENERIC_FOLLOWUPS, GENERIC_OPENERS,
                           GREETING_OPENERS, OPENERS, REFLECT_TEMPLATES,
                           REFLECT_TEMPLATES_ABOUT, SUMMARY_TEMPLATES,
                           TERSE_FOLLOWUPS, TERSE_OPENERS, pool_for)


@dataclass
class Completion:
    text: str = ""
    data: dict[str, Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Provider:
    name = "base"

    def complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 512,
        purpose: str = "generic",
        meta: dict[str, Any] | None = None,
    ) -> Completion:
        raise NotImplementedError


# ---------------------------------------------------------------- Mock


def _rng(*parts: Any) -> random.Random:
    blob = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12], 16))


class MockProvider(Provider):
    name = "mock"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.seed = int(cfg.get("seed", 0))

    def complete(self, messages, schema=None, temperature=0.8, max_tokens=512, purpose="generic", meta=None):
        meta = meta or {}
        handler = getattr(self, "_p_" + purpose.replace(".", "_"), None)
        data = handler(meta) if handler else {"ok": True}
        text = json.dumps(data, ensure_ascii=False)
        return Completion(text=text, data=data, prompt_tokens=len(text) // 3, completion_tokens=len(text) // 3)

    # ---------------- 各 purpose 的假输出

    def _p_memory_importance(self, meta: dict[str, Any]) -> dict[str, Any]:
        """重要度：按文本里的线索给一个 1–10 的分，而不是瞎给。"""
        text = str(meta.get("text", ""))
        score = 3
        for kw, bonus in [
            ("老师", 2), ("答疑", 2), ("争执", 3), ("道歉", 3), ("咖啡", 1),
            ("组会", 2), ("挂科", 4), ("表扬", 3), ("帮忙", 2), ("第一次", 2),
        ]:
            if kw in text:
                score += bonus
        rng = _rng(self.seed, meta.get("agent"), text)
        return {"importance": max(1, min(10, score + rng.randint(-1, 1)))}

    def _p_dialogue_turn(self, meta: dict[str, Any]) -> dict[str, Any]:
        speaker = meta.get("speaker", "某人")
        other = meta.get("other", "对方")
        role, other_role = meta.get("role", "student"), meta.get("other_role", "student")
        turn = int(meta.get("turn", 0))
        topic = meta.get("topic") or "最近的事"
        traits = meta.get("traits") or {}
        rng = _rng(self.seed, speaker, other, turn, topic)

        used = set(meta.get("used") or [])
        if turn == 0:
            # 事务型话题才能提问；寒暄话题走不带 {topic} 的开场白
            table = OPENERS if topic in ACADEMIC_TOPICS else GREETING_OPENERS
        else:
            table = FOLLOWUPS
        pool = pool_for(table, role, other_role)
        rendered = [p.format(n=other, t=teacher_title(other_role, other), topic=topic) for p in pool]
        fresh = [x for x in rendered if x not in used]
        line = rng.choice(fresh or rendered)      # 都用过了就允许重复，但尽量不

        # 内向的人话短——用专门的短句池，而不是把长句砍断
        social = float(traits.get("social", 0.5))
        if social < 0.35 and rng.random() < 0.5:
            # 开场不能拿告别语充数
            line = rng.choice(TERSE_OPENERS if turn == 0 else TERSE_FOLLOWUPS)

        should_end = turn >= 4 or (turn >= 2 and rng.random() < 0.35)
        return {
            "utterance": line,
            "should_end": should_end,
            "feeling": rng.choice(["平静", "愉快", "有点紧张", "被逗笑", "若有所思"]),
        }

    def _p_dialogue_summary(self, meta: dict[str, Any]) -> dict[str, Any]:
        a, b = meta.get("a", "甲"), meta.get("b", "乙")
        topic = meta.get("topic") or "最近的事"
        turns = int(meta.get("turns", 2))
        rng = _rng(self.seed, a, b, topic, turns)
        tpl = rng.choice(SUMMARY_TEMPLATES)
        warm = 0.02 + 0.01 * min(turns, 6)
        if any(k in tpl for k in ("顺利", "收获", "踏实")):
            warm += 0.02
        # 摘要是**第一人称且不指认谁先开口**的——它会被分别写进两个人的记忆
        return {
            "summary": tpl.format(topic=topic),
            "topic": topic,
            "affinity_delta": round(rng.uniform(0.3, 1.0) * warm, 4),
            "mood_delta": round(rng.uniform(-0.04, 0.08), 4),
        }

    def _p_reflect(self, meta: dict[str, Any]) -> dict[str, Any]:
        agent = meta.get("agent", "我")
        mems = meta.get("memories") or []
        subjects = [s for s in (meta.get("subjects") or []) if s]
        rng = _rng(self.seed, agent, len(mems), " ".join(m[:12] for m in mems[:3]))

        pool = list(REFLECT_TEMPLATES)
        if subjects:
            # subject 是"人/地点"这种名词短语，塞进模板不会出病句
            pool = [t.format(s=rng.choice(subjects)) for t in REFLECT_TEMPLATES_ABOUT] + pool
        picks = rng.sample(pool, k=min(2, len(pool)))

        deltas: dict[str, float] = {}
        for key in ("social", "diligence", "initiative", "stability"):
            if rng.random() < 0.4:
                deltas[key] = round(rng.uniform(-0.06, 0.06), 4)
        return {"insights": picks, "trait_deltas": deltas}


def teacher_title(role: str, name: str) -> str:
    """学生对老师用「姓+老师」，同辈之间直呼其名。"""
    if role == "teacher" and name:
        return f"{name[0]}老师"
    return name


# ---------------------------------------------------------------- OpenAI 兼容


class OpenAICompatProvider(Provider):
    """DeepSeek / Qwen / GLM / Ollama 都走这个（都是 OpenAI 兼容接口）。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.name = cfg.get("label", "openai")
        self.base_url = str(cfg.get("base_url", "")).rstrip("/")
        self.model = cfg.get("model", "deepseek-chat")
        self.api_key = os.environ.get(cfg.get("api_key_env", ""), "") or cfg.get("api_key", "")
        self.timeout = float(cfg.get("timeout", 60))
        self.concurrency = int(cfg.get("concurrency", 4))

    def complete(self, messages, schema=None, temperature=0.8, max_tokens=512, purpose="generic", meta=None):
        import httpx

        if not self.base_url:
            raise RuntimeError(f"{self.name} 没有配置 base_url")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = payload.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
