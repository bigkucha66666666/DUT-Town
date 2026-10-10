"""相遇对话：两个智能体碰上，逐句聊，聊完各自写回记忆。

为什么是异步的
--------------
真实 LLM 一轮要 1–3 秒。如果引擎 `step()` 里同步调用，整个世界会卡住——
其他人不动、时钟不走。所以对话被拆成"每次推进一轮"的会话对象，
由世界服务用 `asyncio.to_thread` 驱动；对话期间参与者标记为 `talking`
站着不动，**其他人照常走动、时钟照常推进**。

一次完整对话会留下四样东西：
    1. 双方的逐句记录（前端画气泡用）
    2. 一条对话摘要记忆写进**双方各自的**记忆流（措辞不同：各自视角）
    3. 关系状态变化（熟悉度 / 好感 / 信任）
    4. 人格漂移——**必须带出处**，见 agents/traits.py 的铁律
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "utterance": {"type": "string"},
        "should_end": {"type": "boolean"},
        "feeling": {"type": "string"},
    },
    "required": ["utterance", "should_end"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topic": {"type": "string"},
        "affinity_delta": {"type": "number"},
        "mood_delta": {"type": "number"},
    },
    # 只硬要求 summary。其余字段给默认值——真实模型偶尔会漏一个，
    # 为这个把整条摘要判失败、退化成兜底句，得不偿失。
    "required": ["summary"],
    "additionalProperties": False,
}

def _short_topic(text: str, limit: int = 12) -> str:
    t = text.strip().replace("\n", " ")
    for sep in ("，", "。", "：", "、", " "):
        if sep in t:
            t = t.split(sep)[0]
            break
    return t[:limit] or "最近的事"


def _fmt(minutes: float) -> str:
    m = int(minutes) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass
class Turn:
    speaker: str
    name: str
    text: str
    clock: float
    feeling: str = ""

    def to_dict(self, room: str | None = None) -> dict[str, Any]:
        return {"speaker": self.speaker, "name": self.name, "text": self.text,
                "clock": round(self.clock, 1), "at": _fmt(self.clock),
                "feeling": self.feeling, "room": room}


@dataclass
class DialogueSession:
    id: str
    a: str                     # 发起者 agent_id
    b: str
    topic: str
    started: float             # 游戏分钟
    place: str = ""            # 发生地点（房间名）
    max_turns: int = 6
    turns: list[Turn] = field(default_factory=list)
    done: bool = False
    summary: str = ""
    # 按**游戏时间**节流：一轮对话占多少模拟时间。
    # 不能用真实时间——批量模式下游戏时间跑得飞快，真实时间节流会让
    # 一轮话还没说完，40 游戏分钟已经过去了。
    next_turn_clock: float = 0.0
    # 看门狗：最后一轮推进的真实时刻。对话推进任务若因故停了，
    # 参与者会永远卡在 talking——必须有东西把它们放出来。
    last_progress: float = 0.0

    def participants(self) -> tuple[str, str]:
        return (self.a, self.b)

    def other(self, agent_id: str) -> str:
        return self.b if agent_id == self.a else self.a

    def turn_count(self) -> int:
        return len(self.turns)

    def next_speaker(self) -> str:
        return self.a if len(self.turns) % 2 == 0 else self.b

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "a": self.a, "b": self.b, "topic": self.topic,
            "place": self.place, "started": round(self.started, 1), "at": _fmt(self.started),
            "turns": [t.to_dict(self.place) for t in self.turns],
            "done": self.done, "summary": self.summary,
        }


# ---------------------------------------------------------------- 对话管理


class DialogueManager:
    def __init__(self, gateway: Any, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.gateway = gateway
        self.radius = float(cfg.get("dialogue_engage_radius_m", 2.6))
        self.cooldown = float(cfg.get("dialogue_cooldown_minutes", 25))
        self.max_turns = int(cfg.get("dialogue_max_turns", 6))
        # 每轮对话占多少**游戏秒**（默认 5 秒，接近真人说一句话的节奏）
        self.turn_pacing = float(cfg.get("dialogue_turn_seconds", 5.0)) / 60.0
        self._pair_seen: dict[tuple[str, str], float] = {}
        self.sessions: list[DialogueSession] = []
        self.finished: list[DialogueSession] = []

    # ---------------------------------------------------------- 触发

    def maybe_start(self, agents: list[Any], clock: float, room_name: str = "",
                    rng: Any = None, place_for: Any = None) -> DialogueSession | None:
        """在互相靠近、都不在对话中、且过了冷却的两人之间发起一次对话。

        `place_for(a, b)` 用来在**选出这一对之后**算出地点。不能在外面先算——
        外面还不知道谁会聊起来，拿全体平均位置算出来的房间是错的
        （两个老师在办公室里聊天，地点却被记成"校园里"）。
        """
        busy = {s.participants()[0] for s in self.sessions} | {s.participants()[1] for s in self.sessions}
        free = [a for a in agents if a.state != "talking" and a.id not in busy]

        best = None
        best_d = self.radius
        for i, x in enumerate(free):
            for y in free[i + 1 :]:
                d = ((x.x - y.x) ** 2 + (x.y - y.y) ** 2) ** 0.5
                if d > best_d:
                    continue
                key = tuple(sorted((x.id, y.id)))
                if clock - self._pair_seen.get(key, -1e9) < self.cooldown:
                    continue
                best, best_d = (x, y), d
        if best is None:
            return None

        x, y = best
        place = place_for(x, y) if place_for is not None else room_name
        from llm.templates import NO_TALK_ROOMS
        if place in NO_TALK_ROOMS:
            return None          # 卫生间之类的场合不闲聊
        # 越外向越容易先开口——特质真的影响行为，而不只是写在人设里
        social = (x.personality.values.get("social", .5) + y.personality.values.get("social", .5)) / 2
        if rng is not None and rng.random() > 0.25 + 0.7 * social:
            return None

        key = tuple(sorted((x.id, y.id)))
        self._pair_seen[key] = clock
        topic = self._pick_topic(x, y, room_name)
        s = DialogueSession(
            id=f"d{uuid.uuid4().hex[:10]}", a=x.id, b=y.id, topic=topic,
            started=clock, place=place, max_turns=self.max_turns,
        )
        self.sessions.append(s)
        for ag in (x, y):
            ag.state = "talking"
        return s

    def _pick_topic(self, x: Any, y: Any, room_name: str) -> str:
        """挑一个"能聊的东西"。

        话题必须是名词短语——不能拿一整句记忆当话题，否则插进台词就是
        「我想问一下我在大厅（中庭）」这种病句。优先级：
            1) 之前聊过留下的 topic（记忆里带 topic 字段）
            2) 地点对应的常见话题
            3) 关于对方的记忆里抽一小段
        """
        from llm.templates import ROOM_TOPICS

        for m in reversed(x.memory.about(y.name, k=4)):
            if m.meta.get("topic"):
                return str(m.meta["topic"])[:16]
        if room_name in ROOM_TOPICS:
            return ROOM_TOPICS[room_name]
        for m in reversed(x.memory.about(y.name, k=2)):
            if m.kind == "dialogue":
                return _short_topic(m.text)
        return "最近忙什么"

    # ---------------------------------------------------------- 推进

    async def advance(self, session: DialogueSession, ctx: dict[str, Any]) -> bool:
        """推进一轮。返回 True 表示这次对话结束了。"""
        import time as _t

        clock = ctx["clock"]
        if clock < session.next_turn_clock:
            return False
        session.next_turn_clock = clock + self.turn_pacing
        session.last_progress = _t.perf_counter()

        agents = ctx["agents"]
        speaker = agents[session.next_speaker()]
        listener = agents[session.other(speaker.id)]
        room = ctx.get("room_name", "")

        memories = speaker.memory.retrieve(query=f"{listener.name} {session.topic} {room}", k=4, now=clock)
        relation = ctx["relations"].get(speaker.id, listener.id)
        messages = build_turn_messages(speaker, listener, session, memories, relation, room)

        res = await asyncio.to_thread(
            self.gateway.chat,
            messages,
            TURN_SCHEMA,
            "dialogue.turn",
            {
                "speaker": speaker.name, "other": listener.name,
                "role": speaker.role, "other_role": listener.role,
                "topic": session.topic, "turn": session.turn_count(),
                "traits": speaker.personality.values,
                "used": [t.text for t in session.turns],
            },
        )

        if not res.ok:
            # 模型抽风不能让两个人僵在原地
            ctx["on_event"]({"type": "dialogue_error", "dialogue": session.id,
                             "message": res.error or "模型无响应"})
            self._close(session, ctx, forced=True)
            return True

        utterance = str(res.get("utterance", "")).strip() or "……"
        session.turns.append(Turn(speaker=speaker.id, name=speaker.name, text=utterance,
                                  clock=clock, feeling=str(res.get("feeling", ""))))

        ctx["on_event"]({
            "type": "dialogue_line", "dialogue": session.id,
            "speaker": speaker.id, "name": speaker.name,
            "listener": listener.id, "text": utterance,
            "x": round(speaker.x, 2), "y": round(speaker.y, 2),
            "topic": session.topic, "room": room, "turn": session.turn_count(),
        })

        should_end = bool(res.get("should_end")) or session.turn_count() >= session.max_turns
        if should_end:
            await self._summarize(session, ctx)
            self._close(session, ctx)
            return True
        return False

    # ---------------------------------------------------------- 收尾

    async def _summarize(self, session: DialogueSession, ctx: dict[str, Any]) -> None:
        agents = ctx["agents"]
        a, b = agents[session.a], agents[session.b]
        transcript = "\n".join(f"{t.name}：{t.text}" for t in session.turns)
        messages = [
            {"role": "system", "content": "你在为一通校园对话写摘要，只输出 JSON。"},
            {"role": "user", "content":
                f"{a.name}（{a.persona.get('identity','')}）和{b.name}（{b.persona.get('identity','')}）"
                f"在{ session.place or '校园里'}聊了「{session.topic}」。对话如下：\n{transcript}\n\n"
                "请给出 summary（一句话，第一人称「我」的视角，30 字内）、topic、"
                "affinity_delta（这次聊完好感变化，0–0.15）、mood_delta（情绪变化，-0.1–0.1）。"},
        ]
        res = await asyncio.to_thread(
            self.gateway.chat, messages, SUMMARY_SCHEMA, "dialogue.summary",
            {"a": a.name, "b": b.name, "topic": session.topic, "turns": session.turn_count()},
        )
        if res.ok:
            session.summary = str(res.get("summary", "")).strip()[:80] or f"和{b.name}聊了聊。"
            topic = str(res.get("topic") or session.topic)[:20]
            try:
                aff = float(res.get("affinity_delta", 0.02))
            except (TypeError, ValueError):
                aff = 0.02
            try:
                mood = float(res.get("mood_delta", 0.0))
            except (TypeError, ValueError):
                mood = 0.0
        else:
            # 摘要失败也要留下痕迹，别让"台词很好但记忆是兜底句"这种情况静默发生
            session.summary = f"和{b.name}聊了{session.topic}。"
            topic, aff, mood = session.topic, 0.02, 0.0
            ctx["on_event"]({"type": "dialogue_error", "dialogue": session.id,
                             "message": f"摘要失败：{res.error or '未知'}",
                             "raw": (res.text or "")[:300]})
        session.topic = topic

        clock = ctx["clock"]
        fam = 0.05 + 0.015 * session.turn_count()

        # 1) 双方各自写一条摘要记忆——措辞是各自视角，不是同一句话复制两份
        for me, other in ((a, b), (b, a)):
            me.memory.add(
                f"{session.summary}（和{other.name}，在{session.place or '校园里'}）",
                kind="dialogue", clock=clock, subject=other.name,
                meta={"dialogue": session.id, "topic": topic, "turns": session.turn_count()},
            )

        # 2) 关系状态
        ctx["relations"].update(a.id, b.id, clock, reason=f"聊了{topic}",
                                familiarity=fam, affinity=aff, trust=aff * 0.6)

        # 3) 人格漂移——每一处都写清出处
        changes: list[tuple[str, Any]] = []
        for me in (a, b):
            other_id = session.other(me.id)
            if me.id == session.a:
                c = me.personality.nudge("initiative", 0.02, f"主动找{other_name(me, session, ctx)}搭话",
                                         "dialogue", clock, counterpart=other_id)
                if c: changes.append((me.id, c))
            said = sum(1 for t in session.turns if t.speaker == me.id)
            if said >= 2:
                c = me.personality.nudge("social", 0.012 * min(said, 4),
                                         f"这次一口气说了 {said} 句", "dialogue", clock,
                                         counterpart=other_id)
                if c: changes.append((me.id, c))
            c = me.personality.nudge("stability", mood * 0.6, f"聊完心情变化 {mood:+.2f}",
                                     "dialogue", clock, counterpart=other_id)
            if c: changes.append((me.id, c))

        for agent_id, c in changes:
            ctx["on_event"]({"type": "trait_change", "agent": agent_id,
                             "name": ctx["agents"][agent_id].name, **c.to_dict()})

        ctx["on_event"]({
            "type": "dialogue_end", "dialogue": session.id,
            "a": a.id, "b": b.id, "a_name": a.name, "b_name": b.name,
            "topic": topic, "summary": session.summary,
            "turns": session.turn_count(), "room": session.place,
            "x": round((a.x + b.x) / 2, 2), "y": round((a.y + b.y) / 2, 2),
            "affinity": round(ctx["relations"].get(a.id, b.id).affinity, 3),
            "familiarity": round(ctx["relations"].get(a.id, b.id).familiarity, 3),
        })

    def reap_stalled(self, ctx: dict[str, Any], timeout: float = 30.0) -> int:
        """把停滞太久的会话强制收掉，免得参与者被永久冻结在 talking。"""
        import time as _t

        n = 0
        for session in list(self.sessions):
            if _t.perf_counter() - session.last_progress > timeout:
                self._close(session, ctx_for(self, session, ctx), forced=True)
                n += 1
        return n

    def _close(self, session: DialogueSession, ctx: dict[str, Any], forced: bool = False) -> None:
        session.done = True
        agents = ctx["agents"]
        for aid in session.participants():
            ag = agents.get(aid)
            if ag is not None:
                ag.state = "walking" if ag.path else "idle"
        if session in self.sessions:
            self.sessions.remove(session)
        self.finished.append(session)
        if forced:
            ctx["on_event"]({"type": "dialogue_end", "dialogue": session.id,
                             "a": session.a, "b": session.b, "topic": session.topic,
                             "summary": "（对话中断）", "turns": session.turn_count(),
                             "room": session.place})


def ctx_for(manager: "DialogueManager", session: DialogueSession, ctx: dict[str, Any]) -> dict[str, Any]:
    return ctx


def other_of(me: Any, session: DialogueSession) -> str:
    return session.other(me.id)


def other_name(me: Any, session: DialogueSession, ctx: dict[str, Any]) -> str:
    other = ctx["agents"].get(session.other(me.id))
    return other.name if other else "对方"


# ---------------------------------------------------------------- prompt 构造

def build_turn_messages(speaker: Any, listener: Any, session: DialogueSession,
                        memories: list[Any], relation: Any, room: str) -> list[dict[str, str]]:
    """给真实模型用的 prompt。MockProvider 会忽略它、只读 meta。"""
    p = speaker.persona
    sys_prompt = (
        "你在扮演中国大学校园里的一个人，参与一次自然的口头对话。\n"
        "要求：\n"
        "1. 只输出 JSON，字段为 utterance（这一句台词）、should_end（这句之后对话是否自然结束）、feeling（此刻情绪，两字）。\n"
        "2. 台词必须口语化、符合身份与性格，15–45 字，不要旁白、不要动作描写。\n"
        "3. 不要复述对方刚说过的话，不要每句都提问。\n"
        "4. 性格内向的人话短、少主动；性格外向的人话多、爱接话。\n"
    )
    mem_txt = "\n".join(f"- [{m.kind}] {m.text}" for m in memories) or "（想不起什么相关的）"
    transcript = "\n".join(f"{t.name}：{t.text}" for t in session.turns) or "（还没开口）"
    user = (
        f"你是{speaker.name}，{p.get('identity','')}。\n"
        f"你的性格：{speaker.personality.describe()}。"
        f"说话风格：{p.get('speech_style','')}。\n"
        f"你当前的目标：{p.get('goal','')}。\n"
        f"对方是{listener.name}，{listener.persona.get('identity','')}。"
        f"你们的关系：熟悉度{relation.familiarity:.2f}、好感{relation.affinity:.2f}、信任{relation.trust:.2f}。\n"
        f"地点：{room or '校园里'}。话题：{session.topic}。\n\n"
        f"你脑子里冒出来的相关记忆：\n{mem_txt}\n\n"
        f"到目前为止的对话：\n{transcript}\n\n"
        "现在轮到你说话，只输出 JSON。"
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
