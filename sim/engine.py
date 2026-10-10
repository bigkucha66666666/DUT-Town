"""世界引擎：tick 时钟、智能体调度、事件与回放轨迹。

时间模型
--------
* 逻辑时间 1 tick = 1 游戏分钟（与设计文档 §5.3 一致）
* 引擎内部把 1 tick 细分成 60 个 substep，1 substep = 1 游戏秒，
  移动与感知都跑在 substep 上，否则 1 分钟一动会瞬移。
* 回放轨迹按"位移 ≥ 0.8m 或每 15 游戏秒"自适应采样：既不会长到跑不动，
  又保证相邻两帧之间不会横穿墙体（墙厚 2m，帧间距远小于它）。

回合结构
--------
    每个 substep：
      brain.decide()  → 若无目标则给出目标房间      （= 将来的 LLM 规划）
      find_path()     → A* 变成一条路径            （确定性）
      agent.step()    → 沿路径推进                 （确定性）
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agents.brain import Brain, RuleBrain
from agents.dialogue import DialogueManager
from agents.memory import Memory, MemoryStream
from agents.reflect import Reflector
from agents.runtime import Agent
from agents.traits import Personality, RelationBook
from world.map_loader import MapData, load_map, rasterize
from world.pathfinding import find_path, nearest_walkable

PERSONA_GLOB = ("students/*.json", "teachers/*.json", "staff/*.json")


@dataclass
class Engine:
    map_path: str | Path = "world/maps/sem_1f.json"
    personas_dir: str | Path = "agents/personas"
    start_clock: str = "07:40"
    seed: int = 42
    brain: Brain | None = None
    gateway: Any = None                     # LLM 网关；None 时认知层走启发式兜底
    llm_cfg: dict[str, Any] = field(default_factory=dict)
    store: Any = None                       # 落盘（可选）
    # 批处理（导出回放）需要逐帧记录；常驻世界服务不需要，它会话级地推快照
    record_frames: bool = True

    # 运行时状态
    map: MapData = field(init=False)
    grid: Any = field(init=False)
    agents: list[Agent] = field(init=False, default_factory=list)
    clock: float = field(init=False, default=0.0)          # 当日第几分钟
    day: int = field(init=False, default=0)                # 第几天（跨天比较新旧要用）
    frames: list[dict[str, Any]] = field(init=False, default_factory=list)
    events: list[dict[str, Any]] = field(init=False, default_factory=list)
    _last_pos: dict[str, tuple[float, float]] = field(init=False, default_factory=dict)
    _last_frame_clock: float = field(init=False, default=0.0)
    _last_states: dict[str, str] = field(init=False, default_factory=dict)

    FRAME_MOVE = 0.8        # 位移超过这个距离就记一帧（米）
    FRAME_MAX_GAP = 15 / 60  # 最长也要每 15 游戏秒记一帧（分钟）

    def __post_init__(self) -> None:
        from agents.brain import parse_hhmm

        self.map = load_map(self.map_path)
        self.grid = rasterize(self.map)
        self.clock = float(parse_hhmm(self.start_clock))
        self._last_frame_clock = self.clock
        self.brain = self.brain or RuleBrain()
        if self.gateway is None:
            # 没给网关就兜一个 Mock：批量模式与测试也能自然地产生对话，
            # 而不是让认知层静默失效。
            from llm.gateway import LLMGateway
            self.gateway = LLMGateway({"provider": "mock"})
        self.rng = random.Random(self.seed)
        self.relations = RelationBook()
        self.dialogues = DialogueManager(self.gateway, self.llm_cfg.get("budget", self.llm_cfg))
        self.reflector = Reflector(self.gateway, self.llm_cfg.get("budget", self.llm_cfg))
        self.dialogue_log: list[dict[str, Any]] = []
        self._last_ambient = 0.0
        self._last_trail = 0.0
        # 每个人的滚动足迹——"独特的轨迹"最直白的可视化就是它
        self.trails: dict[str, list[list[float]]] = {a.id: [] for a in self.agents}
        self._persisted_events = 0
        # 落盘靠"对历史做差分"，而不是让各个认知模块自己登记待写项——
        # 后者只要有一条代码路径忘了登记，那次变化就永远丢了。
        self._persisted_traits: dict[str, int] = {}
        self._persisted_relation_changes = 0
        self._persisted_dialogues = 0
        self._spawn_agents()
        if self.store is not None:
            self.resume()
        self._record_frame(force=True)

    # ------------------------------------------------------------ 出生

    def _spawn_agents(self) -> None:
        spawns = {s["id"]: s for s in self.map.spawns}
        for persona in self._load_personas():
            sp = spawns.get(persona.get("start_spawn", ""), None)
            if sp is None:
                sp = self.map.spawns[0] if self.map.spawns else {"pos": [1, 1]}
            jitter = random.Random(f"{self.seed}:{persona['id']}")
            x = sp["pos"][0] + jitter.uniform(-3.5, 3.5)
            y = sp["pos"][1] + jitter.uniform(-2.5, 2.5)
            snapped = nearest_walkable(self.grid, x, y) or (float(sp["pos"][0]), float(sp["pos"][1]))
            self.agents.append(
                Agent(
                    id=persona["id"],
                    name=persona["name"],
                    role=persona["role"],
                    persona=persona,
                    x=snapped[0],
                    y=snapped[1],
                    speed=float(persona.get("speed", 1.3)),
                    # 重要度打分默认走启发式：LLM 调用是同步阻塞的，
                    # 放进实时循环会卡住整个世界。反思阶段再让 LLM 精修。
                    memory=MemoryStream(persona["id"], gateway=None, store=self.store),
                    personality=Personality.from_persona(persona),
                )
            )

    def _load_personas(self) -> list[dict[str, Any]]:
        root = Path(self.personas_dir)
        out: list[dict[str, Any]] = []
        for pattern in PERSONA_GLOB:
            for path in sorted(root.glob(pattern)):
                with path.open(encoding="utf-8") as fh:
                    out.append(json.load(fh))
        if not out:
            raise FileNotFoundError(f"在 {root} 下没找到任何人设文件（{PERSONA_GLOB}）")
        return out

    # ------------------------------------------------------------ 推进

    def step(self, dt: float = 1.0) -> None:
        """推进 dt 游戏秒。"""
        self.clock += dt / 60.0

        for agent in self.agents:
            if agent.state == "talking":
                continue                      # 正在聊天的人站着不动
            if agent.state == "idle" and not agent.path:
                goal = self.brain.decide(agent, self, self.clock)
                if goal:
                    self._dispatch(agent, goal)
            ev = agent.step(dt)
            if ev:
                ev["clock"] = self.clock_str()
                ev["t"] = round((self.clock - self._start_minutes()) * 60, 1)
                self.events.append(ev)
                self._remember_arrival(agent, ev)

        self._ambient_observation()
        self._remember_trail()
        self._maybe_talk()
        self._record_frame()

    # ------------------------------------------------------------ 记忆写入

    def _remember_arrival(self, agent: Agent, ev: dict[str, Any]) -> None:
        room = self.map.room_name_of(ev.get("room")) if hasattr(self.map, "room_name_of") else None
        room = room or self._room_name(ev.get("room"))
        what = ev.get("why") or "去做点事"
        agent.memory.add(
            f"我到了{room}，来{what}。",
            kind="event", clock=self.now_minutes, subject=room,
            meta={"room": ev.get("room"), "why": what},
        )

    def _place_of_pair(self, a: Any, b: Any) -> str:
        """这一对所在的地点——取两人中点落进的房间。"""
        room = self.room_at((a.x + b.x) / 2, (a.y + b.y) / 2)
        return room.name if room else "校园里"

    def _room_name(self, room_id: str | None) -> str:
        r = next((x for x in self.map.rooms if x.id == room_id), None)
        return r.name if r else (room_id or "校园里")

    def _ambient_observation(self) -> None:
        """每 12 游戏分钟给闲着的智能体记一条环境观察，让记忆流有底色，
        不然只有"出发/到达"会显得干巴巴。"""
        if self.clock - self._last_ambient < 12:
            return
        self._last_ambient = self.clock
        for agent in self.agents:
            if agent.state == "talking":
                continue
            room = self.room_at(agent.x, agent.y)
            here = room.name if room else "楼外"
            others = [a for a in self.agents
                      if a.id != agent.id and math.hypot(a.x - agent.x, a.y - agent.y) < 6
                      and (a.state != "talking")]
            who = "，".join(a.name for a in others[:3])
            text = f"我在{here}" + (f"，{who}也在附近。" if who else "，旁边没什么人。")
            agent.memory.add(text, kind="observation", importance=2, clock=self.now_minutes, subject=here)

    def _remember_trail(self) -> None:
        """每 6 游戏秒记一个足迹点，滚动保留最近 300 个（约半天）。"""
        if self.clock - self._last_trail < 6 / 60.0:
            return
        self._last_trail = self.clock
        for a in self.agents:
            trail = self.trails.setdefault(a.id, [])
            if trail and math.dist(trail[-1], (a.x, a.y)) < 0.5:
                continue                       # 站着不动就别堆点
            trail.append([round(a.x, 2), round(a.y, 2)])
            if len(trail) > 300:
                del trail[: len(trail) - 300]

    def trail(self, agent_id: str) -> list[list[float]]:
        return self.trails.get(agent_id, [])

    # ------------------------------------------------------------ 相遇对话

    def _maybe_talk(self) -> None:
        if not self.agents:
            return
        s = self.dialogues.maybe_start(
            self.agents, self.now_minutes, rng=self.rng,
            place_for=self._place_of_pair,
        )
        if s is None:
            return
        a = self.agent_by_id(s.a)
        b = self.agent_by_id(s.b)
        self._emit({
            "type": "dialogue_start", "dialogue": s.id,
            "a": s.a, "b": s.b, "a_name": a.name, "b_name": b.name,
            "topic": s.topic, "room": s.place,
            "x": round((a.x + b.x) / 2, 2), "y": round((a.y + b.y) / 2, 2),
        })

    async def advance_reflections(self, limit: int = 1) -> int:
        """让攒够重要度的智能体反思。每次最多处理 limit 个——
        反思不急，别让它把模型配额和 tick 都占了。"""
        done = 0
        for agent in self.reflector.candidates(self.agents, self.now_minutes)[:limit]:
            ctx = self._dialogue_ctx_stub()
            if await self.reflector.reflect(agent, ctx) is not None:
                done += 1
        return done

    async def advance_dialogues(self) -> int:
        """推进所有进行中的对话各一轮。由世界服务 await，**不阻塞世界推进**。"""
        done_count = 0
        for session in list(self.dialogues.sessions):
            ctx = self._dialogue_ctx(session)
            if await self.dialogues.advance(session, ctx):
                self.dialogue_log.append(session.to_dict())
                done_count += 1
        return done_count

    def reap_stalled_dialogues(self, timeout: float = 30.0) -> int:
        """世界服务在对话任务之外兜底调用：把卡死的会话放掉。"""
        ctx = self._dialogue_ctx_stub()
        return self.dialogues.reap_stalled(ctx, timeout)

    def _dialogue_ctx_stub(self) -> dict[str, Any]:
        return {
            "agents": {ag.id: ag for ag in self.agents},
            "clock": self.now_minutes,
            "clock_str": self.clock_str(),
            "relations": self.relations,
            "room_name": "校园里",
            "on_event": self._emit,
            "store": self.store,
            "day": self.day,
        }

    def _dialogue_ctx(self, session: Any) -> dict[str, Any]:
        a = self.agent_by_id(session.a)
        room = self.room_at(a.x, a.y) if a else None
        return {
            "agents": {ag.id: ag for ag in self.agents},
            "clock": self.now_minutes,
            "clock_str": self.clock_str(),
            "relations": self.relations,
            "room_name": session.place or (room.name if room else "校园里"),
            "on_event": self._emit,
            "store": self.store,
            "day": self.day,
        }

    def _emit(self, ev: dict[str, Any]) -> None:
        ev.setdefault("clock", self.clock_str())
        ev.setdefault("t", round((self.clock - self._start_minutes()) * 60, 1))
        self.events.append(ev)

    def _dispatch(self, agent: Agent, goal: dict[str, Any]) -> None:
        room = next((r for r in self.map.rooms if r.id == goal["room"]), None)
        if room is None:
            self.events.append({
                "type": "error", "agent": agent.id, "clock": self.clock_str(),
                "message": f"目标房间不存在：{goal['room']}",
            })
            return

        target = self._point_in(room, agent)
        path = find_path(self.grid, (agent.x, agent.y), target)
        if not path:
            self.events.append({
                "type": "error", "agent": agent.id, "clock": self.clock_str(),
                "message": f"到「{room.name}」找不到路",
            })
            agent.goal = None
            return

        agent.assign(path, goal)
        self.events.append({
            "type": "depart",
            "agent": agent.id,
            "name": agent.name,
            "room": room.id,
            "room_name": room.name,
            "why": goal.get("why", ""),
            "distance": round(sum(math.dist(a, b) for a, b in zip(
                [(agent.x, agent.y)] + path, path)), 1),
            "clock": self.clock_str(),
            "t": round((self.clock - self._start_minutes()) * 60, 1),
        })

    def _point_in(self, room: Any, agent: Agent) -> tuple[float, float]:
        """在房间里挑一个可走的落脚点。用确定性随机，保证可复现。"""
        rng = random.Random(f"{self.seed}:{agent.id}:{room.id}:{int(self.clock)}")
        rect = room.rect
        for _ in range(60):
            x = rng.uniform(rect.x + 1.0, rect.x2 - 1.0)
            y = rng.uniform(rect.y + 1.0, rect.y2 - 1.0)
            if self.grid.is_walkable(x, y):
                return (x, y)
        return nearest_walkable(self.grid, *rect.center()) or rect.center()

    def run(self, minutes: float, substep: float = 1.0) -> None:
        """批量跑一段。**同时驱动对话**——否则智能体一聊起来就永远卡在
        talking，谁也不会再动（批处理模式没有世界服务那条对话任务）。"""
        import asyncio

        steps = int(round(minutes * 60 / substep))

        async def _go() -> None:
            for _ in range(steps):
                self.step(substep)
                await self.advance_dialogues()

        asyncio.run(_go())

    # ------------------------------------------------------------ 外部干预
    # 世界服务层通过这些方法响应浏览器/脚本的指令。它们只改"目标"，
    # 不绕过 A* 与移动逻辑——所以人工干预和自主行为走的是同一条腿。

    @property
    def now_minutes(self) -> float:
        """绝对游戏分钟（day*1440 + 当日分钟）。

        记忆与人格轨迹要跨天比较新旧；只存当日分钟的话，
        第二天 07:40 的 now 会比昨天 08:00 的记忆还小，时效性算出来是负的。
        """
        return self.day * 1440.0 + self.clock

    def agent_by_id(self, aid: str) -> Agent | None:
        return next((a for a in self.agents if a.id == aid), None)

    def nearest_agent(self, x: float, y: float) -> Agent | None:
        if not self.agents:
            return None
        return min(self.agents, key=lambda a: math.hypot(a.x - x, a.y - y))

    def room_at(self, x: float, y: float) -> Any | None:
        """点所在的房间；不在任何房间里返回 None。"""
        for r in self.map.rooms:
            if r.rect.x <= x <= r.rect.x2 and r.rect.y <= y <= r.rect.y2:
                return r
        return None

    def send_to_point(self, agent: Agent, x: float, y: float, why: str = "接到指令前往") -> bool:
        """把智能体派到某个坐标（不可通行时会吸附到最近的可走格）。"""
        target = nearest_walkable(self.grid, x, y)
        if target is None:
            return False
        path = find_path(self.grid, (agent.x, agent.y), target)
        if not path:
            self._log("intervene_failed", agent, f"到 ({x:.0f}, {y:.0f}) 找不到路")
            return False
        room = self.room_at(*target)
        agent.assign(path, {"room": room.id if room else None, "why": why, "kind": "manual"})
        self._log("dispatch", agent, why, room=room.id if room else None,
                  room_name=room.name if room else "场地", target=[round(x, 1), round(y, 1)])
        return True

    def stop_agent(self, agent: Agent) -> None:
        agent.path = []
        agent.goal = None
        agent.state = "idle"
        self._log("stop", agent, "被叫停")

    def _log(self, kind: str, agent: Agent, why: str = "", **extra: Any) -> None:
        ev = {
            "type": kind, "agent": agent.id, "name": agent.name,
            "why": why, "clock": self.clock_str(),
            "t": round((self.clock - self._start_minutes()) * 60, 1),
        }
        ev.update(extra)
        self.events.append(ev)

    # ------------------------------------------------------------ 落盘

    def flush(self) -> dict[str, int]:
        """把内存里的变化写进 SQLite。由世界服务定期调用。

        一律用"历史游标 + 差分"：记下每个序列已经写到哪，下次只写新增的部分。
        不管人格是被对话改的、被反思改的，还是被测试直接调 nudge() 改的，
        都会在这里被写下去——**没有哪条路径需要记得来登记**。
        """
        if self.store is None:
            return {}
        n_mem = sum(a.memory.flush(self.day) for a in self.agents if a.memory)
        n_ev = self.store.save_events(self.events[self._persisted_events:], self.day)
        self._persisted_events = len(self.events)

        n_trait = 0
        for a in self.agents:
            if a.personality is None:
                continue
            done = self._persisted_traits.get(a.id, 0)
            for ch in a.personality.history[done:]:
                self.store.save_trait_change(a.id, ch, self.day)
                n_trait += 1
            self._persisted_traits[a.id] = len(a.personality.history)

        for ch in self.relations.history[self._persisted_relation_changes:]:
            self.store.save_relation_change(ch, self.day)
        self._persisted_relation_changes = len(self.relations.history)
        for (x, y), rel in self.relations._rel.items():
            self.store.save_relation(x, y, rel, self.now_minutes)

        for d in self.dialogue_log[self._persisted_dialogues:]:
            self.store.save_dialogue(d, self.day)
        self._persisted_dialogues = len(self.dialogue_log)

        return {"memories": n_mem, "events": n_ev, "traits": n_trait}

    def resume(self) -> dict[str, int]:
        """从库里恢复记忆与人格。**这就是"持久记忆"的意思**：
        重启世界之后，智能体还记得上次发生过什么。"""
        st = self.store
        if not st.has_data():
            return {"memories": 0, "traits": 0, "relations": 0}   # 空库不算"过了一天"
        self.day = st.latest_day() + 1
        loaded = {"memories": 0, "traits": 0, "relations": 0}
        for agent in self.agents:
            for row in st.load_memories(agent.id):
                m = Memory(
                    id=row["id"], agent_id=agent.id, kind=row["kind"], text=row["text"],
                    importance=row["importance"], created=row["created"],
                    last_access=row["last_access"], subject=row["subject"], meta=row["meta"],
                )
                agent.memory.items.append(m)
                agent.memory.total_importance += m.importance
                loaded["memories"] += 1
            traits = st.load_traits(agent.id)
            for k, v in traits.items():
                if k in agent.personality.values:
                    agent.personality.values[k] = v
                    loaded["traits"] += 1
        for (x, y), (fam, aff, tru) in st.load_relations().items():
            rel = self.relations.get(x, y)
            rel.familiarity, rel.affinity, rel.trust = fam, aff, tru
            loaded["relations"] += 1
        for agent in self.agents:
            if agent.personality is not None:
                self._persisted_traits[agent.id] = len(agent.personality.history)
        self._persisted_relation_changes = len(self.relations.history)
        if loaded["memories"] or loaded["traits"]:
            self._emit({"type": "resume", "day": self.day, **loaded})
        return loaded

    # ------------------------------------------------------------ 实时快照

    def snapshot(self, path_points: int = 16) -> dict[str, Any]:
        """给世界服务层用的当前状态快照（不含历史帧）。"""
        from agents.runtime import FACING_INDEX

        active = {s.participants()[0]: s for s in self.dialogues.sessions}
        active.update({s.participants()[1]: s for s in self.dialogues.sessions})
        return {
            "clock": self.clock_str(),
            "t": round((self.clock - self._start_minutes()) * 60, 1),
            "day_minutes": round(self.clock, 2),
            "dialogues": [
                {
                    "id": s.id, "a": s.a, "b": s.b, "topic": s.topic, "place": s.place,
                    "turns": s.turn_count(),
                    "last": (s.turns[-1].to_dict(s.place) if s.turns else None),
                }
                for s in self.dialogues.sessions
            ],
            "agents": [
                {
                    "id": a.id, "name": a.name, "role": a.role,
                    "x": round(a.x, 2), "y": round(a.y, 2),
                    "f": FACING_INDEX[a.facing],
                    "m": 1 if a.state == "walking" else 0,
                    "state": a.state,
                    "room": (a.goal or {}).get("room"),
                    "why": (a.goal or {}).get("why", ""),
                    "kind": (a.goal or {}).get("kind", "routine"),
                    "path": [[round(p[0], 1), round(p[1], 1)] for p in a.path[:path_points]],
                    "memory": (a.memory.stats() if a.memory else None),
                    "personality": (a.personality.snapshot() if a.personality else None),
                    "talking": (active[a.id].id if a.id in active else None),
                }
                for a in self.agents
            ],
        }

    # ------------------------------------------------------------ 轨迹

    def _start_minutes(self) -> float:
        from agents.brain import parse_hhmm
        return float(parse_hhmm(self.start_clock))

    def clock_str(self) -> str:
        m = int(self.clock) % (24 * 60)
        return f"{m // 60:02d}:{m % 60:02d}"

    def _record_frame(self, force: bool = False) -> None:
        if not self.record_frames:
            return
        moved = any(
            agent.id not in self._last_pos
            or math.dist(self._last_pos[agent.id], (agent.x, agent.y)) >= self.FRAME_MOVE
            for agent in self.agents
        )
        switched = any(self._last_states.get(a.id) != a.state for a in self.agents)
        stale = (self.clock - self._last_frame_clock) >= self.FRAME_MAX_GAP
        if not (force or moved or switched or stale):
            return

        self.frames.append({
            "t": round((self.clock - self._start_minutes()) * 60, 1),
            "clock": self.clock_str(),
            "a": {a.id: a.snapshot() for a in self.agents},
        })
        self._last_pos = {a.id: (a.x, a.y) for a in self.agents}
        self._last_states = {a.id: a.state for a in self.agents}
        self._last_frame_clock = self.clock

    def trace(self) -> dict[str, Any]:
        return {
            "meta": {
                "map": self.map.meta["id"],
                "map_name": self.map.meta["name"],
                "map_file": str(self.map_path),
                "start_clock": self.start_clock,
                "end_clock": self.clock_str(),
                "duration_game_seconds": round((self.clock - self._start_minutes()) * 60, 1),
                "frames": len(self.frames),
                "brain": getattr(self.brain, "name", "unknown"),
                "seed": self.seed,
                "note": "由 tools/simulate.py 生成。RuleBrain 是占位大脑，M2 会换成 LLM 认知层。",
            },
            "agents": [
                {
                    "id": a.id, "name": a.name, "role": a.role,
                    "identity": a.persona.get("identity", ""),
                    "goal": a.persona.get("goal", ""),
                    "traits": a.persona.get("traits", []),
                    "speech_style": a.persona.get("speech_style", ""),
                    "look": a.look,
                    "speed": a.speed,
                    "walked_m": round(a.walked, 1),
                    "arrivals": len(a.arrivals),
                }
                for a in self.agents
            ],
            "frames": self.frames,
            "events": self.events,
        }

    def write_trace(self, out: str | Path) -> Path:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.trace(), fh, ensure_ascii=False, separators=(",", ":"))
        return path
