"""常驻世界服务：实时推进引擎、广播状态、接收指令。

这是设计文档 §4 里"世界服务层"的最小实现：把一个原本跑完就退出的
批处理引擎，包成一个常驻的异步服务。

时间与步长
----------
真实时间与游戏时间的换算：`游戏秒 = 真实秒 × speed`。
每个 tick 把这段游戏时间**细分到不超过 1 游戏秒**再喂给引擎——
这样既有连续平滑的移动（不像批处理那样一次跳 1 秒），
又不会因为一次卡顿攒下几十秒而让世界"暴走"。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from sim.engine import Engine

# 可选的时间流速（游戏秒 / 真实秒）
SPEEDS = [1, 2, 4, 8, 16, 60]
# 单次 tick 最多补多少游戏秒（防止挂起后一次性推进过多）
MAX_CATCHUP = 5.0


def _coord(cmd: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return float(cmd["x"]), float(cmd["y"])
    except (KeyError, TypeError, ValueError):
        return None


class WorldService:
    def __init__(self, engine: Engine, speed: float = 8.0, tick_hz: float = 20.0) -> None:
        self.engine = engine
        self.speed = float(speed)
        self.tick_hz = float(tick_hz)
        self.paused = False

        self._subs: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._running = False
        self._last = 0.0
        self._event_cursor = 0
        self._last_flush = 0.0
        self.stats = {"steps": 0, "ticks": 0, "started_at": time.time()}

    # ------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last = time.perf_counter()
        self._task = asyncio.create_task(self._run(), name="world-loop")
        # 对话单独一条任务：真实 LLM 一轮要 1–3 秒，
        # 塞进 tick 循环会把整个世界一起卡住
        self._dlg_task = asyncio.create_task(self._dialogue_loop(), name="dialogue-loop")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            for task in (self._task, getattr(self, "_dlg_task", None)):
                if task is None:
                    continue
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._task = None
            self._dlg_task = None
        try:
            self.engine.flush()                    # 退出前把最后一笔写掉
            if getattr(self.engine, "store", None) is not None:
                self.engine.store.close()
        except Exception as exc:                   # noqa: BLE001
            print(f"关停落盘失败：{exc!r}")

    async def _run(self) -> None:
        interval = 1.0 / self.tick_hz
        while self._running:
            await asyncio.sleep(interval)
            now = time.perf_counter()
            dt_real = now - self._last
            self._last = now
            self.stats["ticks"] += 1

            if not self.paused and dt_real > 0:
                self._advance(min(dt_real * self.speed, MAX_CATCHUP))
                self._flush_events()
            if now - self._last_flush > 3.0:       # 定期落盘，别攒着
                self._last_flush = now
                try:
                    self.engine.flush()
                except Exception as exc:           # noqa: BLE001
                    print(f"落盘失败：{exc!r}")
            await self._broadcast_state()

    async def _dialogue_loop(self) -> None:
        """独立推进对话。await 的是线程里的模型调用，世界循环不受影响。"""
        while self._running:
            try:
                await self._dialogue_tick()
            except Exception as exc:                      # noqa: BLE001
                print(f"对话推进异常：{exc!r}")
            await asyncio.sleep(0.12)

    async def _dialogue_tick(self) -> None:
        """一轮对话推进。抽出来是为了能单独测——暂停时**绝不能**跑看门狗。

        暂停时对话本来就不会推进，停滞时间必然超时；如果看门狗照跑，
        一按暂停就会把所有正在进行的对话误杀掉。
        """
        if self.paused:
            return
        if self.engine.dialogues.sessions:
            await self.engine.advance_dialogues()
        self.engine.reap_stalled_dialogues()
        await self.engine.advance_reflections()

    def _advance(self, game_seconds: float) -> None:
        """把 game_seconds 游戏时间细分推进，单步不超过 1 游戏秒。"""
        remaining = game_seconds
        while remaining > 1e-6:
            piece = min(remaining, 1.0)
            self.engine.step(piece)
            self.stats["steps"] += 1
            remaining -= piece

    def _flush_events(self) -> None:
        new = self.engine.events[self._event_cursor:]
        if not new:
            return
        self._event_cursor = len(self.engine.events)
        self._publish({"type": "events", "events": new})

    # ------------------------------------------------------------ 广播

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def _publish(self, msg: dict[str, Any]) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # 客户端跟不上就丢最旧的一帧，绝不阻塞世界
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    async def _broadcast_state(self) -> None:
        if not self._subs:
            return
        snap = self.engine.snapshot()
        self._publish({
            "type": "state",
            "paused": self.paused,
            "speed": self.speed,
            **snap,
        })

    # ------------------------------------------------------------ 指令

    def hello(self) -> dict[str, Any]:
        return {
            "map": self.engine.map.meta["id"],
            "map_name": self.engine.map.meta["name"],
            "start_clock": self.engine.start_clock,
            "speed": self.speed,
            "speeds": SPEEDS,
            "tick_hz": self.tick_hz,
            "paused": self.paused,
            "brain": getattr(self.engine.brain, "name", "unknown"),
            # 让后连上来的客户端也能看到刚才发生了什么，而不是空白的"动态"
            "recent_events": self.engine.events[-15:],
            "clock": self.engine.clock_str(),
            "agents": [
                {
                    "id": a.id, "name": a.name, "role": a.role,
                    "identity": a.persona.get("identity", ""),
                    "goal": a.persona.get("goal", ""),
                    "traits": a.persona.get("traits", []),
                    "speech_style": a.persona.get("speech_style", ""),
                    "look": a.look, "speed": a.speed,
                }
                for a in self.engine.agents
            ],
        }

    def apply(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """同步执行一条指令。REST 与 WebSocket 走的是同一条路径，便于测试。

        这是对外接口，**绝不能因为输入畸形就抛异常**——一律返回 ok=False。
        """
        try:
            return self._apply(cmd)
        except Exception as exc:                       # noqa: BLE001
            return self._err(f"指令执行失败：{exc}")

    def _apply(self, cmd: dict[str, Any]) -> dict[str, Any]:
        kind = (cmd or {}).get("cmd")

        if kind == "set_paused":
            self.paused = bool(cmd.get("value", True))
            return self._ok("已暂停" if self.paused else "已继续")

        if kind == "toggle_paused":
            self.paused = not self.paused
            return self._ok("已暂停" if self.paused else "已继续")

        if kind == "set_speed":
            v = float(cmd.get("value", self.speed))
            if v <= 0:
                return self._err("速度必须为正")
            self.speed = v
            return self._ok(f"时间流速 {v:g}×")

        if kind == "dispatch":
            agent = self.engine.agent_by_id(cmd.get("agent", ""))
            if agent is None:
                return self._err(f"没有这个智能体：{cmd.get('agent')}")
            pt = _coord(cmd)
            if pt is None:
                return self._err("x / y 必须是数字")
            x, y = pt
            why = cmd.get("why") or "你让他过去"
            if self.engine.send_to_point(agent, x, y, why):
                return self._ok(f"{agent.name} 出发了", agent=agent.id)
            return self._err(f"{agent.name} 过不去那里")

        if kind == "dispatch_nearest":
            pt = _coord(cmd)
            if pt is None:
                return self._err("x / y 必须是数字")
            x, y = pt
            agent = self.engine.nearest_agent(x, y)
            if agent is None:
                return self._err("场上没有智能体")
            if self.engine.send_to_point(agent, x, y, cmd.get("why") or "被派往此处"):
                return self._ok(f"派了最近的 {agent.name} 过去", agent=agent.id)
            return self._err("最近的智能体也过不去")

        if kind == "stop":
            agent = self.engine.agent_by_id(cmd.get("agent", ""))
            if agent is None:
                return self._err("没有这个智能体")
            self.engine.stop_agent(agent)
            return self._ok(f"{agent.name} 停下了", agent=agent.id)

        if kind == "goto_room":
            agent = self.engine.agent_by_id(cmd.get("agent", ""))
            room = next((r for r in self.engine.map.rooms if r.id == cmd.get("room")), None)
            if agent is None or room is None:
                return self._err("智能体或房间不存在")
            cx, cy = room.rect.center()
            if self.engine.send_to_point(agent, cx, cy, cmd.get("why") or f"去{room.name}"):
                return self._ok(f"{agent.name} 去{room.name}", agent=agent.id)
            return self._err(f"{agent.name} 去不了{room.name}")

        return self._err(f"未知指令：{kind}")

    def _ok(self, message: str, **extra: Any) -> dict[str, Any]:
        return {"ok": True, "message": message, "paused": self.paused, "speed": self.speed, **extra}

    def _err(self, message: str) -> dict[str, Any]:
        return {"ok": False, "message": message, "paused": self.paused, "speed": self.speed}
