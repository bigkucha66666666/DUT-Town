"""FastAPI 应用：静态资源 + REST + WebSocket。

    python -m server            # 起在 http://127.0.0.1:8123/web/

浏览器连上 /ws 之后就是**实时世界**：服务端常驻推进引擎，
前端只渲染收到的状态。连不上就自动退回回放模式（见 web/js/live.js），
所以 tools/serve.py 的纯静态模式仍然可用。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from llm.gateway import LLMGateway
from storage.store import Store
from server.world_service import WorldService
from sim.engine import Engine

ROOT = Path(__file__).resolve().parents[1]


def create_app(
    map_path: str | Path = "world/maps/sem_1f.json",
    personas_dir: str | Path = "agents/personas",
    start_clock: str = "07:40",
    speed: float = 8.0,
    seed: int = 42,
    config_path: str | Path = "config.json",
    save_path: str | Path | None = None,      # 默认不落盘：由调用方决定，测试才不会写进真实存档
    resume: bool = True,
    provider: str | None = None,
) -> FastAPI:
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = ROOT / cfg_file
    full_cfg: dict[str, Any] = {}
    if cfg_file.exists():
        import json as _json
        full_cfg = _json.loads(cfg_file.read_text(encoding="utf-8"))
    gateway = LLMGateway(full_cfg.get("llm", {}))
    if provider:
        gateway.switch(provider)
    store = None
    if save_path:
        sp = Path(save_path)
        store = Store(sp if sp.is_absolute() else ROOT / sp)
        if not resume:
            store.conn.executescript(
                "DELETE FROM memories; DELETE FROM trait_changes; DELETE FROM relations;"
                " DELETE FROM relation_changes; DELETE FROM dialogues; DELETE FROM events;")
            store.conn.commit()
    budget_cfg = {"budget": full_cfg.get("budget", {})}

    engine = Engine(
        map_path=ROOT / map_path if not Path(map_path).is_absolute() else map_path,
        personas_dir=ROOT / personas_dir if not Path(personas_dir).is_absolute() else personas_dir,
        start_clock=start_clock,
        seed=seed,
        gateway=gateway,
        llm_cfg=budget_cfg,
        store=store,
        record_frames=False,          # 实时模式不攒历史帧，只推快照
    )
    service = WorldService(engine, speed=speed)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        print(f"世界服务已启动：{engine.map.meta['name']}，"
              f"{len(engine.agents)} 个智能体，{start_clock} 起，{speed:g}× 流速，"
              f"LLM={gateway.provider_name}")
        if gateway.provider_name != "mock":
            print(f"⚠ 正在用真实模型（{gateway.provider_name}），每一次对话都会花钱。")
        yield
        await service.stop()

    app = FastAPI(title="DUT 小镇 · 世界服务", lifespan=lifespan)
    app.state.service = service
    app.state.engine = engine

    # ---------------------------------------------------------- REST

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "map": engine.map.meta["id"],
            "agents": len(engine.agents),
            "clock": engine.clock_str(),
            "paused": service.paused,
            "speed": service.speed,
            "subscribers": len(service._subs),
            **service.stats,
        }

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return {"type": "state", "paused": service.paused, "speed": service.speed, **engine.snapshot()}

    @app.get("/api/store")
    def store_stats() -> dict[str, Any]:
        return store.stats() if store else {"enabled": False}

    @app.get("/api/timeline")
    def timeline(agent: str | None = None) -> dict[str, Any]:
        if not store:
            return {"agents": {}, "enabled": False}
        out: dict[str, Any] = {}
        for a in engine.agents:
            if agent and a.id != agent:
                continue
            rows = store.trait_timeline(a.id)
            out[a.id] = {
                "name": a.name,
                "current": a.personality.snapshot(),
                "series": rows,
            }
        return {"agents": out, "enabled": True}

    @app.get("/api/trail")
    def trail(agent: str, day: int | None = None) -> dict[str, Any]:
        """某个智能体的空间轨迹。这就是"她今天走了哪些路"。"""
        a = engine.agent_by_id(agent)
        if a is None:
            return {"ok": False, "message": f"没有这个智能体：{agent}"}
        return {"ok": True, "agent": agent, "name": a.name,
                "trail": engine.trail(agent), "clock": engine.clock_str()}

    @app.get("/api/llm")
    def llm_stats() -> dict[str, Any]:
        return {"provider": gateway.provider_name, **gateway.summary()}

    @app.get("/api/memory")
    def memory_dump(agent: str | None = None, limit: int = 40) -> dict[str, Any]:
        out = {}
        for a in engine.agents:
            if agent and a.id != agent:
                continue
            out[a.id] = {
                "name": a.name,
                "personality": a.personality.snapshot(),
                "describe": a.personality.describe(),
                "stats": a.memory.stats(),
                "recent": [m.to_dict() for m in a.memory.recent(limit)][::-1],
                "trait_history": [c.to_dict() for c in a.personality.history][-20:],
            }
        return {"agents": out, "relations": engine.relations.to_dict(),
                "dialogues": engine.dialogue_log[-10:]}

    @app.get("/api/hello")
    def hello() -> dict[str, Any]:
        return {"type": "hello", **service.hello()}

    @app.post("/api/command")
    async def command(cmd: dict[str, Any]) -> JSONResponse:
        result = service.apply(cmd)
        return JSONResponse(result, status_code=200 if result["ok"] else 400)

    # ---------------------------------------------------------- WebSocket

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = service.subscribe()
        await websocket.send_json({"type": "hello", **service.hello()})
        await websocket.send_json({
            "type": "state", "paused": service.paused, "speed": service.speed, **engine.snapshot(),
        })

        async def pump() -> None:
            while True:
                msg = await queue.get()
                await websocket.send_json(msg)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                data = await websocket.receive_json()
                if isinstance(data, dict) and "cmd" in data:
                    res = service.apply(data)
                    await websocket.send_json({"type": "ack", **res})
        except WebSocketDisconnect:
            pass
        except Exception as exc:                      # noqa: BLE001 - 客户端出错不该拖垮服务
            print(f"WebSocket 异常：{exc!r}")
        finally:
            pump_task.cancel()
            service.unsubscribe(queue)

    # ---------------------------------------------------------- 静态资源

    @app.middleware("http")
    async def no_store(request, call_next):
        """开发期禁缓存。StaticFiles 默认不发 Cache-Control，浏览器会按
        Last-Modified 做启发式缓存——改完 JS 刷新看到的还是旧的，
        排查时会白白怀疑代码。"""
        resp = await call_next(request)
        if request.url.path.startswith(("/web", "/world", "/saves")):
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    app.mount("/web", StaticFiles(directory=ROOT / "web", html=True), name="web")
    app.mount("/world", StaticFiles(directory=ROOT / "world"), name="world")
    app.mount("/saves", StaticFiles(directory=ROOT / "saves"), name="saves")

    @app.get("/")
    def index() -> RedirectResponse:
        return RedirectResponse("/web/")

    return app
