"""寻路与世界引擎的回归测试。

这些是确定性代码——正是最值得钉住的部分。LLM 认知层接上来之后，
寻路仍然必须是对的，否则很难分辨"智能体行为怪"是大脑的问题还是腿的问题。

    /opt/anaconda3/envs/Agent/bin/python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.engine import Engine                                    # noqa: E402
from world.map_loader import load_map, rasterize                 # noqa: E402
from world.pathfinding import find_path, has_line_of_sight, nearest_walkable  # noqa: E402

MAP = ROOT / "world/maps/sem_1f.json"


@pytest.fixture(scope="module")
def world():
    m = load_map(MAP)
    return m, rasterize(m)


def room_center(m, rid):
    r = next(x for x in m.rooms if x.id == rid)
    return r.rect.center()


# ---------------------------------------------------------------- 基础

def test_南门到大厅可达(world):
    m, grid = world
    door = (54, 66)
    path = find_path(grid, door, room_center(m, "rm_lobby"))
    assert path, "从南门进大厅应该找得到路"
    assert grid.is_walkable(*path[-1])


def test_路径每一段都不穿墙(world):
    """关键回归：路径抽稀之后，任意相邻途经点之间必须全程可走。"""
    m, grid = world
    cases = [
        ((54, 74), room_center(m, "rm_classroom")),
        ((54, 74), room_center(m, "rm_office_a")),
        ((54, 74), room_center(m, "rm_restroom")),
        (room_center(m, "rm_coffee"), room_center(m, "rm_classroom")),
        (room_center(m, "rm_seminar"), room_center(m, "rm_stairs")),
        (room_center(m, "rm_lobby"), (54, 82)),
    ]
    for start, goal in cases:
        path = find_path(grid, start, goal)
        assert path, f"{start} → {goal} 应该有路"
        anchor = nearest_walkable(grid, *start)
        for pt in path:
            assert grid.is_walkable(*pt), f"途经点 {pt} 落在不可通行格上"
            assert has_line_of_sight(grid, anchor, pt), f"{anchor} → {pt} 之间穿过了障碍"
            anchor = pt
        assert math.dist(anchor, nearest_walkable(grid, *goal)) < 1.5


def test_目标落在障碍里会自动吸附(world):
    m, grid = world
    bookshelf = next(f for f in m.furniture if f.category == "bookshelf")
    inside = bookshelf.rect.center()
    assert not grid.is_walkable(*inside), "测试前提：书架应该是实体"
    snapped = nearest_walkable(grid, *inside)
    assert snapped is not None and grid.is_walkable(*snapped)
    path = find_path(grid, (54, 74), inside)
    assert path and grid.is_walkable(*path[-1])


def test_起点终点相同返回单点(world):
    _, grid = world
    p = find_path(grid, (54, 74), (54.2, 74.2))
    assert p is not None and len(p) == 1


def test_图外目标不会崩(world):
    _, grid = world
    # 图外会被 to_tile 夹到边界，可能无解也可能吸附到边界草地，但绝不能抛异常
    find_path(grid, (54, 74), (-50, -50))
    find_path(grid, (54, 74), (9999, 9999))


# ---------------------------------------------------------------- 引擎

def test_引擎可复现():
    a = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=42)
    b = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=42)
    a.run(20)
    b.run(20)
    assert [x["t"] for x in a.frames] == [x["t"] for x in b.frames]
    assert [f["a"] for f in a.frames] == [f["a"] for f in b.frames]


def test_六个智能体都按课表出发并到达():
    eng = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=42)
    eng.run(40)
    events = eng.events
    assert not [e for e in events if e["type"] == "error"], "不应有寻路错误"
    for agent in eng.agents:
        departs = [e for e in events if e.get("agent") == agent.id and e["type"] == "depart"]
        assert departs, f"{agent.name} 一次都没出发"
        assert agent.arrivals, f"{agent.name} 一次都没到达"
        assert agent.walked > 5, f"{agent.name} 几乎没走动"
    # 不变量：正在 talking 的人，必须确实是某个进行中对话的参与者。
    # （跑完时正好有话没聊完是正常的；永久卡住才是 bug。）
    active = set()
    for s in eng.dialogues.sessions:
        active |= set(s.participants())
    stuck = [a.name for a in eng.agents if a.state == "talking" and a.id not in active]
    assert not stuck, f"卡在 talking 却不在任何对话里：{stuck}"
    assert eng.dialogue_log, "跑了 40 游戏分钟，总该有人聊过天"


def test_智能体不会走进墙里():
    eng = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=42)
    eng.run(40)
    bad = []
    for frame in eng.frames:
        for aid, snap in frame["a"].items():
            if not eng.grid.is_walkable(snap[0], snap[1]):
                bad.append((frame["t"], aid, snap))
    assert not bad, f"轨迹里有 {len(bad)} 个不可通行的点，例如 {bad[:3]}"


def test_相邻两帧不会横穿墙体():
    """自适应采样的目的就是这个：帧间距必须远小于最薄的障碍。"""
    eng = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=42)
    eng.run(40)
    worst = 0.0
    for f0, f1 in zip(eng.frames, eng.frames[1:]):
        for aid, a in f0["a"].items():
            b = f1["a"][aid]
            worst = max(worst, math.dist(a[:2], b[:2]))
    assert worst < 2.0, f"最大帧间距 {worst:.2f}m，可能横穿 2m 厚的墙"


def test_轨迹结构完整(tmp_path):
    eng = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=42)
    eng.run(10)
    out = eng.write_trace(tmp_path / "trace.json")
    import json
    tr = json.loads(out.read_text(encoding="utf-8"))
    assert tr["meta"]["map"] == "sem_1f"
    assert tr["meta"]["brain"] == "rule"
    assert len(tr["agents"]) == 6
    assert tr["frames"] and set(tr["agents"][0]) >= {"id", "name", "role", "look"}
    first = tr["frames"][0]["a"]
    assert set(first) == {a["id"] for a in tr["agents"]}
    assert all(len(v) == 4 for v in first.values())


# ---------------------------------------------------------------- 世界服务

def test_世界服务指令():
    """把服务层的指令路径也钉住——这是浏览器唯一能碰到世界的入口。"""
    from server.world_service import WorldService

    eng = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=7)
    svc = WorldService(eng, speed=8.0)

    assert svc.hello()["agents"], "hello 应带上人设"
    assert svc.apply({"cmd": "set_speed", "value": 30})["speed"] == 30
    assert svc.apply({"cmd": "set_paused", "value": True})["paused"] is True
    assert svc.apply({"cmd": "toggle_paused"})["paused"] is False

    # 派一个智能体去咖啡角，然后推进，应当真的走到
    coffee = next(r for r in eng.map.rooms if r.id == "rm_coffee")
    cx, cy = coffee.rect.center()
    res = svc.apply({"cmd": "dispatch", "agent": "stu_01", "x": cx, "y": cy, "why": "测试派发"})
    assert res["ok"], res["message"]
    agent = eng.agent_by_id("stu_01")
    assert agent.state == "walking" and agent.path
    assert agent.goal["kind"] == "manual" and agent.goal["room"] == "rm_coffee"

    # 注意：路上碰到人可能聊起来把人冻住，所以要驱动对话把人放出来；
    # 判断也用"到达记录"而不是 state——到了之后可能正聊着。
    import asyncio

    async def drive(seconds: int) -> None:
        for _ in range(seconds):
            svc._advance(1)
            await eng.advance_dialogues()

    asyncio.run(drive(300))
    assert any(e.get("room") == "rm_coffee" for e in agent.arrivals), \
        f"应该走到过咖啡角，实际到达：{[e.get('room') for e in agent.arrivals]}"

    # 叫停
    svc._advance(1)
    svc.apply({"cmd": "dispatch", "agent": "stu_02", "x": cx, "y": cy})
    assert eng.agent_by_id("stu_02").state == "walking"
    assert svc.apply({"cmd": "stop", "agent": "stu_02"})["ok"]
    assert eng.agent_by_id("stu_02").state == "idle"

    # 错误分支要返回 ok=False 而不是抛异常
    assert svc.apply({"cmd": "dispatch", "agent": "不存在", "x": 1, "y": 1})["ok"] is False
    assert svc.apply({"cmd": "dispatch", "agent": "stu_01", "x": [1, 2], "y": 3})["ok"] is False
    assert svc.apply({"cmd": "dispatch", "agent": "stu_01"})["ok"] is False
    assert svc.apply({"cmd": "dispatch_nearest", "x": None, "y": "abc"})["ok"] is False
    assert svc.apply({"cmd": "set_speed", "value": 0})["ok"] is False
    assert svc.apply({"cmd": "goto_room", "agent": "stu_01", "room": "不存在"})["ok"] is False
    assert svc.apply({"cmd": "胡搞"})["ok"] is False

    # 派最近的
    svc.apply({"cmd": "stop", "agent": "stu_03"})
    r = svc.apply({"cmd": "dispatch_nearest", "x": cx, "y": cy})
    assert r["ok"] and r["agent"]


def test_快照结构():
    eng = Engine(map_path=MAP, personas_dir=ROOT / "agents/personas", start_clock="07:40", seed=3)
    eng.run(1)
    snap = eng.snapshot()
    assert set(snap) >= {"clock", "t", "agents"}
    assert len(snap["agents"]) == 6
    a = snap["agents"][0]
    assert set(a) >= {"id", "name", "role", "x", "y", "f", "m", "state", "path"}
    assert 0 <= a["f"] <= 3


def test_websocket_握手与指令():
    """浏览器唯一的数据通道：握手要给人设，指令要有 ack。"""
    from fastapi.testclient import TestClient

    from server.app import create_app

    app = create_app(speed=8.0)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["ok"] is True
        hello = client.get("/api/hello").json()
        assert hello["type"] == "hello" and len(hello["agents"]) == 6
        assert "look" in hello["agents"][0]

        snap = client.get("/api/state").json()
        assert snap["type"] == "state" and len(snap["agents"]) == 6

        with client.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            assert first["type"] == "hello"
            assert first["agents"], "握手应带人设"
            seen_state = False
            for _ in range(5):
                msg = ws.receive_json()
                if msg["type"] == "state":
                    seen_state = True
                    break
            assert seen_state, "握手后应立即推一帧状态"

            ws.send_json({"cmd": "set_speed", "value": 30})
            got = None
            for _ in range(40):
                msg = ws.receive_json()
                if msg["type"] == "ack":
                    got = msg
                    break
            assert got and got["ok"] and got["speed"] == 30

            ws.send_json({"cmd": "dispatch", "agent": "stu_01", "x": 78, "y": 46, "why": "测试"})
            for _ in range(40):
                msg = ws.receive_json()
                if msg["type"] == "ack" and msg.get("agent") == "stu_01":
                    assert msg["ok"]
                    break
            else:
                raise AssertionError("派发没有收到 ack")

        # 错误指令走 REST 应返回 400 而不是 500
        bad = client.post("/api/command", json={"cmd": "胡搞"})
        assert bad.status_code == 400 and bad.json()["ok"] is False
