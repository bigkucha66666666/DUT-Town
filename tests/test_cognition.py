"""认知层回归测试：记忆检索、人格漂移约束、相遇对话写回。

这些是"LLM 换进来之后也必须成立"的不变量——所以要用确定性的 Mock
把它们钉死，否则以后分不清"行为怪"是大脑的问题还是这层地基的问题。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.memory import MemoryStream, relevance                       # noqa: E402
from agents.traits import MAX_STEP, Personality, RelationBook            # noqa: E402
from llm.gateway import LLMGateway, extract_json, validate_schema        # noqa: E402
from sim.engine import Engine                                            # noqa: E402

MAP = ROOT / "world/maps/sem_1f.json"
PERSONAS = ROOT / "agents/personas"


# ---------------------------------------------------------------- 网关

def test_schema_校验能抓出问题():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert validate_schema({"a": "x"}, schema) == []
    assert validate_schema({}, schema)
    assert validate_schema({"a": 1}, schema)
    assert validate_schema("不是对象", schema)


def test_能从模型的废话里抠出_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的，这是结果：{"a": 2} 希望有用') == {"a": 2}
    assert extract_json("完全不是 JSON") is None


def test_网关重试与降级():
    class Bad:
        name = "bad"
        def __init__(self): self.n = 0
        def complete(self, *a, **k):
            from llm.providers import Completion
            self.n += 1
            return Completion(text="这不是 JSON")

    gw = LLMGateway({"provider": "mock"})
    gw.provider = Bad()
    res = gw.chat([{"role": "user", "content": "x"}],
                  schema={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]})
    assert res.ok is False and res.error, "坏输出应当降级为 ok=False 而不是抛异常"
    assert gw.provider.n == 3, "应当重试 max_retries 次"


def test_缓存命中():
    gw = LLMGateway({"provider": "mock"})
    kw = dict(schema={"type": "object", "properties": {"importance": {"type": "integer"}}, "required": ["importance"]},
              purpose="memory.importance", meta={"agent": "a", "text": "被老师表扬了"})
    a = gw.chat([{"role": "user", "content": "q"}], **kw)
    b = gw.chat([{"role": "user", "content": "q"}], **kw)
    assert a.ok and b.cached and gw.stats["cache_hits"] == 1


# ---------------------------------------------------------------- 记忆

def test_记忆检索三因子():
    ms = MemoryStream("stu_01")
    ms.add("我在图书馆自习了一下午", kind="observation", importance=3, clock=100)
    ms.add("王老师当众表扬了我的作业", kind="event", importance=9, clock=101)
    ms.add("在走廊碰到郭佳仑，聊了两句", kind="dialogue", importance=4, clock=102)

    top = ms.retrieve(query="王老师 作业", k=1, now=102)
    assert "表扬" in top[0].text, "相关性高的记忆应当排第一"

    # 时间久远会让权重衰减
    ms2 = MemoryStream("x")
    ms2.add("很久以前的小事", importance=5, clock=0)
    ms2.add("刚才的小事", importance=5, clock=2000)
    assert "刚才" in ms2.retrieve(query="小事", k=1, now=2000)[0].text


def test_记忆重要度启发式有区分度():
    ms = MemoryStream("a")
    assert ms.importance_of("我走过了走廊") < ms.importance_of("王老师当众表扬了我的作业")
    assert 1 <= ms.importance_of("随便一件事") <= 10


def test_关于某人_与反思阈值():
    ms = MemoryStream("a")
    ms.add("和李子雨一起做小组作业", subject="李子雨", importance=5, clock=10)
    ms.add("无关的事", subject="图书馆", importance=5, clock=11)
    assert len(ms.about("李子雨")) == 1
    assert ms.importance_since_reflection() == 10
    ms.mark_reflected()
    assert ms.importance_since_reflection() == 0


# ---------------------------------------------------------------- 人格

def test_人格漂移必须有出处():
    p = Personality.from_persona({"id": "a", "traits": ["内向"]})
    with pytest.raises(ValueError):
        p.nudge("social", 0.05, "随便变变", "magic", 10)          # 非法来源
    with pytest.raises(ValueError):
        p.nudge("social", 0.05, "", "dialogue", 10)               # 没理由
    with pytest.raises(KeyError):
        p.nudge("不存在", 0.05, "x", "dialogue", 10)


def test_单次漂移有上限且不会越界():
    p = Personality.from_persona({"id": "a"})
    before = p.values["social"]
    ch = p.nudge("social", 0.9, "一次极端经历", "dialogue", 10)
    assert ch is not None
    assert abs(ch.new - before) <= MAX_STEP + 1e-9, "单次变化必须被 MAX_STEP 卡住"
    for _ in range(200):
        p.nudge("social", 0.9, "反复冲击", "reflection", 10)
    assert p.values["social"] <= 0.95, "不能漂到极端"


def test_人格历史可追溯():
    p = Personality.from_persona({"id": "a"})
    p.nudge("diligence", 0.03, "连着三天去自习", "reflection", 100)
    p.nudge("social", -0.02, "答疑时被怼了", "dialogue", 130)
    assert len(p.history) == 2
    assert all(c.reason and c.source in {"dialogue", "reflection"} for c in p.history)
    assert [c.clock for c in p.history] == [100, 130]


def test_关系是对称的():
    rb = RelationBook()
    rb.update("a", "b", 10, "聊了一次", familiarity=0.1, affinity=0.05)
    assert rb.get("b", "a").familiarity == pytest.approx(0.1)
    rb.update("b", "a", 20, "又聊了一次", familiarity=0.1)
    assert rb.get("a", "b").familiarity == pytest.approx(0.2)
    assert len(rb.history) == 2
    assert rb.neighbors("a")[0][0] == "b"


# ---------------------------------------------------------------- 相遇对话

class AlwaysTalk:
    """确定性的 rng：让"越外向越容易开口"的概率门控必然通过。
    否则按 seed 偶发不开话，测试就会随机红。"""

    def random(self) -> float:
        return 0.0



def finish_dialogue(eng, max_game_seconds: int = 600) -> int:
    """把进行中的对话跑完。

    `advance_dialogues()` **一次只推进一轮**（按游戏时钟节流），
    所以必须和 `eng.step()` 交替驱动——只调它一次是聊不完的。
    """
    async def go() -> int:
        done = 0
        for _ in range(max_game_seconds):
            eng.step(1.0)
            done += await eng.advance_dialogues()
            if done and not eng.dialogues.sessions:
                break
        return done
    return asyncio.run(go())

def make_engine(seed=5, **budget):
    gw = LLMGateway({"provider": "mock"})
    cfg = {"dialogue_engage_radius_m": 2.6, "dialogue_cooldown_minutes": 25,
           "dialogue_max_turns": 4, "dialogue_turn_seconds": 3.0}
    cfg.update(budget)
    return Engine(map_path=MAP, personas_dir=PERSONAS, start_clock="07:40", seed=seed,
                  gateway=gw, llm_cfg={"budget": cfg}, record_frames=False), gw


def run_dialogue(eng, seconds=200):
    async def go():
        for _ in range(seconds):
            eng.step(1.0)
            await eng.advance_dialogues()
            if len(eng.dialogue_log) >= 2:
                break
    asyncio.run(go())


def test_靠近才会触发对话():
    eng, _ = make_engine()
    for i, a in enumerate(eng.agents):        # 全部集中到走廊
        a.x, a.y = 30.0 + i * 0.5, 35.0
    run_dialogue(eng)
    assert eng.dialogue_log, "站在一起应该聊起来"
    assert eng.relations.history, "聊完关系状态应当变化"


def test_离得远不会触发对话():
    eng, _ = make_engine()
    positions = [(10, 10), (60, 20), (90, 50), (20, 70), (80, 70), (40, 50)]
    for a, (x, y) in zip(eng.agents, positions):
        a.x, a.y = float(x), float(y)
    run_dialogue(eng, seconds=120)
    assert not eng.dialogue_log, "天各一方不该凭空聊起来"


def test_对话写回双方记忆且摘要一致():
    eng, _ = make_engine(seed=3)
    for i, a in enumerate(eng.agents[:2]):
        a.x, a.y = 30.0 + i * 0.6, 35.0
    run_dialogue(eng, seconds=200)
    assert eng.dialogue_log, "应当发生过对话"
    d = eng.dialogue_log[0]
    a, b = eng.agent_by_id(d["a"]), eng.agent_by_id(d["b"])
    for ag, other in ((a, b), (b, a)):
        hits = [m for m in ag.memory.items if m.kind == "dialogue"]
        assert hits, f"{ag.name} 应当留下对话记忆"
        assert any(other.name in m.text for m in hits), "记忆里应当提到对方"
        assert other.name in hits[-1].subject or other.name in hits[-1].text


def test_每一次人格漂移都有出处且来自经历():
    eng, _ = make_engine(seed=9)
    for i, a in enumerate(eng.agents):
        a.x, a.y = 30.0 + i * 0.5, 35.0
    run_dialogue(eng)
    changes = [c for ag in eng.agents for c in ag.personality.history]
    assert changes, "聊完应当有人格变化"
    for c in changes:
        assert c.source in {"dialogue", "reflection"}
        assert c.reason, "每一处漂移都要写清为什么"
        assert abs(c.new - c.old) <= MAX_STEP + 1e-9


def test_聊天的人站着不动其他人照常走():
    """对话绝不能把整个世界冻住——这是异步推进的全部理由。"""
    eng, _ = make_engine(seed=4)
    for i, a in enumerate(eng.agents[:2]):
        a.x, a.y = 30.0 + i * 0.6, 35.0
    eng.step(1.0)
    eng.dialogues.maybe_start(eng.agents, eng.clock, room_name="走廊（东西主轴）", rng=AlwaysTalk())
    assert eng.dialogues.sessions, "应当开起一次对话"
    talking = {s.participants()[0] for s in eng.dialogues.sessions} | \
              {s.participants()[1] for s in eng.dialogues.sessions}

    x0 = {a.id: (a.x, a.y) for a in eng.agents}
    clock0 = eng.clock
    for _ in range(30):
        eng.step(1.0)
    for aid in talking:
        assert (eng.agent_by_id(aid).x, eng.agent_by_id(aid).y) == x0[aid], "聊天的人不该移动"
    assert eng.clock > clock0 + 0.4, "时钟必须继续走"


def test_对话中途模型挂了不会僵在原地():
    eng, gw = make_engine(seed=6)
    for i, a in enumerate(eng.agents[:2]):
        a.x, a.y = 30.0 + i * 0.6, 35.0

    class Dead:
        name = "dead"
        def complete(self, *a, **k):
            raise RuntimeError("模型挂了")

    eng.dialogue_log.clear()
    eng.dialogues.gateway.provider = Dead()
    run_dialogue(eng, seconds=30)
    assert eng.dialogues.sessions == [], "模型挂了也要把会话收掉，不能卡住"
    assert any(e["type"] == "dialogue_error" for e in eng.events)


def test_暂停时对话不会被看门狗误杀():
    """真实踩过的坑：暂停后对话无法推进，停滞必然超时，
    看门狗若照跑就会把正在进行的对话全部误杀。"""
    from server.world_service import WorldService

    eng, _ = make_engine(seed=2)
    for i, a in enumerate(eng.agents[:2]):
        a.x, a.y = 30.0 + i * 0.6, 35.0
    eng.step(1.0)
    eng.dialogues.maybe_start(eng.agents, eng.clock, room_name="走廊（东西主轴）", rng=AlwaysTalk())
    assert eng.dialogues.sessions, "应当开起一次对话"

    svc = WorldService(eng, speed=8)
    for s in eng.dialogues.sessions:          # 假装已经停滞很久
        s.last_progress -= 9999

    svc.paused = True
    asyncio.run(svc._dialogue_tick())
    assert eng.dialogues.sessions, "暂停时不该把对话收掉"

    # 恢复运行、且确实推不动时，看门狗才该出手
    svc.paused = False

    async def stuck(session, ctx):
        return False                       # 假装推进失败，且不刷新停滞计时

    eng.dialogues.advance = stuck
    asyncio.run(svc._dialogue_tick())
    assert not eng.dialogues.sessions, "真的卡死时看门狗应当把会话放掉"


# ---------------------------------------------------------------- 反思

def test_反思由累计重要度触发而不是条数():
    from agents.reflect import Reflector

    eng, gw = make_engine(seed=1)
    r = Reflector(gw, {"reflect_importance_threshold": 20, "reflect_min_interval_minutes": 0})
    a = eng.agents[0]
    assert not r.due(a, eng.now_minutes), "一开始不该触发"

    for i in range(3):                       # 三条低重要度的琐事，不该够
        a.memory.add("我走过走廊", kind="observation", importance=2, clock=eng.now_minutes + i)
    assert not r.due(a, eng.now_minutes), "琐事攒不出反思"

    a.memory.add("王老师当众表扬了我", kind="event", importance=15, clock=eng.now_minutes)
    assert r.due(a, eng.now_minutes), "累计重要度够了就该触发"


def test_反思产出洞察并写回记忆且能改人格():
    import asyncio

    from agents.reflect import Reflector

    eng, gw = make_engine(seed=2)
    r = Reflector(gw, {"reflect_importance_threshold": 1, "reflect_min_interval_minutes": 0})
    a = eng.agents[0]
    for i in range(4):
        a.memory.add(f"第 {i+1} 次在办公室门口没敢进去", kind="event",
                     importance=6, clock=eng.now_minutes + i)
    before = dict(a.personality.values)

    events: list[dict] = []
    ctx = eng._dialogue_ctx_stub()
    ctx["on_event"] = events.append
    ref = asyncio.run(r.reflect(a, ctx))

    assert ref is not None and ref.insights, "应当产出洞察"
    kinds = [m.kind for m in a.memory.items]
    assert "reflection" in kinds, "洞察应当写进记忆流"
    assert a.memory.importance_since_reflection() == 0, "反思后阈值应当清零"
    assert any(e["type"] == "reflection" for e in events)
    # 人格变化必须仍然带出处
    for c in a.personality.history:
        assert c.source in {"dialogue", "reflection"} and c.reason


def test_反思有最小间隔不会刷屏():
    from agents.reflect import Reflector

    eng, gw = make_engine(seed=3)
    r = Reflector(gw, {"reflect_importance_threshold": 1, "reflect_min_interval_minutes": 60})
    a = eng.agents[0]
    a.memory.add("很重要的事", kind="event", importance=10, clock=eng.now_minutes)
    assert r.due(a, eng.now_minutes)
    r._last[a.id] = eng.now_minutes
    assert not r.due(a, eng.now_minutes + 30), "间隔没到不该再反思"


# ---------------------------------------------------------------- 落盘

def test_落盘后能恢复记忆与人格(tmp_path):
    from storage.store import Store

    db = tmp_path / "world.db"
    store = Store(db)

    eng, _ = make_engine(seed=4)
    eng.store = store
    for a in eng.agents:
        a.memory.store = store
    a = eng.agents[0]
    a.memory.add("王老师当堂表扬了我的作业", kind="event", importance=9, clock=460)
    a.memory.add("和郭佳仑在走廊聊了几句", kind="dialogue", importance=5, clock=462, subject="郭佳仑")
    a.personality.nudge("social", 0.03, "聊得挺顺", "dialogue", 462)
    eng.relations.update(a.id, eng.agents[1].id, 462, "聊了一次", familiarity=0.1, affinity=0.05)
    eng.flush()

    assert store.stats()["memories"] == 2
    assert store.stats()["trait_changes"] == 1
    assert store.stats()["relations"] == 1

    # 新开一个引擎，从库里恢复
    eng2, _ = make_engine(seed=4)
    eng2.store = store
    for ag in eng2.agents:
        ag.memory.store = store
    loaded = eng2.resume()
    assert loaded["memories"] == 2 and loaded["traits"] >= 1
    revived = eng2.agent_by_id(a.id)
    assert any("表扬" in m.text for m in revived.memory.items), "重启后应当还记得"
    assert revived.personality.values["social"] > 0.2, "人格也应当恢复"
    assert eng2.day == 1, "跨了一个新的一天"
    store.close()


def test_跨天时间戳不会倒流(tmp_path):
    """绝对分钟的作用：第二天的记忆必须比第一天新。"""
    from storage.store import Store

    store = Store(tmp_path / "w.db")
    eng, _ = make_engine(seed=5)
    eng.store = store
    for a in eng.agents:
        a.memory.store = store
    a = eng.agents[0]
    a.memory.add("第一天的事", kind="event", importance=8, clock=eng.now_minutes)
    eng.flush()

    eng2, _ = make_engine(seed=5)
    eng2.store = store
    for ag in eng2.agents:
        ag.memory.store = store
    eng2.resume()
    assert eng2.day == 1
    revived = eng2.agent_by_id(a.id)
    old = max(m.created for m in revived.memory.items)
    assert old < eng2.now_minutes, "昨天的记忆必须比今天早"
    recent = revived.memory.retrieve(query="事", k=1, now=eng2.now_minutes)
    assert recent, "检索仍然能命中旧记忆"
    store.close()


def test_空库启动不该被当成新的一天(tmp_path):
    from storage.store import Store

    store = Store(tmp_path / "empty.db")
    eng, _ = make_engine(seed=8)
    eng.store = store
    for a in eng.agents:
        a.memory.store = store
    loaded = eng.resume()
    assert loaded == {"memories": 0, "traits": 0, "relations": 0}
    assert eng.day == 0, "空库启动应当还是第 0 天"
    store.close()


# ---------------------------------------------------------------- 真实 provider

def test_openai兼容provider的请求与解析():
    """用一个假的 OpenAI 兼容服务端，把"切真模型"这条代码路径先测通。

    否则等你配好 Key 第一次调用时，分不清失败是鉴权问题还是请求写错了。
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # 静音
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(n) or b"{}")
            payload = {
                "choices": [{"message": {"content": json.dumps(
                    {"utterance": "好，我记一下。", "should_end": False, "feeling": "平静"},
                    ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7},
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    try:
        import os
        os.environ["DUT_TEST_KEY"] = "sk-test-123"
        gw = LLMGateway({
            "provider": "cloud",
            "cloud": {"base_url": f"http://127.0.0.1:{port}/v1", "model": "test-model",
                      "api_key_env": "DUT_TEST_KEY", "label": "cloud"},
        })
        res = gw.chat(
            [{"role": "user", "content": "随便说句"}],
            schema={"type": "object",
                    "properties": {"utterance": {"type": "string"},
                                   "should_end": {"type": "boolean"},
                                   "feeling": {"type": "string"}},
                    "required": ["utterance", "should_end"]},
            purpose="dialogue.turn")

        assert res.ok, res.error
        assert res.get("utterance") == "好，我记一下。"
        assert res.prompt_tokens == 42 and res.completion_tokens == 7, "token 计量要读 usage"
        assert seen["path"].endswith("/v1/chat/completions")
        assert seen["auth"] == "Bearer sk-test-123", "API Key 要从环境变量读出来并带上"
        assert seen["body"]["model"] == "test-model"
        assert seen["body"]["response_format"] == {"type": "json_object"}, "要求 JSON 输出"
        assert seen["body"]["messages"][0]["role"] == "user"
    finally:
        srv.shutdown()


def test_云端没配_key_时不崩只是失败():
    gw = LLMGateway({"provider": "cloud",
                     "cloud": {"base_url": "http://127.0.0.1:9/v1", "model": "x",
                               "api_key_env": "绝不存在的变量"}})
    res = gw.chat([{"role": "user", "content": "x"}], purpose="generic")
    assert res.ok is False and res.error, "连不上应当降级为 ok=False，不能抛穿"


# ---------------------------------------------------------------- 记忆影响行为

class RecordingProvider:
    """把送进模型的 prompt 原样记下来——这样才能验证"记忆有没有真的被用上"，
    而不是只看它有没有被存下来。"""

    name = "recording"

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def complete(self, messages, **kw):
        self.calls.append({"messages": messages, "purpose": kw.get("purpose"), "meta": kw.get("meta")})
        return self.inner.complete(messages, **kw)

    def prompts(self, purpose: str, between: tuple[str, str] | None = None) -> list[str]:
        """取某类调用的 prompt。between 用来只看某一对说的话——
        maybe_start 挑的是全场最近的一对，驱动世界时别人也会碰上。"""
        out = []
        for c in self.calls:
            if c["purpose"] != purpose:
                continue
            if between:
                who = {c["meta"].get("speaker"), c["meta"].get("other")}
                if not who & set(between):
                    continue
            out.append("\n".join(m["content"] for m in c["messages"]))
        return out


def test_对话prompt里带上记忆人格与关系():
    """这是"记忆影响行为"的实证：不是存下来就算数，得真的出现在生成用的 prompt 里。"""
    eng, gw = make_engine(seed=12)
    rec = RecordingProvider(gw.provider)
    gw.provider = rec

    a, b = eng.agents[0], eng.agents[1]        # 这一对才是被挪到一起的人
    for i, a_ in enumerate(eng.agents[:2]):
        a_.x, a_.y = 30.0 + i * 0.6, 35.0
    # 先塞一条关于对方的、很具体的记忆
    a.memory.add(f"上次{b.name}说我推导跳步，让我把定义域写清楚", kind="dialogue",
                 importance=8, clock=eng.now_minutes, subject=b.name)
    a.memory.add("我最近总在拖延", kind="reflection", importance=8, clock=eng.now_minutes)

    eng.dialogues.maybe_start(eng.agents, eng.now_minutes,
                              room_name="走廊（东西主轴）", rng=AlwaysTalk())
    assert eng.dialogues.sessions, "应当开起一次对话"
    asyncio.run(eng.advance_dialogues())

    prompts = rec.prompts("dialogue.turn")
    assert prompts, "应当发生过 dialogue.turn 调用"
    p = prompts[0]
    assert "定义域" in p, "相关记忆必须出现在 prompt 里"
    assert b.name in p, "prompt 里必须写清对方是谁"
    assert "熟悉度" in p and "好感" in p, "关系状态要进 prompt"
    assert "你的性格" in p, "人格要进 prompt"


def test_聊过一次之后_第二次会带上上次的记忆():
    """记忆的回环：第一次对话写下的摘要，第二次对话时应当被检索到。"""
    eng, gw = make_engine(seed=13)
    rec = RecordingProvider(gw.provider)
    gw.provider = rec

    for i, a in enumerate(eng.agents[:2]):
        a.x, a.y = 30.0 + i * 0.6, 35.0
    eng.dialogues.maybe_start(eng.agents, eng.now_minutes,
                              room_name="走廊（东西主轴）", rng=AlwaysTalk())
    assert finish_dialogue(eng), "第一场应当聊到结束"

    first = [m for m in eng.agents[0].memory.items if m.kind == "dialogue"]
    assert first, "第一次对话应当留下记忆摘要"

    # 冷却过后再碰一次面
    for i, a in enumerate(eng.agents[:2]):
        a.x, a.y = 30.0 + i * 0.6, 35.0
    later = eng.now_minutes + 60
    eng.clock = later - eng.day * 1440
    s2 = eng.dialogues.maybe_start(eng.agents, later,
                                   room_name="走廊（东西主轴）", rng=AlwaysTalk())
    assert s2 is not None, "冷却过后应当能再聊一次"
    assert finish_dialogue(eng), "第二场也应当聊到结束"

    duo = (eng.agents[0].name, eng.agents[1].name)
    prompts = rec.prompts("dialogue.turn", between=duo)
    assert len(prompts) >= 2, f"这一对应当聊过两场，实际记录 {len(prompts)} 条"
    # 第二场里，"上次聊过"的摘要应当被检索进 prompt——这就是记忆的回环
    assert any(("聊" in p or "说开" in p or "交换" in p) for p in prompts[1:]), \
        "第二次对话的 prompt 里应当带上上次聊过的内容（记忆回环）"


def test_跨三次启动_天数递增且记忆与人格持续累积(tmp_path):
    """持久记忆的完整证据：连开关三次，天数是 0/1/2，记忆只增不减。"""
    from storage.store import Store

    store = Store(tmp_path / "multi.db")
    counts, days, socials = [], [], []

    for _ in range(3):
        eng, _ = make_engine(seed=21)
        eng.store = store
        for a in eng.agents:
            a.memory.store = store
        eng.resume()
        days.append(eng.day)
        for i, a in enumerate(eng.agents[:2]):        # 制造一次对话
            a.x, a.y = 30.0 + i * 0.6, 35.0
        eng.dialogues.maybe_start(eng.agents, eng.now_minutes,
                                  room_name="走廊（东西主轴）", rng=AlwaysTalk())
        assert finish_dialogue(eng), "每轮都应当聊到结束，才会留下记忆"
        eng.flush()
        counts.append(store.stats()["memories"])
        socials.append(eng.agents[0].personality.values["social"])

    assert days == [0, 1, 2], f"天数应当逐次递增，实际 {days}"
    assert counts == sorted(counts) and counts[-1] > counts[0], f"记忆应当只增不减：{counts}"
    assert socials[0] != socials[-1], f"人格应当随经历变化：{socials}"
    assert store.stats()["trait_changes"] >= 2
    store.close()


# ---------------------------------------------------------------- 真实模型的容错

def test_类型宽容_把数字字符串转成数字():
    """真实模型常把 0.05 写成 "0.05"。严格校验会直接判失败，
    然后整条链路退化成兜底值——白白浪费一次调用。"""
    from llm.gateway import coerce

    schema = {"type": "object",
              "properties": {"a": {"type": "number"}, "b": {"type": "integer"},
                             "c": {"type": "boolean"}, "d": {"type": "string"}},
              "required": ["a", "b", "c", "d"]}
    got = coerce({"a": "0.05", "b": "12", "c": "true", "d": 7}, schema)
    assert got == {"a": 0.05, "b": 12, "c": True, "d": "7"}
    assert validate_schema(got, schema) == []
    assert coerce({"a": "不是数字"}, {"type": "object", "properties": {"a": {"type": "number"}}})["a"] == "不是数字"


def test_摘要缺字段也算成功():
    """真实模型偶尔漏一个可选字段。为这个把整条摘要判失败、
    让记忆退化成兜底句，得不偿失。"""
    from llm.providers import Completion

    class Sparse:
        name = "sparse"
        def complete(self, *a, **k):
            return Completion(text='{"summary": "聊了聊课上的例题，有点收获。"}')

    gw = LLMGateway({"provider": "mock"})
    gw.provider = Sparse()
    from agents.dialogue import SUMMARY_SCHEMA
    res = gw.chat([{"role": "user", "content": "x"}], schema=SUMMARY_SCHEMA,
                  purpose="dialogue.summary")
    assert res.ok, res.error
    assert res.get("summary").startswith("聊了聊")


def test_对话记忆重要度不低于5():
    """重要度打分是启发式的，看不出来「聊了一场」有多重；
    如果只有 2-3，聊半天也攒不出一次反思。"""
    ms = MemoryStream("a")
    assert ms.add("和某人聊了几句", kind="dialogue", clock=100).importance >= 5
    assert ms.add("走过走廊", kind="observation", clock=100).importance < 5


def test_对话地点取说话两人的位置():
    """踩过的坑：原来用 agents[0] 的房间，于是两个老师在楼里聊天，
    地点却被记成了首位智能体所在的「校园里」。"""
    eng, _ = make_engine(seed=31)
    a, b = eng.agents[3], eng.agents[4]                     # 两位老师
    a.x, a.y = 60.0, 20.0
    b.x, b.y = 60.8, 20.2
    eng.agents[0].x, eng.agents[0].y = 54.0, 74.0           # 首位智能体在广场

    # 直接测这个函数：地点只看这一对
    assert eng._place_of_pair(a, b) == "教师办公室 A"

    # 端到端：其余人分散放远，保证它俩是最近的一对
    # 其余人各放一个互不重合的远点——放重了它们自己就成了"最近的一对"
    far = {0: (95.0, 95.0), 1: (5.0, 95.0), 2: (95.0, 5.0), 5: (5.0, 5.0)}
    for i, pos in far.items():
        eng.agents[i].x, eng.agents[i].y = pos
    eng._maybe_talk()
    assert eng.dialogues.sessions, "两位老师站在一起应当聊起来"
    assert eng.dialogues.sessions[0].place == "教师办公室 A", \
        f"地点应当是办公室，实际 {eng.dialogues.sessions[0].place}"


# ---------------------------------------------------------------- id 唯一性

def test_记忆_id_跨会话不冲突(tmp_path):
    """踩过的坑（严重）：id 原来是进程内自增 m000001…，重启后又从 1 开始，
    `INSERT OR REPLACE` 会把上一轮的记忆**静默覆盖**——
    看起来在累积，其实在覆写。这直接推翻了"持久记忆"。"""
    from storage.store import Store

    store = Store(tmp_path / "s.db")
    for session in range(3):
        ms = MemoryStream("stu_01", store=store)
        for i in range(4):
            ms.add(f"第 {session} 轮的第 {i} 条", kind="event", clock=100 + i)
        ms.flush(day=session)

    assert store.stats()["memories"] == 12, "三轮各写 4 条，一条都不能少"
    rows = store.load_memories(limit=100)
    assert len({r["id"] for r in rows}) == 12, "id 必须互不重复"
    assert {r["created"] for r in rows} == {100.0, 101.0, 102.0, 103.0}
    store.close()


def test_对话_id_也不复用():
    eng, _ = make_engine(seed=41, dialogue_cooldown_minutes=0)
    seen = set()
    for k in range(3):
        for i, a in enumerate(eng.agents[:2]):
            a.x, a.y = 30.0 + i * 0.6, 35.0
        s = eng.dialogues.maybe_start(eng.agents, eng.now_minutes + k * 100,
                                      room_name="走廊（东西主轴）", rng=AlwaysTalk())
        assert s is not None, f"第 {k} 次应当能开起对话"
        seen.add(s.id)
        eng.dialogues.sessions.clear()          # 收掉，好开下一场
        for a in eng.agents[:2]:
            a.state = "idle"
    assert len(seen) == 3, f"三场对话的 id 应当互不相同，实际 {seen}"
