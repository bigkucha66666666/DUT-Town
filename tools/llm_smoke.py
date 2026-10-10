"""真实模型体检：只发一次调用，验证 Key、地址、JSON 输出、延迟与计费。

切到真实模型之前先跑这个——比直接起整个世界再排查快得多。

    # 用环境变量里的 Key
    export DEEPSEEK_API_KEY=sk-...
    python tools/llm_smoke.py

    # 或者临时从某个 .env 读（不会打印 Key 本身）
    python tools/llm_smoke.py --env-file ../Agent/traffic-agent-lab/.env

    # 换通道 / 换模型
    python tools/llm_smoke.py --provider local --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.gateway import LLMGateway, validate_schema   # noqa: E402

TURN_SCHEMA = {
    "type": "object",
    "properties": {"utterance": {"type": "string"},
                   "should_end": {"type": "boolean"},
                   "feeling": {"type": "string"}},
    "required": ["utterance", "should_end"],
    "additionalProperties": False,
}

# 跟真实对话同构的一次调用：学生找老师问上节课的例题
SYSTEM = (
    "你在扮演中国大学校园里的一个人，参与一次自然的口头对话。\n"
    "只输出 JSON，字段为 utterance（这一句台词）、should_end（这句之后对话是否自然结束）、"
    "feeling（此刻情绪，两字）。台词 15–45 字，口语化，不要旁白。"
)
USER = (
    "你是李子雨，大二 · 软件工程。\n"
    "你的性格：社交活跃低、学业投入高、主动性低、情绪稳定中。说话风格：说话简短，"
    "习惯用「嗯……」开头。\n"
    "你当前的目标：这学期绩点进专业前 10%。\n"
    "对方是王士鹏（副教授 · 经济管理学院）。你们的关系：熟悉度 0.12、好感 0.05、信任 0.03。\n"
    "地点：教师办公室 A。话题：作业和答疑。\n\n"
    "你脑子里冒出来的相关记忆：\n"
    "- [dialogue] 上次王老师说我推导跳步，让我把定义域写清楚。\n\n"
    "到目前为止的对话：\n（还没开口）\n\n现在轮到你说话，只输出 JSON。"
)


def load_env_file(path: str) -> int:
    """把 .env 里的键值读进环境变量。返回读进来几条（不打印值）。"""
    p = Path(path)
    if not p.exists():
        print(f"✗ 找不到 {p}")
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="真实模型体检（只发一次调用）")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--provider", choices=["mock", "cloud", "local"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--env-file", default=None, help="从 .env 读 Key（只读键值，不打印）")
    args = ap.parse_args(argv)

    if args.env_file:
        n = load_env_file(args.env_file)
        print(f"从 {args.env_file} 读入 {n} 个环境变量（值不显示）")

    gw = LLMGateway.from_file(ROOT / args.config)
    if args.provider:
        gw.switch(args.provider)
    if args.model:
        gw.provider.model = args.model           # type: ignore[attr-defined]

    prov = gw.provider
    print(f"通道　　：{gw.provider_name}  模型：{getattr(prov, 'model', '（mock）')}")
    print(f"地址　　：{getattr(prov, 'base_url', '（mock，无网络）')}")
    key_env = None
    for name in ("cloud", "local"):
        for m in (gw.cfg.get(name, {}),):
            if m.get("model") == getattr(prov, "model", None):
                key_env = m.get("api_key_env")
    has_key = bool(getattr(prov, "api_key", ""))
    print(f"API Key ：{'已读到（不显示）' if has_key else '未读到'}"
          + (f"  来源变量：{key_env}" if key_env else ""))
    if gw.provider_name != "mock" and not has_key:
        print("\n✗ 没有读到 API Key。请先 `export <KEY_ENV>=...` 或加 --env-file。")
        return 2

    print("\n发一次 dialogue.turn 形状的调用……")
    t0 = time.perf_counter()
    res = gw.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
                  schema=TURN_SCHEMA, purpose="dialogue.turn", max_tokens=300)
    dt = time.perf_counter() - t0

    if not res.ok:
        print(f"\n✗ 失败（{dt:.1f}s，试了 {res.tries} 次）：{res.error}")
        print("  常见原因：Key 无效 / 额度不足 / 地址不对 / 模型名写错")
        return 1

    errs = validate_schema(res.data or {}, TURN_SCHEMA)
    print(f"\n✓ 成功（{dt:.1f}s，{res.tries} 次尝试"
          + ("，命中缓存" if res.cached else "") + "）")
    print(f"  token：prompt {res.prompt_tokens} / completion {res.completion_tokens}")
    print(f"  台词：{res.get('utterance')}")
    print(f"  结束：{res.get('should_end')}    情绪：{res.get('feeling')}")
    if errs:
        print(f"\n✗ JSON 结构不符：{errs}")
        return 1
    print("\nJSON 结构校验通过。可以 `python -m server` 起真实世界了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
