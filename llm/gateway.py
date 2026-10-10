"""LLM 网关：统一入口、结构化输出、重试降级、缓存、计量。

设计要点
--------
* 所有决策类调用都要求 **JSON Schema**，解析失败把错误回灌重试，
  再失败就返回 ok=False 让上层走兜底——**绝不让一次模型抽风拖垮仿真**。
* provider 可切换（mock / cloud / local），上层代码不知道用的是哪个。
* 同一个 prompt 命中缓存直接返回，回放与调试友好。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm.providers import MockProvider, OpenAICompatProvider, Provider


@dataclass
class LLMResult:
    text: str = ""
    data: dict[str, Any] | None = None
    ok: bool = False
    provider: str = ""
    purpose: str = ""
    tries: int = 0
    latency_ms: int = 0
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return (self.data or {}).get(key, default)


# ---------------------------------------------------------------- schema 校验

_TYPE_MAP = {
    "object": dict, "array": list, "string": str,
    "number": (int, float), "integer": int, "boolean": bool,
}


def validate_schema(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """极简 JSON Schema 校验（只支持我们用到的子集），返回错误列表。"""
    errs: list[str] = []
    t = schema.get("type")
    if t:
        py = _TYPE_MAP.get(t)
        if py and (not isinstance(data, py) or (t != "boolean" and isinstance(data, bool))):
            return [f"{path} 应为 {t}，实际是 {type(data).__name__}"]
    if t == "object":
        for key in schema.get("required", []):
            if key not in data:
                errs.append(f"{path}.{key} 缺失")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in data:
                errs += validate_schema(data[key], sub, f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in props:
                    errs.append(f"{path}.{key} 是多余字段")
    elif t == "array" and "items" in schema:
        for i, item in enumerate(data):
            errs += validate_schema(item, schema["items"], f"{path}[{i}]")
    elif "enum" in schema and data not in schema["enum"]:
        errs.append(f"{path} 必须是 {schema['enum']} 之一，实际是 {data!r}")
    if "const" in schema and data != schema["const"]:
        errs.append(f"{path} 必须是 {schema['const']!r}")
    return errs


def coerce(data: Any, schema: dict[str, Any]) -> Any:
    """按 schema 做一次宽容转换。

    真实模型经常把 0.05 写成 "0.05"、"true" 写成 true 的字符串形式，
    严格校验会直接判失败——然后整条链路退化成兜底值，白白浪费一次调用。
    这里只做**不改语义**的转换，改不了的就交给校验去报错。
    """
    t = schema.get("type")
    if t == "object" and isinstance(data, dict):
        props = schema.get("properties", {})
        return {k: (coerce(v, props[k]) if k in props else v) for k, v in data.items()}
    if t == "array" and isinstance(data, list) and "items" in schema:
        return [coerce(x, schema["items"]) for x in data]
    if t in ("number", "integer") and isinstance(data, str):
        try:
            f = float(data.strip().rstrip("%"))
            return int(f) if t == "integer" else f
        except ValueError:
            return data
    if t == "boolean" and isinstance(data, str):
        low = data.strip().lower()
        if low in ("true", "yes", "1", "是"):
            return True
        if low in ("false", "no", "0", "否"):
            return False
        return data
    if t == "string" and isinstance(data, (int, float)) and not isinstance(data, bool):
        return str(data)
    return data


def extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON——容忍 ```json 围栏和前后废话。"""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s[3:]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 退一步：找第一个 { 到最后一个 }
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------- 网关

DEFAULT_CONFIG = Path("config.json")


class LLMGateway:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.cfg = cfg
        self.provider_name = cfg.get("provider", "mock")
        self.temperature = float(cfg.get("temperature", 0.8))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.use_cache = bool(cfg.get("cache", True))
        self.provider: Provider = self._make_provider(self.provider_name)
        self._cache: dict[str, LLMResult] = {}
        self.stats: dict[str, Any] = {
            "calls": 0, "cache_hits": 0, "failures": 0, "retries": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "by_purpose": {},
        }

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_CONFIG) -> "LLMGateway":
        p = Path(path)
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                return cls(json.load(fh).get("llm", {}))
        return cls({})

    def _make_provider(self, name: str) -> Provider:
        if name == "mock":
            return MockProvider(self.cfg.get("mock", {}))
        if name in ("cloud", "local"):
            sub = dict(self.cfg.get(name, {}))
            sub.setdefault("label", name)
            return OpenAICompatProvider(sub)
        raise ValueError(f"未知 provider：{name}")

    def switch(self, name: str) -> None:
        """运行期换 provider（mock ↔ cloud ↔ local），上层无感。"""
        self.provider = self._make_provider(name)
        self.provider_name = name
        self._cache.clear()

    def _key(self, messages: list[dict], schema: dict | None, purpose: str) -> str:
        blob = json.dumps([self.provider_name, purpose, messages, schema], ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        purpose: str = "generic",
        meta: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int = 512,
    ) -> LLMResult:
        key = self._key(messages, schema, purpose)
        if self.use_cache and key in self._cache:
            hit = self._cache[key]
            self.stats["cache_hits"] += 1
            return LLMResult(**{**hit.__dict__, "cached": True})

        self.stats["calls"] += 1
        per = self.stats["by_purpose"].setdefault(purpose, {"calls": 0, "fails": 0})
        per["calls"] += 1

        work = list(messages)
        last_err = None
        last_raw = ""
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            if attempt:
                self.stats["retries"] += 1
            try:
                out = self.provider.complete(
                    work, schema=schema, temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens, purpose=purpose, meta=meta or {},
                )
            except Exception as exc:                       # noqa: BLE001 - provider 的锅不该炸穿仿真
                last_err = f"provider 异常：{exc}"
                break

            res = LLMResult(
                text=out.text, ok=True, provider=self.provider_name, purpose=purpose,
                tries=attempt + 1, prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
            )
            self.stats["prompt_tokens"] += out.prompt_tokens
            self.stats["completion_tokens"] += out.completion_tokens

            if schema is None:
                res.latency_ms = int((time.perf_counter() - t0) * 1000)
                return self._store(key, res)

            data = out.data if out.data is not None else extract_json(out.text)
            if data is not None:
                data = coerce(data, schema)          # 先宽容转换，再严格校验
            errs = validate_schema(data, schema) if data is not None else ["不是合法 JSON"]
            if not errs:
                res.data = data
                res.latency_ms = int((time.perf_counter() - t0) * 1000)
                return self._store(key, res)

            last_err = "；".join(errs[:4])
            last_raw = out.text[:600]        # 记模型的原话，不是我们回灌的修复提示
            # 把错误回灌，让模型自己修
            work = list(messages) + [
                {"role": "assistant", "content": out.text[:800]},
                {"role": "user", "content": f"上面的输出不符合要求：{last_err}。请只输出修正后的 JSON，不要解释。"},
            ]

        self.stats["failures"] += 1
        per["fails"] += 1
        return LLMResult(
            ok=False, provider=self.provider_name, purpose=purpose,
            text=last_raw,
            tries=self.max_retries + 1, latency_ms=int((time.perf_counter() - t0) * 1000),
            error=last_err or "未知错误",
        )

    def _store(self, key: str, res: LLMResult) -> LLMResult:
        if self.use_cache:
            self._cache[key] = res
        return res

    def summary(self) -> dict[str, Any]:
        return dict(self.stats, provider=self.provider_name, cache_size=len(self._cache))
