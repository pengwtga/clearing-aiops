"""LLM 客户端 —— 多 Provider 容灾链，只用标准库（urllib），不引 SDK / httpx

三个设计点：
1. **链式容灾**：primary(DeepSeek) → zhipu(GLM-4-Flash) → siliconflow(Qwen)，
   逐个尝试，任一成功即返回；全挂抛 LLMError，由归因层降级到规则基线。
2. **JSON 模式兼容**：部分免费模型不支持 response_format=json_object，遇 400 自动
   去掉该参数重试，并用宽松提取（剥 ``` 代码围栏 / 截首个平衡花括号）兜底。
3. **限流退避**：免费档 429 很常见，按 Retry-After 或指数退避重试，不把限流当故障。

为什么不用 SDK：一份依赖树就是一份升级/漏洞/体积负担。OpenAI 兼容接口的
chat/completions 用 20 行 urllib 就能调通，对单机服务没有理由引 SDK。
"""

import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from app.config import ProviderConfig, load_settings

logger = logging.getLogger(__name__)

settings = load_settings()


_RATE_LOCK = threading.Lock()
_last_call_at = 0.0


def _throttle() -> None:
    """全局最小请求间隔。免费档按 RPM 限流时，光靠减小并发不够——
    多个线程仍可能在同一个时间窗里打满，这里用一把锁把它们均匀摊开。"""
    interval = settings.min_interval_ms / 1000.0
    if interval <= 0:
        return
    global _last_call_at
    with _RATE_LOCK:
        wait = _last_call_at + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


class LLMError(Exception):
    pass


class LLMNotConfiguredError(LLMError):
    pass


class _JSONModeUnsupported(Exception):
    pass


@dataclass
class LLMResult:
    data: dict
    provider: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    notes: list[str] = field(default_factory=list)


def _post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _sleep_for_retry(attempt: int, retry_after: str | None) -> None:
    if retry_after:
        try:
            time.sleep(min(float(retry_after), 20.0))
            return
        except ValueError:
            pass
    time.sleep(min(2 ** attempt + random.random(), 10.0))


def chat_json(system_prompt: str, user_payload: str) -> LLMResult:
    """遍历容灾链调用 LLM，返回解析后的 JSON 与调用元信息。全链失败抛 LLMError。"""
    chain = settings.providers
    if not chain:
        raise LLMNotConfiguredError("未配置任何 LLM Provider 的 API Key（见 .env.example）")

    errors: list[str] = []
    for provider in chain:
        started = time.perf_counter()
        try:
            result = _call_with_retry(provider, system_prompt, user_payload)
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001 —— 容灾链要吞掉一切异常，继续下一个
            errors.append(f"{provider.name}: {exc}")
            logger.warning("[llm] provider=%s 失败，切换下一个：%s", provider.name, exc)
    raise LLMError("所有 LLM Provider 均失败：" + "; ".join(errors))


def _call_with_retry(provider: ProviderConfig, system_prompt: str, user_payload: str) -> LLMResult:
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(settings.max_retries + 1):
        attempts = attempt + 1
        try:
            return _call(provider, system_prompt, user_payload, use_json_mode=True)
        except _JSONModeUnsupported:
            # 不支持 json_object：立即切宽松模式，不消耗重试次数
            try:
                res = _call(provider, system_prompt, user_payload, use_json_mode=False)
                res.notes.append("provider 不支持 json_object，已用宽松提取")
                return res
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (429, 500, 502, 503, 504):
                retry_after = (exc.headers or {}).get("Retry-After") if exc.headers else None
                logger.info("[llm] %s 限流/服务端错误 %s，退避重试", provider.name, exc.code)
                _sleep_for_retry(attempt, retry_after)
                continue
            if exc.code in (401, 403):
                raise LLMError(f"{provider.name} 鉴权失败({exc.code})，检查 API Key") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            last_error = exc
            _sleep_for_retry(attempt, None)
    raise LLMError(str(last_error))


def _call(provider: ProviderConfig, system_prompt: str, user_payload: str, use_json_mode: bool) -> LLMResult:
    body: dict = {
        "model": provider.model,
        "temperature": settings.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
    }
    if use_json_mode:
        body["response_format"] = {"type": "json_object"}

    _throttle()
    try:
        resp = _post(
            f"{provider.base_url.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {provider.api_key}"},
            body,
            settings.timeout,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 400 and use_json_mode:
            raise _JSONModeUnsupported() from exc
        raise

    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage") or {}
    return LLMResult(
        data=_extract_json(content),
        provider=provider.name,
        latency_ms=0,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def _extract_json(content: str) -> dict:
    """宽松 JSON 提取：直接解析 → 剥 markdown 代码围栏 → 截取首个平衡花括号块"""
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = content.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(content[start:i + 1])
    raise json.JSONDecodeError("no json object found", content, 0)


def is_configured() -> bool:
    return len(settings.providers) > 0


def configured_providers() -> list[str]:
    return [p.name for p in settings.providers]
