"""归因层 —— LLM 归因 + 输出校验 + 置信度门禁 + 规则兜底

四道关，缺一不可：
1. **可解析**：模型返回的必须是 JSON（客户端已做宽松提取）
2. **枚举合法**：subcause 必须落在字典里。模型答"资金交收延迟"这种自由文本，
   下游的处置动作映射就直接断了——这比答错更危险，因为看起来像答对了。
3. **证据可核对**：evidence 里引用的字段名必须在差异单里真实存在。模型编一个
   `settlement_batch_no` 出来，人是看不出来的；能机检的就不靠人眼。
4. **置信度门禁**：< 阈值转人工，不硬答。宁可少自动化 30%，不能给错处方——
   对账场景把"重复清算"判成"舍入差异"，处置动作就是"调平"，差额直接变损失。

任何一关不过 → 退回规则兜底，并标记 need_human。服务永不因为模型不行而不可用。
"""

import logging
from dataclasses import asdict, dataclass, field

from taxonomy import CATEGORIES, SUBCAUSES

from app import llm_client, rules
from app.config import load_settings
from app.prompt import build_system_prompt, build_user_payload

logger = logging.getLogger(__name__)
settings = load_settings()

# 不采信的三种原因（评测里分开统计，因为它们对应的修法完全不同）
REASON_LLM_UNAVAILABLE = "llm_unavailable"   # 全链失败 → 修容灾/额度
REASON_INVALID_OUTPUT = "invalid_output"     # 枚举越界/证据编造 → 改 prompt 或加约束解码
REASON_LOW_CONFIDENCE = "low_confidence"     # 置信度不足 → 补证据字段或改判据


@dataclass
class Attribution:
    diff_id: str
    subcause: str                 # 最终结论（可能来自规则兜底）
    category: str
    category_cn: str
    subcause_name: str
    confidence: float
    evidence: list[dict] = field(default_factory=list)
    suggested_action: str = ""
    source: str = "llm"           # llm | rule_fallback
    need_human: bool = False
    fallback_reason: str | None = None
    llm_subcause: str | None = None   # 被门禁拦下时保留模型原始判断，供人工参考
    llm_confidence: float | None = None
    provider: str | None = None
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _valid_evidence(items, diff: dict) -> tuple[list[dict], list[str]]:
    """只保留引用了真实字段的证据；返回 (有效证据, 被丢弃的说明)"""
    valid, dropped = [], []
    keys = set(diff.keys())
    if not isinstance(items, list):
        return [], ["evidence 不是数组"]

    for item in items:
        if not isinstance(item, dict):
            dropped.append(f"evidence 项不是对象：{item!r}")
            continue
        name = str(item.get("field") or "").strip()
        if name not in keys:
            dropped.append(f"引用了不存在的字段 {name!r}")
            continue
        valid.append({
            "field": name,
            "value": str(item.get("value", diff.get(name))),
            "why": str(item.get("why") or "").strip(),
        })
    return valid, dropped


def _parse_confidence(value) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf != conf:          # NaN
        return None
    if conf > 1.0:            # 有模型会输出 90 表示 90%
        conf = conf / 100.0 if conf <= 100 else 1.0
    return max(0.0, min(1.0, conf))


def _rule_result(diff: dict, reason: str, notes: list[str] | None = None) -> Attribution:
    subcause, why = rules.rule_predict(diff)
    meta = SUBCAUSES[subcause]
    return Attribution(
        diff_id=diff.get("diff_id", ""),
        subcause=subcause,
        category=meta["category"],
        category_cn=CATEGORIES[meta["category"]],
        subcause_name=meta["name"],
        confidence=0.5,                     # 规则兜底给固定低置信度，不假装有把握
        evidence=[{"field": "diff_type", "value": str(diff.get("diff_type")), "why": why}],
        suggested_action=meta["action"],
        source="rule_fallback",
        need_human=True,                    # 兜底结论一律要人复核
        fallback_reason=reason,
        notes=[*(notes or []), f"规则兜底依据：{why}"],
    )


def attribute(diff: dict, prompt_level: str = "full") -> Attribution:
    """对一张差异单做归因。永不抛异常——最差也会返回规则兜底结果。"""
    try:
        result = llm_client.chat_json(build_system_prompt(prompt_level), build_user_payload(diff))
    except llm_client.LLMNotConfiguredError as exc:
        return _rule_result(diff, REASON_LLM_UNAVAILABLE, [str(exc)])
    except llm_client.LLMError as exc:
        return _rule_result(diff, REASON_LLM_UNAVAILABLE, [str(exc)])

    payload = result.data if isinstance(result.data, dict) else {}
    notes: list[str] = []

    # ---- 关 2：枚举合法
    raw_subcause = str(payload.get("subcause") or "").strip()
    subcause = raw_subcause.upper()
    if subcause not in SUBCAUSES:
        notes.append(f"subcause 不在字典内：{raw_subcause!r}")
        fb = _rule_result(diff, REASON_INVALID_OUTPUT, notes)
        fb.provider = result.provider
        fb.latency_ms = result.latency_ms
        fb.llm_subcause = raw_subcause or None
        return fb

    # ---- 关 3：证据可核对
    evidence, dropped = _valid_evidence(payload.get("evidence"), diff)
    if dropped:
        notes.extend(dropped)
    if not evidence:
        notes.append("无可核对的证据，不采信模型结论")
        fb = _rule_result(diff, REASON_INVALID_OUTPUT, notes)
        fb.provider = result.provider
        fb.latency_ms = result.latency_ms
        fb.llm_subcause = subcause
        return fb

    # ---- 置信度解析（缺失按 0 处理：不给分就是没把握）
    confidence = _parse_confidence(payload.get("confidence"))
    if confidence is None:
        notes.append(f"confidence 不可解析：{payload.get('confidence')!r}")
        confidence = 0.0

    meta = SUBCAUSES[subcause]
    action = str(payload.get("suggested_action") or "").strip() or meta["action"]

    attribution = Attribution(
        diff_id=diff.get("diff_id", ""),
        subcause=subcause,
        category=meta["category"],
        category_cn=CATEGORIES[meta["category"]],
        subcause_name=meta["name"],
        confidence=confidence,
        evidence=evidence,
        suggested_action=action,
        source="llm",
        need_human=False,
        provider=result.provider,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        notes=notes + result.notes,
    )

    # ---- 关 4：置信度门禁
    if confidence < settings.gate_threshold:
        fb = _rule_result(diff, REASON_LOW_CONFIDENCE, notes)
        fb.provider = result.provider
        fb.latency_ms = result.latency_ms
        fb.prompt_tokens = result.prompt_tokens
        fb.completion_tokens = result.completion_tokens
        fb.llm_subcause = subcause
        fb.llm_confidence = confidence
        fb.notes.append(f"模型给 {subcause}（{confidence:.2f}），低于门禁 {settings.gate_threshold}，转人工")
        return fb

    return attribution


def attribute_many(diffs: list[dict], workers: int = 4, prompt_level: str = "full") -> list[Attribution]:
    """并发归因。免费档限流常见，并发不要开太大（默认 4）。"""
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    fn = partial(attribute, prompt_level=prompt_level)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, diffs))
