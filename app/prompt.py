"""Prompt 构造 —— 由差错字典自动生成，不手写第二份字典

关键决策：**prompt 里的子类表、判定信号、标准处置全部从 taxonomy.py 渲染**。
手写 prompt 是工程上最常见的漂移源：字典加了一个子类，prompt 忘了改，模型永远
答不出这个类，而评测集里却有它的样本。这里是同一份数据渲染出来的，改字典即生效。

另一个决策：**只让模型预测 subcause（子类），大类由字典查出来**。
让模型同时输出 category 和 subcause，就多了一处"两个字段互相矛盾"的失败模式，
而这个映射本来就是确定的字典关系，没必要交给模型算。
"""

from functools import lru_cache

from taxonomy import CATEGORIES, SUBCAUSES

OUTPUT_SCHEMA = """{
  "subcause": "字典里的子类 code（必须完全一致）",
  "confidence": 0.0 到 1.0 的小数,
  "evidence": [
    {"field": "差异单里的字段名", "value": "该字段的实际值", "why": "这个值说明了什么"}
  ],
  "suggested_action": "处置动作，与字典里该子类的标准处置一致，可补充本单具体对象"
}"""

# 三个示例覆盖三种最容易判错的形态：单字段可判 / 要靠批次证据 / 现象相同但成因不同
_EXAMPLES = [
    {
        "input": {
            "diff_type": "AMOUNT_MISMATCH",
            "trade_qty": 300, "clear_qty": 300,
            "trade_amount": 503355.0, "clear_amount": 503355.01, "diff_amount": 0.01,
            "qty_ratio": None, "batch_rerun_count": 1, "corp_action_today": False,
            "remark": "差一分钱，四舍五入方向不同",
        },
        "output": {
            "subcause": "CAL_ROUNDING",
            "confidence": 0.93,
            "evidence": [
                {"field": "diff_amount", "value": "0.01", "why": "金额差恰为 1 分钱，在 0.01 容差内"},
                {"field": "clear_qty", "value": "300", "why": "数量与成交侧完全一致，排除单位换算问题"},
            ],
            "suggested_action": "容差内调平，不追上游；把舍入规则写进对账口径文档",
        },
    },
    {
        "input": {
            "diff_type": "MISSING_IN_CLEARING",
            "legs_present": ["trade"],
            "account": "A88123012",
            "batch_missed_shards": ["A88123012"], "batch_rerun_count": 1,
            "trade_time": "10:12:03",
            "remark": "这个客户今天全部没对账",
        },
        "output": {
            "subcause": "SYS_SHARD_MISSING",
            "confidence": 0.9,
            "evidence": [
                {"field": "batch_missed_shards", "value": '["A88123012"]', "why": "批次记录显示该账户所属分片任务未完成"},
                {"field": "remark", "value": "这个客户今天全部没对账", "why": "整账户成片缺失，不是随机单笔漏发"},
            ],
            "suggested_action": "重跑该账户所属分片的清算任务，补推后重对账；给分片任务加完成度校验",
        },
    },
    {
        "input": {
            "diff_type": "MISSING_IN_CLEARING",
            "legs_present": ["trade"],
            "account": "A88123007",
            "batch_missed_shards": [], "batch_rerun_count": 1,
            "trade_time": "14:57:41",
            "remark": "成交里有，清算里翻不到",
        },
        "output": {
            "subcause": "TML_CROSS_DAY",
            "confidence": 0.86,
            "evidence": [
                {"field": "trade_time", "value": "14:57:41", "why": "贴近日终切分点，该笔很可能被计入下一清算日"},
                {"field": "batch_missed_shards", "value": "[]", "why": "没有分片漏跑记录，排除系统侧问题"},
            ],
            "suggested_action": "归入正确清算日重跑，不需要补推数据",
        },
    },
]


_PRIORITY_BLOCK = """【判别优先级】（多个子类都像时，按这个顺序排除；这是老师傅的判断顺序）
1. `batch_rerun_count > 1` 且差异表现为**重复行 / 多出一份 / 费用翻倍** → SYS_IDEMPOTENT。
   不要因为差异单写着 DUP 就判 SRC_DUP_LEG——重复下发与重跑的区别就在这个字段。
2. `batch_missed_shards` 含本单账户，且整账户多笔成片缺失 → SYS_SHARD_MISSING。
3. 单笔缺失、批次无异常、成交时间 ≥ 14:50 或清算日与成交日不一致 → TML_CROSS_DAY。
4. 单笔缺失、以上都不成立 → SRC_MISSING_LEG。
5. 交收未完成：`batch_pending_count ≥ 5` 或 `file_arrive_time` 晚于 20:00 → TML_LATE_FILE；
   仅单笔未完成、批次正常 → TML_STATUS_LAG。
6. 数量差：`qty_ratio` ≈ 100 或 ≈ 0.01 → SRC_UNIT_MISMATCH；≈1.2~1.4 → BUS_CORP_ACTION；
   小于 1 且同委托多笔成交 → BUS_PARTIAL_FILL。
7. 金额差：`amount_ratio` ≈ 100 → SRC_UNIT_MISMATCH；`|diff_amount| ≤ 0.02` → CAL_ROUNDING。
8. 费用差：`|diff_fee| ≤ 0.02` 且全批次同向 → CAL_TRUNCATION；
   差额与成交金额成比例（费率不对）→ CAL_FEE_VERSION；
   金额无差异而费用恰好翻倍且 `batch_rerun_count > 1` → SYS_IDEMPOTENT。
9. 成交侧已撤单而清算仍计账 → BUS_CANCEL_ASYNC；交收记录生成时间早于清算记录 → SYS_OUT_OF_ORDER。
10. `remark` 只作辅助：它常写"算错了""对不上"，**不等于**计算类；与结构化字段冲突时以字段为准。

"""


@lru_cache(maxsize=2)
def build_system_prompt(level: str = "full") -> str:
    """level=basic 只给字典与判断顺序（第一版 prompt）；
    level=full 追加"判别优先级"（把专家排除法写进 prompt）——评测里用于对照实验。"""
    rows = []
    for code, meta in SUBCAUSES.items():
        rows.append(
            f"| {code} | {CATEGORIES[meta['category']]} | {meta['name']} | {meta['signal']} | {meta['action']} | {meta['severity']} |"
        )
    dictionary = "\n".join(rows)

    examples = []
    for i, ex in enumerate(_EXAMPLES, 1):
        examples.append(
            f"示例 {i}\n输入：{_compact(ex['input'])}\n输出：{_compact(ex['output'])}"
        )
    examples_block = "\n\n".join(examples)
    priority_block = _PRIORITY_BLOCK if level == "full" else ""

    return f"""你是券商清算部的对账差错归因助手。给你一张日终对账产生的差异单（JSON），
你要判断它的根因属于下列子类中的哪一个，并给出处置动作。

【子类字典】（共 {len(SUBCAUSES)} 个，只能从中选一个）
| code | 类别 | 名称 | 判定信号 | 标准处置 | 严重级 |
|---|---|---|---|---|---|
{dictionary}

【判断顺序】（按此顺序看证据，不要跳步）
1. **时点字段**：trade_time / clearing_date / clear_created_at / settle_created_at / file_arrive_time。
   时点类差错的唯一判据在这里——数据本身是对的，只是出现在错误的时点。
2. **批次级证据**：batch_rerun_count（>1 说明批次重跑过，费用可能被重复累加）、
   batch_missed_shards（分片漏跑）、batch_pending_count（大面积待交收＝文件晚到）、
   corp_action_today（权益变动）。单笔字段判不了的问题，答案在这一组。
3. **三方数值**：legs_present（缺哪条腿）、qty_ratio / amount_ratio（是否成 100 倍、
   1.3 倍等整数关系）、diff_amount / diff_fee（差多少、是否同向）。
4. **remark 自由文本**：值班或客服原话，信息量最大但可能误导（"算错了"往往不是计算问题）。

{priority_block}【硬规则】
- subcause 必须是上面字典里的 code，原样输出，不要自创、不要翻译成中文。
- evidence 必须引用输入里**真实存在**的字段名，禁止编造字段；每条说明"这个值说明了什么"。
- evidence 只保留与该子类判定信号**直接相关**的字段（2~3 条即可），不要罗列无关字段凑数。
- 现象相同、成因不同是常态：「清算侧缺行」可能是整笔漏发、可能是分片漏跑、也可能是
  跨日切分，必须用批次证据和时间区分，**不要默认判成漏发**。
- 拿不准就给低 confidence（< 0.85），宁可交给人工，也不要为了显得果断而硬答。
- suggested_action 与字典里该子类的标准处置保持一致，可以补一句本单的具体对象。

【输出格式】只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块：
{OUTPUT_SCHEMA}

【示例】
{examples_block}
"""


def _compact(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=1)
def build_user_template_hint() -> str:
    """放在 user 消息尾部的一句话，重复强调输出契约（长 prompt 尾部指令权重更高）"""
    return "只输出 JSON：subcause / confidence / evidence / suggested_action。"


def build_user_payload(diff: dict) -> str:
    return f"差异单：\n{_compact(diff)}\n\n{build_user_template_hint()}"
