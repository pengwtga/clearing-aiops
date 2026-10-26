"""清算对账差错字典 —— 整个项目的单一事实来源（single source of truth）

结构：5 大类（label，模型要预测的粒度）× 15 子类（subcause，判定依据与处置动作）。

设计原则（面试可讲）：
1. **互斥**：一张差异单只能落一个子类，否则规则与模型会互相打架；
2. **可判定**：每个子类都有明确的 evidence 字段组合，不依赖"人的经验"；
3. **可处置**：每个子类对应一条标准处置动作，模型输出必须能落到动作上，而不是只给个标签。

⚠️ 费率、容差、SLA 均为示例值，生产环境必须版本化（见 FEES / THRESHOLDS 注释）。
"""

from decimal import Decimal

# ---------------------------------------------------------------- 5 大类
CATEGORIES = {
    "source_data": "数据源",
    "timing": "时点",
    "calculation": "计算",
    "business": "业务",
    "system": "系统",
}

CATEGORY_ORDER = ["source_data", "timing", "calculation", "business", "system"]

# ---------------------------------------------------------------- 15 子类
# signal：判定依据（写进 prompt，也用于规则基线对照）
# action：标准处置动作（模型输出必须落到这里）
SUBCAUSES = {
    # ---- 数据源类：文件内容本身不对 ----
    "SRC_MISSING_LEG": {
        "category": "source_data",
        "name": "清算侧整笔漏发",
        "signal": "成交侧有该笔，清算文件无该 trade_id；同批次文件条数校验不平；后续批次也不会补发",
        "action": "向清算方补推该笔并重跑对账，同时核查文件条数/金额双重校验是否生效",
        "severity": "P2",
    },
    "SRC_DUP_LEG": {
        "category": "source_data",
        "name": "清算文件重复行",
        "signal": "同一 clear_id/trade_id 在同一文件出现多行，金额完全相同",
        "action": "按 clear_id 幂等去重后重跑，确认清算方出文件是否重发",
        "severity": "P3",
    },
    "SRC_CODE_FORMAT": {
        "category": "source_data",
        "name": "编码格式不一致",
        "signal": "归一化后能匹配，但原始字符串不同（带 .SH/.SZ 后缀、缺失前导零、账户号位数不同）",
        "action": "走标准化映射表修正，不改数据；推动上游统一编码规范",
        "severity": "P4",
    },
    "SRC_UNIT_MISMATCH": {
        "category": "source_data",
        "name": "数量/金额单位不一致",
        "signal": "数量或金额成 100 / 10000 倍关系，价格与费率一致，同批次其他笔正常",
        "action": "按单位换算（手↔股、元↔万元）后重跑，要求上游在文件头声明单位",
        "severity": "P2",
    },
    # ---- 时点类：数据是对的，出现在错误的时点 ----
    "TML_CROSS_DAY": {
        "category": "timing",
        "name": "跨日切分错位",
        "signal": "成交时间贴近日终切分点（14:5x 后），该笔被计入下一清算日",
        "action": "归入正确清算日重跑，不需要补推数据",
        "severity": "P3",
    },
    "TML_LATE_FILE": {
        "category": "timing",
        "name": "交收文件晚到",
        "signal": "交收文件到达时间晚于 SLA（示例 20:00），同批次大面积待交收（pending 笔数远高于日常）",
        "action": "等待补发文件到齐后重跑，禁止按失败处理、禁止重复出单",
        "severity": "P2",
    },
    "TML_STATUS_LAG": {
        "category": "timing",
        "name": "交收状态回传滞后",
        "signal": "文件按时到达，仅单笔状态仍为 PENDING；同批次其余笔已 SETTLED",
        "action": "查询清算方交收状态接口，确认后再决定是否重推，不做资金调账",
        "severity": "P3",
    },
    # ---- 计算类：同一个数算出来不一样 ----
    "CAL_ROUNDING": {
        "category": "calculation",
        "name": "金额舍入差异",
        "signal": "金额差 |d| ≤ 0.01，方向随机（有正有负），数量、费率完全一致",
        "action": "容差内调平，不追上游；把舍入规则写进对账口径文档",
        "severity": "P4",
    },
    "CAL_TRUNCATION": {
        "category": "calculation",
        "name": "费用截断/舍入模式不一致",
        "signal": "费用差 |d| ≤ 0.01 且同批次全部同向（我方高、对方低），金额无差异",
        "action": "统一舍入模式（ROUND_HALF_UP）后按差额调平，历史差额集中一次性处理",
        "severity": "P3",
    },
    "CAL_FEE_VERSION": {
        "category": "calculation",
        "name": "费率版本不一致",
        "signal": "费用差远大于 0.01，且差额与成交金额成比例；费/金额比值与当前生效费率不符",
        "action": "确认费率生效日期，按正确版本重算并补收/退还差额",
        "severity": "P1",
    },
    # ---- 业务类：规则/事件层面的差异 ----
    "BUS_CANCEL_ASYNC": {
        "category": "business",
        "name": "撤单未同步到清算",
        "signal": "成交侧状态已 CANCELLED，清算侧仍按原数量计账",
        "action": "重推撤单消息，确认清算侧撤单回执后重跑",
        "severity": "P2",
    },
    "BUS_PARTIAL_FILL": {
        "category": "business",
        "name": "部分成交未拆分",
        "signal": "同一 order_id 对应多笔成交，清算侧合并为一行（clear_id 关联委托号而非成交号）",
        "action": "按成交笔数拆分清算明细后对账，要求清算侧按成交编号出账",
        "severity": "P3",
    },
    "BUS_CORP_ACTION": {
        "category": "business",
        "name": "权益变动导致数量差异",
        "signal": "当日为除权除息/送转股登记日，数量成送转比例（如 1.3），价格同步调整",
        "action": "用复权后数量对账，权益变动单独出权益台账，不纳入差错统计",
        "severity": "P4",
    },
    # ---- 系统类：调度/幂等/顺序问题 ----
    "SYS_OUT_OF_ORDER": {
        "category": "system",
        "name": "任务乱序（交收先于清算）",
        "signal": "交收记录生成时间早于对应清算记录生成时间，数据本身金额无误",
        "action": "检查任务依赖与调度顺序，重跑交收任务，修复 DAG 依赖",
        "severity": "P1",
    },
    "SYS_IDEMPOTENT": {
        "category": "system",
        "name": "批次重跑产生重复/累加数据",
        "signal": "同一批次重跑次数 > 1：交收侧出现重复 settle_id，或费用被重复累加（费用差额恰为整笔佣金）",
        "action": "按 (batch_id, settle_id) 幂等键去重、按批次回滚重算，重跑任务写幂等表",
        "severity": "P2",
    },
    "SYS_SHARD_MISSING": {
        "category": "system",
        "name": "清算分片任务漏跑",
        "signal": "同一账户当日全部成交在清算侧成片缺失（不是随机单笔），批次记录显示某分片任务未完成",
        "action": "重跑该账户所属分片的清算任务，补推后重对账；给分片任务加完成度校验",
        "severity": "P1",
    },
}

# ---------------------------------------------------------------- 配置（示例值，须版本化）
FEES = {
    "version": "2026-08-01",
    "commission_rate": Decimal("0.00025"),   # 佣金（双向）
    "commission_min": Decimal("5.00"),       # 最低佣金
    "stamp_rate": Decimal("0.0005"),         # 印花税（仅卖出）
    "transfer_rate": Decimal("0.00001"),     # 过户费（沪市）
    "rounding": "ROUND_HALF_UP",
}

# 旧费率版本，用于构造"费率版本不一致"差错
FEES_OLD = {**FEES, "version": "2025-12-01", "commission_rate": Decimal("0.0003")}

THRESHOLDS = {
    "amount_tolerance": Decimal("0.01"),      # 金额/费用容差
    "pending_count_alert": 5,                 # 同批次待交收笔数告警阈值（超过视为文件级问题）
    "file_arrive_sla": "20:00",               # 交收文件到达 SLA
}

SEVERITY_ORDER = ["P1", "P2", "P3", "P4"]
