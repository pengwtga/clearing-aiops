"""规则兜底 —— 确定性映射，不依赖任何模型

两条用途：
1. **降级通道**：LLM 全链失败或输出不合格时，服务仍然能给一个可执行结论；
2. **门禁的下家**：置信度 < 阈值时，由规则接手并打上"需人工复核"，而不是硬答。

这里比 script/rule_baseline.py 的 12 条启发式多走一步——要落到**子类**，
所以会看批次证据（rerun_count / missed_shards / pending_count）和时点字段。
但仍然是"现象 → 类型"的确定性映射，不做语义理解，因此在 hard 集上同样吃亏，
这正是它只能当兜底、不能当主链路的原因。
"""

from decimal import Decimal, InvalidOperation


def _dec(value) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def rule_predict(diff: dict) -> tuple[str, str]:
    """返回 (subcause, 判定依据简述)"""
    d = diff
    diff_type = d.get("diff_type")
    rerun = d.get("batch_rerun_count") or 1
    missed = d.get("batch_missed_shards") or []
    pending = d.get("batch_pending_count") or 0
    same_sign = d.get("batch_fee_diff_same_sign_ratio")
    qty_ratio = _dec(d.get("qty_ratio"))
    amount_ratio = _dec(d.get("amount_ratio"))
    diff_amount = _dec(d.get("diff_amount"))
    diff_fee = _dec(d.get("diff_fee"))
    trade_time = d.get("trade_time") or ""
    trade_date = d.get("trade_date")
    clearing_date = d.get("clearing_date")

    if diff_type == "MISSING_IN_CLEARING":
        if missed:
            return "SYS_SHARD_MISSING", f"批次记录显示分片 {missed} 未完成"
        if trade_time and trade_time >= "14:50":
            return "TML_CROSS_DAY", f"成交时间 {trade_time} 贴近日终切分点"
        return "SRC_MISSING_LEG", "单笔在清算侧缺失，无批次异常证据"

    if diff_type == "DUP_IN_CLEARING":
        if rerun > 1:
            return "SYS_IDEMPOTENT", f"批次重跑 {rerun} 次"
        return "SRC_DUP_LEG", "清算文件内同 clear_id 多行"

    if diff_type == "DUP_IN_SETTLEMENT":
        if rerun > 1:
            return "SYS_IDEMPOTENT", f"批次重跑 {rerun} 次导致交收重复"
        return "SRC_DUP_LEG", "交收文件重复下发"

    if diff_type == "CODE_FORMAT_MISMATCH":
        return "SRC_CODE_FORMAT", "编码归一化后能匹配"

    if diff_type == "QTY_MISMATCH":
        if qty_ratio is not None:
            if qty_ratio >= Decimal("10"):
                return "SRC_UNIT_MISMATCH", f"数量成 {qty_ratio} 倍关系"
            if Decimal("1.2") <= qty_ratio <= Decimal("1.4"):
                return "BUS_CORP_ACTION", f"数量比例 {qty_ratio} 接近送转比例"
            if qty_ratio < 1:
                return "BUS_PARTIAL_FILL", f"数量比例 {qty_ratio}，疑似多笔成交被合并"
        return "BUS_PARTIAL_FILL", "同一委托多笔成交在清算侧合并"

    if diff_type == "AMOUNT_MISMATCH":
        if amount_ratio is not None and amount_ratio >= Decimal("10"):
            return "SRC_UNIT_MISMATCH", f"金额成 {amount_ratio} 倍关系，价格未变"
        if diff_amount is not None and abs(diff_amount) <= Decimal("0.02"):
            return "CAL_ROUNDING", f"金额差 {diff_amount}，在分位容差内"
        return "CAL_ROUNDING", "金额差异但无单位/费率证据"

    if diff_type == "FEE_MISMATCH":
        if diff_amount is not None and diff_fee is not None and diff_amount == 0 and rerun > 1:
            return "SYS_IDEMPOTENT", f"金额无差异而费用翻倍，且批次重跑 {rerun} 次"
        if diff_fee is not None and abs(diff_fee) <= Decimal("0.02") and (same_sign or 1) >= 0.9:
            return "CAL_TRUNCATION", f"费用差 {diff_fee} 且全批次同向"
        return "CAL_FEE_VERSION", f"费用差 {diff_fee}，与成交金额成比例"

    if diff_type == "CLEARING_DATE_MISMATCH":
        return "TML_CROSS_DAY", f"成交日 {trade_date} 被计入清算日 {clearing_date}"

    if diff_type in ("SETTLE_PENDING", "SETTLE_STATUS_MISMATCH"):
        if pending >= 5:
            return "TML_LATE_FILE", f"同批次待交收 {pending} 笔，疑似文件晚到"
        return "TML_STATUS_LAG", "文件按时到，仅单笔状态未回传"

    if diff_type == "CANCELLED_NOT_REFLECTED":
        return "BUS_CANCEL_ASYNC", "成交侧已撤单，清算侧未同步"

    if diff_type == "PARTIAL_FILL_AGGREGATED":
        return "BUS_PARTIAL_FILL", "同一 order_id 多笔成交被合并为一行"

    if diff_type == "SEQUENCE_VIOLATION":
        return "SYS_OUT_OF_ORDER", "交收记录生成早于清算记录"

    return "SRC_MISSING_LEG", "未知差异类型，按数据源缺失兜底"
