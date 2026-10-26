"""三方对账引擎：成交（交易所/券商）↔ 清算（中国结算口径）↔ 交收（资金）

纯函数实现，只用标准库。输入三方明细，输出差异单（diff ticket）事件 —— 也就是后面对账
归因模型的输入。

四条工程设计（都是真实对账系统踩过的点）：
1. **匹配优先级**：1:1（成交编号）→ 1:N（委托号聚合，识别"部分成交未拆分"）→ 未匹配。
   少了 1:N 这一步，一笔部分成交会炸出 4 张差异单（3 笔成交漏发 + 1 笔孤儿），
   假阳性直接淹没真差错。
2. **编码归一化后再匹配**：证券代码去掉 .SH/.SZ、补前导零。编码格式差异单独出 P4 单，
   不能让它伪装成"漏发"——这是"数据源类"里最容易被误判的一种。
3. **一张差异单只承载一个结论**：按 撤单 > 编码 > 日期 > 数量 > 金额 > 费用 的优先级
   取第一个命中，避免同一笔重复出单。
4. **差异单自带批次上下文**：文件到达时间、重跑次数、同批次待交收笔数、同向比例。
   单笔字段判不了的（文件晚到 vs 状态滞后），只能靠批次级证据判。

用法：
    python scripts/reconcile.py data/            # 读三方 CSV，打印差异单统计
"""

import csv
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

EPS = Decimal("0.001")


# ---------------------------------------------------------------- 基础工具
def dec(v):
    if v is None or v == "":
        return None
    return Decimal(str(v))


def f(v):
    """Decimal / None → JSON 友好的 float / None（保留 2 位语义）"""
    return None if v is None else float(v)


def norm_code(code):
    """证券代码归一化：600519.SH / sh600519 / 600519 → 600519"""
    if not code:
        return ""
    c = str(code).strip().upper()
    for suf in (".SH", ".SZ", ".BJ"):
        if c.endswith(suf):
            c = c[: -len(suf)]
            break
    if len(c) > 6 and c[:2] in ("SH", "SZ", "BJ"):
        c = c[2:]
    return c.zfill(6) if c.isdigit() else c


def _qty(v):
    if v is None:
        return None
    d = Decimal(str(v))
    return int(d) if d == d.to_integral_value() else float(d)


# ---------------------------------------------------------------- 事件构造
def _base_event(trade=None, clearing=None, settle=None):
    """差异单公共字段：尽量把三方的可比字段摊平在一行，方便模型与规则同时消费"""
    src = trade or {}
    cl = clearing or {}
    st = settle or {}
    trade_date = src.get("trade_date") or cl.get("trade_date") or st.get("trade_date")
    return {
        "diff_id": None,               # 生成阶段统一编号
        "diff_type": None,
        "severity": None,
        "batch_id": ("B" + trade_date.replace("-", "")) if trade_date else None,
        "trade_date": trade_date,
        "clearing_date": cl.get("clearing_date"),
        "settle_date": cl.get("settle_date") or st.get("settle_date"),
        "account": src.get("account") or cl.get("account") or st.get("account"),
        "security_code": norm_code(src.get("security_code") or cl.get("security_code")),
        "security_name": src.get("security_name") or cl.get("security_name"),
        "side": src.get("side") or cl.get("side") or st.get("side"),
        "trade_id": src.get("trade_id"),
        "order_id": src.get("order_id"),
        "clear_id": cl.get("clear_id") or st.get("clear_id"),
        "trade_qty": _qty(src.get("qty")),
        "clear_qty": _qty(cl.get("qty")),
        "trade_amount": f(dec(src.get("amount"))),
        "clear_amount": f(dec(cl.get("amount"))),
        "trade_fee": f(dec(src.get("fee_total"))),
        "clear_fee": f(dec(cl.get("fee_total"))),
        "settle_amount": f(dec(st.get("settle_amount"))),
        "settle_status": st.get("settle_status"),
        "qty_ratio": None,
        "amount_ratio": None,
        "diff_qty": None,
        "diff_amount": None,
        "diff_fee": None,
        "trade_time": src.get("trade_time"),
        "clear_created_at": cl.get("created_at"),
        "settle_created_at": st.get("created_at"),
        "file_arrive_time": st.get("file_arrive_time"),
        "batch_rerun_count": None,
        "batch_pending_count": None,
        "batch_fee_diff_same_sign_ratio": None,
        "corp_action_today": None,
        "batch_missed_shards": None,
        "trade_status": src.get("status"),
        "legs_present": [k for k, v in (("trade", trade), ("clearing", clearing), ("settlement", settle)) if v],
        "remark": None,                # 值班/客服登记的自由文本（语义信息主要在这里）
        "_key": None,                  # 内部：关联真值用，落盘前删除
    }


def _compare_trade(trade, clearing):
    """成交 vs 清算，按优先级返回第一个命中结论"""
    ev = _base_event(trade=trade, clearing=clearing)

    # 1. 撤单未同步：成交侧已撤，清算仍计账
    if (trade.get("status") or "").upper() == "CANCELLED":
        return {**ev, "diff_type": "CANCELLED_NOT_REFLECTED", "severity": "P2"}

    # 2. 编码格式差异（归一化后能匹配，说明不是漏发）
    raw_code = str(clearing.get("security_code") or "")
    if raw_code != str(trade.get("security_code") or "") and norm_code(raw_code) == norm_code(trade.get("security_code")):
        return {**ev, "diff_type": "CODE_FORMAT_MISMATCH", "severity": "P4"}

    # 3. 清算日期错位（跨日切分）
    if clearing.get("clearing_date") and clearing["clearing_date"] != trade.get("trade_date"):
        return {**ev, "diff_type": "CLEARING_DATE_MISMATCH", "severity": "P3"}

    # 4. 数量
    tq, cq = dec(trade.get("qty")), dec(clearing.get("qty"))
    if tq and cq is not None and cq != tq:
        return {
            **ev, "diff_type": "QTY_MISMATCH", "severity": "P2",
            "qty_ratio": round(float(cq / tq), 6), "diff_qty": f(cq - tq),
        }

    # 5. 金额
    ta, ca = dec(trade.get("amount")), dec(clearing.get("amount"))
    if ta and ca is not None and abs(ca - ta) > EPS:
        return {
            **ev, "diff_type": "AMOUNT_MISMATCH", "severity": "P1",
            "diff_amount": f(ca - ta), "amount_ratio": round(float(ca / ta), 6),
        }

    # 6. 费用
    tf, cf = dec(trade.get("fee_total")), dec(clearing.get("fee_total"))
    if tf is not None and cf is not None and abs(cf - tf) > EPS:
        return {**ev, "diff_type": "FEE_MISMATCH", "severity": "P3", "diff_fee": f(cf - tf)}

    return None


def _compare_settle(clearing, settle):
    """清算 vs 交收"""
    ev = _base_event(clearing=clearing, settle=settle)

    if (settle.get("settle_status") or "").upper() == "PENDING":
        return {**ev, "diff_type": "SETTLE_PENDING", "severity": "P3"}

    cs, ss = dec(clearing.get("created_at_ts")), dec(settle.get("created_at_ts"))
    if cs is not None and ss is not None and ss < cs:
        return {**ev, "diff_type": "SEQUENCE_VIOLATION", "severity": "P1"}

    amount = dec(clearing.get("amount")) or Decimal("0")
    fee = dec(clearing.get("fee_total")) or Decimal("0")
    expected = amount + fee if clearing.get("side") == "B" else amount - fee
    actual = dec(settle.get("settle_amount"))
    if actual is not None and abs(actual - expected) > EPS:
        return {
            **ev, "diff_type": "SETTLE_AMOUNT_MISMATCH", "severity": "P1",
            "diff_amount": f(actual - expected),
        }
    return None


# ---------------------------------------------------------------- 主流程
def build_events(trades, clearing, settlements, manifest=None):
    """三方明细 → 差异单列表

    manifest: {batch_id: {"rerun_count": 1, "corp_action": [...], "file_arrive_time": "19:35"}}
    """
    manifest = manifest or {}

    cl_by_trade = defaultdict(list)
    cl_by_id = defaultdict(list)
    for c in clearing:
        cl_by_id[c["clear_id"]].append(c)
        if c.get("trade_id"):
            cl_by_trade[c["trade_id"]].append(c)

    st_by_clear = defaultdict(list)
    for s in settlements:
        st_by_clear[s["clear_id"]].append(s)

    order_groups = defaultdict(list)
    for t in trades:
        order_groups[t["order_id"]].append(t)

    events = []
    consumed = set()
    handled_orders = set()

    # --- 阶段一：成交 → 清算
    for t in trades:
        rows = cl_by_trade.get(t["trade_id"]) or []

        if len(rows) == 1:
            consumed.add(rows[0]["clear_id"])
            ev = _compare_trade(t, rows[0])
            if ev:
                ev["_key"] = ("trade", t["trade_id"])
                events.append(ev)
            continue

        if len(rows) > 1:                       # 同一成交编号多行 → 重复行
            consumed.update(r["clear_id"] for r in rows)
            ev = {**_base_event(trade=t, clearing=rows[0]),
                  "diff_type": "DUP_IN_CLEARING", "severity": "P3"}
            ev["_key"] = ("trade", t["trade_id"])
            events.append(ev)
            continue

        # 无 1:1 → 尝试 1:N 聚合（部分成交未拆分）
        oid = t["order_id"]
        group = order_groups[oid]
        merged = cl_by_id.get("C" + (oid[1:] if oid.startswith("O") else oid))
        if len(group) > 1 and merged:
            if oid not in handled_orders:
                handled_orders.add(oid)
                consumed.add("C" + (oid[1:] if oid.startswith("O") else oid))
                sum_qty = sum(dec(x["qty"]) for x in group)
                sum_amt = sum(dec(x["amount"]) for x in group)
                ev = {**_base_event(trade=group[0], clearing=merged[0]),
                      "diff_type": "PARTIAL_FILL_AGGREGATED", "severity": "P3"}
                ev["fill_count"] = len(group)
                ev["trade_qty"] = _qty(sum_qty)
                ev["trade_amount"] = f(sum_amt)
                ev["_key"] = ("order", oid)
                events.append(ev)
            continue

        ev = {**_base_event(trade=t), "diff_type": "MISSING_IN_CLEARING", "severity": "P2"}
        ev["_key"] = ("trade", t["trade_id"])
        events.append(ev)

    # 孤儿清算行（成交侧找不到对应，且未被 1:N 消费）
    for cid, rows in cl_by_id.items():
        if cid in consumed:
            continue
        ev = {**_base_event(clearing=rows[0]), "diff_type": "ORPHAN_IN_CLEARING", "severity": "P2"}
        ev["_key"] = ("clear", cid)
        events.append(ev)

    # --- 阶段二：清算 → 交收
    for cid, rows in cl_by_id.items():
        c = rows[0]
        sts = st_by_clear.get(cid) or []

        if len(sts) > 1:
            ev = {**_base_event(clearing=c, settle=sts[0]),
                  "diff_type": "DUP_IN_SETTLEMENT", "severity": "P2"}
            ev["_key"] = ("settle", cid)
            events.append(ev)
            continue

        if not sts:
            ev = {**_base_event(clearing=c), "diff_type": "MISSING_IN_SETTLEMENT", "severity": "P2"}
            ev["_key"] = ("settle", cid)
            events.append(ev)
            continue

        ev = _compare_settle(c, sts[0])
        if ev:
            ev["_key"] = ("settle", cid)
            events.append(ev)

    # --- 阶段三：批次上下文（单笔字段判不了的问题，只能靠批次级证据）
    pending = defaultdict(int)
    for s in settlements:
        if (s.get("settle_status") or "").upper() == "PENDING":
            pending["B" + s["trade_date"].replace("-", "")] += 1

    fee_signs = defaultdict(list)
    for ev in events:
        if ev["diff_type"] == "FEE_MISMATCH" and ev["diff_fee"]:
            fee_signs[ev["batch_id"]].append(1 if float(ev["diff_fee"]) > 0 else -1)

    # 交收侧信息回填：成交/清算阶段的差异单也要能看到交收状态（对账员看的是全链路）
    for ev in events:
        cid = ev.get("clear_id")
        if ev.get("settle_status") is None and cid in st_by_clear:
            st = st_by_clear[cid][0]
            ev["settle_status"] = st.get("settle_status")
            ev["settle_amount"] = f(dec(st.get("settle_amount")))
            ev["settle_created_at"] = st.get("created_at")

    for ev in events:
        b = ev["batch_id"]
        meta = manifest.get(b, {})
        ev["batch_rerun_count"] = meta.get("rerun_count", 1)
        ev["batch_pending_count"] = pending.get(b, 0)
        signs = fee_signs.get(b) or []
        ev["batch_fee_diff_same_sign_ratio"] = (
            round(max(signs.count(1), signs.count(-1)) / len(signs), 3) if signs else None
        )
        ev["corp_action_today"] = ev["security_code"] in (meta.get("corp_action") or [])
        ev["batch_missed_shards"] = meta.get("missed_shards") or []
        if ev["file_arrive_time"] is None:
            ev["file_arrive_time"] = meta.get("file_arrive_time")

    return events


def number_events(events):
    """按批次编号，稳定顺序：交易日 + 类型"""
    events.sort(key=lambda e: (e["trade_date"] or "", e["diff_type"] or "", e["clear_id"] or ""))
    seq = defaultdict(int)
    for ev in events:
        b = (ev["trade_date"] or "00000000").replace("-", "")
        seq[b] += 1
        ev["diff_id"] = f"DF{b}-{seq[b]:04d}"
    return events


# ---------------------------------------------------------------- CSV 读写（直接运行用）
def load_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_bundle(data_dir):
    d = Path(data_dir)
    manifest_file = d / "batch_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    return {
        "trades": load_csv(d / "trades.csv"),
        "clearing": load_csv(d / "clearing.csv"),
        "settlements": load_csv(d / "settlement.csv"),
        "manifest": manifest,
    }


def reconcile_dir(data_dir):
    b = load_bundle(data_dir)
    return number_events(build_events(b["trades"], b["clearing"], b["settlements"], b["manifest"]))


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data")
    events = reconcile_dir(data_dir)
    print(f"差异单 {len(events)} 张（数据目录：{data_dir}）")
    by_type = defaultdict(int)
    for e in events:
        by_type[e["diff_type"]] += 1
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {k:<26} {v}")


if __name__ == "__main__":
    main()
