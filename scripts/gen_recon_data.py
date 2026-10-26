"""合成对账数据生成器 —— 三方明细 + 差错注入 + 差异单

产出（easy 一套 / hard 一套）：
    trades.csv          成交明细（模拟交易所/券商回报）
    clearing.csv        清算明细（模拟中国结算口径文件）
    settlement.csv      资金交收明细（含状态、文件到达时间）
    batch_manifest.json 批次上下文（重跑次数、权益登记日）
    diffs.jsonl         对账引擎跑出来的差异单 + 真值（label / subcause / expected_action）

为什么先生成"正确的三方数据"再注入差错：这样每条差错都有确定的真值和唯一成因，
差异单是引擎真跑出来的，不是手写标注——数据可信，模型评测才有意义。

用法：
    python scripts/gen_recon_data.py            # 两套都生成
    python scripts/gen_recon_data.py easy       # 只生成 easy
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from reconcile import build_events, number_events
from remarks import REMARKS
from taxonomy import FEES, FEES_OLD, SUBCAUSES

ROOT = Path(__file__).parent.parent
CENT = Decimal("0.01")

SECURITIES = [
    ("600519", "贵州茅台", Decimal("1672.00")),
    ("601318", "中国平安", Decimal("52.30")),
    ("600036", "招商银行", Decimal("38.90")),
    ("601899", "紫金矿业", Decimal("17.80")),
    ("688981", "中芯国际", Decimal("88.40")),
    ("000001", "平安银行", Decimal("11.20")),
    ("000858", "五粮液", Decimal("140.20")),
    ("002594", "比亚迪", Decimal("245.60")),
    ("300750", "宁德时代", Decimal("182.50")),
    ("510300", "沪深300ETF", Decimal("3.95")),
]

ACCOUNTS = [f"A88123{i:03d}" for i in range(1, 13)]

NEXT_DAY = {
    "2026-10-12": "2026-10-13", "2026-10-13": "2026-10-14", "2026-10-14": "2026-10-15",
    "2026-10-15": "2026-10-16", "2026-10-16": "2026-10-19",
    "2026-10-19": "2026-10-20", "2026-10-20": "2026-10-21", "2026-10-21": "2026-10-22",
    "2026-10-22": "2026-10-23", "2026-10-23": "2026-10-26",
    "2026-10-26": "2026-10-27", "2026-10-27": "2026-10-28",
}

CONFIGS = {
    "easy": {
        "out": ROOT / "data",
        "seed": 20261026,
        "days": ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"],
        "trades_per_day": 60,
        "partial_orders_per_day": 1,
        "hard": False,
        "counts": {
            "SRC_MISSING_LEG": 8, "SRC_DUP_LEG": 6, "SRC_CODE_FORMAT": 7, "SRC_UNIT_MISMATCH": 7,
            "TML_CROSS_DAY": 7, "TML_LATE_FILE": 7, "TML_STATUS_LAG": 6,
            "CAL_ROUNDING": 8, "CAL_TRUNCATION": 6, "CAL_FEE_VERSION": 7,
            "BUS_CANCEL_ASYNC": 6, "BUS_PARTIAL_FILL": 5, "BUS_CORP_ACTION": 6,
            "SYS_OUT_OF_ORDER": 6, "SYS_IDEMPOTENT": 5,
        },
    },
    "hard": {
        "out": ROOT / "data" / "hard",
        "seed": 20261027,
        "days": ["2026-10-13", "2026-10-14", "2026-10-15"],
        "trades_per_day": 45,
        "partial_orders_per_day": 1,
        "hard": True,
        "counts": {k: 2 for k in SUBCAUSES},
    },
}


# ---------------------------------------------------------------- 金额/费用
def calc_fees(amount: Decimal, side: str, code: str, cfg=None):
    cfg = cfg or FEES
    comm = max(amount * cfg["commission_rate"], cfg["commission_min"]).quantize(CENT, ROUND_HALF_UP)
    stamp = (amount * cfg["stamp_rate"]).quantize(CENT, ROUND_HALF_UP) if side == "S" else Decimal("0.00")
    transfer = (
        (amount * cfg["transfer_rate"]).quantize(CENT, ROUND_HALF_UP)
        if code.startswith(("60", "68")) else Decimal("0.00")
    )
    return comm, stamp, transfer, (comm + stamp + transfer)


def ts(stamp_str: str) -> int:
    return int(datetime.strptime(stamp_str, "%Y-%m-%d %H:%M:%S").timestamp())


# ---------------------------------------------------------------- 原始数据
def gen_trade(day: str, seq: int, order_id: str) -> dict:
    code, name, base = random.choice(SECURITIES)
    price = (base * Decimal(str(random.uniform(0.97, 1.03)))).quantize(CENT, ROUND_HALF_UP)
    qty = random.randrange(1, 40) * 100
    amount = (price * qty).quantize(CENT, ROUND_HALF_UP)
    side = random.choice(["B", "S"])
    if random.random() < 0.5:
        hh = random.randint(9, 11)
        mm = random.randint(30 if hh == 9 else 0, 59)
    else:
        hh = random.randint(13, 14)
        mm = random.randint(0, 59)
    return {
        "trade_id": f"T{day.replace('-', '')}{seq:05d}",
        "order_id": order_id,
        "account": random.choice(ACCOUNTS),
        "security_code": code,
        "security_name": name,
        "side": side,
        "price": price,
        "qty": qty,
        "amount": amount,
        "trade_date": day,
        "trade_time": f"{hh:02d}:{mm:02d}:{random.randint(0, 59):02d}",
        "status": "FILLED",
        "channel": random.choice(["WEB", "APP", "API"]),
    }


def gen_day_trades(day: str, n_orders: int, n_partial: int):
    trades, partial = [], []
    order_slots = list(range(n_orders))
    planned = set(random.sample(order_slots, n_partial))
    seq = 0
    for i in order_slots:
        oid = f"O{day.replace('-', '')}{i + 1:04d}"
        fills = random.randint(2, 4) if i in planned else 1
        if fills > 1:
            partial.append(oid)
        for _ in range(fills):
            seq += 1
            trades.append(gen_trade(day, seq, oid))
    return trades, partial


def build_clearing(trades, partial_orders):
    """成交 → 清算。部分成交的委托，清算侧合并成一行（clear_id 关联委托号）"""
    groups = defaultdict(list)
    for t in trades:
        groups[t["order_id"]].append(t)

    rows = []
    for oid, ts_list in groups.items():
        first = ts_list[0]
        if oid in partial_orders:
            qty = sum(x["qty"] for x in ts_list)
            amount = sum(x["amount"] for x in ts_list)
            fee = sum(calc_fees(x["amount"], x["side"], x["security_code"])[3] for x in ts_list)
            rows.append({
                "clear_id": "C" + oid[1:],
                "trade_id": "",
                "order_id": oid,
                "account": first["account"], "security_code": first["security_code"],
                "security_name": first["security_name"], "side": first["side"],
                "price": first["price"], "qty": qty, "amount": amount,
                "fee_total": fee,
                "trade_date": first["trade_date"], "clearing_date": first["trade_date"],
                "settle_date": NEXT_DAY[first["trade_date"]],
                "created_at": None, "created_at_ts": None,
            })
        else:
            comm, stamp, transfer, total = calc_fees(first["amount"], first["side"], first["security_code"])
            rows.append({
                "clear_id": "C" + first["trade_id"][1:],
                "trade_id": first["trade_id"],
                "order_id": oid,
                "account": first["account"], "security_code": first["security_code"],
                "security_name": first["security_name"], "side": first["side"],
                "price": first["price"], "qty": first["qty"], "amount": first["amount"],
                "fee_total": total,
                "trade_date": first["trade_date"], "clearing_date": first["trade_date"],
                "settle_date": NEXT_DAY[first["trade_date"]],
                "created_at": None, "created_at_ts": None,
            })
    return rows


def build_settlements(clearing):
    """清算 → 交收，按 clear_id 去重（清算文件重复行不应在交收侧再复制一份）"""
    rows, seen = [], set()
    for c in clearing:
        if c["clear_id"] in seen:
            continue
        seen.add(c["clear_id"])
        amount = c["amount"]
        fee = c["fee_total"]
        settle_amount = (amount + fee) if c["side"] == "B" else (amount - fee)
        rows.append({
            "settle_id": "S" + c["clear_id"][1:],
            "clear_id": c["clear_id"],
            "account": c["account"], "security_code": c["security_code"], "side": c["side"],
            "settle_amount": settle_amount, "settle_date": c["settle_date"],
            "settle_status": "SETTLED",
            "trade_date": c["trade_date"],
            "created_at": None, "created_at_ts": None,
            "file_arrive_time": "19:35",
        })
    return rows


def stamp_clearing(clearing):
    """清算任务在 T 日 18:05~19:10 跑批"""
    for c in clearing:
        s = f"{c['trade_date']} 18:{random.randint(5, 59):02d}:{random.randint(0, 59):02d}"
        c["created_at"], c["created_at_ts"] = s, ts(s)


def stamp_settle(settlements):
    """交收任务在 T+1 日 19:20~20:05 跑批（正常顺序：先清算后交收）"""
    for st in settlements:
        s = f"{st['settle_date']} 19:{random.randint(20, 59):02d}:{random.randint(0, 59):02d}"
        st["created_at"], st["created_at_ts"] = s, ts(s)


# ---------------------------------------------------------------- 差错注入
def make_taker(trades, partial_orders):
    """返回一个取目标成交的函数：不重复、可按条件筛选"""
    normal = [t for t in trades if t["order_id"] not in partial_orders]
    random.shuffle(normal)
    state = {"i": 0, "used": set()}

    def take(n=1, pred=None):
        out = []
        while len(out) < n and state["i"] < len(normal):
            t = normal[state["i"]]
            state["i"] += 1
            if t["trade_id"] in state["used"]:
                continue
            if pred and not pred(t):
                continue
            state["used"].add(t["trade_id"])
            out.append(t)
        if len(out) < n:
            raise RuntimeError("可注入的成交不足，请增加 trades_per_day")
        return out

    return take


def cid_of(trade):
    return "C" + trade["trade_id"][1:]


def inject_clearing(trades, clearing, manifest, partial_orders, cfg, truth, take):
    """第一阶段注入：只改成交/清算侧。必须在生成交收明细之前跑完，
    否则清算侧的金额差错会向下传染，多出一堆假的"清算↔交收不一致"。"""
    hard = cfg["hard"]
    counts = cfg["counts"]
    cl_idx = {c["clear_id"]: c for c in clearing}
    drop = set()

    # ---- 数据源类
    for t in take(counts["SRC_MISSING_LEG"]):
        drop.add(cid_of(t))
        truth[("trade", t["trade_id"])] = "SRC_MISSING_LEG"

    for i, t in enumerate(take(counts["SRC_DUP_LEG"])):
        clearing.append({**cl_idx[cid_of(t)]})
        # hard 集：其中一条其实是"批次重跑导致的重复"，唯一证据是重跑次数
        if hard and i == 0:
            manifest["B" + t["trade_date"].replace("-", "")]["rerun_count"] = 2
            truth[("trade", t["trade_id"])] = "SYS_IDEMPOTENT"
        else:
            truth[("trade", t["trade_id"])] = "SRC_DUP_LEG"

    for t in take(counts["SRC_CODE_FORMAT"]):
        row = cl_idx[cid_of(t)]
        code = row["security_code"]
        row["security_code"] = code + (".SH" if code.startswith(("60", "68")) else ".SZ")
        truth[("trade", t["trade_id"])] = "SRC_CODE_FORMAT"

    for i, t in enumerate(take(counts["SRC_UNIT_MISMATCH"])):
        row = cl_idx[cid_of(t)]
        if hard and i == 0:
            row["amount"] = (Decimal(row["amount"]) * 100).quantize(CENT)   # 金额单位错
        elif hard and i == 1:
            row["qty"] = max(int(Decimal(row["qty"]) / 100), 1)             # 股 → 手（反向）
        else:
            row["qty"] = int(Decimal(row["qty"]) * 100)                     # 手 → 股
        truth[("trade", t["trade_id"])] = "SRC_UNIT_MISMATCH"

    # ---- 时点类（跨日切分）
    for i, t in enumerate(take(counts["TML_CROSS_DAY"])):
        row = cl_idx[cid_of(t)]
        nxt = NEXT_DAY[row["clearing_date"]]
        t["trade_time"] = f"14:5{random.randint(0, 9)}:{random.randint(10, 59):02d}"
        truth[("trade", t["trade_id"])] = "TML_CROSS_DAY"
        if hard and i == 0:
            # hard：该笔被切到次日批次、编号也换了 → 差异单上是"漏发"+次日"孤儿行"，
            # 只有把两张单连起来看、再看成交时间，才知道是切分问题
            new_cid = "C" + nxt.replace("-", "") + "X" + row["clear_id"][-5:]
            row["clear_id"] = new_cid
            row["trade_id"] = ""
            truth[("clear", new_cid)] = "TML_CROSS_DAY"
        row["clearing_date"] = nxt
        row["settle_date"] = NEXT_DAY[nxt]

    # ---- 计算类
    for i, t in enumerate(take(counts["CAL_ROUNDING"])):
        row = cl_idx[cid_of(t)]
        row["amount"] = Decimal(row["amount"]) + (CENT if i % 2 == 0 else -CENT)
        truth[("trade", t["trade_id"])] = "CAL_ROUNDING"

    for t in take(counts["CAL_TRUNCATION"]):
        row = cl_idx[cid_of(t)]
        row["fee_total"] = max(Decimal(row["fee_total"]) - CENT, Decimal("0.00"))
        truth[("trade", t["trade_id"])] = "CAL_TRUNCATION"

    # 费率版本：只挑成交金额够大的（小额会撞最低佣金，改了费率也看不出差）
    for t in take(counts["CAL_FEE_VERSION"], pred=lambda x: x["amount"] > Decimal("200000")):
        row = cl_idx[cid_of(t)]
        comm, stamp, transfer, total = calc_fees(Decimal(row["amount"]), row["side"], row["security_code"], FEES_OLD)
        if total == Decimal(row["fee_total"]):        # 兜底：撞最低佣金则跳过
            truth.pop(("trade", t["trade_id"]), None)
            continue
        row["fee_total"] = total
        truth[("trade", t["trade_id"])] = "CAL_FEE_VERSION"

    # ---- 业务类
    for t in take(counts["BUS_CANCEL_ASYNC"]):
        t["status"] = "CANCELLED"
        truth[("trade", t["trade_id"])] = "BUS_CANCEL_ASYNC"

    for oid in sorted(partial_orders):   # 每个多笔成交委托都是一张差异单
        truth[("order", oid)] = "BUS_PARTIAL_FILL"

    for t in take(counts["BUS_CORP_ACTION"]):
        row = cl_idx[cid_of(t)]
        row["qty"] = int(Decimal(row["qty"]) * Decimal("1.3"))
        if not hard:   # hard 集屏蔽权益日历，只能靠数量比例推断
            manifest["B" + t["trade_date"].replace("-", "")].setdefault("corp_action", []).append(row["security_code"])
        truth[("trade", t["trade_id"])] = "BUS_CORP_ACTION"

    if hard:
        # 重跑导致费用累加：看起来是"费用算错"，根因其实在调度
        t = take(1)[0]
        row = cl_idx[cid_of(t)]
        row["fee_total"] = Decimal(row["fee_total"]) * 2
        manifest["B" + t["trade_date"].replace("-", "")]["rerun_count"] = 2
        truth[("trade", t["trade_id"])] = "SYS_IDEMPOTENT"

        # 分片漏跑：同一账户当日成片缺失（不是随机单笔），批次记录有分片未完成标记
        cand = {}
        for tr in trades:
            if ("trade", tr["trade_id"]) in truth or tr["order_id"] in partial_orders or tr["status"] != "FILLED":
                continue
            cand.setdefault((tr["trade_date"], tr["account"]), []).append(tr)
        pick = next((v for _, v in sorted(cand.items()) if 2 <= len(v) <= 5), None)
        if pick:
            day, acc = pick[0]["trade_date"], pick[0]["account"]
            manifest["B" + day.replace("-", "")].setdefault("missed_shards", []).append(acc)
            for tr in pick:
                drop.add(cid_of(tr))
                truth[("trade", tr["trade_id"])] = "SYS_SHARD_MISSING"

    # 漏发：清算与交收一起缺（符合"整笔未下发"的现实）
    return [c for c in clearing if c["clear_id"] not in drop]


def inject_settle(settlements, manifest, cfg, truth, take, late_day):
    """第二阶段注入：只改交收侧（状态、顺序、重复）"""
    counts = cfg["counts"]
    st_idx = {s["clear_id"]: s for s in settlements if s["settle_status"] == "SETTLED"}

    # 交收文件晚到：同一批次多笔同时未回状态（批次级证据，不是单笔问题）
    late = [s for s in settlements if s["trade_date"] == late_day][: counts["TML_LATE_FILE"]]
    for s in late:
        s["settle_status"] = "PENDING"
        s["file_arrive_time"] = "21:47"
        truth[("settle", s["clear_id"])] = "TML_LATE_FILE"

    late_ids = {s["clear_id"] for s in late}

    # 状态回传滞后：文件按时到，只单笔 PENDING
    for t in take(counts["TML_STATUS_LAG"], pred=lambda x: x["trade_date"] != late_day):
        st = st_idx.get(cid_of(t))
        if not st or st["clear_id"] in late_ids:
            continue
        st["settle_status"] = "PENDING"
        truth[("settle", cid_of(t))] = "TML_STATUS_LAG"

    # 任务乱序：交收记录生成时间早于清算记录（同一天，早于 18:05 的清算批次）
    for t in take(counts["SYS_OUT_OF_ORDER"]):
        st = st_idx.get(cid_of(t))
        if not st or st["settle_status"] != "SETTLED":
            continue
        new_ts = ts(f"{t['trade_date']} 15:{random.randint(10, 50):02d}:00")
        st["created_at_ts"] = new_ts
        st["created_at"] = datetime.fromtimestamp(new_ts).strftime("%Y-%m-%d %H:%M:%S")
        truth[("settle", cid_of(t))] = "SYS_OUT_OF_ORDER"

    # 重复交收记录：可能是批次重跑（系统），也可能是文件重复下发（数据源）
    for i, t in enumerate(take(counts["SYS_IDEMPOTENT"])):
        st = st_idx.get(cid_of(t))
        if not st or st["settle_status"] != "SETTLED":
            continue
        settlements.append({**st})
        if cfg["hard"] and i == 0:
            truth[("settle", cid_of(t))] = "SRC_DUP_LEG"      # 批次未重跑，纯文件重复
        else:
            manifest["B" + t["trade_date"].replace("-", "")]["rerun_count"] = 2
            truth[("settle", cid_of(t))] = "SYS_IDEMPOTENT"

    return settlements


# ---------------------------------------------------------------- 输出
def attach_truth(events, truth, cfg):
    """给差异单挂真值 + 自由文本备注；hard 集屏蔽部分字段"""
    out, missing = [], []
    hard = cfg["hard"]
    for ev in events:
        key = ev.pop("_key", None)
        sub = truth.get(key)
        if not sub:
            missing.append(ev)
            continue
        meta = SUBCAUSES[sub]
        ev["remark"] = random.choice(REMARKS[sub]["hard" if hard else "easy"]).format(
            amount=f"{ev['trade_amount']:.2f}" if ev.get("trade_amount") else "-",
            qty=ev.get("trade_qty") or "-",
            fee_diff=f"{ev['diff_fee']:.2f}" if ev.get("diff_fee") else "-",
            fill_count=ev.get("fill_count") or "-",
        )
        ev["severity"] = meta["severity"]
        if hard:
            # 差异单不带成交流水时间、不带权益日历：逼着从数量比例/批次证据推理
            ev["trade_time"] = None
            ev["corp_action_today"] = None
            ev["file_arrive_time"] = None if sub in ("TML_LATE_FILE", "TML_STATUS_LAG") else ev["file_arrive_time"]
        out.append({
            "diff": ev,
            "label": meta["category"],
            "subcause": sub,
            "subcause_name": meta["name"],
            "expected_action": meta["action"],
        })
    return out, missing


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (format(v, "f") if isinstance(v, Decimal) else v)) for k, v in r.items()})


TRADE_FIELDS = ["trade_id", "order_id", "account", "security_code", "security_name", "side",
                "price", "qty", "amount", "fee_total", "trade_date", "trade_time", "status", "channel"]
CLEARING_FIELDS = ["clear_id", "trade_id", "order_id", "account", "security_code", "security_name",
                   "side", "price", "qty", "amount", "fee_total", "trade_date", "clearing_date",
                   "settle_date", "created_at", "created_at_ts"]
SETTLE_FIELDS = ["settle_id", "clear_id", "account", "security_code", "side", "settle_amount",
                 "settle_date", "settle_status", "trade_date", "created_at", "created_at_ts",
                 "file_arrive_time"]


def run(name):
    cfg = CONFIGS[name]
    random.seed(cfg["seed"])

    trades, partial_orders = [], []
    for day in cfg["days"]:
        day_trades, day_partial = gen_day_trades(day, cfg["trades_per_day"], cfg["partial_orders_per_day"])
        trades += day_trades
        partial_orders += day_partial

    for t in trades:
        t["fee_total"] = calc_fees(t["amount"], t["side"], t["security_code"])[3]

    partial_set = set(partial_orders)
    clearing = build_clearing(trades, partial_set)
    stamp_clearing(clearing)

    manifest = {"B" + d.replace("-", ""): {"rerun_count": 1, "corp_action": [], "file_arrive_time": "19:35"}
                for d in cfg["days"]}

    truth = {}
    take = make_taker(trades, partial_set)
    clearing = inject_clearing(trades, clearing, manifest, partial_set, cfg, truth, take)

    # 交收明细基于"已注入差错的清算结果"生成 —— 现实里交收就是清算的下游
    settlements = build_settlements(clearing)
    stamp_settle(settlements)
    settlements = inject_settle(settlements, manifest, cfg, truth, take, late_day=cfg["days"][1])

    events = number_events(build_events(trades, clearing, settlements, manifest))
    rows, missing = attach_truth(events, truth, cfg)

    out = cfg["out"]
    write_csv(out / "trades.csv", trades, TRADE_FIELDS)
    write_csv(out / "clearing.csv", clearing, CLEARING_FIELDS)
    write_csv(out / "settlement.csv", settlements, SETTLE_FIELDS)
    (out / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "diffs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report(name, cfg, trades, clearing, settlements, rows, missing, truth)
    return rows


def report(name, cfg, trades, clearing, settlements, rows, missing, truth):
    print(f"\n=== {name} ===")
    print(f"成交 {len(trades)} 笔 | 清算 {len(clearing)} 行 | 交收 {len(settlements)} 行 | 差异单 {len(rows)} 张")
    if missing:
        print(f"⚠️  未匹配真值的差异单 {len(missing)} 张（数据集真值不完整）")
        for m in missing[:8]:
            print("   ", m["diff_type"], m["clear_id"], m["trade_id"], m.get("settle_status"))

    by_cat = Counter(r["label"] for r in rows)
    print("按大类：", dict(by_cat))
    by_sub = Counter(r["subcause"] for r in rows)
    for k in SUBCAUSES:
        print(f"  {k:<20} {by_sub.get(k, 0)}")

    cross = defaultdict(Counter)
    for r in rows:
        cross[r["diff"]["diff_type"]][r["label"]] += 1
    print("\ndiff_type × 大类（同一差异类型对应多种根因，是规则失效的根源）：")
    cats = ["source_data", "timing", "calculation", "business", "system"]
    print(f"  {'diff_type':<26}" + "".join(f"{c[:6]:>8}" for c in cats))
    for dt in sorted(cross, key=lambda x: -sum(cross[x].values())):
        print(f"  {dt:<26}" + "".join(f"{cross[dt].get(c, 0):>8}" for c in cats))


def main():
    targets = sys.argv[1:] or ["easy", "hard"]
    for name in targets:
        run(name)


if __name__ == "__main__":
    main()
