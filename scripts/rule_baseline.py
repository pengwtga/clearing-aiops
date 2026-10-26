"""规则基线：把"不上大模型、先写一套 if-else"能做到什么程度量出来

这不是为了证明规则不好，而是为了给后续 LLM 归因一个**诚实的对照组**：
- easy 集（模板化表达、字段齐全）：规则几乎全对——这是自己考自己，只能当冒烟测试；
- hard 集（字段被屏蔽、同一差异类型多因、口语化描述）：规则掉得厉害，
  因为它的判据是"差异类型 + 单字段阈值"，而真实差错恰恰是同象不同因。

规则设计刻意保持"第一版真实会写成什么样"：12 条基于 diff_type + 数量比例阈值的启发式，
不读 remark（自由文本），不看批次上下文。任何超出这个范围的规则都会让对照组失真。

用法：
    python scripts/rule_baseline.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
CATS = ["source_data", "timing", "calculation", "business", "system"]
CN = {"source_data": "数据源", "timing": "时点", "calculation": "计算", "business": "业务", "system": "系统"}


def predict(ev):
    """12 条启发式：只看差异类型和数量比例"""
    t = ev["diff_type"]
    if t in ("MISSING_IN_CLEARING", "DUP_IN_CLEARING", "CODE_FORMAT_MISMATCH"):
        return "source_data"
    if t == "QTY_MISMATCH":
        return "source_data" if (ev.get("qty_ratio") or 1) >= 10 else "business"
    if t in ("AMOUNT_MISMATCH", "FEE_MISMATCH"):
        return "calculation"
    if t in ("CLEARING_DATE_MISMATCH", "SETTLE_PENDING"):
        return "timing"
    if t in ("CANCELLED_NOT_REFLECTED", "PARTIAL_FILL_AGGREGATED"):
        return "business"
    if t in ("SEQUENCE_VIOLATION", "DUP_IN_SETTLEMENT"):
        return "system"
    return "source_data"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate(name, rows):
    labels = [r["label"] for r in rows]
    preds = [predict(r["diff"]) for r in rows]
    correct = sum(1 for a, b in zip(labels, preds) if a == b)
    print(f"\n=== {name}  n={len(rows)}  准确率 {correct / len(rows):.1%} ===")

    print(f"  {'大类':<8}{'precision':>11}{'recall':>9}{'f1':>8}{'样本':>6}")
    per = {}
    for c in CATS:
        tp = sum(1 for a, p in zip(labels, preds) if a == c and p == c)
        fp = sum(1 for a, p in zip(labels, preds) if a != c and p == c)
        fn = sum(1 for a, p in zip(labels, preds) if a == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[c] = (pr, rc, f1)
        print(f"  {CN[c]:<8}{pr:>10.1%}{rc:>9.1%}{f1:>8.2f}{labels.count(c):>6}")

    print("\n  混淆矩阵（行=真值，列=规则判断）")
    print("  " + " " * 9 + "".join(f"{CN[c]:>7}" for c in CATS))
    matrix = defaultdict(Counter)
    for a, p in zip(labels, preds):
        matrix[a][p] += 1
    for c in CATS:
        print(f"  {CN[c]:<9}" + "".join(f"{matrix[c].get(x, 0):>7}" for x in CATS))

    bad = [(r, p) for r, p in zip(rows, preds) if r["label"] != p]
    if bad:
        print("\n  规则判错的样本（人工看一眼就知道为什么规则不行）：")
        for r, p in bad[:6]:
            ev = r["diff"]
            print(f"    {ev['diff_type']:<24} 真值={CN[r['label']]} 规则判={CN[p]} "
                  f"| ratio={ev.get('qty_ratio')} diff_fee={ev.get('diff_fee')} "
                  f"| {ev['remark']}")
    return correct / len(rows)


def main():
    print("规则基线（12 条启发式，不看自由文本、不看批次上下文）")
    easy = evaluate("easy 集", load(ROOT / "data" / "diffs.jsonl"))
    hard = evaluate("hard 集", load(ROOT / "data" / "hard" / "diffs.jsonl"))
    print(f"\n结论：easy {easy:.1%} → hard {hard:.1%}，下降 {(easy - hard) * 100:.1f} 个点。")
    print("模板化数据上规则几乎全对（自己考自己），真实语境下同象不同因才是常态。")


if __name__ == "__main__":
    main()
