"""置信度校准 —— 回答"0.85 这个阈值是怎么来的"

门禁阈值不能拍脑袋。做法是把它当成分拣问题来算：
1. 把模型自报的置信度分桶，看每个桶里的**实际准确率**（可靠性表）；
2. 横扫阈值，得到每个阈值下的「自动覆盖率 × 自动子集准确率」曲线；
3. 在满足目标准确率（默认 95%）的前提下，取覆盖率最大的那个阈值。

这一步会暴露一个真实问题：**小模型普遍过度自信**——它给的 0.9 未必真有 9 成把握。
如果分桶后发现 0.85~0.95 桶的实际准确率远低于 0.85，说明模型自报分数不能直接当概率用，
要么把阈值调高（牺牲覆盖率），要么换成"证据充分性"这类结构性信号做二次判定。

用法：
    python scripts/eval_attributor.py        # 先跑评测，产出 data/eval_*.jsonl
    python scripts/calibrate.py              # 再看校准曲线与建议阈值
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUCKETS = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.001)]
TARGET_ACC = 0.95


def load(name: str) -> list[dict]:
    path = ROOT / "data" / f"eval_{name}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def model_confidence(row: dict) -> float | None:
    """模型自己给的分数：正常判定取 confidence；被门禁拦下的取 llm_confidence"""
    if row["source"] == "llm":
        return row["confidence"]
    if row.get("llm_confidence") is not None:
        return row["llm_confidence"]
    return None


def reliability(rows: list[dict]) -> None:
    print(f"\n{'置信度区间':<14}{'样本':>6}{'大类准确率':>12}{'子类准确率':>12}   {'校准偏差':>8}")
    print("-" * 60)
    for lo, hi in BUCKETS:
        bucket = [r for r in rows if (c := model_confidence(r)) is not None and lo <= c < hi]
        if not bucket:
            continue
        cat = sum(r["category_correct"] for r in bucket) / len(bucket)
        sub = sum(r["subcause_correct"] for r in bucket) / len(bucket)
        mid = (lo + min(hi, 1.0)) / 2
        gap = cat - mid
        flag = "过度自信" if gap < -0.1 else ("偏保守" if gap > 0.1 else "较准")
        print(f"[{lo:.2f}, {min(hi, 1.0):.2f}){'':<2}{len(bucket):>6}{cat:>11.1%}{sub:>12.1%}   {gap:+.2f} {flag}")


def sweep(rows: list[dict], target: float) -> None:
    print(f"\n阈值扫描（目标：自动子集准确率 ≥ {target:.0%}）")
    print(f"  {'阈值':>6}{'自动条数':>10}{'覆盖率':>10}{'自动子集准确率':>16}{'被拦条数':>10}")
    print("  " + "-" * 54)
    best = None
    for t in [0.0, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98]:
        auto = [r for r in rows if (c := model_confidence(r)) is not None and c >= t]
        if not auto:
            continue
        acc = sum(r["category_correct"] for r in auto) / len(auto)
        cov = len(auto) / len(rows)
        mark = ""
        if acc >= target and (best is None or cov > best[2]):
            best = (t, acc, cov)
        print(f"  {t:>6.2f}{len(auto):>10}{cov:>10.1%}{acc:>16.1%}{len(rows) - len(auto):>10}{mark}")

    if best:
        t, acc, cov = best
        print(f"\n  建议阈值 {t:.2f}：自动判定 {cov:.1%}，该子集大类准确率 {acc:.1%}")
        print(f"  含义：机器独立扛 {cov:.0%} 的差异单（准确率 {acc:.0%}），其余转人工复核。")
    else:
        print(f"\n  ⚠ 任何阈值都达不到 {target:.0%} 自动准确率 —— 模型自报置信度不可信，"
              f"需要引入结构性信号（证据充分性、一致性投票）或换更强模型。")


def main() -> None:
    easy, hard = load("easy"), load("hard")
    if not (easy or hard):
        print("没有评测结果，先跑 python scripts/eval_attributor.py")
        return

    # 分集看：easy 集几乎全对，混在一起会把 hard 集的问题稀释掉
    for name, rows in (("easy", easy), ("hard", hard)):
        scored = [r for r in rows if model_confidence(r) is not None]
        if not scored:
            continue
        print(f"\n{'#' * 60}\n# {name} 集：{len(scored)} 条有模型置信度分数\n{'#' * 60}")
        print("\n准确率 vs 自报置信度（可靠性表）")
        reliability(scored)
        sweep(scored, TARGET_ACC)

    rows = [r for r in easy + hard if model_confidence(r) is not None]
    if rows:
        print(f"\n{'#' * 60}\n# 合并（{len(rows)} 条）：生产上要按这条曲线定阈值\n{'#' * 60}")
        reliability(rows)
        sweep(rows, TARGET_ACC)


if __name__ == "__main__":
    main()
