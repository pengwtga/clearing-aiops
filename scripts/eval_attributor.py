"""归因层评测 —— 与规则基线在同一批样本上对比，并把门禁的代价算清楚

评测回答四个问题，缺一个都不算讲清楚了这个系统：
1. **准不准**：大类准确率、子类准确率（子类更难，但处置动作靠它）
2. **敢不敢自动**：门禁后自动判定的覆盖率，以及该子集上的准确率
   —— 覆盖率 100% 说明门禁形同虚设；覆盖率 30% 说明系统没有实用价值
3. **拦对了吗**：被拦下的样本里，模型原本判错的比例（拦截精确率）
4. **拦错了多少**：模型判对但被拦下转人工的比例（门禁的机会成本）

用法：
    python scripts/eval_attributor.py                 # easy + hard 全量
    python scripts/eval_attributor.py --set hard      # 只跑 hard
    python scripts/eval_attributor.py --limit 20      # 抽 20 条先试水
    python scripts/eval_attributor.py --workers 6     # 并发（免费档别开太大）
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm_client                                    # noqa: E402
from app.attributor import (                                  # noqa: E402
    REASON_INVALID_OUTPUT, REASON_LLM_UNAVAILABLE, REASON_LOW_CONFIDENCE,
    attribute_many,
)
from app.config import load_settings                          # noqa: E402
from taxonomy import CATEGORIES, SUBCAUSES                     # noqa: E402
from rule_baseline import predict as rule_category_predict     # noqa: E402

CN = {k: v for k, v in CATEGORIES.items()}
SETS = {
    "easy": ROOT / "data" / "diffs.jsonl",
    "hard": ROOT / "data" / "hard" / "diffs.jsonl",
}


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def evaluate(name: str, rows: list[dict], workers: int, prompt_level: str = "full") -> dict:
    settings = load_settings()
    diffs = [r["diff"] for r in rows]
    print(f"\n{'=' * 78}\n{name} 集：n={len(diffs)}，并发 {workers}，prompt={prompt_level}，"
          f"门禁阈值 {settings.gate_threshold}\n{'=' * 78}", flush=True)

    results = attribute_many(diffs, workers=workers, prompt_level=prompt_level)

    truth_label = [r["label"] for r in rows]
    truth_sub = [r["subcause"] for r in rows]
    pred_label = [a.category for a in results]
    pred_sub = [a.subcause for a in results]

    n = len(rows)
    cat_acc = sum(a == b for a, b in zip(truth_label, pred_label)) / n
    sub_acc = sum(a == b for a, b in zip(truth_sub, pred_sub)) / n
    rule_acc = sum(a == rule_category_predict(d) for a, d in zip(truth_label, diffs)) / n

    auto = [i for i, a in enumerate(results) if a.source == "llm"]
    gated = [i for i, a in enumerate(results) if a.source != "llm"]
    auto_acc = (sum(truth_label[i] == pred_label[i] for i in auto) / len(auto)) if auto else 0.0
    auto_sub_acc = (sum(truth_sub[i] == pred_sub[i] for i in auto) / len(auto)) if auto else 0.0

    low_conf = [i for i, a in enumerate(results) if a.fallback_reason == REASON_LOW_CONFIDENCE]
    invalid = [i for i, a in enumerate(results) if a.fallback_reason == REASON_INVALID_OUTPUT]
    unavailable = [i for i, a in enumerate(results) if a.fallback_reason == REASON_LLM_UNAVAILABLE]

    # 门禁的"拦对了"与"拦错了"
    caught_wrong = [i for i in low_conf if (results[i].llm_subcause or "") and
                    SUBCAUSES.get(results[i].llm_subcause, {}).get("category") != truth_label[i]]
    caught_right = [i for i in low_conf if i not in caught_wrong]
    gate_precision = len(caught_wrong) / len(low_conf) if low_conf else 0.0

    print(f"  规则基线（大类）      {pct(rule_acc):>7}")
    print(f"  LLM 归因（大类）      {pct(cat_acc):>7}   {'↑' if cat_acc > rule_acc else '↓'} {(cat_acc - rule_acc) * 100:+.1f} pt")
    print(f"  LLM 归因（子类）      {pct(sub_acc):>7}   ← 处置动作靠这个粒度")
    print()
    print(f"  自动判定覆盖率        {pct(len(auto) / n):>7}   ({len(auto)}/{n})")
    print(f"  ├ 该子集大类准确率    {pct(auto_acc):>7}   ← 机器独立扛的这批有多准")
    print(f"  └ 该子集子类准确率    {pct(auto_sub_acc):>7}")
    print()
    print(f"  转人工 {len(gated)} 条：低置信度 {len(low_conf)} / 输出不合格 {len(invalid)} / LLM 不可用 {len(unavailable)}")
    if low_conf:
        print(f"  门禁拦截精确率        {pct(gate_precision):>7}   （拦下的里面原本就判错的占比）")
        print(f"  门禁机会成本          {len(caught_right)} 条判对但被转人工（{pct(len(caught_right) / n)} 的样本）")

    # ---- 分类别指标
    print(f"\n  {'大类':<8}{'precision':>11}{'recall':>9}{'f1':>8}{'样本':>6}")
    for c in CATEGORIES:
        tp = sum(1 for a, p in zip(truth_label, pred_label) if a == c and p == c)
        fp = sum(1 for a, p in zip(truth_label, pred_label) if a != c and p == c)
        fn = sum(1 for a, p in zip(truth_label, pred_label) if a == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"  {CN[c]:<8}{pr:>10.1%}{rc:>9.1%}{f1:>8.2f}{truth_label.count(c):>6}")

    print("\n  混淆矩阵（行=真值，列=归因结果）")
    print("  " + " " * 9 + "".join(f"{CN[c]:>7}" for c in CATEGORIES))
    matrix = defaultdict(Counter)
    for a, p in zip(truth_label, pred_label):
        matrix[a][p] += 1
    for c in CATEGORIES:
        print(f"  {CN[c]:<9}" + "".join(f"{matrix[c].get(x, 0):>7}" for x in CATEGORIES))

    # ---- 子类层面的错判（这一层才是处置动作对不对的关键）
    sub_bad = [(r, a) for r, a in zip(rows, results) if r["subcause"] != a.subcause]
    if sub_bad:
        print(f"\n  子类判错 {len(sub_bad)} 条（大类可能对、处置动作会错）：")
        for r, a in sub_bad[:10]:
            ev = r["diff"]
            print(f"    {ev['diff_id']} 真值={r['subcause']:<18} 预测={a.subcause:<18} "
                  f"conf={a.confidence:.2f} src={a.source:<13} | {ev['remark']}")

    latencies = [a.latency_ms for a in results if a.latency_ms]
    providers = Counter(a.provider or "-" for a in results)
    tokens_in = sum(a.prompt_tokens for a in results)
    tokens_out = sum(a.completion_tokens for a in results)
    if latencies:
        print(f"\n  延迟 p50 {statistics.median(latencies):.0f}ms / p95 {_p95(latencies)}ms"
              f"    Provider 分布 {dict(providers)}")
    if tokens_in or tokens_out:
        print(f"  Token 合计：输入 {tokens_in} / 输出 {tokens_out}"
              f"（单条均值 {(tokens_in + tokens_out) / n:.0f}）")

    suffix = "" if prompt_level == "full" else f"_{prompt_level}"
    out_path = ROOT / "data" / f"eval_{name}{suffix}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r, a in zip(rows, results):
            fh.write(json.dumps({
                "diff_id": r["diff"]["diff_id"],
                "diff_type": r["diff"]["diff_type"],
                "remark": r["diff"]["remark"],
                "truth_label": r["label"], "truth_subcause": r["subcause"],
                **a.to_dict(),
                "category_correct": a.category == r["label"],
                "subcause_correct": a.subcause == r["subcause"],
            }, ensure_ascii=False) + "\n")
    print(f"  逐条结果已写入 {out_path.relative_to(ROOT)}")

    return {
        "n": n, "cat_acc": cat_acc, "sub_acc": sub_acc, "rule_acc": rule_acc,
        "coverage": len(auto) / n, "auto_acc": auto_acc, "auto_sub_acc": auto_sub_acc,
        "gated": len(gated), "low_conf": len(low_conf), "invalid": len(invalid),
        "unavailable": len(unavailable), "gate_precision": gate_precision,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["easy", "hard", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prompt", choices=["basic", "full"], default="full",
                    help="basic=只给字典；full=追加专家判别优先级")
    args = ap.parse_args()

    if not llm_client.is_configured():
        print("未配置任何 LLM Provider（复制 .env.example 为 .env 并填 Key）")
        sys.exit(2)
    print(f"Provider 容灾链：{llm_client.configured_providers()}")

    targets = ["easy", "hard"] if args.set == "both" else [args.set]
    summary = {}
    for name in targets:
        rows = load(SETS[name])
        if args.limit:
            rows = rows[: args.limit]
        summary[name] = evaluate(name, rows, args.workers, args.prompt)

    if len(summary) == 2:
        e, h = summary["easy"], summary["hard"]
        print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
        print(f"  {'指标':<22}{'easy':>10}{'hard':>10}")
        print(f"  {'规则基线（大类）':<20}{pct(e['rule_acc']):>11}{pct(h['rule_acc']):>11}")
        print(f"  {'LLM（大类）':<21}{pct(e['cat_acc']):>11}{pct(h['cat_acc']):>11}")
        print(f"  {'LLM（子类）':<21}{pct(e['sub_acc']):>11}{pct(h['sub_acc']):>11}")
        print(f"  {'自动覆盖率':<21}{pct(e['coverage']):>11}{pct(h['coverage']):>11}")
        print(f"  {'自动子集准确率':<19}{pct(e['auto_acc']):>11}{pct(h['auto_acc']):>11}")
        print(f"\n  对照：规则基线 hard 集 {pct(h['rule_acc'])} → LLM {pct(h['cat_acc'])}"
              f"，提升 {(h['cat_acc'] - h['rule_acc']) * 100:+.1f} pt")


if __name__ == "__main__":
    main()
