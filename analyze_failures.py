"""
Mind2Web Failure Autopsy — Failure Analysis

Reads evaluation results and categorizes every failure into
specific, actionable failure modes.
"""

import json
import sys
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).parent / "results"


def classify_failure(pred: dict) -> str:
    """Classify a single failure into a failure mode."""

    gt_tag = pred["gt_element_tag"].lower()
    pred_tag = pred["pred_element_tag"].lower()
    gt_op = pred["gt_op"]
    pred_op = pred["pred_op"]
    gt_value = pred["gt_value"].lower().strip()
    pred_value = pred["pred_value"].lower().strip()

    # 1. Parse failure — model didn't produce a valid response
    if pred["pred_element_idx"] == -1 or not pred_op:
        return "parse_failure"

    # 2. Element correct, operation wrong
    if pred["element_correct"] and not pred["op_correct"]:
        if gt_op == "TYPE" and pred_op == "CLICK":
            return "op_confusion:clicked_instead_of_typed"
        elif gt_op == "SELECT" and pred_op == "CLICK":
            return "op_confusion:clicked_instead_of_selected"
        elif gt_op == "CLICK" and pred_op == "TYPE":
            return "op_confusion:typed_instead_of_clicked"
        elif gt_op == "CLICK" and pred_op == "SELECT":
            return "op_confusion:selected_instead_of_clicked"
        else:
            return f"op_confusion:{gt_op}_vs_{pred_op}"

    # 3. Element correct, op correct, value wrong
    if pred["element_correct"] and pred["op_correct"] and not pred["value_correct"]:
        if gt_value in pred_value or pred_value in gt_value:
            return "value_partial_match"
        else:
            return "value_wrong"

    # 4. Wrong element — the big category. Sub-classify.
    if not pred["element_correct"]:
        # same tag type (e.g., both <button>, both <input>)
        same_tag = (gt_tag == pred_tag)

        if same_tag:
            return f"wrong_element:same_tag_{gt_tag}"
        else:
            return f"wrong_element:{gt_tag}_vs_{pred_tag}"

    return "unknown"


def analyze(results_path: str):
    data = json.loads(Path(results_path).read_text())

    model = data["model"]
    total = data["total_steps"]
    correct = data["correct_steps"]
    accuracy = data["accuracy"]
    predictions = data["predictions"]

    failures = [p for p in predictions if not p["step_correct"]]

    print(f"{'='*70}")
    print(f"FAILURE AUTOPSY: {model}")
    print(f"{'='*70}")
    print(f"Total steps: {total}")
    print(f"Correct: {correct} ({accuracy:.1%})")
    print(f"Failures: {len(failures)} ({1-accuracy:.1%})")

    # classify every failure
    failure_modes = Counter()
    failure_examples = {}

    for f in failures:
        mode = classify_failure(f)
        failure_modes[mode] += 1
        if mode not in failure_examples:
            failure_examples[mode] = []
        if len(failure_examples[mode]) < 3:
            failure_examples[mode].append({
                "website": f["website"],
                "task": f["task"][:80],
                "step": f["step"],
                "action_repr": f["action_repr"],
                "gt": f"{f['gt_op']} on <{f['gt_element_tag']}>",
                "pred": f"{f['pred_op']} on <{f['pred_element_tag']}>",
                "gt_value": f["gt_value"][:50] if f["gt_value"] else "",
                "pred_value": f["pred_value"][:50] if f["pred_value"] else "",
            })

    # group into high-level categories
    categories = {
        "wrong_element": 0,
        "op_confusion": 0,
        "value_wrong": 0,
        "value_partial_match": 0,
        "parse_failure": 0,
        "unknown": 0,
    }
    for mode, count in failure_modes.items():
        for cat in categories:
            if mode.startswith(cat):
                categories[cat] += count
                break

    print(f"\n{'='*70}")
    print(f"HIGH-LEVEL FAILURE CATEGORIES")
    print(f"{'='*70}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / len(failures) * 100
            bar = "█" * int(pct / 2)
            print(f"  {cat:30s} {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*70}")
    print(f"DETAILED FAILURE MODES")
    print(f"{'='*70}")
    for mode, count in failure_modes.most_common():
        pct = count / len(failures) * 100
        print(f"\n  {mode}: {count} ({pct:.1f}%)")
        for ex in failure_examples[mode]:
            print(f"    Website: {ex['website']}")
            print(f"    Task: {ex['task']}")
            print(f"    Expected: {ex['gt']}" + (f" value=\"{ex['gt_value']}\"" if ex['gt_value'] else ""))
            print(f"    Got:      {ex['pred']}" + (f" value=\"{ex['pred_value']}\"" if ex['pred_value'] else ""))
            print(f"    Action:   {ex['action_repr']}")
            print()

    # accuracy by operation type
    print(f"{'='*70}")
    print(f"ACCURACY BY OPERATION TYPE")
    print(f"{'='*70}")
    by_op = {}
    for p in predictions:
        op = p["gt_op"]
        by_op.setdefault(op, {"total": 0, "correct": 0, "elem_correct": 0, "op_correct": 0})
        by_op[op]["total"] += 1
        if p["step_correct"]:
            by_op[op]["correct"] += 1
        if p["element_correct"]:
            by_op[op]["elem_correct"] += 1
        if p["op_correct"]:
            by_op[op]["op_correct"] += 1

    for op, stats in sorted(by_op.items()):
        t = stats["total"]
        print(f"\n  {op} ({t} steps):")
        print(f"    Step accuracy:    {stats['correct']/t:.1%}")
        print(f"    Element accuracy: {stats['elem_correct']/t:.1%}")
        print(f"    Operation accuracy: {stats['op_correct']/t:.1%}")

    # accuracy by domain
    print(f"\n{'='*70}")
    print(f"ACCURACY BY DOMAIN")
    print(f"{'='*70}")
    by_domain = {}
    for p in predictions:
        d = p["domain"]
        by_domain.setdefault(d, {"total": 0, "correct": 0})
        by_domain[d]["total"] += 1
        if p["step_correct"]:
            by_domain[d]["correct"] += 1

    for d, stats in sorted(by_domain.items()):
        acc = stats["correct"] / stats["total"]
        print(f"  {d:20s} {stats['correct']}/{stats['total']} ({acc:.1%})")

    # accuracy by website (top 10 worst)
    print(f"\n{'='*70}")
    print(f"HARDEST WEBSITES (worst accuracy)")
    print(f"{'='*70}")
    by_site = {}
    for p in predictions:
        s = p["website"]
        by_site.setdefault(s, {"total": 0, "correct": 0})
        by_site[s]["total"] += 1
        if p["step_correct"]:
            by_site[s]["correct"] += 1

    sorted_sites = sorted(by_site.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1))
    for site, stats in sorted_sites[:10]:
        if stats["total"] >= 3:
            acc = stats["correct"] / stats["total"]
            print(f"  {site:30s} {stats['correct']}/{stats['total']} ({acc:.1%})")

    # save analysis
    analysis = {
        "model": model,
        "total_steps": total,
        "correct_steps": correct,
        "accuracy": accuracy,
        "num_failures": len(failures),
        "high_level_categories": categories,
        "detailed_modes": dict(failure_modes.most_common()),
        "examples": failure_examples,
        "by_operation": by_op,
        "by_domain": {d: {"accuracy": s["correct"]/s["total"]} for d, s in by_domain.items()},
    }

    out_path = RESULTS_DIR / f"analysis_{model.replace('/', '_').replace(':', '_')}.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # find most recent results file
        files = sorted(RESULTS_DIR.glob("*.json"))
        files = [f for f in files if not f.name.startswith("analysis_")]
        if not files:
            print("No results files found. Run evaluate.py first.")
            sys.exit(1)
        results_path = str(files[-1])
        print(f"Using most recent results: {results_path}")
    else:
        results_path = sys.argv[1]

    analyze(results_path)
