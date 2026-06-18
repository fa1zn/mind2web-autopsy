"""
Mind2Web Failure Autopsy — Deep Analysis

Goes beyond surface-level failure categories to find the actual
mechanisms behind web agent failures.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(path):
    data = json.loads(Path(path).read_text())
    return data["model"], data["predictions"]


def positional_bias_analysis(model, preds):
    """Do models disproportionately pick certain candidate positions?"""
    print(f"\n{'='*70}")
    print(f"1. POSITIONAL BIAS — {model}")
    print(f"{'='*70}")

    position_chosen = Counter()
    position_correct = Counter()
    total = 0
    correct_by_gt_pos = defaultdict(lambda: {"total": 0, "correct": 0})

    for p in preds:
        idx = p["pred_element_idx"]
        gt_idx = None
        # reconstruct gt position from element_correct
        if p["element_correct"]:
            gt_idx = idx
        position_chosen[idx] += 1
        total += 1

    # distribution of chosen positions
    print(f"\n  Predicted element index distribution (top 15):")
    for pos, count in position_chosen.most_common(15):
        pct = count / total * 100
        bar = "█" * int(pct)
        print(f"    [{pos:2d}] {count:4d} ({pct:5.1f}%) {bar}")

    # check if model favors early positions
    early = sum(count for pos, count in position_chosen.items() if 0 <= pos <= 4)
    mid = sum(count for pos, count in position_chosen.items() if 5 <= pos <= 14)
    late = sum(count for pos, count in position_chosen.items() if 15 <= pos <= 29)
    invalid = sum(count for pos, count in position_chosen.items() if pos < 0 or pos >= 30)

    print(f"\n  Position bands:")
    print(f"    Early [0-4]:   {early:4d} ({early/total*100:.1f}%)")
    print(f"    Mid   [5-14]:  {mid:4d} ({mid/total*100:.1f}%)")
    print(f"    Late  [15-29]: {late:4d} ({late/total*100:.1f}%)")
    print(f"    Invalid/OOB:   {invalid:4d} ({invalid/total*100:.1f}%)")

    # what would random chance look like?
    avg_candidates = sum(p["num_candidates"] for p in preds) / len(preds)
    random_early = min(5, avg_candidates) / avg_candidates * 100
    print(f"\n  Random baseline for early [0-4]: {random_early:.1f}%")
    print(f"  Model actual for early [0-4]:    {early/total*100:.1f}%")
    bias_ratio = (early/total*100) / random_early if random_early > 0 else 0
    print(f"  Bias ratio: {bias_ratio:.2f}x (>1 = favors early positions)")

    return {"early_pct": early/total*100, "bias_ratio": bias_ratio}


def candidate_count_vs_accuracy(model, preds):
    """Does accuracy drop as the number of candidates increases?"""
    print(f"\n{'='*70}")
    print(f"2. CANDIDATE COUNT vs ACCURACY — {model}")
    print(f"{'='*70}")

    by_count = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in preds:
        n = p["num_candidates"]
        bucket = f"{(n//5)*5}-{(n//5)*5+4}"
        by_count[bucket]["total"] += 1
        if p["step_correct"]:
            by_count[bucket]["correct"] += 1

    print(f"\n  {'Candidates':>12s}  {'Total':>6s}  {'Correct':>8s}  {'Accuracy':>9s}")
    for bucket in sorted(by_count.keys()):
        stats = by_count[bucket]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        bar = "█" * int(acc * 40)
        print(f"  {bucket:>12s}  {stats['total']:6d}  {stats['correct']:8d}  {acc:8.1%}  {bar}")


def step_position_analysis(model, preds):
    """Are later steps in a task harder?"""
    print(f"\n{'='*70}")
    print(f"3. STEP POSITION vs ACCURACY — {model}")
    print(f"{'='*70}")

    by_step = defaultdict(lambda: {"total": 0, "correct": 0})
    for p in preds:
        s = p["step"]
        by_step[s]["total"] += 1
        if p["step_correct"]:
            by_step[s]["correct"] += 1

    print(f"\n  {'Step':>6s}  {'Total':>6s}  {'Correct':>8s}  {'Accuracy':>9s}")
    for step in sorted(by_step.keys()):
        if by_step[step]["total"] >= 3:
            stats = by_step[step]
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            bar = "█" * int(acc * 40)
            print(f"  {step:6d}  {stats['total']:6d}  {stats['correct']:8d}  {acc:8.1%}  {bar}")

    # first step vs rest
    first = by_step.get(1, {"total": 0, "correct": 0})
    rest_total = sum(s["total"] for k, s in by_step.items() if k > 1)
    rest_correct = sum(s["correct"] for k, s in by_step.items() if k > 1)
    first_acc = first["correct"] / first["total"] if first["total"] > 0 else 0
    rest_acc = rest_correct / rest_total if rest_total > 0 else 0
    print(f"\n  Step 1 accuracy: {first_acc:.1%} ({first['correct']}/{first['total']})")
    print(f"  Steps 2+ accuracy: {rest_acc:.1%} ({rest_correct}/{rest_total})")


def tag_confusion_matrix(model, preds):
    """What element tags get confused with what?"""
    print(f"\n{'='*70}")
    print(f"4. TAG CONFUSION MATRIX — {model}")
    print(f"{'='*70}")

    confusion = Counter()
    for p in preds:
        if not p["element_correct"] and p["pred_element_tag"]:
            gt = p["gt_element_tag"].lower()
            pred = p["pred_element_tag"].lower()
            confusion[(gt, pred)] += 1

    print(f"\n  {'Expected':>12s} → {'Predicted':>12s}  {'Count':>6s}  {'%':>6s}")
    total_confused = sum(confusion.values())
    for (gt, pred), count in confusion.most_common(20):
        pct = count / total_confused * 100
        print(f"  {gt:>12s} → {pred:>12s}  {count:6d}  {pct:5.1f}%")


def operation_confusion_detail(model, preds):
    """When the model gets the element right but the operation wrong, what happens?"""
    print(f"\n{'='*70}")
    print(f"5. OPERATION CONFUSION DETAIL — {model}")
    print(f"{'='*70}")

    op_confusion = Counter()
    examples = defaultdict(list)
    for p in preds:
        if p["element_correct"] and not p["op_correct"]:
            key = f"{p['gt_op']} → {p['pred_op']}"
            op_confusion[key] += 1
            if len(examples[key]) < 2:
                examples[key].append({
                    "website": p["website"],
                    "task": p["task"][:60],
                    "action": p["action_repr"],
                })

    if not op_confusion:
        print("  No operation confusions (element correct but op wrong)")
        return

    print(f"\n  {'Confusion':>25s}  {'Count':>6s}")
    for key, count in op_confusion.most_common():
        print(f"  {key:>25s}  {count:6d}")
        for ex in examples[key]:
            print(f"    {ex['website']}: {ex['task']}")
            print(f"    Action: {ex['action']}")


def value_error_analysis(model, preds):
    """When element and op are right but value is wrong, what's the pattern?"""
    print(f"\n{'='*70}")
    print(f"6. VALUE ERROR ANALYSIS — {model}")
    print(f"{'='*70}")

    value_errors = []
    for p in preds:
        if p["element_correct"] and p["op_correct"] and not p["value_correct"]:
            value_errors.append(p)

    if not value_errors:
        print("  No value errors")
        return

    print(f"\n  Total value errors: {len(value_errors)}")

    # categorize
    categories = Counter()
    for v in value_errors:
        gt = v["gt_value"].lower().strip()
        pred = v["pred_value"].lower().strip()
        if not pred:
            categories["empty_prediction"] += 1
        elif gt in pred:
            categories["over_specified"] += 1
        elif pred in gt:
            categories["under_specified"] += 1
        elif gt.split() and pred.split() and set(gt.split()) & set(pred.split()):
            categories["partial_word_overlap"] += 1
        else:
            categories["completely_different"] += 1

    print(f"\n  Value error types:")
    for cat, count in categories.most_common():
        print(f"    {cat:30s} {count:4d} ({count/len(value_errors)*100:.0f}%)")

    print(f"\n  Examples (gt_value → pred_value):")
    for v in value_errors[:8]:
        print(f"    \"{v['gt_value'][:40]}\" → \"{v['pred_value'][:40]}\"")
        print(f"      {v['website']}: {v['task'][:50]}")


def cross_model_agreement(models_preds):
    """Do both models fail on the same steps? Reveals genuine difficulty vs model weakness."""
    print(f"\n{'='*70}")
    print(f"7. CROSS-MODEL AGREEMENT")
    print(f"{'='*70}")

    if len(models_preds) < 2:
        print("  Need 2+ models for cross-model analysis")
        return

    (m1, p1), (m2, p2) = list(models_preds.items())[:2]

    # index by (task_id, step)
    r1 = {(p["task_id"], p["step"]): p["step_correct"] for p in p1}
    r2 = {(p["task_id"], p["step"]): p["step_correct"] for p in p2}

    common_keys = set(r1.keys()) & set(r2.keys())
    both_right = sum(1 for k in common_keys if r1[k] and r2[k])
    both_wrong = sum(1 for k in common_keys if not r1[k] and not r2[k])
    only_m1 = sum(1 for k in common_keys if r1[k] and not r2[k])
    only_m2 = sum(1 for k in common_keys if not r1[k] and r2[k])

    print(f"\n  Compared {len(common_keys)} common steps between {m1} and {m2}")
    print(f"\n  {'':30s}  {m2[:20]:>20s}")
    print(f"  {'':30s}  {'Correct':>10s} {'Wrong':>10s}")
    print(f"  {m1[:20]:>20s} Correct  {both_right:10d} {only_m1:10d}")
    print(f"  {'':>20s} Wrong    {only_m2:10d} {both_wrong:10d}")

    total = len(common_keys)
    print(f"\n  Both correct:       {both_right:4d} ({both_right/total*100:.1f}%) — easy steps")
    print(f"  Both wrong:         {both_wrong:4d} ({both_wrong/total*100:.1f}%) — genuinely hard steps")
    print(f"  Only {m1[:15]} right: {only_m1:4d} ({only_m1/total*100:.1f}%)")
    print(f"  Only {m2[:15]} right: {only_m2:4d} ({only_m2/total*100:.1f}%)")

    # what domains are hardest for both?
    both_wrong_domains = Counter()
    both_wrong_websites = Counter()
    for k in common_keys:
        if not r1[k] and not r2[k]:
            p = [p for p in p1 if (p["task_id"], p["step"]) == k][0]
            both_wrong_domains[p["domain"]] += 1
            both_wrong_websites[p["website"]] += 1

    print(f"\n  Domains where BOTH models fail most:")
    for domain, count in both_wrong_domains.most_common(10):
        print(f"    {domain:20s} {count:4d}")

    print(f"\n  Websites where BOTH models fail most:")
    for site, count in both_wrong_websites.most_common(10):
        print(f"    {site:25s} {count:4d}")

    # element agreement on failures
    elem_agree = 0
    elem_disagree = 0
    for k in common_keys:
        if not r1[k] and not r2[k]:
            pp1 = [p for p in p1 if (p["task_id"], p["step"]) == k][0]
            pp2 = [p for p in p2 if (p["task_id"], p["step"]) == k][0]
            if pp1["pred_element_idx"] == pp2["pred_element_idx"]:
                elem_agree += 1
            else:
                elem_disagree += 1

    if elem_agree + elem_disagree > 0:
        print(f"\n  When both fail, do they pick the SAME wrong element?")
        print(f"    Same wrong element: {elem_agree:4d} ({elem_agree/(elem_agree+elem_disagree)*100:.1f}%)")
        print(f"    Different wrong:    {elem_disagree:4d} ({elem_disagree/(elem_agree+elem_disagree)*100:.1f}%)")

    return {
        "both_right": both_right, "both_wrong": both_wrong,
        "only_m1": only_m1, "only_m2": only_m2,
    }


def domain_difficulty_analysis(models_preds):
    """Which domains/websites are hardest and why?"""
    print(f"\n{'='*70}")
    print(f"8. DOMAIN DIFFICULTY ANALYSIS")
    print(f"{'='*70}")

    for model, preds in models_preds.items():
        print(f"\n  --- {model} ---")
        by_domain = defaultdict(lambda: {"total": 0, "correct": 0, "sites": set()})
        for p in preds:
            d = p["domain"]
            by_domain[d]["total"] += 1
            if p["step_correct"]:
                by_domain[d]["correct"] += 1
            by_domain[d]["sites"].add(p["website"])

        sorted_domains = sorted(by_domain.items(),
                                key=lambda x: x[1]["correct"]/max(x[1]["total"],1))
        print(f"\n  {'Domain':>20s}  {'Acc':>7s}  {'Correct':>8s}  {'Total':>6s}  Sites")
        for domain, stats in sorted_domains:
            if stats["total"] >= 5:
                acc = stats["correct"] / stats["total"]
                sites = ", ".join(sorted(stats["sites"])[:3])
                print(f"  {domain:>20s}  {acc:6.1%}  {stats['correct']:8d}  {stats['total']:6d}  {sites}")


def raw_response_quality(model, preds):
    """Analyze the raw model responses for formatting issues."""
    print(f"\n{'='*70}")
    print(f"9. RESPONSE FORMAT QUALITY — {model}")
    print(f"{'='*70}")

    has_element = 0
    has_op = 0
    has_value = 0
    has_all_three = 0
    has_thinking = 0
    empty_response = 0
    extra_text = 0

    for p in preds:
        raw = p["raw_response"]
        if not raw:
            empty_response += 1
            continue
        if "<think>" in raw.lower() or "let me" in raw.lower() or "i need to" in raw.lower():
            has_thinking += 1
        lines = [l.strip().lower() for l in raw.split("\n") if l.strip()]
        e = any(l.startswith("element:") for l in lines)
        o = any(l.startswith("operation:") for l in lines)
        v = any(l.startswith("value:") for l in lines)
        if e: has_element += 1
        if o: has_op += 1
        if v: has_value += 1
        if e and o and v: has_all_three += 1
        if len(lines) > 3:
            extra_text += 1

    total = len(preds)
    print(f"\n  Total responses: {total}")
    print(f"  Empty responses: {empty_response} ({empty_response/total*100:.1f}%)")
    print(f"  Has 'Element:' line: {has_element} ({has_element/total*100:.1f}%)")
    print(f"  Has 'Operation:' line: {has_op} ({has_op/total*100:.1f}%)")
    print(f"  Has 'Value:' line: {has_value} ({has_value/total*100:.1f}%)")
    print(f"  All three fields: {has_all_three} ({has_all_three/total*100:.1f}%)")
    print(f"  Contains thinking/reasoning: {has_thinking} ({has_thinking/total*100:.1f}%)")
    print(f"  Extra text (>3 lines): {extra_text} ({extra_text/total*100:.1f}%)")


def error_cascade_analysis(model, preds):
    """After an error, does the model recover or keep failing?"""
    print(f"\n{'='*70}")
    print(f"10. ERROR CASCADE — {model}")
    print(f"{'='*70}")

    # group by task
    tasks = defaultdict(list)
    for p in preds:
        tasks[p["task_id"]].append(p)

    after_error_total = 0
    after_error_correct = 0
    after_correct_total = 0
    after_correct_correct = 0
    streak_lengths = []

    for task_id, steps in tasks.items():
        steps = sorted(steps, key=lambda x: x["step"])
        current_streak = 0
        for i in range(1, len(steps)):
            prev_correct = steps[i-1]["step_correct"]
            curr_correct = steps[i]["step_correct"]

            if prev_correct:
                after_correct_total += 1
                if curr_correct:
                    after_correct_correct += 1
            else:
                after_error_total += 1
                if curr_correct:
                    after_error_correct += 1

            if not curr_correct:
                current_streak += 1
            else:
                if current_streak > 0:
                    streak_lengths.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streak_lengths.append(current_streak)

    if after_error_total > 0:
        print(f"\n  P(correct | previous step correct): {after_correct_correct/after_correct_total:.1%}" if after_correct_total > 0 else "")
        print(f"  P(correct | previous step wrong):   {after_error_correct/after_error_total:.1%}")
        print(f"\n  → Error recovery rate: {after_error_correct/after_error_total:.1%}")

    if streak_lengths:
        avg_streak = sum(streak_lengths) / len(streak_lengths)
        max_streak = max(streak_lengths)
        print(f"\n  Consecutive error streaks:")
        print(f"    Average length: {avg_streak:.1f} steps")
        print(f"    Max length: {max_streak} steps")
        streak_dist = Counter(streak_lengths)
        for length in sorted(streak_dist.keys())[:8]:
            print(f"    {length} steps: {streak_dist[length]} times")


def task_difficulty_analysis(models_preds):
    """Which specific tasks are hardest and what characterizes them?"""
    print(f"\n{'='*70}")
    print(f"11. HARDEST TASKS (across all models)")
    print(f"{'='*70}")

    task_scores = defaultdict(lambda: {"scores": [], "website": "", "task": "", "num_steps": 0})

    for model, preds in models_preds.items():
        by_task = defaultdict(lambda: {"total": 0, "correct": 0})
        for p in preds:
            by_task[p["task_id"]]["total"] += 1
            if p["step_correct"]:
                by_task[p["task_id"]]["correct"] += 1
            task_scores[p["task_id"]]["website"] = p["website"]
            task_scores[p["task_id"]]["task"] = p["task"]
            task_scores[p["task_id"]]["num_steps"] = max(
                task_scores[p["task_id"]]["num_steps"], by_task[p["task_id"]]["total"])

        for tid, stats in by_task.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            task_scores[tid]["scores"].append(acc)

    # tasks where ALL models score 0
    zero_tasks = [(tid, info) for tid, info in task_scores.items()
                  if all(s == 0 for s in info["scores"]) and info["num_steps"] >= 3]
    zero_tasks.sort(key=lambda x: -x[1]["num_steps"])

    print(f"\n  Tasks where ALL models scored 0% ({len(zero_tasks)} tasks):")
    for tid, info in zero_tasks[:10]:
        print(f"    [{info['num_steps']} steps] {info['website']}: {info['task'][:65]}")

    # tasks where models disagree most
    if len(list(models_preds.keys())) >= 2:
        disagree = [(tid, info, max(info["scores"]) - min(info["scores"]))
                    for tid, info in task_scores.items() if len(info["scores"]) >= 2]
        disagree.sort(key=lambda x: -x[2])
        print(f"\n  Tasks with biggest accuracy gap between models:")
        for tid, info, gap in disagree[:10]:
            scores_str = " / ".join(f"{s:.0%}" for s in info["scores"])
            print(f"    [{scores_str}] {info['website']}: {info['task'][:55]}")


def website_complexity_analysis(model, preds):
    """What makes certain websites harder?"""
    print(f"\n{'='*70}")
    print(f"12. WEBSITE COMPLEXITY — {model}")
    print(f"{'='*70}")

    by_site = defaultdict(lambda: {
        "total": 0, "correct": 0,
        "avg_candidates": [],
        "op_dist": Counter(),
        "elem_correct_rate": [],
        "op_correct_rate": [],
    })

    for p in preds:
        site = p["website"]
        by_site[site]["total"] += 1
        if p["step_correct"]:
            by_site[site]["correct"] += 1
        by_site[site]["avg_candidates"].append(p["num_candidates"])
        by_site[site]["op_dist"][p["gt_op"]] += 1
        by_site[site]["elem_correct_rate"].append(1 if p["element_correct"] else 0)
        by_site[site]["op_correct_rate"].append(1 if p["op_correct"] else 0)

    # sort by accuracy
    sites = [(site, stats) for site, stats in by_site.items() if stats["total"] >= 5]
    sites.sort(key=lambda x: x[1]["correct"] / x[1]["total"])

    print(f"\n  {'Website':>20s}  {'Acc':>6s}  {'ElemAcc':>8s}  {'OpAcc':>6s}  {'AvgCand':>8s}  {'Steps':>5s}  Ops")
    for site, stats in sites:
        acc = stats["correct"] / stats["total"]
        elem_acc = sum(stats["elem_correct_rate"]) / len(stats["elem_correct_rate"])
        op_acc = sum(stats["op_correct_rate"]) / len(stats["op_correct_rate"])
        avg_cand = sum(stats["avg_candidates"]) / len(stats["avg_candidates"])
        ops = "+".join(f"{op}:{c}" for op, c in stats["op_dist"].most_common(3))
        print(f"  {site:>20s}  {acc:5.1%}  {elem_acc:7.1%}  {op_acc:5.1%}  {avg_cand:7.1f}  {stats['total']:5d}  {ops}")


def summary_findings(models_preds, cross_model_stats):
    """Synthesize all analyses into key findings."""
    print(f"\n{'='*70}")
    print(f"KEY FINDINGS — SYNTHESIS")
    print(f"{'='*70}")

    findings = []
    for model, preds in models_preds.items():
        total = len(preds)
        correct = sum(1 for p in preds if p["step_correct"])
        elem_failures = sum(1 for p in preds if not p["element_correct"])
        findings.append(f"  {model}: {correct/total:.1%} step accuracy, "
                       f"{elem_failures/total:.1%} element selection failure rate")

    for f in findings:
        print(f)

    if cross_model_stats:
        total = sum(cross_model_stats.values())
        bw = cross_model_stats["both_wrong"]
        print(f"\n  Cross-model: {bw/total:.1%} of steps are hard for ALL models")
        print(f"  → These are the steps that need better representations, not just better models")


if __name__ == "__main__":
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    result_files = [f for f in result_files if not f.name.startswith("analysis_")]

    if not result_files:
        print("No results files found. Run evaluate.py first.")
        sys.exit(1)

    models_preds = {}
    for f in result_files:
        model, preds = load_results(f)
        models_preds[model] = preds
        print(f"Loaded {model}: {len(preds)} predictions")

    # per-model analyses
    for model, preds in models_preds.items():
        positional_bias_analysis(model, preds)
        candidate_count_vs_accuracy(model, preds)
        step_position_analysis(model, preds)
        tag_confusion_matrix(model, preds)
        operation_confusion_detail(model, preds)
        value_error_analysis(model, preds)
        raw_response_quality(model, preds)
        error_cascade_analysis(model, preds)
        website_complexity_analysis(model, preds)

    # cross-model analyses
    cross_stats = cross_model_agreement(models_preds)
    domain_difficulty_analysis(models_preds)
    task_difficulty_analysis(models_preds)
    summary_findings(models_preds, cross_stats)

    # save full analysis
    out_path = RESULTS_DIR / "deep_analysis.txt"
    print(f"\n\nRe-run with: python3 deep_analysis.py > results/deep_analysis.txt")
