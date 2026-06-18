"""
Mind2Web Shaped Reward Function

A verifiable, decomposable reward for web agent training.
Replaces binary (right/wrong) scoring with a gradient that
distinguishes between failure modes.

Components:
  - element_score (0.4 weight): tag match → instance match
  - operation_score (0.3 weight): correct action type
  - value_score (0.3 weight): correct input value
"""

import json
from pathlib import Path


def element_reward(pred_tag: str, gt_tag: str, element_correct: bool) -> float:
    if element_correct:
        return 1.0
    if pred_tag.lower() == gt_tag.lower():
        return 0.5  # same tag type, wrong instance (near miss)
    return 0.0  # wrong tag type entirely


def operation_reward(pred_op: str, gt_op: str) -> float:
    if pred_op == gt_op:
        return 1.0
    return 0.0


def value_reward(pred_value: str, gt_value: str, gt_op: str) -> float:
    if gt_op == "CLICK":
        return 1.0
    pred = pred_value.lower().strip()
    gt = gt_value.lower().strip()
    if pred == gt:
        return 1.0
    if not pred:
        return 0.0
    # partial credit for substring containment (over/under specification)
    if gt in pred or pred in gt:
        return 0.5
    # partial credit for word overlap
    gt_words = set(gt.split())
    pred_words = set(pred.split())
    if gt_words and pred_words:
        overlap = len(gt_words & pred_words) / len(gt_words | pred_words)
        if overlap > 0:
            return 0.3 * overlap
    return 0.0


def shaped_reward(pred: dict, w_elem=0.4, w_op=0.3, w_val=0.3) -> dict:
    e = element_reward(pred["pred_element_tag"], pred["gt_element_tag"], pred["element_correct"])
    o = operation_reward(pred["pred_op"], pred["gt_op"])
    v = value_reward(pred["pred_value"], pred["gt_value"], pred["gt_op"])

    total = w_elem * e + w_op * o + w_val * v

    return {
        "total": round(total, 3),
        "element": round(e, 3),
        "operation": round(o, 3),
        "value": round(v, 3),
        "binary": 1.0 if pred["step_correct"] else 0.0,
    }


def analyze_reward_distribution(results_path: str):
    data = json.loads(Path(results_path).read_text())
    model = data["model"]
    preds = data["predictions"]

    print(f"{'='*70}")
    print(f"SHAPED vs BINARY REWARD — {model}")
    print(f"{'='*70}")

    rewards = [shaped_reward(p) for p in preds]

    # binary distribution
    binary_zero = sum(1 for r in rewards if r["binary"] == 0)
    binary_one = sum(1 for r in rewards if r["binary"] == 1)
    print(f"\n  Binary reward distribution:")
    print(f"    0.0 (wrong): {binary_zero} ({binary_zero/len(rewards)*100:.1f}%)")
    print(f"    1.0 (right): {binary_one} ({binary_one/len(rewards)*100:.1f}%)")

    # shaped distribution (bucketed)
    buckets = {"0.0": 0, "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0, "1.0": 0}
    for r in rewards:
        t = r["total"]
        if t == 0: buckets["0.0"] += 1
        elif t < 0.2: buckets["0.0-0.2"] += 1
        elif t < 0.4: buckets["0.2-0.4"] += 1
        elif t < 0.6: buckets["0.4-0.6"] += 1
        elif t < 0.8: buckets["0.6-0.8"] += 1
        elif t < 1.0: buckets["0.8-1.0"] += 1
        else: buckets["1.0"] += 1

    print(f"\n  Shaped reward distribution:")
    for bucket, count in buckets.items():
        pct = count / len(rewards) * 100
        bar = "█" * int(pct)
        print(f"    {bucket:>8s}: {count:4d} ({pct:5.1f}%) {bar}")

    # key comparison: how many "binary 0" cases get partial credit?
    binary_zero_shaped = [r["total"] for r in rewards if r["binary"] == 0]
    if binary_zero_shaped:
        got_partial = sum(1 for s in binary_zero_shaped if s > 0)
        avg_partial = sum(binary_zero_shaped) / len(binary_zero_shaped)
        print(f"\n  Among {len(binary_zero_shaped)} binary-zero (wrong) predictions:")
        print(f"    Got partial credit: {got_partial} ({got_partial/len(binary_zero_shaped)*100:.1f}%)")
        print(f"    Average shaped reward: {avg_partial:.3f}")
        print(f"    → Binary sees {len(binary_zero_shaped)} identical zeros.")
        print(f"    → Shaped sees a gradient from {min(binary_zero_shaped):.3f} to {max(binary_zero_shaped):.3f}")

    # component breakdown for failures
    failures = [(r, p) for r, p in zip(rewards, preds) if r["binary"] == 0]

    elem_partial = sum(1 for r, _ in failures if r["element"] == 0.5)
    elem_full = sum(1 for r, _ in failures if r["element"] == 1.0)
    op_correct = sum(1 for r, _ in failures if r["operation"] == 1.0)
    val_partial = sum(1 for r, _ in failures if 0 < r["value"] < 1.0)

    print(f"\n  Component analysis of failures:")
    print(f"    Element near-miss (same tag, wrong instance): {elem_partial} ({elem_partial/len(failures)*100:.1f}%)")
    print(f"    Element fully correct but failed on op/val:   {elem_full} ({elem_full/len(failures)*100:.1f}%)")
    print(f"    Operation correct despite failure:            {op_correct} ({op_correct/len(failures)*100:.1f}%)")
    print(f"    Value partial credit:                         {val_partial} ({val_partial/len(failures)*100:.1f}%)")

    # simulate GRPO batch signal
    print(f"\n  {'='*70}")
    print(f"  SIMULATED GRPO BATCH ANALYSIS")
    print(f"  {'='*70}")

    import random
    random.seed(42)
    batch_size = 8
    num_batches = 50
    binary_dead = 0
    shaped_dead = 0

    for _ in range(num_batches):
        batch = random.sample(rewards, min(batch_size, len(rewards)))
        binary_scores = [r["binary"] for r in batch]
        shaped_scores = [r["total"] for r in batch]

        # dead batch = all same score (no spread for GRPO)
        if len(set(binary_scores)) == 1:
            binary_dead += 1
        if max(shaped_scores) - min(shaped_scores) < 0.05:
            shaped_dead += 1

    print(f"\n  Simulated {num_batches} random batches of {batch_size}:")
    print(f"    Binary: {binary_dead}/{num_batches} dead batches ({binary_dead/num_batches*100:.0f}%) — no learning signal")
    print(f"    Shaped: {shaped_dead}/{num_batches} dead batches ({shaped_dead/num_batches*100:.0f}%) — no learning signal")
    print(f"    → Shaped reward gives {((binary_dead - shaped_dead) / max(binary_dead, 1) * 100):.0f}% fewer dead batches")

    return rewards


if __name__ == "__main__":
    import sys
    results_dir = Path(__file__).parent / "results"
    result_files = sorted(results_dir.glob("*.json"))
    result_files = [f for f in result_files if not f.name.startswith("analysis_") and not f.name.startswith("deep_")]

    for f in result_files:
        analyze_reward_distribution(str(f))
        print()
