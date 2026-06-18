"""
Mind2Web Failure Autopsy — Evaluation Pipeline

Runs models on Mind2Web tasks, scores every prediction deterministically,
and saves every failure with full context for analysis.
"""

import json
import time
import re
import sys
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = Path(__file__).parent / "data" / "tasks_cache.json"


# ── Data structures ──────────────────────────────────────────

@dataclass
class Prediction:
    task_id: str
    step: int
    website: str
    domain: str
    task: str
    # ground truth
    gt_element_tag: str
    gt_element_attrs: str
    gt_op: str
    gt_value: str
    # model prediction
    pred_element_idx: int
    pred_element_tag: str
    pred_op: str
    pred_value: str
    # scoring
    element_correct: bool
    op_correct: bool
    value_correct: bool
    step_correct: bool
    # context for failure analysis
    num_candidates: int
    action_repr: str
    raw_response: str


# ── Prompt construction ──────────────────────────────────────

def format_candidate(idx: int, cand: dict) -> str:
    tag = cand.get("tag", "unknown")
    attrs = cand.get("attributes", "")
    if isinstance(attrs, str):
        try:
            attrs_dict = json.loads(attrs)
        except (json.JSONDecodeError, TypeError):
            attrs_dict = {}
    else:
        attrs_dict = attrs or {}

    # extract useful attributes
    parts = [f"[{idx}] <{tag}>"]
    for key in ["id", "class", "placeholder", "aria-label", "title", "name", "type", "href", "value", "role"]:
        val = attrs_dict.get(key, "")
        if val:
            parts.append(f'{key}="{str(val)[:80]}"')

    text_content = attrs_dict.get("text", attrs_dict.get("innerText", ""))
    if text_content:
        parts.append(f'text="{str(text_content)[:60]}"')

    return " ".join(parts)


def build_prompt(task: str, action_reprs: list[str], step_idx: int,
                 candidates: list[dict], candidate_indices: list[int]) -> str:
    # previous actions as context
    prev_actions = ""
    if step_idx > 0:
        prev_lines = [f"  Step {i+1}: {r}" for i, r in enumerate(action_reprs[:step_idx])]
        prev_actions = f"\nPrevious actions taken:\n" + "\n".join(prev_lines) + "\n"

    # format candidates
    cand_lines = []
    for display_idx, real_idx in enumerate(candidate_indices):
        cand_lines.append(format_candidate(display_idx, candidates[real_idx]))

    candidates_text = "\n".join(cand_lines)

    return f"""You are a web agent. You must complete a task on a website by selecting the right element and action.

Task: {task}
{prev_actions}
The page has the following candidate elements. Pick ONE element and ONE operation.

{candidates_text}

Respond in exactly this format, nothing else:
Element: <number>
Operation: CLICK or TYPE or SELECT
Value: <value if TYPE or SELECT, otherwise "none">"""


# ── Model calling ────────────────────────────────────────────

def call_model(prompt: str, client: OpenAI, model: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ── Response parsing ─────────────────────────────────────────

def parse_response(response: str) -> tuple[int, str, str]:
    """Parse model response into (element_idx, operation, value)."""
    # strip thinking tags if present (qwen3 does this)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    element_idx = -1
    operation = ""
    value = ""

    for line in response.split("\n"):
        line = line.strip()
        if line.lower().startswith("element:"):
            match = re.search(r'(\d+)', line)
            if match:
                element_idx = int(match.group(1))
        elif line.lower().startswith("operation:"):
            op = line.split(":", 1)[1].strip().upper()
            for valid_op in ["CLICK", "TYPE", "SELECT"]:
                if valid_op in op:
                    operation = valid_op
                    break
        elif line.lower().startswith("value:"):
            value = line.split(":", 1)[1].strip()
            if value.lower() == "none" or value == "":
                value = ""

    return element_idx, operation, value


# ── Scoring ──────────────────────────────────────────────────

def score_prediction(pred_idx: int, pred_op: str, pred_value: str,
                     gt_pos_idx: int, gt_op: str, gt_value: str) -> tuple[bool, bool, bool]:
    element_correct = (pred_idx == gt_pos_idx)
    op_correct = (pred_op == gt_op)

    if gt_op == "CLICK":
        value_correct = True
    else:
        value_correct = (pred_value.lower().strip() == gt_value.lower().strip())

    return element_correct, op_correct, value_correct


# ── Main evaluation loop ────────────────────────────────────

def load_dataset_cached():
    if CACHE_PATH.exists():
        print(f"Loading Mind2Web from local cache...")
        start = time.time()
        with open(CACHE_PATH) as f:
            ds = json.load(f)
        print(f"Loaded {len(ds)} tasks in {time.time()-start:.1f}s")
        return ds

    print(f"Loading Mind2Web from HuggingFace (first run)...")
    from datasets import load_dataset
    hf_ds = load_dataset("osunlp/Mind2Web", split="train")
    ds = []
    for i in range(len(hf_ds)):
        ex = hf_ds[i]
        ds.append({
            'annotation_id': ex['annotation_id'],
            'confirmed_task': ex['confirmed_task'],
            'website': ex['website'],
            'domain': ex['domain'],
            'action_reprs': ex['action_reprs'],
            'actions': [{'operation': a['operation'],
                         'pos_candidates': a['pos_candidates'],
                         'neg_candidates': a['neg_candidates']} for a in ex['actions']],
        })
    Path(CACHE_PATH).parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(ds))
    print(f"Cached {len(ds)} tasks to {CACHE_PATH}")
    return ds


def prepare_steps(ds, max_tasks, max_candidates, seed):
    random.seed(seed)
    task_indices = list(range(len(ds)))
    random.shuffle(task_indices)
    task_indices = task_indices[:max_tasks]

    steps = []
    for task_idx in task_indices:
        ex = ds[task_idx]
        rng = random.Random(seed + task_idx)
        for step_idx, action in enumerate(ex["actions"]):
            pos_candidates = action["pos_candidates"]
            neg_candidates = action["neg_candidates"]
            if not pos_candidates:
                continue

            all_neg = list(range(len(neg_candidates)))
            rng.shuffle(all_neg)
            sampled_neg_indices = all_neg[:max_candidates - 1]

            all_candidates = pos_candidates + [neg_candidates[i] for i in sampled_neg_indices]
            candidate_order = list(range(len(all_candidates)))
            rng.shuffle(candidate_order)

            gt_pos_display_idx = candidate_order.index(0)
            ordered_candidates = [all_candidates[i] for i in candidate_order]

            prompt = build_prompt(
                ex["confirmed_task"], ex["action_reprs"], step_idx,
                ordered_candidates, list(range(len(ordered_candidates)))
            )

            gt_op_info = action["operation"]
            steps.append({
                "prompt": prompt,
                "task_id": ex["annotation_id"],
                "step_idx": step_idx,
                "website": ex["website"],
                "domain": ex["domain"],
                "task": ex["confirmed_task"],
                "gt_op": gt_op_info["op"],
                "gt_value": gt_op_info.get("value", ""),
                "gt_pos_display_idx": gt_pos_display_idx,
                "gt_tag": pos_candidates[0].get("tag", "?"),
                "gt_attrs": str(pos_candidates[0].get("attributes", ""))[:200],
                "ordered_candidates": ordered_candidates,
                "action_repr": ex["action_reprs"][step_idx] if step_idx < len(ex["action_reprs"]) else "",
                "num_candidates": len(ordered_candidates),
            })
    return steps


def run_evaluation(
    model: str = "qwen3:8b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    max_tasks: int = 50,
    max_candidates: int = 30,
    seed: int = 42,
    concurrency: int = 1,
    output_suffix: str = "",
):
    ds = load_dataset_cached()
    steps = prepare_steps(ds, max_tasks, max_candidates, seed)

    client = OpenAI(base_url=base_url, api_key=api_key)
    model_label = model.replace("/", "_").replace(":", "_")

    print(f"Running {model} on {max_tasks} tasks ({len(steps)} steps, concurrency={concurrency})...")
    print(f"{'='*60}")
    run_start = time.time()

    def process_step(idx, step):
        response = call_model(step["prompt"], client, model)
        pred_idx, pred_op, pred_value = parse_response(response)
        element_correct, op_correct, value_correct = score_prediction(
            pred_idx, pred_op, pred_value,
            step["gt_pos_display_idx"], step["gt_op"], step["gt_value"]
        )
        step_correct = element_correct and op_correct and value_correct

        pred_tag = ""
        if 0 <= pred_idx < len(step["ordered_candidates"]):
            pred_tag = step["ordered_candidates"][pred_idx].get("tag", "?")

        return asdict(Prediction(
            task_id=step["task_id"],
            step=step["step_idx"] + 1,
            website=step["website"],
            domain=step["domain"],
            task=step["task"],
            gt_element_tag=step["gt_tag"],
            gt_element_attrs=step["gt_attrs"],
            gt_op=step["gt_op"],
            gt_value=step["gt_value"],
            pred_element_idx=pred_idx,
            pred_element_tag=pred_tag,
            pred_op=pred_op,
            pred_value=pred_value,
            element_correct=element_correct,
            op_correct=op_correct,
            value_correct=value_correct,
            step_correct=step_correct,
            num_candidates=step["num_candidates"],
            action_repr=step["action_repr"],
            raw_response=response[:500],
        ))

    results = [None] * len(steps)
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process_step, i, s): i for i, s in enumerate(steps)}
        for future in as_completed(futures):
            idx = futures[future]
            pred = future.result()
            results[idx] = pred
            completed += 1

            elapsed = time.time() - run_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(steps) - completed) / rate if rate > 0 else 0
            icon = "✅" if pred["step_correct"] else "❌"
            print(f"  [{completed}/{len(steps)}] {pred['website']}  step {pred['step']}: {icon} elem={'✓' if pred['element_correct'] else '✗'} op={'✓' if pred['op_correct'] else '✗'} val={'✓' if pred['value_correct'] else '✗'} | gt={pred['gt_op']} pred={pred['pred_op']} [{elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s]")

    total = len(results)
    correct = sum(1 for r in results if r["step_correct"])
    acc = correct / total if total > 0 else 0
    elapsed = time.time() - run_start

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} steps correct ({acc:.1%}) in {elapsed:.0f}s")
    print(f"{'='*60}")

    output = {
        "model": model,
        "total_steps": total,
        "correct_steps": correct,
        "accuracy": acc,
        "max_tasks": max_tasks,
        "max_candidates": max_candidates,
        "predictions": results,
    }

    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = RESULTS_DIR / f"{model_label}_{max_tasks}tasks{suffix}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved to {out_path}")

    failures = [p for p in results if not p["step_correct"]]
    if failures:
        print(f"\n--- Quick Failure Breakdown ({len(failures)} failures) ---")
        elem_only = sum(1 for f in failures if not f["element_correct"])
        op_only = sum(1 for f in failures if f["element_correct"] and not f["op_correct"])
        val_only = sum(1 for f in failures if f["element_correct"] and f["op_correct"] and not f["value_correct"])
        print(f"  Wrong element: {elem_only} ({elem_only/len(failures)*100:.0f}%)")
        print(f"  Right element, wrong operation: {op_only} ({op_only/len(failures)*100:.0f}%)")
        print(f"  Right element+op, wrong value: {val_only} ({val_only/len(failures)*100:.0f}%)")

    by_op = {}
    for p in results:
        op = p["gt_op"]
        by_op.setdefault(op, {"total": 0, "correct": 0})
        by_op[op]["total"] += 1
        if p["step_correct"]:
            by_op[op]["correct"] += 1

    print(f"\n--- Accuracy by Operation ---")
    for op, stats in sorted(by_op.items()):
        a = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {op}: {stats['correct']}/{stats['total']} ({a:.1%})")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-suffix", default="", help="Suffix for output filename")
    args = parser.parse_args()

    run_evaluation(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tasks=args.max_tasks,
        max_candidates=args.max_candidates,
        concurrency=args.concurrency,
        output_suffix=args.output_suffix,
    )
