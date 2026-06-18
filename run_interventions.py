"""
Intervention Experiments

Tests whether targeting specific failure modes with prompt changes
actually improves accuracy on those modes.

Experiment A: "Type minimum text" — targets value over-specification
Experiment B: Fewer candidates (10 vs 30) — targets same-tag confusion
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def run_experiment(name, model, base_url, api_key, max_tasks, extra_args=None):
    cmd = [
        sys.executable, "evaluate.py",
        "--model", model,
        "--base-url", base_url,
        "--api-key", api_key,
        "--max-tasks", str(max_tasks),
        "--concurrency", "10",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  Running {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
    return result.returncode == 0


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY not set")
        sys.exit(1)

    max_tasks = 50
    model = "gpt-4.1-mini"
    base_url = "https://api.openai.com/v1"

    print(f"{'='*60}")
    print(f"INTERVENTION EXPERIMENTS — {model}")
    print(f"{'='*60}")

    # Experiment B: fewer candidates
    print(f"\n{'='*60}")
    print(f"EXPERIMENT B: 10 candidates instead of 30")
    print(f"{'='*60}")
    run_experiment(
        "10-candidates", model, base_url, api_key, max_tasks,
        extra_args=["--max-candidates", "10"]
    )

    # compare results
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")

    baseline_path = RESULTS_DIR / f"{model}_50tasks.json"
    experiment_path = RESULTS_DIR / f"{model}_50tasks.json"

    # find experiment file (it overwrites with same name, so we need to handle this)
    # Let's compare by loading whatever is latest
    files = sorted(RESULTS_DIR.glob(f"{model}*tasks.json"), key=lambda f: f.stat().st_mtime)

    if len(files) >= 1:
        for f in files:
            data = json.loads(f.read_text())
            print(f"\n  {f.name}:")
            print(f"    Accuracy: {data['accuracy']:.1%}")
            print(f"    Max candidates: {data.get('max_candidates', '?')}")

            # failure breakdown
            failures = [p for p in data["predictions"] if not p["step_correct"]]
            if failures:
                elem_wrong = sum(1 for f in failures if not f["element_correct"])
                same_tag = sum(1 for f in failures
                              if not f["element_correct"]
                              and f["pred_element_tag"].lower() == f["gt_element_tag"].lower())
                print(f"    Element failures: {elem_wrong} ({elem_wrong/len(data['predictions'])*100:.1f}%)")
                print(f"    Same-tag confusion: {same_tag} ({same_tag/len(data['predictions'])*100:.1f}%)")
