"""
Run all models in parallel. One command, walk away, come back to results.

Usage:
  python3 run_all.py                          # all models
  python3 run_all.py --models gpt-4.1-mini    # single model
  python3 run_all.py --max-tasks 20           # fewer tasks for quick test
"""

import subprocess
import sys
import os
import time
import argparse

MODELS = {
    "gpt-4.1-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "concurrency": 10,
    },
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "concurrency": 10,
    },
    "qwen3:8b": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
        "api_key": "ollama",
        "concurrency": 1,
    },
}

def run_model(model_name, config, max_tasks):
    if config.get("api_key_env"):
        api_key = os.environ.get(config["api_key_env"], "")
        if not api_key:
            print(f"  SKIP {model_name}: {config['api_key_env']} not set")
            return None
    else:
        api_key = config.get("api_key", "")

    cmd = [
        sys.executable, "evaluate.py",
        "--model", model_name,
        "--base-url", config["base_url"],
        "--api-key", api_key,
        "--max-tasks", str(max_tasks),
        "--concurrency", str(config["concurrency"]),
    ]
    print(f"  START {model_name} (concurrency={config['concurrency']})")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    models_to_run = args.models or list(MODELS.keys())

    print(f"{'='*60}")
    print(f"Mind2Web Failure Autopsy — Running {len(models_to_run)} models")
    print(f"Tasks: {args.max_tasks} | Parallel model runs")
    print(f"{'='*60}\n")

    start = time.time()
    procs = {}

    for name in models_to_run:
        if name not in MODELS:
            print(f"  Unknown model: {name}")
            continue
        proc = run_model(name, MODELS[name], args.max_tasks)
        if proc:
            procs[name] = proc

    if not procs:
        print("No models to run. Check your API keys.")
        sys.exit(1)

    print(f"\n  {len(procs)} models running in parallel. Waiting...\n")

    for name, proc in procs.items():
        output, _ = proc.communicate()
        elapsed = time.time() - start
        status = "DONE" if proc.returncode == 0 else "FAIL"
        print(f"\n{'='*60}")
        print(f"[{status}] {name} ({elapsed:.0f}s)")
        print(f"{'='*60}")
        lines = output.strip().split("\n")
        for line in lines[-20:]:
            print(f"  {line}")

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_time:.0f}s")
    print(f"{'='*60}")
    print(f"\nNext: python3 analyze_failures.py")
