# Mind2Web Autopsy

A failure-mode autopsy of web agents on OSU NLP's [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) dataset, plus a shaped verifiable reward function designed to replace binary scoring for GRPO training.

## Motivation

Web agent benchmarks typically report aggregate accuracy — a model gets 30% of actions right, another gets 11%. But accuracy alone doesn't tell you *why* models fail or *how* to fix them. And binary reward signals (correct/incorrect) discard the gradient between failure types, producing dead GRPO batches where the optimizer has nothing to learn from.

This project does two things:

1. **Autopsy**: Decompose every failure across two models into a taxonomy of element selection errors, operation errors, and value errors — then analyze patterns across 12 dimensions.
2. **Shaped Reward**: Replace binary pass/fail with a verifiable reward function that gives partial credit based on failure mode, designed for GRPO training of web agents.

## Results

| Model | Steps | Accuracy | Element Errors | Operation Errors | Value Errors |
|-------|-------|----------|---------------|-----------------|-------------|
| GPT-4.1-mini | 407 | 30.0% | 62% of failures | 4% | 3% |
| Qwen2.5-7B-Instruct-Turbo | 407 | 11.5% | 81% of failures | 8% | 2% |

### Key Findings

- **67.8% of steps are hard for both models.** They fail on the same steps but pick different wrong elements 71% of the time — the bottleneck is candidate representation, not model capability.
- **Same-tag confusion dominates.** GPT-4.1-mini picks the right element type (e.g., a link) but the wrong instance 19% of the time. Binary scoring treats this identically to picking a completely wrong element type.
- **Shaped reward eliminates dead batches.** In simulated GRPO batches, binary scoring produces 40% dead batches (zero variance → no gradient signal). The shaped reward drops this to 2%.

### Intervention Experiment

To validate that the failure taxonomy is actionable, not just descriptive:

| Condition | Candidates | Accuracy | Same-Tag Confusions |
|-----------|-----------|----------|-------------------|
| Baseline | 30 | 30.0% | 19% of failures |
| Intervention | 10 | 49.6% | -42% reduction |

Reducing candidates from 30 to 10 directly targets same-tag confusion. Accuracy jumps from 30% to 50%, confirming the failure mode is real and addressable.

## Shaped Reward Function

The core idea: a model that picks the right tag but the wrong instance is closer to correct than a model that doesn't understand the page at all. The reward should know the difference.

```python
def shaped_reward(pred, gt, w_elem=0.4, w_op=0.3, w_val=0.3):
    return (w_elem * element_reward(pred, gt) +
            w_op * operation_reward(pred, gt) +
            w_val * value_reward(pred, gt))
```

**Element reward** (weight 0.4):
- Exact match → 1.0
- Same tag, wrong instance → 0.5 (partial credit for same-tag confusion)
- Wrong tag → 0.0

**Operation reward** (weight 0.3):
- Exact match → 1.0
- Wrong operation → 0.0

**Value reward** (weight 0.3):
- Exact match → 1.0
- Substring containment → 0.7
- Word overlap (Jaccard) → proportional
- No match → 0.0

All sub-rewards are deterministic — no LLM judge, fully reproducible. This is relevant to [WebJudge-7B](https://arxiv.org/abs/2410.09305): the binary signal it provides discards the gradient between failure types that could accelerate GRPO training.

## 12-Dimension Deep Analysis

The full analysis (`deep_analysis.py`) covers:

1. **Positional bias** — do models prefer certain candidate positions?
2. **Candidate count vs. accuracy** — how does the number of candidates affect performance?
3. **Step position vs. accuracy** — do models degrade on later steps in a task?
4. **Tag confusion matrix** — which element types get confused with which?
5. **Operation confusion** — CLICK/TYPE/SELECT error patterns
6. **Value error analysis** — over-specification, under-specification, wrong values
7. **Cross-model agreement** — which steps are universally hard?
8. **Domain difficulty** — which website categories are hardest?
9. **Response format quality** — how often do models fail to follow the output format?
10. **Error cascade** — do early errors in a task cause later errors?
11. **Task difficulty distribution** — per-task accuracy spread
12. **Website complexity** — accuracy by website

## Project Structure

```
├── evaluate.py           # Core evaluation pipeline (concurrent, cached)
├── reward.py             # Shaped reward function + GRPO batch simulation
├── analyze_failures.py   # Basic failure categorization
├── deep_analysis.py      # 12-dimension analysis
├── run_all.py            # Parallel multi-model runner
├── run_interventions.py  # Intervention experiments
├── download_data.py      # Dataset exploration
└── results/              # Full evaluation results (JSON)
```

## Usage

### Run evaluation

```bash
# GPT-4.1-mini via OpenAI
python evaluate.py --model gpt-4.1-mini \
  --base-url https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY \
  --max-tasks 50 --concurrency 10

# Qwen2.5-7B via Together AI
python evaluate.py --model Qwen/Qwen2.5-7B-Instruct-Turbo \
  --base-url https://api.together.xyz/v1 \
  --api-key $TOGETHER_API_KEY \
  --max-tasks 50 --concurrency 10
```

### Run reward analysis

```bash
python reward.py  # Analyzes reward distribution, simulates GRPO batches
```

### Run deep analysis

```bash
python deep_analysis.py  # Generates 12-dimension analysis from results/
```

## Dataset

Uses [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) (OSU NLP Group) — 1,009 tasks across 73 websites and 7,775 action steps with ground-truth annotations. The evaluation pipeline caches the dataset locally on first run (~1.1GB).

## Related Work

- [Mind2Web: Towards a Generalist Agent for the Web](https://arxiv.org/abs/2306.06070) — Deng et al., 2023
- [WebJudge-7B](https://arxiv.org/abs/2410.09305) — LLM-based binary judge for web agent evaluation
- [RedlineBench](https://github.com/fa1zn/redlinebench) — RL environment for contract negotiation with verifiable rewards (companion project)
