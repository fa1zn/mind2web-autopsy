"""Download Mind2Web dataset and explore its structure."""

from datasets import load_dataset
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Loading Mind2Web training split from HuggingFace...")
ds = load_dataset("osunlp/Mind2Web", split="train")

print(f"\nDataset size: {len(ds)} tasks")
print(f"Columns: {ds.column_names}")

# Look at one example in detail
ex = ds[0]
print(f"\n{'='*60}")
print(f"EXAMPLE TASK")
print(f"{'='*60}")
print(f"Website: {ex['website']}")
print(f"Domain: {ex['domain']}")
print(f"Subdomain: {ex['subdomain']}")
print(f"Task: {ex['confirmed_task']}")
print(f"Number of actions: {len(ex['actions'])}")
print(f"\nAction representations (human-readable):")
for i, rep in enumerate(ex['action_reprs']):
    print(f"  Step {i+1}: {rep}")

# Look at the first action in detail
action = ex['actions'][0]
print(f"\n{'='*60}")
print(f"FIRST ACTION - DETAILED")
print(f"{'='*60}")
print(f"Operation: {action['operation']}")
print(f"Number of positive candidates (correct): {len(action['pos_candidates'])}")
print(f"Number of negative candidates (wrong): {len(action['neg_candidates'])}")

if action['pos_candidates']:
    pos = action['pos_candidates'][0]
    print(f"\nCorrect element:")
    print(f"  Tag: {pos['tag']}")
    print(f"  Attributes: {pos['attributes'][:200] if pos.get('attributes') else 'N/A'}")

if action['neg_candidates']:
    print(f"\nFirst 3 wrong elements:")
    for neg in action['neg_candidates'][:3]:
        print(f"  Tag: {neg['tag']}, Attrs: {str(neg.get('attributes', ''))[:100]}")

print(f"\nCleaned HTML length: {len(action.get('cleaned_html', ''))}")

# Save a few examples as JSON for inspection
print(f"\n{'='*60}")
print("Saving 5 sample tasks to data/samples.json...")
samples = []
for i in range(5):
    ex = ds[i]
    sample = {
        "annotation_id": ex["annotation_id"],
        "website": ex["website"],
        "domain": ex["domain"],
        "confirmed_task": ex["confirmed_task"],
        "action_reprs": ex["action_reprs"],
        "num_actions": len(ex["actions"]),
        "actions_summary": []
    }
    for j, action in enumerate(ex["actions"]):
        sample["actions_summary"].append({
            "step": j + 1,
            "operation": action["operation"],
            "num_pos_candidates": len(action["pos_candidates"]),
            "num_neg_candidates": len(action["neg_candidates"]),
            "pos_tags": [p["tag"] for p in action["pos_candidates"]],
            "cleaned_html_length": len(action.get("cleaned_html", "")),
        })
    samples.append(sample)

(DATA_DIR / "samples.json").write_text(json.dumps(samples, indent=2))
print("Done. Check data/samples.json to see the structure.")

# Dataset stats
print(f"\n{'='*60}")
print("DATASET STATS")
print(f"{'='*60}")
domains = {}
websites = set()
total_actions = 0
ops = {"CLICK": 0, "TYPE": 0, "SELECT": 0}

for ex in ds:
    d = ex["domain"]
    domains[d] = domains.get(d, 0) + 1
    websites.add(ex["website"])
    for action in ex["actions"]:
        total_actions += 1
        op = action["operation"]["op"]
        if op in ops:
            ops[op] += 1

print(f"Total tasks: {len(ds)}")
print(f"Total action steps: {total_actions}")
print(f"Unique websites: {len(websites)}")
print(f"Unique domains: {len(domains)}")
print(f"\nOperations breakdown:")
for op, count in sorted(ops.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count} ({count/total_actions*100:.1f}%)")
print(f"\nDomains:")
for d, count in sorted(domains.items(), key=lambda x: -x[1]):
    print(f"  {d}: {count} tasks")
