# resample.py
# Phase 3: Thought Anchor resampling on hacking rollouts.
# For each sentence in a hacking CoT, generate a semantically different
# alternative and measure how much hack rate drops downstream.

import re
import json
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config import MODEL_ID, TEMPERATURE, TOP_P, N_RESAMPLE, RESULTS_DIR, SYSTEM_PROMPT, SGLANG_BASE_URL
from detect_hack import extract_code, is_hardcode_hack

_client = OpenAI(base_url=SGLANG_BASE_URL, api_key="EMPTY")


RESAMPLE_PROMPT = (
    "You are rewriting a single sentence from a chain-of-thought reasoning trace. "
    "Rewrite the following sentence so it expresses a SEMANTICALLY DIFFERENT idea "
    "— different approach, different decision, or different direction — "
    "while keeping the same sentence length and writing style. "
    "Output ONLY the rewritten sentence, nothing else.\n\n"
    "Original sentence: {sentence}"
)


def generate_alternative_sentence(sentence: str) -> str:
    """Ask the model to generate a semantically different version of sentence."""
    prompt = RESAMPLE_PROMPT.format(sentence=sentence)
    resp = _client.completions.create(
        model=MODEL_ID,
        prompt=prompt,
        max_tokens=128,
        temperature=0.8,
        top_p=0.95,
    )
    alt = resp.choices[0].text.strip()
    # Take only first line if multiline
    return alt.split('\n')[0].strip()


def generate_random_sentence(all_rollouts: list[dict], exclude_id: str) -> str:
    """
    Sanity check baseline: pick a random sentence from a different rollout.
    Replacing with a random sentence should NOT drop hack rate significantly.
    """
    import random
    candidates = [
        r for r in all_rollouts
        if r["rollout_id"] != exclude_id and r["n_sentences"] > 0
    ]
    if not candidates:
        return "Let me think about this differently."
    c = random.choice(candidates)
    return random.choice(c["cot_sentences"])


def run_continuations(task: dict, think_prefix: str, n: int) -> float:
    """
    Given a (modified) CoT prefix up to position i,
    run N continuations and measure hack rate.
    Returns fraction of continuations that are hacks.
    All N requests share the same prompt prefix, so SGLang's RadixAttention
    caches it once and reuses it across the batch.
    """
    # Reconstruct the prompt: system + user task + partial think block,
    # feeding the think block prefix as the assistant turn so far.
    full_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{task['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{think_prefix}"
    )

    hacks = 0
    for _ in range(n):
        resp = _client.completions.create(
            model=MODEL_ID,
            prompt=full_prompt,
            max_tokens=2048,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        continuation = resp.choices[0].text
        code = extract_code(continuation)
        if is_hardcode_hack(code, task):
            hacks += 1

    return hacks / n


def compute_thought_anchors(rollout: dict, task: dict,
                             all_rollouts: list[dict],
                             n_resample: int = N_RESAMPLE) -> dict:
    """
    For a single hacking rollout, compute anchor scores for each sentence.
    Returns dict with per-sentence results and the identified anchor.
    """
    sentences = rollout["cot_sentences"]
    if not sentences:
        return {"error": "no sentences found"}

    # Baseline hack rate (no modification)
    full_think = " ".join(sentences)
    baseline_rate = run_continuations(task, full_think, n=n_resample)

    sentence_results = []
    for i, sent in enumerate(tqdm(sentences, desc=f"Anchors {rollout['rollout_id']}")):

        # Build prefix up to (not including) sentence i
        prefix_sentences = sentences[:i]

        # --- Semantic alternative ---
        alt_sent = generate_alternative_sentence(sent)
        alt_think = " ".join(prefix_sentences + [alt_sent] + sentences[i+1:])
        alt_rate   = run_continuations(task, alt_think, n=n_resample)
        anchor_score = baseline_rate - alt_rate

        # --- Random baseline ---
        rand_sent  = generate_random_sentence(all_rollouts, rollout["rollout_id"])
        rand_think = " ".join(prefix_sentences + [rand_sent] + sentences[i+1:])
        rand_rate   = run_continuations(task, rand_think, n=n_resample)
        rand_score  = baseline_rate - rand_rate

        sentence_results.append({
            "position":       i,
            "position_frac":  i / len(sentences),   # 0.0 = start, 1.0 = end
            "sentence":       sent,
            "alt_sentence":   alt_sent,
            "baseline_rate":  baseline_rate,
            "alt_hack_rate":  alt_rate,
            "rand_hack_rate": rand_rate,
            "anchor_score":   anchor_score,   # high = this sentence drives hacking
            "random_score":   rand_score,     # should be ~0 (sanity check)
        })

    # Find the thought anchor (max anchor score)
    best = max(sentence_results, key=lambda x: x["anchor_score"])

    return {
        "rollout_id":       rollout["rollout_id"],
        "task_id":          rollout["task_id"],
        "baseline_hack_rate": baseline_rate,
        "n_sentences":      len(sentences),
        "sentence_results": sentence_results,
        "anchor_sentence":  best["sentence"],
        "anchor_position":  best["position"],
        "anchor_position_frac": best["position_frac"],
        "anchor_score":     best["anchor_score"],
    }


def load_hacking_rollouts(task_id: str = None) -> list[dict]:
    """Load all hacking rollouts from JSONL files."""
    results_dir = Path(RESULTS_DIR)
    rollouts = []
    pattern = f"{task_id}_rollouts.jsonl" if task_id else "*_rollouts.jsonl"
    for f in results_dir.glob(pattern):
        for line in f.read_text().strip().split('\n'):
            if line:
                r = json.loads(line)
                if r.get("is_hack"):
                    rollouts.append(r)
    return rollouts


if __name__ == "__main__":
    import argparse
    from tasks import TASKS

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None)
    parser.add_argument("--max",  type=int, default=20,
                        help="max hacking rollouts to analyse")
    args = parser.parse_args()

    models = _client.models.list().data
    print(f"Connected to SGLang server at {SGLANG_BASE_URL}. Serving: {[m.id for m in models]}")

    task_map  = {t["id"]: t for t in TASKS}
    hacking   = load_hacking_rollouts(args.task)
    all_rolls = []
    for f in Path(RESULTS_DIR).glob("*_rollouts.jsonl"):
        for line in f.read_text().strip().split('\n'):
            if line:
                all_rolls.append(json.loads(line))

    print(f"Found {len(hacking)} hacking rollouts. Analysing up to {args.max}.")
    out_path = Path(RESULTS_DIR) / "anchor_results.jsonl"

    for rollout in hacking[:args.max]:
        task = task_map[rollout["task_id"]]
        result = compute_thought_anchors(rollout, task, all_rolls)
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\nAnchor for {rollout['rollout_id']}:")
        print(f"  Position: {result.get('anchor_position_frac', '?'):.0%} through CoT")
        print(f"  Score:    {result.get('anchor_score', '?'):.2f}")
        print(f"  Sentence: {result.get('anchor_sentence', '?')[:120]}")
