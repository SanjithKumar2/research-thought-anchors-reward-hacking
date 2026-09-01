# lazy_coding/lazy_run_rollouts.py
import argparse
import json
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer
from openai import OpenAI

from lazy_config import MODEL_ID, N_ROLLOUTS, RESULTS_DIR, VLLM_BASE_URL
from lazy_agent import run_agent_rollout

_probe_client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=60.0, max_retries=1)


def load_tokenizer():
    print(f"Loading tokenizer for {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    models = _probe_client.models.list().data
    print(f"Connected to vLLM server at {VLLM_BASE_URL}. Serving: {[m.id for m in models]}")
    return tok


def write_readable_dump(results: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"Rollout {r['rollout_id']} | label={r['label']} | turns={r['n_turns_used']} "
                    f"| declared_done={r['declared_done']} | hit_turn_limit={r['hit_turn_limit']}\n")
            f.write(f"canonical_mypy_errors_remaining={r['canonical_mypy_errors_remaining']} "
                    f"type_ignore_count={r['type_ignore_count']} "
                    f"annotated_defs={r['annotated_defs']}/{r['total_defs']} "
                    f"committed_changes={r['committed_changes']}\n")
            if r.get("error"):
                f.write(f"ERROR: {r['error']}\n")
            f.write(f"{'-'*70}\n")
            for t in r["transcript"]:
                f.write(f"\n--- turn {t['turn']} ---\n")
                if t.get("think"):
                    f.write("THINK:\n" + t["think"] + "\n")
                f.write("VISIBLE:\n" + (t.get("visible") or "") + "\n")
                f.write("COMMAND:\n" + (t.get("command") or "(none)") + "\n")
                if t.get("observation"):
                    f.write("OBSERVATION:\n" + t["observation"] + "\n")
            f.write(f"\nFINAL PRE-COMMIT HOOK CONTENT:\n{r['hook_content']}\n")
            f.write(f"{'='*70}\n")


def run_batch(tok, n: int, concurrency: int, out_path: Path) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    results = []
    failures = []

    def _do(i):
        try:
            r = run_agent_rollout(tok, i)
        except Exception as e:
            print(f"  [WARN] rollout {i} failed: {type(e).__name__}: {e}", flush=True)
            failures.append(i)
            return None
        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(r) + "\n")
        print(f"  done rollout {i}: label={r['label']} turns={r['n_turns_used']} "
              f"errors_left={r['canonical_mypy_errors_remaining']}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_do, i) for i in range(n)]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda r: r["rollout_id"])

    if results:
        n_res = len(results)
        counts = {}
        for r in results:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        avg_turns = sum(r["n_turns_used"] for r in results) / n_res
        print(f"\n=== SUMMARY (n={n_res}, failures={len(failures)}) ===")
        for label, c in sorted(counts.items()):
            print(f"  {label}: {c}/{n_res} ({c/n_res:.1%})")
        print(f"  avg turns used: {avg_turns:.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_ROLLOUTS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--single", action="store_true", help="run exactly 1 rollout, verbose")
    args = parser.parse_args()

    tok = load_tokenizer()
    results_dir = Path(RESULTS_DIR)
    out_path = results_dir / "lazy_rollouts.jsonl"

    n = 1 if args.single else args.n
    results = run_batch(tok, n=n, concurrency=(1 if args.single else args.concurrency), out_path=out_path)

    readable_path = results_dir / "lazy_rollouts_readable.txt"
    write_readable_dump(results, readable_path)
    print(f"\nWrote {out_path} and {readable_path}")

    if not args.single:
        subprocess.run(["git", "add", "-A"], cwd="/marimo/mats")
        subprocess.run(["git", "commit", "-m", f"lazy_coding: {n}-rollout pilot batch on DeepSeek-R1-0528-Qwen3-8B"], cwd="/marimo/mats")
        subprocess.run(["git", "push", "origin", "main"], cwd="/marimo/mats")
        print("LAZY_BATCH_DONE")
