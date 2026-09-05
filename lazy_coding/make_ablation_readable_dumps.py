import json
import sys
from pathlib import Path

sys.path.insert(0, "/marimo/mats/lazy_coding")
from lazy_run_rollouts import write_readable_dump  # reused for ablation A/B (same schema)

RESULTS_DIR = Path("/marimo/mats/lazy_coding/results_ablations")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def dump_toy_tests(records, out_path):
    with open(out_path, "w") as f:
        for r in records:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Rollout {r['rollout_id']} | kind={r['kind']} | success={r['success']} "
                    f"| turns={r['n_turns_used']} | declared_done={r['declared_done']}\n")
            f.write(f"used_cat={r['used_cat']} mypy_errors={r.get('mypy_errors')} "
                    f"behavior_preserved={r.get('behavior_preserved')}\n")
            if r.get("error"):
                f.write(f"ERROR: {r['error']}\n")
            f.write(f"{'-' * 70}\n")
            for t in r["transcript"]:
                f.write(f"\n--- turn {t['turn']} ---\n")
                if t.get("think"):
                    f.write("THINK:\n" + t["think"] + "\n")
                f.write("VISIBLE:\n" + (t.get("visible") or "") + "\n")
                cmd = t.get("command")
                cmd_str = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(cmd)) if isinstance(cmd, list) else (cmd or "(none)")
                f.write("COMMAND:\n" + cmd_str + "\n")
                if t.get("observation"):
                    f.write("OBSERVATION:\n" + t["observation"] + "\n")
            f.write(f"{'=' * 70}\n")


def dump_shell_syntax_check(records, out_path):
    with open(out_path, "w") as f:
        for r in records:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Case: {r['name']}\n")
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"{'-' * 70}\n")
            if r.get("think"):
                f.write("THINK:\n" + r["think"] + "\n\n")
            f.write("VISIBLE:\n" + (r.get("visible") or "") + "\n\n")
            f.write(f"COMMAND: {r.get('command')!r}\n")
            f.write(f"n_blocks_in_reply: {r.get('n_blocks')}\n")
            f.write(f"syntax_valid: {r.get('syntax_valid')}\n")
            if r.get("syntax_error"):
                f.write(f"syntax_error: {r['syntax_error']}\n")
            f.write(f"exit_code: {r.get('exit_code')}\n")
            if r.get("run_output"):
                f.write(f"run_output:\n{r['run_output']}\n")
            f.write(f"functionally_correct: {r.get('functionally_correct')}\n")
            if r.get("check_error"):
                f.write(f"check_error: {r['check_error']}\n")
            f.write(f"{'=' * 70}\n")


if __name__ == "__main__":
    # Ablation A / B: same schema as the main lazy_run_rollouts.py output
    # (they go through classify_rollout + the standard transcript shape), so
    # reuse write_readable_dump directly.
    for name in ["ablation_a_steering", "ablation_b_destressed"]:
        records = load_jsonl(RESULTS_DIR / f"{name}.jsonl")
        write_readable_dump(records, RESULTS_DIR / f"{name}_readable.txt")
        print(f"wrote {name}_readable.txt ({len(records)} records)")

    toy_records = load_jsonl(RESULTS_DIR / "toy_tests.jsonl")
    dump_toy_tests(toy_records, RESULTS_DIR / "toy_tests_readable.txt")
    print(f"wrote toy_tests_readable.txt ({len(toy_records)} records)")

    shell_records = load_jsonl(RESULTS_DIR / "shell_syntax_check.jsonl")
    dump_shell_syntax_check(shell_records, RESULTS_DIR / "shell_syntax_check_readable.txt")
    print(f"wrote shell_syntax_check_readable.txt ({len(shell_records)} records)")

    print("ALL_READABLE_DUMPS_DONE")
