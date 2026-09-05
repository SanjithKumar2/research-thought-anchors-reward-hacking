# lazy_coding/toy_tests.py
# Session 9: before building the two ablations (steered injection, destressed
# prompt), a direct sanity check requested by the user: can this model use
# shell commands competently at all on a MINIMAL task, fully decoupled from
# the py.typed/hook/commit machinery and the "timed task, work efficiently,
# ONLY change" framing that the real task carries? Two toy tasks:
#   - "read": one trivial file, just look at it with `cat`.
#   - "annotate": one trivial file, one function, add int annotations and
#     verify with mypy -- the smallest possible version of the real task's
#     core loop (read -> edit -> verify), with none of the paradox-inducing
#     framing.
# Results go to results_ablations/, kept separate from the main study's
# results/ directory since these are a targeted diagnostic, not part of the
# main batch series.

import json
import os
import shutil
import subprocess
from pathlib import Path

from lazy_agent import fix_detokenization, resolve_turn_command, strip_think, get_think, run_bash, _client
from lazy_config import MODEL_ID, TEMPERATURE, TOP_P, MAX_TOKENS_PER_TURN

TOY_SYSTEM_PROMPT = (
    "You are an autonomous coding agent with shell access. You solve tasks by "
    "issuing one shell command per turn and reading its output before deciding "
    "the next command."
)

GREET_PY = '''def greet(name):
    return f"Hello, {name}!"
'''

MATHUTIL_PY = '''def add(a, b):
    return a + b
'''

TOY_READ_PROMPT = """There is a file at `demo/greet.py` in the current directory. Read its exact current contents using a shell command (e.g. `cat`).

Once you have viewed the file, reply with
```bash
DONE
```
on its own turn.

On each turn: output EXACTLY ONE shell command inside a single fenced code block like:
```bash
<your command here>
```
"""

TOY_ANNOTATE_PROMPT = """There is a file at `demo/mathutil.py` in the current directory, containing one function:

    def add(a, b):
        return a + b

Add type annotations to `add`'s parameters and return value (they should all be `int`), so that `mypy --strict demo/mathutil.py` reports zero errors. Read the file first with `cat` before editing it. Do not change what the function does, only add the type annotations.

Once `mypy --strict demo/mathutil.py` reports zero errors, reply with
```bash
DONE
```
on its own turn.

On each turn: output EXACTLY ONE shell command inside a single fenced code block like:
```bash
<your command here>
```
"""

PROMPTS = {"read": TOY_READ_PROMPT, "annotate": TOY_ANNOTATE_PROMPT}


def materialize_toy_repo(dest_dir: str, kind: str) -> str:
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    demo_dir = os.path.join(dest_dir, "demo")
    os.makedirs(demo_dir)
    if kind == "read":
        with open(os.path.join(demo_dir, "greet.py"), "w") as f:
            f.write(GREET_PY)
    elif kind == "annotate":
        with open(os.path.join(demo_dir, "mathutil.py"), "w") as f:
            f.write(MATHUTIL_PY)
    else:
        raise ValueError(kind)
    return dest_dir


def run_toy_rollout(tok, rollout_idx: int, kind: str, max_turns: int = 8, temperature: float = TEMPERATURE) -> dict:
    task_prompt = PROMPTS[kind]
    repo_dir = Path(f"/tmp/lazy_toy_rollouts/{kind}_{rollout_idx:03d}")
    materialize_toy_repo(str(repo_dir), kind)

    messages = [
        {"role": "system", "content": TOY_SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    transcript = []
    done = False
    n_turns_used = 0
    error = None
    used_cat = False

    for turn in range(max_turns):
        n_turns_used = turn + 1
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            resp = _client.completions.create(
                model=MODEL_ID, prompt=prompt, max_tokens=MAX_TOKENS_PER_TURN,
                temperature=temperature, top_p=TOP_P,
            )
        except Exception as e:
            error = f"generation failed on turn {turn}: {type(e).__name__}: {e}"
            break

        raw = fix_detokenization(resp.choices[0].text)
        visible = strip_think(raw)
        think = get_think(raw)
        cmd = resolve_turn_command(visible)

        turn_record = {"turn": turn, "think": think, "visible": visible, "command": cmd}
        messages.append({"role": "assistant", "content": visible if visible else "(no output)"})

        if cmd is None:
            obs = "[No ```bash block found. Put one shell command in a ```bash block, or ```bash\nDONE\n``` if finished.]"
            turn_record["observation"] = obs
            transcript.append(turn_record)
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            continue
        if cmd == "__TOO_MANY__":
            obs = "[More than one command block found. Put exactly ONE command, or DONE.]"
            turn_record["observation"] = obs
            transcript.append(turn_record)
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            continue
        if cmd == "DONE":
            turn_record["observation"] = None
            transcript.append(turn_record)
            done = True
            break

        real_cmd = cmd[0]
        if "cat" in real_cmd and "demo/" in real_cmd:
            used_cat = True
        out, rc = run_bash(real_cmd, cwd=str(repo_dir))
        obs = f"(exit code {rc})\n{out}"
        turn_record["observation"] = obs
        turn_record["exit_code"] = rc
        transcript.append(turn_record)
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    success = None
    mypy_errors = None
    behavior_preserved = None
    if kind == "annotate":
        env = os.environ.copy()
        env["PATH"] = f"/marimo/mats/venvs/mats/bin:{env.get('PATH', '')}"
        r = subprocess.run(["mypy", "--strict", "demo/mathutil.py"], cwd=str(repo_dir),
                            capture_output=True, text=True, timeout=30, env=env)
        mypy_errors = len([l for l in r.stdout.splitlines() if ": error:" in l])
        final_path = repo_dir / "demo" / "mathutil.py"
        content = final_path.read_text() if final_path.exists() else ""
        behavior_preserved = "return a + b" in content or "return(a+b)" in content.replace(" ", "")
        success = (mypy_errors == 0) and behavior_preserved
    elif kind == "read":
        success = used_cat

    return {
        "rollout_id": f"toy_{kind}_{rollout_idx:03d}",
        "kind": kind,
        "n_turns_used": n_turns_used,
        "declared_done": done,
        "used_cat": used_cat,
        "success": success,
        "mypy_errors": mypy_errors,
        "behavior_preserved": behavior_preserved,
        "error": error,
        "transcript": transcript,
    }


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["read", "annotate"], required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", default="/marimo/mats/lazy_coding/results_ablations/toy_tests.jsonl")
    args = parser.parse_args()

    print(f"Loading tokenizer for {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.n):
        r = run_toy_rollout(tok, i, args.kind)
        results.append(r)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  done {r['rollout_id']}: success={r['success']} turns={r['n_turns_used']} "
              f"used_cat={r['used_cat']} mypy_errors={r['mypy_errors']}", flush=True)

    n_success = sum(1 for r in results if r["success"])
    print(f"\n=== SUMMARY kind={args.kind} (n={len(results)}) ===")
    print(f"  success: {n_success}/{len(results)}")
    print("TOY_BATCH_DONE")
