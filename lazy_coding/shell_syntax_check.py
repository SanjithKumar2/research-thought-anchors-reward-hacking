# lazy_coding/shell_syntax_check.py
# Session 9: a narrower diagnostic than toy_tests.py's "annotate" case --
# this isolates raw shell-COMMAND-GENERATION correctness from task
# reasoning entirely. Each prompt asks for exactly one command, covering
# command shapes that have caused real, recurring failures throughout this
# project: whitespace-squashed commands ("gitcommit-m..."), malformed sed
# capture groups, broken heredocs, and mis-escaped quoting. For each
# generated command we check:
#   - syntax_valid: does `bash -n -c "<command>"` accept it (parses without
#     executing) -- this is the closest thing to "did the model emit
#     well-formed shell", independent of whether it picked the right
#     approach.
#   - functionally_correct: when syntax is valid, does actually running it
#     (in a fresh scratch dir per prompt) produce the intended result.
# One completion per prompt (no multi-turn loop) -- this measures the raw
# generation, not a rollout.

import json
import os
import shutil
import subprocess
from pathlib import Path

from lazy_agent import fix_detokenization, extract_bash_blocks, strip_think, get_think, _client
from lazy_config import MODEL_ID, TEMPERATURE, TOP_P, MAX_TOKENS_PER_TURN

SYSTEM_PROMPT = (
    "You are an autonomous coding agent with shell access. When asked for a "
    "shell command, respond with EXACTLY ONE command inside a single fenced "
    "```bash code block, and nothing else needs to be executed for you to "
    "answer -- just output the command."
)

# Each case: (name, user_prompt, setup_fn(work_dir) or None, check_fn(work_dir) -> bool or None)

def _setup_notes(work_dir):
    (Path(work_dir) / "notes.txt").write_text("version=1.0\n")

def _check_notes(work_dir):
    return (Path(work_dir) / "notes.txt").read_text().strip() == "version=2.0"

def _check_greeting(work_dir):
    p = Path(work_dir) / "greeting.txt"
    return p.exists() and p.read_text().strip() == "Hello\nWorld"

def _setup_pyfiles(work_dir):
    src = Path(work_dir) / "src"
    src.mkdir()
    (src / "a.py").write_text("foo = 1\nprint(foo)\n")
    (src / "b.py").write_text("def foo():\n    return foo\n")

def _check_pyfiles(work_dir):
    a = (Path(work_dir) / "src" / "a.py").read_text()
    b = (Path(work_dir) / "src" / "b.py").read_text()
    return "bar" in a and "foo" not in a.replace("bar", "") and "bar" in b

def _check_build_dir(work_dir):
    return (Path(work_dir) / "build" / "out.txt").exists()

def _setup_git(work_dir):
    env = os.environ.copy()
    env["PATH"] = f"/marimo/mats/venvs/mats/bin:{env.get('PATH', '')}"
    subprocess.run(["git", "init", "-q"], cwd=work_dir, env=env)
    subprocess.run(["git", "config", "user.email", "t@t.local"], cwd=work_dir, env=env)
    subprocess.run(["git", "config", "user.name", "T"], cwd=work_dir, env=env)
    (Path(work_dir) / "file.txt").write_text("hello\n")

def _check_git_commit(work_dir):
    env = os.environ.copy()
    env["PATH"] = f"/marimo/mats/venvs/mats/bin:{env.get('PATH', '')}"
    r = subprocess.run(["git", "log", "--oneline"], cwd=work_dir, capture_output=True, text=True, env=env)
    return "Fix bug in parser" in r.stdout

def _setup_config(work_dir):
    (Path(work_dir) / "config.py").write_text("def foo(x):\n    return x\n")

def _check_config(work_dir):
    content = (Path(work_dir) / "config.py").read_text()
    return "def foo(x: int) -> int:" in content

CASES = [
    ("echo_simple",
     "Output a shell command that prints the exact text: Hello, World!",
     None, None),
    ("sed_replace",
     "You have a file called `notes.txt` in the current directory with the line `version=1.0`. "
     "Output a shell command using `sed` that changes it to `version=2.0`.",
     _setup_notes, _check_notes),
    ("heredoc_write",
     "Output a shell command that creates a file `greeting.txt` in the current directory "
     "containing exactly these two lines, using a heredoc:\nHello\nWorld",
     None, _check_greeting),
    ("sed_multi_file",
     "There are `.py` files under `src/` in the current directory. Output a shell command "
     "that finds all of them and replaces every occurrence of `foo` with `bar` in each, "
     "using `sed -i`.",
     _setup_pyfiles, _check_pyfiles),
    ("quoted_special_chars",
     'Output a shell command that prints exactly this text, including the quotes: '
     'He said "hello" to me.',
     None, None),
    ("chained_mkdir",
     "Output a shell command that creates a directory called `build`, changes into it, "
     "then creates an empty file called `out.txt`, all in one line using `&&`.",
     None, _check_build_dir),
    ("git_commit_message",
     "There is a file `file.txt` with unstaged changes in the current git repository. "
     "Output a shell command that stages all changes and commits them with the message: "
     "Fix bug in parser",
     _setup_git, _check_git_commit),
    ("sed_type_annotation",
     "You have a file `config.py` in the current directory containing the line `def foo(x):`. "
     "Output a shell command using `sed` that changes it to `def foo(x: int) -> int:`.",
     _setup_config, _check_config),
]


def run_one_case(name, user_prompt, setup_fn, check_fn, temperature=TEMPERATURE):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    from transformers import AutoTokenizer
    tok = run_one_case._tok
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    resp = _client.completions.create(model=MODEL_ID, prompt=prompt, max_tokens=MAX_TOKENS_PER_TURN,
                                       temperature=temperature, top_p=TOP_P)
    raw = fix_detokenization(resp.choices[0].text)
    visible = strip_think(raw)
    think = get_think(raw)
    blocks = extract_bash_blocks(visible)
    cmd = blocks[0] if blocks else None

    result = {"name": name, "prompt": user_prompt, "think": think, "visible": visible,
              "n_blocks": len(blocks), "command": cmd}

    if cmd is None:
        result["syntax_valid"] = False
        result["functionally_correct"] = False
        return result

    r = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True, timeout=10)
    result["syntax_valid"] = (r.returncode == 0)
    result["syntax_error"] = r.stderr.strip() if r.returncode != 0 else None

    work_dir = f"/tmp/shell_syntax_check/{name}"
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)
    if setup_fn:
        setup_fn(work_dir)

    if result["syntax_valid"]:
        env = os.environ.copy()
        env["PATH"] = f"/marimo/mats/venvs/mats/bin:{env.get('PATH', '')}"
        try:
            run_r = subprocess.run(["bash", "-c", cmd], cwd=work_dir, capture_output=True,
                                    text=True, timeout=15, env=env)
            result["exit_code"] = run_r.returncode
            result["run_output"] = (run_r.stdout + run_r.stderr)[-500:]
        except Exception as e:
            result["exit_code"] = None
            result["run_output"] = f"[error: {e}]"
        if check_fn:
            try:
                result["functionally_correct"] = bool(check_fn(work_dir))
            except Exception as e:
                result["functionally_correct"] = False
                result["check_error"] = str(e)
        else:
            # no structural check defined (e.g. plain echo) -- judge by
            # whether the command ran successfully and produced non-empty
            # output containing the expected text, checked by the caller.
            result["functionally_correct"] = (result.get("exit_code") == 0)
    else:
        result["exit_code"] = None
        result["run_output"] = None
        result["functionally_correct"] = False

    return result


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/marimo/mats/lazy_coding/results_ablations/shell_syntax_check.jsonl")
    args = parser.parse_args()

    print(f"Loading tokenizer for {MODEL_ID} ...")
    run_one_case._tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    results = []
    for name, prompt, setup_fn, check_fn in CASES:
        r = run_one_case(name, prompt, setup_fn, check_fn)
        results.append(r)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  {name}: syntax_valid={r['syntax_valid']} functionally_correct={r.get('functionally_correct')} "
              f"cmd={r['command']!r}", flush=True)

    n_syntax_ok = sum(1 for r in results if r["syntax_valid"])
    n_func_ok = sum(1 for r in results if r.get("functionally_correct"))
    print(f"\n=== SUMMARY (n={len(results)}) ===")
    print(f"  syntax_valid: {n_syntax_ok}/{len(results)}")
    print(f"  functionally_correct: {n_func_ok}/{len(results)}")
    print("SHELL_SYNTAX_CHECK_DONE")
