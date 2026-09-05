# lazy_coding/lazy_agent.py
# Minimal ReAct-style bash-tool agent loop for the lazy-coding replication.
# Up to MAX_COMMANDS_PER_TURN shell commands per model turn, executed for
# real in an isolated repo copy under /tmp, in order, stopping at the first
# failure; combined observation fed back; loop until DONE or MAX_TURNS.

import os
import re
import subprocess
from pathlib import Path

from openai import OpenAI

from lazy_config import (
    MODEL_ID, TEMPERATURE, TOP_P, MAX_TOKENS_PER_TURN, MAX_TURNS,
    MAX_COMMANDS_PER_TURN, VLLM_BASE_URL, SYSTEM_PROMPT, TASK_PROMPT,
)
from lazy_repo_gen import materialize_repo, materialize_repo_scaled, materialize_repo_tiny
from lazy_classify import classify_rollout

VARIANTS = {
    "full": (materialize_repo, "lazy_r"),
    "scaled": (materialize_repo_scaled, "lazy_scaled_r"),
    "tiny": (materialize_repo_tiny, "lazy_tiny_r"),
}

_client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=1800.0, max_retries=3)

BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)


def fix_detokenization(text: str) -> str:
    """DeepSeek-R1-0528-Qwen3-8B's HF repo declares tokenizer_class=
    LlamaTokenizerFast, but tokenizer.json's actual pretokenizer/decoder is
    GPT2-style byte-level BPE -- a mismatch baked into the model repo itself
    (confirmed: AutoTokenizer resolves to LlamaTokenizerFast and drops
    spaces entirely on decode; vLLM's own detokenizer leaks the raw
    byte-level markers instead). Verified fix: the standard GPT2 byte-level
    BPE markers for space/newline/tab decode correctly with a plain
    substitution -- confirmed against known-good text on this box."""
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def extract_bash_blocks(text: str):
    """Returns the list of raw (stripped) fenced ```bash block contents, in
    the order they appear."""
    return [m.strip() for m in BASH_BLOCK_RE.findall(text)]


def resolve_turn_command(text: str):
    """Decide what this turn actually does, from the raw ```bash blocks in
    the model's VISIBLE answer (caller must pass the post-<think> portion --
    see the Session 9 note below on why). Returns one of:
      - None            -- no bash block at all
      - "DONE"          -- the turn should end the rollout
      - "__TOO_MANY__"  -- more than MAX_COMMANDS_PER_TURN real command
                           blocks; ambiguous, so nothing is executed
      - [cmd, ...]      -- an ordered list of 1..MAX_COMMANDS_PER_TURN real
                           commands to run in sequence

    Session 8 background (small model, R1-0528-Qwen3-8B): a real command
    block followed by a separate bare ```bash\\nDONE\\n``` block in the same
    reply -- a premature completion claim tacked on after drafting the
    actual command(s), not a deliberate decision the task is finished. Safe
    to resolve automatically: the bare DONE is just filtered out of the
    returned list, and DONE gets declared honestly later once its result is
    visible.

    Session 9 fix (important): the caller must pass the VISIBLE answer only,
    not the full raw completion. Scanning the full raw text (Session 6-8
    behavior) picked up illustrative/candidate bash blocks the model drafts
    while reasoning inside <think> -- e.g. trying a few regex variants before
    settling on one -- and wrongly counted those as additional real commands,
    triggering false-positive multi-command rejections on turns whose actual
    visible answer was a single clean command the whole time. Confirmed on a
    real rollout: 6/7 non-DONE turns each had exactly one visible block but
    were rejected solely due to extra blocks embedded in <think>.

    Session 9 also raises the cap from 1 to MAX_COMMANDS_PER_TURN: the
    original strict single-command rule turned out to be enforcing an
    artifact of the old think-scanning bug more often than a genuine
    multi-step-planning problem, and unnecessarily forced obviously-
    sequential short plans (edit, verify, commit) across turns with no
    benefit. `__TOO_MANY__` still exists for the genuinely-unbounded case
    (e.g. a 14B-class model drafting 5+ real steps in one reply), where
    picking a subset would still be guessing at intent.
    """
    blocks = extract_bash_blocks(text)
    if not blocks:
        return None
    non_done = [b for b in blocks if b != "DONE"]
    if not non_done:
        return "DONE"
    if len(non_done) > MAX_COMMANDS_PER_TURN:
        return "__TOO_MANY__"
    return non_done

def strip_think(text: str) -> str:
    """What goes back into conversation history for later turns: the
    <think> block is dropped (standard practice for R1-family multi-turn
    use), keeping only the model's visible action/answer."""
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


def get_think(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[-1].split("</think>", 1)[0].strip()
    if "</think>" in text:
        return text.split("</think>", 1)[0].strip()
    return ""


def run_bash(cmd: str, cwd: str, timeout: int = 30):
    """Non-login shell (`-c`, not `-l`) is deliberate: a login shell sources
    profile scripts that silently reset PATH on this box, wiping out the
    venv containing mypy regardless of what's passed via env= -- confirmed
    directly (bash -lc drops back to a bare system PATH; bash -c preserves
    the injected one). That bug made mypy invisible to every command the
    agent ran, including its own self-authored hook, for the entire first
    two batches -- do not revert this without re-verifying that fix."""
    env = os.environ.copy()
    env["PATH"] = f"/marimo/mats/venvs/mats/bin:{env.get('PATH', '')}"
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        return out[-4000:], r.returncode
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]", -1
    except Exception as e:
        return f"[error running command: {e}]", -1


def run_commands_sequence(cmds: list[str], cwd: str):
    """Run a list of commands in order, stopping at the first nonzero exit
    (so the model sees exactly where a multi-step plan broke, rather than
    every subsequent command failing against a bad precondition). Returns
    (combined_observation_text, per_command_results, last_exit_code)."""
    results = []
    for i, cmd in enumerate(cmds):
        out, rc = run_bash(cmd, cwd=cwd)
        results.append({"command": cmd, "exit_code": rc, "output": out})
        if rc != 0:
            break

    parts = []
    for i, r in enumerate(results):
        label = f"[command {i + 1}/{len(cmds)}]" if len(cmds) > 1 else ""
        parts.append(f"{label} $ {r['command']}\n(exit code {r['exit_code']})\n{r['output']}".strip())
    if len(results) < len(cmds):
        skipped = len(cmds) - len(results)
        parts.append(f"[{skipped} remaining command(s) in this turn were skipped because the one above failed]")

    return "\n\n".join(parts), results, results[-1]["exit_code"]


def run_agent_rollout(tok, rollout_idx: int, max_turns: int = MAX_TURNS, temperature: float = TEMPERATURE,
                       variant: str = "full") -> dict:
    materializer, prefix = VARIANTS[variant]
    rollout_id = f"{prefix}{rollout_idx:03d}"
    repo_dir = Path(f"/tmp/lazy_coding_rollouts/{rollout_id}/widgetkit-repo")
    materializer(str(repo_dir))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TASK_PROMPT},
    ]
    transcript = []
    done = False
    n_turns_used = 0
    error = None

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
            obs = ("[No ```bash block found in your last reply. Put one to "
                   f"{MAX_COMMANDS_PER_TURN} shell commands, each in its own ```bash block, "
                   "or ```bash\nDONE\n``` if the task is finished.]")
            turn_record["observation"] = obs
            transcript.append(turn_record)
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            continue

        if cmd == "__TOO_MANY__":
            n_blocks = len([b for b in extract_bash_blocks(visible) if b != "DONE"])
            obs = (f"[Your last reply contained {n_blocks} separate ```bash command blocks, "
                   f"more than the {MAX_COMMANDS_PER_TURN}-command limit per turn. NONE of them were "
                   "executed this turn -- nothing you wrote actually ran, so do not assume any of it "
                   f"happened. Put at most {MAX_COMMANDS_PER_TURN} commands (each in its own single "
                   "```bash block, in the order you want them to run) this turn, or ```bash\nDONE\n``` "
                   "alone if the task is already finished.]")
            turn_record["observation"] = obs
            transcript.append(turn_record)
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            continue

        if cmd == "DONE":
            turn_record["observation"] = None
            transcript.append(turn_record)
            done = True
            break

        obs, per_command_results, last_exit_code = run_commands_sequence(cmd, cwd=str(repo_dir))
        turn_record["observation"] = obs
        turn_record["exit_code"] = last_exit_code
        turn_record["per_command_results"] = per_command_results
        transcript.append(turn_record)
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    classification = classify_rollout(str(repo_dir), variant)

    return {
        "rollout_id": rollout_id,
        "variant": variant,
        "n_turns_used": n_turns_used,
        "hit_turn_limit": (n_turns_used >= max_turns and not done),
        "declared_done": done,
        "error": error,
        "transcript": transcript,
        "temperature": temperature,
        **classification,
    }
