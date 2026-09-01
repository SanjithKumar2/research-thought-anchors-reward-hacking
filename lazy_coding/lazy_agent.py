# lazy_coding/lazy_agent.py
# Minimal ReAct-style bash-tool agent loop for the lazy-coding replication.
# One shell command per model turn, executed for real in an isolated repo
# copy under /tmp; observation fed back; loop until DONE or MAX_TURNS.

import os
import re
import subprocess
from pathlib import Path

from openai import OpenAI

from lazy_config import (
    MODEL_ID, TEMPERATURE, TOP_P, MAX_TOKENS_PER_TURN, MAX_TURNS,
    VLLM_BASE_URL, SYSTEM_PROMPT, TASK_PROMPT,
)
from lazy_repo_gen import materialize_repo
from lazy_classify import classify_rollout

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


def extract_last_bash_command(text: str):
    """Returns the command to run this turn.

    Defensive rule (observed failure mode in testing): if the model emits
    several ```bash blocks in one reply and the LAST one is bare `DONE`
    while an EARLIER block contains a real command, that's the model
    tacking on a premature completion claim after drafting the actual
    command it meant to run -- not a deliberate decision that the task is
    finished. In that case we run the last real (non-DONE) block instead
    and let DONE be declared honestly on a later turn once its result is
    visible. DONE only takes effect when it is the sole block, or the
    last block among several that are ALL bare DONE.
    """
    matches = [m.strip() for m in BASH_BLOCK_RE.findall(text)]
    if not matches:
        return None
    non_done = [m for m in matches if m != "DONE"]
    if matches[-1] == "DONE" and non_done:
        return non_done[-1]
    return matches[-1]


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
    env = os.environ.copy()
    env["PATH"] = f"/marimo/mats/venvs/mats/bin:{env.get('PATH', '')}"
    try:
        r = subprocess.run(["bash", "-lc", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        return out[-4000:], r.returncode
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]", -1
    except Exception as e:
        return f"[error running command: {e}]", -1


def run_agent_rollout(tok, rollout_idx: int, max_turns: int = MAX_TURNS, temperature: float = TEMPERATURE) -> dict:
    rollout_id = f"lazy_r{rollout_idx:03d}"
    repo_dir = Path(f"/tmp/lazy_coding_rollouts/{rollout_id}/widgetkit-repo")
    materialize_repo(str(repo_dir))

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
        cmd = extract_last_bash_command(raw)

        turn_record = {"turn": turn, "think": think, "visible": visible, "command": cmd}

        messages.append({"role": "assistant", "content": visible if visible else "(no output)"})

        if cmd is None:
            obs = ("[No ```bash block found in your last reply. Put exactly one shell "
                   "command in a ```bash block, or ```bash\\nDONE\\n``` if the task is finished.]")
            turn_record["observation"] = obs
            transcript.append(turn_record)
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            continue

        if cmd.strip() == "DONE":
            turn_record["observation"] = None
            transcript.append(turn_record)
            done = True
            break

        out, rc = run_bash(cmd, cwd=str(repo_dir))
        obs = f"(exit code {rc})\n{out}"
        turn_record["observation"] = obs
        turn_record["exit_code"] = rc
        transcript.append(turn_record)
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    classification = classify_rollout(str(repo_dir))

    return {
        "rollout_id": rollout_id,
        "n_turns_used": n_turns_used,
        "hit_turn_limit": (n_turns_used >= max_turns and not done),
        "declared_done": done,
        "error": error,
        "transcript": transcript,
        "temperature": temperature,
        **classification,
    }
