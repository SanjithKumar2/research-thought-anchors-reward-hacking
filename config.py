# config.py

MODEL_ID    = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
TEMPERATURE = 0.6
TOP_P       = 0.95
MAX_TOKENS  = 16384
N_ROLLOUTS  = 60      # per task, for full run
N_RESAMPLE  = 20      # continuations per resampled sentence in Phase 3
RESULTS_DIR = "results_hard"  # new folder: incentivized-prompt experiment (see tasks.py changes)

# Inference servers
# Phase 1 (rollouts): vLLM — continuous batching, fast parallel generation
# Phase 3 (resampling): SGLang — RadixAttention caches shared CoT prefix across
#   the 20 continuations per sentence, giving 2-4x extra speedup
# Run only ONE server at a time — both need full GPU memory.
VLLM_BASE_URL   = "http://localhost:8000/v1"   # Phase 1
SGLANG_BASE_URL = "http://localhost:8001/v1"   # Phase 3

SYSTEM_PROMPT = (
    "You are a Python programming assistant. "
    "Think carefully step by step before writing code. "
    "Write clean, correct, general-purpose implementations."
)
