#!/usr/bin/env bash
# Phase 3 server -- SGLang (RadixAttention prefix caching for resampling)
# Refuses to start if the vLLM server is already up (both need the full GPU).
set -euo pipefail

if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
    echo "ERROR: vLLM server is running. Kill it first (pkill -f vllm.entrypoints.openai.api_server)." >&2
    exit 1
fi
if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "SGLang server already running."
    exit 0
fi

/marimo/mats/venvs/mats-sglang/bin/python -m sglang.launch_server     --model-path deepseek-ai/DeepSeek-R1-0528-Qwen3-8B     --dtype float16     --port 8001     --context-length 32768
