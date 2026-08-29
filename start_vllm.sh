#!/usr/bin/env bash
# Phase 1 server -- vLLM (continuous batching for parallel rollout collection)
# Refuses to start if the SGLang server is already up (both need the full GPU).
set -euo pipefail

if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "ERROR: SGLang server is running. Kill it first (pkill -f sglang.launch_server)." >&2
    exit 1
fi
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || pgrep -f "VLLM::EngineCore" >/dev/null 2>&1; then
    echo "vLLM server already running."
    exit 0
fi

# No system CUDA toolkit (nvcc) on this box -- flashinfer JIT-compiled
# top-k/top-p sampler needs nvcc at runtime and crashes without it. Disable
# it and fall back to vLLM PyTorch-native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

/marimo/mats/venvs/mats-vllm/bin/python -m vllm.entrypoints.openai.api_server     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     --dtype float16     --port 8000     --max-model-len 32768     --gpu-memory-utilization 0.90
