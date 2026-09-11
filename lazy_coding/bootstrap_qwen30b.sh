#!/usr/bin/env bash
set -uo pipefail
cd /marimo/mats

echo "=== git identity ==="
git config --local user.name "SanjithKumar2"
git config --local user.email "kumarsanjith967@gmail.com"

echo "=== rebuild mats-vllm venv (mats venv already set up separately) ==="
rm -rf venvs/mats-vllm
uv venv venvs/mats-vllm --python 3.13

echo "=== install deps: mats-vllm venv ==="
uv pip install --python venvs/mats-vllm/bin/python vllm

echo "=== launch vLLM server (Qwen3-30B-A3B-Thinking-2507) ==="
mkdir -p lazy_coding/logs
export VLLM_USE_FLASHINFER_SAMPLER=0
setsid nohup venvs/mats-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --dtype bfloat16 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > lazy_coding/logs/vllm_qwen30b.log 2>&1 < /dev/null &
disown

echo "=== polling for server readiness (up to 30 min -- large model download) ==="
for i in $(seq 1 360); do
    if curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen"; then
        echo "SERVER_READY after $((i*5))s"
        curl -s http://localhost:8000/v1/models
        break
    fi
    sleep 5
done

echo "BOOTSTRAP_QWEN30B_SCRIPT_DONE"
