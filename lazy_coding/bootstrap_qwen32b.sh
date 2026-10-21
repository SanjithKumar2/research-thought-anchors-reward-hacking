#!/usr/bin/env bash
set -uo pipefail
cd /marimo/mats

echo "=== git identity ==="
git config --local user.name "SanjithKumar2"
git config --local user.email "kumarsanjith967@gmail.com"

echo "=== ensure mats-vllm venv exists (reused as-is if already built) ==="
if [ ! -x venvs/mats-vllm/bin/python ]; then
    uv venv venvs/mats-vllm --python 3.13
    uv pip install --python venvs/mats-vllm/bin/python vllm
fi

echo "=== launch vLLM server (Qwen3-32B, hybrid thinking mode) ==="
mkdir -p lazy_coding/logs
export VLLM_USE_FLASHINFER_SAMPLER=0
setsid nohup venvs/mats-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --dtype bfloat16 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > lazy_coding/logs/vllm_qwen32b.log 2>&1 < /dev/null &
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

echo "BOOTSTRAP_QWEN32B_SCRIPT_DONE"
