#!/usr/bin/env bash
set -uo pipefail
cd /marimo/mats

echo "=== git identity ==="
git config --local user.name "SanjithKumar2"
git config --local user.email "kumarsanjith967@gmail.com"
git config --local --get user.name
git config --local --get user.email

echo "=== rebuild venvs ==="
rm -rf venvs/mats venvs/mats-vllm
uv venv venvs/mats --python 3.13
uv venv venvs/mats-vllm --python 3.13

echo "=== install deps: mats venv ==="
uv pip install --python venvs/mats/bin/python openai tqdm transformers jinja2 mypy

echo "=== install deps: mats-vllm venv ==="
uv pip install --python venvs/mats-vllm/bin/python vllm

echo "=== launch git_sync ==="
setsid nohup bash git_sync.sh > git_sync.log 2>&1 < /dev/null &
disown

echo "=== launch vLLM server (DeepSeek-R1-0528-Qwen3-8B) ==="
mkdir -p lazy_coding/logs
export VLLM_USE_FLASHINFER_SAMPLER=0
setsid nohup venvs/mats-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --dtype float16 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > lazy_coding/logs/vllm.log 2>&1 < /dev/null &
disown

echo "=== polling for server readiness (up to 10 min) ==="
for i in $(seq 1 120); do
    if curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q "DeepSeek"; then
        echo "SERVER_READY after $((i*5))s"
        curl -s http://localhost:8000/v1/models
        break
    fi
    sleep 5
done

echo "BOOTSTRAP_SCRIPT_DONE"
