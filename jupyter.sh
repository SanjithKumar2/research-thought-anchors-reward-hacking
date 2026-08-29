#!/usr/bin/env bash
# Convenience wrapper for the MATS project JupyterLab server (env: mats).
# Usage: ./jupyter.sh start | stop | status | url

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mats

LOG="$HOME/.jupyter/jupyterlab.log"
PORT=8888

status() {
    pgrep -f "jupyter-lab" >/dev/null 2>&1 && echo "running (pid $(pgrep -f jupyter-lab | head -1))" || echo "not running"
}

case "${1:-}" in
    start)
        if pgrep -f "jupyter-lab" >/dev/null 2>&1; then
            echo "JupyterLab already running."
        else
            nohup jupyter lab > "$LOG" 2>&1 &
            disown
            sleep 3
            echo "Started. Log: $LOG"
        fi
        grep -m1 "127.0.0.1:$PORT" "$LOG" || true
        ;;
    stop)
        pkill -f "jupyter-lab" && echo "Stopped." || echo "Not running."
        ;;
    status)
        status
        ;;
    url)
        grep "127.0.0.1:$PORT" "$LOG" | tail -1
        ;;
    *)
        echo "Usage: $0 {start|stop|status|url}"
        exit 1
        ;;
esac
