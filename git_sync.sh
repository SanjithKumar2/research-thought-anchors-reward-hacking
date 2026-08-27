#!/bin/bash
cd /marimo/mats
while true; do
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "auto-sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /dev/null 2>&1
    git push origin main > /dev/null 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) synced"
  fi
  sleep 180
done
