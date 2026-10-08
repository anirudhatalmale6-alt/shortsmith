#!/usr/bin/env bash
# Start Shortsmith (web + background worker in one process).
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
exec .venv/bin/uvicorn shortsmith.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
