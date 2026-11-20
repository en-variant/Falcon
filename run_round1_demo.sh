#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m audit.count_tokens --offline
.venv/bin/python -m streamlit run audit/app.py --server.address 127.0.0.1 --server.port 8501
