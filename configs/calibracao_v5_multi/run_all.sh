#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python3 src/main.py --config configs/calibracao_v5_multi/v5multi_todos_sensores.json
