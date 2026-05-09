#!/usr/bin/env bash
# Bateria v5 — 5 sensores × 3 variantes = 15 experimentos
# Cada run enfileira no ClearML e retorna imediatamente (RUN_REMOTE=true).
# Rode da raiz do projeto:
#   source ../venv/bin/activate
#   export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
#   bash configs/calibracao_v5/run_v5_all.sh

set -euo pipefail
CONFIGS_DIR="configs/calibracao_v5"

sensors=(T5_AVG_A NGP_A NPT_A TC382_03_A TC382_04_A)
variants=(v5a v5b v5c)

for sensor in "${sensors[@]}"; do
  for var in "${variants[@]}"; do
    cfg="${CONFIGS_DIR}/${var}_${sensor}.json"
    echo ""
    echo "========================================"
    echo "Enfileirando: ${var}_${sensor}"
    echo "========================================"
    PYTHONPATH=. python src/main.py --config "${cfg}"
  done
done

echo ""
echo "Todos os 15 experimentos enfileirados no ClearML."
