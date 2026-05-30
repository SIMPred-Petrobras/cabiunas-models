#!/usr/bin/env bash
# Bateria v6 — TIME_STEPS sweep: 96 / 120 / 180 (per_sensor, 17 sensores)
# Cada run enfileira no ClearML e retorna imediatamente (RUN_REMOTE=true).
#
# Rode da raiz do projeto:
#   source ../venv/bin/activate
#   export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
#   bash configs/calibracao_v6_steps/run_v6_steps.sh

set -euo pipefail
CONFIGS_DIR="configs/calibracao_v6_steps"

for cfg in ts096 ts120 ts180; do
  echo ""
  echo "========================================"
  echo "Enfileirando: v6_${cfg}_per_sensor"
  echo "========================================"
  PYTHONPATH=. python src/main.py --config "${CONFIGS_DIR}/v6_${cfg}_per_sensor.json"
done

echo ""
echo "3 experimentos enfileirados no ClearML (ts=96 / 120 / 180)."
echo "Acompanhe em: https://app.clear.ml"
