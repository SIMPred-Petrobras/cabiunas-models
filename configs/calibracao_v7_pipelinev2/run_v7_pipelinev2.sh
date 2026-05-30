#!/usr/bin/env bash
# Bateria v7 pipelinev2 — idéias do pipeline do amigo (v4-modular):
#   1. ENABLE_GRADIENT_SPIKE_MASK=true  (exclui ±60 min ao redor de spikes de gradiente)
#   2. NORMALIZE_ON_STABLE_ONLY=true    (normaliza sobre estado estável, evita bimodalidade)
#   3. EXCLUDE_STARTUP_MINUTES=240      (exclui 4h pós-partida vs 0 no baseline)
# Arquitetura per_sensor idêntica ao baseline (TIME_STEPS=60, STRIDE=10, F1=4, F2=1).
#
# Rode da raiz do projeto:
#   source ../venv/bin/activate
#   export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
#   bash configs/calibracao_v7_pipelinev2/run_v7_pipelinev2.sh

set -euo pipefail
CONFIGS_DIR="configs/calibracao_v7_pipelinev2"

echo ""
echo "========================================"
echo "Enfileirando: v7_pipelinev2_per_sensor"
echo "========================================"
PYTHONPATH=. python src/main.py --config "${CONFIGS_DIR}/v7_pipelinev2_per_sensor.json"

echo ""
echo "Experimento v7 enfileirado no ClearML."
echo "Acompanhe em: https://app.clear.ml"
