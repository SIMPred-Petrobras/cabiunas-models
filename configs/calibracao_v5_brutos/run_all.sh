#!/usr/bin/env bash
# Enqueue all v5_brutos experiments (raw dataset, RUNNING_COL, EXCLUDE_CONSTANT_RUNS)
set -euo pipefail

SCRIPT="src/main.py"
CFGS=(
  configs/calibracao_v5_brutos/v5brutos_T5_AVG_A.json
  configs/calibracao_v5_brutos/v5brutos_TC382_01_A.json
  configs/calibracao_v5_brutos/v5brutos_TC382_02_A.json
  configs/calibracao_v5_brutos/v5brutos_TC382_03_A.json
  configs/calibracao_v5_brutos/v5brutos_TC382_04_A.json
  configs/calibracao_v5_brutos/v5brutos_TC382_05_A.json
  configs/calibracao_v5_brutos/v5brutos_TC382_06_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_351X_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_351Y_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_352X_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_352Y_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_353X_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_353Y_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_354X_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_354Y_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_355X_A.json
  configs/calibracao_v5_brutos/v5brutos_TV_355Y_A.json
)

for cfg in "${CFGS[@]}"; do
  echo "Enfileirando: $cfg"
  python3 "$SCRIPT" --config "$cfg"
done

echo "Todos os experimentos enfileirados."
