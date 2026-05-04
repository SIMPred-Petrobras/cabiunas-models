#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

for sensor in NGP_A NPT_A T5_AVG_A TC382_03_A TC382_04_A; do
  cfg="configs/calibracao_v3_on/record2025_tzm3_interpolado_v3_${sensor}.json"
  out="runs_record2025_tzm3_interpolado_v3_${sensor}"
  mkdir -p "$out"
  echo "[RUN] $sensor"

  if [[ "${CREATE_CLEARML_DATASET:-0}" == "1" ]]; then
    ds_log="$out/clearml_dataset.log"
    DATASET_NAME="Cabiunas 2025"
    ./scripts/create_clearml_dataset_from_config.sh "$cfg" "TesteMLCab" "$DATASET_NAME" > "$ds_log" 2>&1
    export CLEARML_DATASET_ID
    CLEARML_DATASET_ID="$(sed -n 's/^CLEARML_DATASET_ID=//p' "$ds_log" | tail -n 1)"
    echo "[DATASET] CLEARML_DATASET_ID=$CLEARML_DATASET_ID"
  else
    unset CLEARML_DATASET_ID || true
  fi

  PYTHONPATH=. ../venv/bin/python src/main.py --config "$cfg" > "$out/run.log" 2>&1
  echo "[DONE] $sensor -> $out"
done
