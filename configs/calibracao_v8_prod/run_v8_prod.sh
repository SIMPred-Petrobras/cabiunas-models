#!/usr/bin/env bash
# v8_prod_oos: treina no 2025, pontua 2025+2026, avalia OOS no 2026
# MAX_TRIALS=40 | PER_SENSOR_EPOCHS=50 | PATIENCE=12 | TIME_STEPS=96 | TRAIN_END_DATE=2025-12-31
set -e
cd "$(dirname "$0")/../.."
source ../venv/bin/activate
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"

PYTHONPATH=. python src/main.py \
    --config configs/calibracao_v8_prod/v8_prod_oos_per_sensor.json
