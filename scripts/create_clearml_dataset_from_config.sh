#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 <config.json> [dataset_project] [dataset_name]"
  exit 1
fi

CONFIG_PATH="$1"
DATASET_PROJECT="${2:-TesteMLCab}"
DATASET_NAME="${3:-Cabiunas 2025}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config nao encontrado: $CONFIG_PATH"
  exit 1
fi

readarray -t DATA_PATHS < <(
  python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
base = config_path.parent
cwd = Path.cwd()
cfg = json.loads(config_path.read_text(encoding='utf-8'))

for key in ("FEATURES_CSV", "RAW_CSV", "ALARM_CSV"):
    value = cfg.get(key)
    if not value:
        print("")
        continue
    p = Path(value)
    if p.is_absolute():
        print(str(p))
        continue

    p_from_config = (base / p).resolve()
    p_from_cwd = (cwd / p).resolve()

    if p_from_config.exists():
        print(str(p_from_config))
    else:
        print(str(p_from_cwd))
PY
)

FEATURES_PATH="${DATA_PATHS[0]:-}"
RAW_PATH="${DATA_PATHS[1]:-}"
ALARM_PATH="${DATA_PATHS[2]:-}"

for f in "$FEATURES_PATH" "$RAW_PATH" "$ALARM_PATH"; do
  if [[ -z "$f" || ! -f "$f" ]]; then
    echo "Arquivo de entrada ausente: $f"
    exit 1
  fi
done

create_out="$(clearml-data create --project "$DATASET_PROJECT" --name "$DATASET_NAME")"
echo "$create_out"

dataset_id="$(echo "$create_out" | sed -n 's/.*New dataset created id=\([^ ]*\).*/\1/p' | tail -n 1)"
if [[ -z "$dataset_id" ]]; then
  echo "Nao foi possivel extrair o dataset_id da saida do clearml-data create"
  exit 1
fi

clearml-data add --files "$FEATURES_PATH" "$RAW_PATH" "$ALARM_PATH"
clearml-data close

echo "CLEARML_DATASET_ID=$dataset_id"
echo "Dataset criado e fechado com sucesso."
