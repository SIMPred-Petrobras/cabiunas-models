"""
Faz upload do dataset bruto 2025 (30s) para o ClearML.

Arquivos enviados:
  - sensores_brutos_2025_30s.csv   (RAW_CSV)
  - alarmes_selecionados_turbina_a.csv  (ALARM_CSV)

Após o upload, imprime o dataset ID para atualizar os configs em
configs/calibracao_v5_brutos/*.json  (CLEARML_DATASET_ID).

Uso:
    PYTHONPATH=. python scripts/upload_brutos_dataset.py
    PYTHONPATH=. python scripts/upload_brutos_dataset.py --project TesteMLCab --name "Cabiunas 2025 Brutos"
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from clearml import Dataset

DEFAULT_PROJECT = "TesteMLCab"
DEFAULT_DATASET = "Cabiunas 2025 Brutos"

RAW_CSV   = Path("../dados/sensores_brutos_2025_30s.csv")
ALARM_CSV = Path("../dados/alarmes_selecionados_turbina_a.csv")

CONFIGS_DIR = Path("configs/calibracao_v5_brutos")


def upload(project: str, name: str) -> str:
    files = [RAW_CSV, ALARM_CSV]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos não encontrados: {missing}\n"
            "Execute scripts/convert_xlsx_to_csv.py primeiro."
        )

    ds = Dataset.create(dataset_name=name, dataset_project=project)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for src in files:
            dst = tmp_path / src.name
            shutil.copy2(src, dst)
            size_mb = src.stat().st_size / 1e6
            print(f"  Adicionando {src.name} ({size_mb:.1f} MB) ...")

        ds.add_files(str(tmp_path))
        ds.upload()
        ds.finalize()

    dataset_id = ds.id
    print(f"\nDataset criado: {project}/{name}")
    print(f"Dataset ID: {dataset_id}")
    return dataset_id


def update_configs(dataset_id: str) -> None:
    if not CONFIGS_DIR.is_dir():
        print(f"Diretório {CONFIGS_DIR} não encontrado — configs não atualizados.")
        return

    updated = []
    for cfg_path in sorted(CONFIGS_DIR.glob("v5brutos_*.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["CLEARML_DATASET_ID"] = dataset_id
        cfg["USE_CLEARML_DATASET"] = True
        cfg["RUN_REMOTE"] = True
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        updated.append(cfg_path.name)

    if updated:
        print(f"\n{len(updated)} configs atualizados com dataset_id e RUN_REMOTE=true:")
        for name in updated:
            print(f"  {name}")
    else:
        print("Nenhum config v5brutos_*.json encontrado para atualizar.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_DATASET)
    parser.add_argument(
        "--no-update-configs",
        action="store_true",
        help="Não atualiza automaticamente os configs após o upload.",
    )
    args = parser.parse_args()

    dataset_id = upload(args.project, args.name)

    if not args.no_update_configs:
        update_configs(dataset_id)
        print("\nPróximo passo: commitar os configs atualizados e fazer push antes de rodar remotamente.")
    else:
        print(f"\nAtualize CLEARML_DATASET_ID nos configs manualmente: {dataset_id}")


if __name__ == "__main__":
    main()
