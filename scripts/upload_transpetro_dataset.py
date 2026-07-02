"""
upload_transpetro_dataset.py — Faz upload dos feathers Transpetro como ClearML Dataset.

Uso:
    PYTHONPATH=. python scripts/upload_transpetro_dataset.py \
        --feather-dir /home/dvar/transpetro/PROJETO \
        --project TranspetroML \
        --name "Transpetro 2025"
"""
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_FEATHER_DIR = "/home/dvar/transpetro/PROJETO"
DEFAULT_PROJECT = "TranspetroML"
DEFAULT_DATASET = "Transpetro 2025"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload dos feathers Transpetro como ClearML Dataset."
    )
    parser.add_argument(
        "--feather-dir",
        default=DEFAULT_FEATHER_DIR,
        help=f"Diretório com os .feather (padrão: {DEFAULT_FEATHER_DIR})",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_DATASET)
    args = parser.parse_args()

    from clearml import Dataset

    feather_dir = Path(args.feather_dir).resolve()
    if not feather_dir.is_dir():
        raise NotADirectoryError(f"Diretório não encontrado: {feather_dir}")

    files = sorted(feather_dir.glob("*.feather"))
    if not files:
        raise FileNotFoundError(f"Nenhum .feather encontrado em {feather_dir}")

    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"[UPLOAD] {len(files)} arquivos | {total_mb:.0f} MB | projeto: {args.project}/{args.name}")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    ds = Dataset.create(dataset_name=args.name, dataset_project=args.project)
    ds.add_files(str(feather_dir), wildcard="*.feather")
    print("[UPLOAD] Fazendo upload...")
    ds.upload()
    ds.finalize()
    print(f"[UPLOAD] Concluído. Dataset ID: {ds.id}")


if __name__ == "__main__":
    main()
