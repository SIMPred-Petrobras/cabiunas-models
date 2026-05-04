from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from clearml import Dataset


DEFAULT_PROJECT = "TesteMLCab"
DEFAULT_DATASET = "Cabiunas 2025"


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    candidates = [
        (Path.cwd() / path).resolve(),
        (config_path.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _input_files_from_config(config_path: Path) -> list[Path]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    files = [
        _resolve_path(config_path, cfg["FEATURES_CSV"]),
        _resolve_path(config_path, cfg["RAW_CSV"]),
        _resolve_path(config_path, cfg["ALARM_CSV"]),
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Arquivos de entrada ausentes: {missing}")
    return files


def upload_dataset(config_path: Path, project: str, dataset_name: str) -> str:
    files = _input_files_from_config(config_path)

    ds = Dataset.create(dataset_name=dataset_name, dataset_project=project)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for src in files:
            dst = tmp_path / src.name
            shutil.copy2(src, dst)
            print(f"Adding {src} -> {dst.name} ({src.stat().st_size / 1e6:.1f} MB)")

        ds.add_files(str(tmp_path))
        ds.upload()
        ds.finalize()

    print(f"Dataset criado: {project}/{dataset_name} (ID: {ds.id})")
    return ds.id


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Cabiunas input CSVs as a ClearML Dataset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_DATASET)
    args = parser.parse_args()

    upload_dataset(args.config.resolve(), args.project, args.name)


if __name__ == "__main__":
    main()
