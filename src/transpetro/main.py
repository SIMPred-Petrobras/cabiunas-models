"""
main.py — Entrypoint da pipeline CNN1D-AE para dados Transpetro (feather).

Uso:
    PYTHONPATH=. python src/transpetro/main.py --config configs/transpetro/B-8802B.json

Fluxo remoto (RUN_REMOTE: true):
  1. Task.init() cria a task no ClearML e a enfileira.
  2. task.execute_remotely() encerra o processo local — o worker clona o repo do Git e
     executa a partir do mesmo ponto, desta vez com running_locally()=False.
  3. No worker, USE_CLEARML_DATASET=true faz com que io.py baixe o feather do Dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import pickle
from pathlib import Path
from typing import Dict

from src.cnn1d_ae.config import PipelineConfig, cfg_to_dict, update_cfg_from_dict
from src.cnn1d_ae.pipeline import run
from src.transpetro.io import load_data_transpetro


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline CNN1D-AE para dados Transpetro (feather)."
    )
    # Opcional: no worker remoto o ClearML não repassa --config;
    # a config é recuperada via task.connect() abaixo.
    p.add_argument("--config", default=None, help="Caminho para o JSON de configuração.")
    return p.parse_args()


def _save_artifact_local(local_dir: Path, artifact_name: str, artifact_object) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^0-9a-zA-Z_.-]+", "_", artifact_name)

    if isinstance(artifact_object, (str, Path)):
        src = Path(artifact_object)
        if src.exists() and src.is_file():
            target = local_dir / f"{safe_name}{src.suffix}"
            shutil.copy2(src, target)
            return target

    target = local_dir / f"{safe_name}.pkl"
    with target.open("wb") as f:
        pickle.dump(artifact_object, f)
    return target


def _upload_run_artifacts(task, run_info: Dict, local_root: Path) -> None:
    _pub = lambda name, obj: _save_artifact_local(local_root, name, obj)
    _pub("summary_all_sensors_csv", run_info.get("summary_path", ""))
    _pub("time_integrity_report_json", run_info.get("time_report_path", ""))

    if task is None:
        return

    for out in run_info.get("sensor_outputs", []):
        sensor = out.get("sensor") or out.get("group", "unknown")
        output_dir = out.get("output_dir")
        if not output_dir:
            continue
        csv_dir = os.path.join(output_dir, "csv")
        if os.path.isdir(csv_dir):
            for name in sorted(os.listdir(csv_dir)):
                fpath = os.path.join(csv_dir, name)
                if os.path.isfile(fpath):
                    try:
                        task.upload_artifact(f"{sensor}/csv/{name}", fpath)
                    except Exception as exc:
                        print(f"[WARN] upload artifact '{sensor}/csv/{name}': {exc}")
        hp_path = os.path.join(output_dir, "best_model", "best_hyperparameters.json")
        if os.path.isfile(hp_path):
            try:
                task.upload_artifact(f"{sensor}/best_hyperparameters_json", hp_path)
            except Exception as exc:
                print(f"[WARN] upload hyperparameters '{sensor}': {exc}")


def main():
    args = parse_args()

    # --- Carrega config do arquivo (execução local) ---
    cfg = PipelineConfig()
    if args.config:
        path = Path(args.config)
        if not path.exists():
            raise FileNotFoundError(f"Config não encontrado: {path}")
        cfg = update_cfg_from_dict(cfg, json.loads(path.read_text(encoding="utf-8")))
        stem = path.stem
    else:
        stem = cfg.EQUIPMENT_ID or "transpetro"

    task = None
    try:
        from clearml import Task
        # pyarrow é usado por pandas.read_feather mas nunca importado diretamente,
        # então o auto-detector de requirements do ClearML não o captura sozinho.
        Task.add_requirements("pyarrow")
        task_name = f"transpetro::{cfg.EQUIPMENT_ID or stem}"
        task = Task.init(
            project_name=cfg.CLEARML_PROJECT_NAME or "TranspetroML",
            task_name=task_name,
            output_uri=True,
            reuse_last_task_id=False,
        )
        task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)

        # connect() armazena o config localmente e devolve os valores do servidor
        # quando executando no worker (remote) — isso recupera todos os params.
        params = task.connect(cfg_to_dict(cfg), name="pipeline_config")
        cfg = update_cfg_from_dict(cfg, params)

        print(f"[CLEARML] Task: {task_name} | ID: {task.id}")

        if cfg.RUN_REMOTE and task.running_locally():
            print(f"[CLEARML] Enfileirando remotamente na fila '{cfg.REMOTE_QUEUE}'...")
            task.execute_remotely(queue_name=cfg.REMOTE_QUEUE, exit_process=True)
            # O processo local encerra aqui; o worker continua a partir daqui.

    except Exception as e:
        print(f"[CLEARML] Não disponível ou erro na inicialização: {e}")
        task = None

    local_artifact_root = Path("artifacts_local") / (cfg.EQUIPMENT_ID or stem)

    try:
        data = load_data_transpetro(cfg)
        run_info = run(cfg, data=data)
        _upload_run_artifacts(task, run_info, local_artifact_root)
        print(f"\n[DONE] Equipamento: {cfg.EQUIPMENT_ID or path.stem}")
        if task:
            task.mark_completed(status_message="Pipeline transpetro concluída.")
    except Exception as exc:
        if task:
            task.mark_failed(status_reason=str(exc))
        raise
    finally:
        if task:
            task.close()


if __name__ == "__main__":
    main()
