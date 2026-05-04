from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from clearml import Task
from src.cnn1d_ae.config import PipelineConfig, cfg_to_dict, update_cfg_from_dict
from src.cnn1d_ae.pipeline import run


def parse_args():
    p = argparse.ArgumentParser(description="Run CNN1D-AE pipeline (local or operacional).")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    return p.parse_args()


def _safe_upload_file(task: Task, artifact_name: str, file_path: str) -> None:
    if os.path.isfile(file_path):
        try:
            task.upload_artifact(name=artifact_name, artifact_object=file_path)
        except Exception as exc:
            print(f"[WARN] Falha no upload ClearML do artifact '{artifact_name}': {exc}")


def _upload_run_artifacts(task: Task, run_info: Dict) -> None:
    _safe_upload_file(task, "summary_all_sensors_csv", run_info.get("summary_path", ""))
    _safe_upload_file(task, "time_integrity_report_json", run_info.get("time_report_path", ""))

    sensor_outputs: List[Dict] = run_info.get("sensor_outputs", [])
    for out in sensor_outputs:
        sensor = out.get("sensor", "unknown_sensor")
        output_dir = out.get("output_dir")
        if not output_dir:
            continue

        csv_dir = os.path.join(output_dir, "csv")
        if os.path.isdir(csv_dir):
            for name in sorted(os.listdir(csv_dir)):
                file_path = os.path.join(csv_dir, name)
                if os.path.isfile(file_path):
                    artifact_name = f"{sensor}/csv/{name}"
                    _safe_upload_file(task, artifact_name, file_path)

        hp_path = os.path.join(output_dir, "best_model", "best_hyperparameters.json")
        _safe_upload_file(task, f"{sensor}/best_hyperparameters_json", hp_path)


def main():
    args = parse_args()
    cfg = PipelineConfig()

    path = Path(args.config)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = update_cfg_from_dict(cfg, d)

    dataset_id = os.getenv("CLEARML_DATASET_ID", "").strip() or cfg.CLEARML_DATASET_ID
    if dataset_id:
        cfg.CLEARML_DATASET_ID = dataset_id
        d["CLEARML_DATASET_ID"] = dataset_id

    Task.add_requirements("clearml")
    Task.add_requirements("numpy")
    Task.add_requirements("pandas")
    Task.add_requirements("matplotlib")
    Task.add_requirements("tensorflow")
    Task.add_requirements("keras-tuner")

    project_name = cfg.CLEARML_PROJECT_NAME
    task_name = f"cnn1d-ae::{path.stem}"
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        output_uri=True,
        reuse_last_task_id=False,
    )
    task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
    task.connect(cfg_to_dict(cfg), name="pipeline_config")

    if dataset_id:
        task.set_parameter("clearml/dataset_id", dataset_id)
        task.get_logger().report_text(f"Using ClearML Dataset ID: {dataset_id}")

    if cfg.RUN_REMOTE and task.running_locally():
        task.get_logger().report_text(
            f"Enqueuing task for remote execution on queue: {cfg.REMOTE_QUEUE}"
        )
        task.execute_remotely(queue_name=cfg.REMOTE_QUEUE, exit_process=True)

    try:
        run_info = run(cfg)
        _upload_run_artifacts(task, run_info)
        task.mark_completed(status_message="Pipeline concluida com sucesso.")
    except Exception as exc:
        task.mark_failed(status_reason=str(exc))
        raise
    finally:
        task.close()


if __name__ == "__main__":
    main()

