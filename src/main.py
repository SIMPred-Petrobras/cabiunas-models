from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List

from clearml import Task
from src.cnn1d_ae.config import PipelineConfig, cfg_to_dict, update_cfg_from_dict
from src.cnn1d_ae.pipeline import run
from src.cnn1d_ae.pipeline_multi import run_pipeline_multivariado
from src.cnn1d_ae.io import load_data
from src.cnn1d_ae.pipeline import discover_sensors


def parse_args():
    p = argparse.ArgumentParser(description="Run CNN1D-AE pipeline (local or operacional).")
    p.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
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


def _publish_artifact(
    task: Task,
    artifact_name: str,
    artifact_object,
    local_dir: Path,
    upload_to_clearml: bool = True,
) -> Path | None:
    local_path = None
    try:
        local_path = _save_artifact_local(local_dir, artifact_name, artifact_object)
    except Exception as exc:
        print(f"[WARN] Falha ao salvar artifact local '{artifact_name}': {exc}")

    if not upload_to_clearml:
        return local_path

    try:
        task.upload_artifact(name=artifact_name, artifact_object=artifact_object)
    except Exception as exc:
        print(f"[WARN] Falha no upload ClearML do artifact '{artifact_name}': {exc}")
        if local_path is not None:
            print(f"[INFO] Artifact '{artifact_name}' preservado localmente em: {local_path}")
    return local_path


def _upload_run_artifacts(task: Task, run_info: Dict) -> None:
    local_task_dir = Path("artifacts_local") / task.id
    summary_path = run_info.get("summary_path", "")
    if summary_path:
        _publish_artifact(task, "summary_all_sensors_csv", summary_path, local_task_dir)
    time_report_path = run_info.get("time_report_path", "")
    if time_report_path:
        _publish_artifact(task, "time_integrity_report_json", time_report_path, local_task_dir)

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
                    artifact_name = f"{sensor}_csv_{name}"
                    _publish_artifact(task, artifact_name, file_path, local_task_dir / sensor)

        figs_dir = os.path.join(output_dir, "figs")
        if os.path.isdir(figs_dir):
            logger = task.get_logger()
            for name in sorted(os.listdir(figs_dir)):
                file_path = os.path.join(figs_dir, name)
                if os.path.isfile(file_path) and name.lower().endswith(".png"):
                    artifact_name = f"{sensor}_figs_{name}"
                    _publish_artifact(task, artifact_name, file_path, local_task_dir / sensor)
                    try:
                        stem = os.path.splitext(name)[0]
                        logger.report_image(
                            title=f"{sensor}/{stem}",
                            series=sensor,
                            local_path=file_path,
                        )
                    except Exception as exc:
                        print(f"[WARN] report_image falhou para '{name}': {exc}")

        hp_path = os.path.join(output_dir, "best_model", "best_hyperparameters.json")
        _publish_artifact(task, f"{sensor}_best_hyperparameters_json", hp_path, local_task_dir / sensor)

        bundle_path = os.path.join(output_dir, "best_model", "inference_bundle.json")
        if os.path.isfile(bundle_path):
            _publish_artifact(task, f"{sensor}_inference_bundle_json", bundle_path, local_task_dir / sensor)


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

    project_name = cfg.CLEARML_PROJECT_NAME
    # '::' gera path duplo-encoded (%253A) e quebra o download de artefatos no fileserver.
    task_name = f"cnn1d-ae-{path.stem}"
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
        if cfg.MULTIVARIATE_JOINT:
            df_alarm, df_feat, df_raw, _ = load_data(cfg)
            sensors = discover_sensors(cfg, df_feat, df_raw)
            print(f"[MULTI] Sensores para treino conjunto: {sensors}")
            run_info = run_pipeline_multivariado(cfg, df_alarm, df_feat, df_raw, sensors)
        else:
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

