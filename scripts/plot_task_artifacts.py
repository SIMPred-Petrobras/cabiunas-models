from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task


def _copy_artifact(task: Task, artifact_name: str, out_dir: Path) -> Path:
    if artifact_name not in task.artifacts:
        raise KeyError(f"Artifact nao encontrado na task: {artifact_name}")

    local_path = Path(task.artifacts[artifact_name].get_local_copy())
    target = out_dir / Path(urlparse(task.artifacts[artifact_name].url).path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if local_path.resolve() != target.resolve():
        shutil.copy2(local_path, target)
    return target


def _pipeline_params(task: Task) -> dict:
    params = task.get_parameters(cast=True)
    prefix = "pipeline_config/"
    return {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}


def _resolve_dataset_file(dataset_root: Path, configured_path: str) -> Path:
    raw_path = Path(configured_path)
    candidates = [dataset_root / raw_path.name]
    if not raw_path.is_absolute():
        candidates.append(dataset_root / raw_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = [p for p in dataset_root.rglob(raw_path.name) if p.is_file()]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Arquivo {configured_path!r} nao encontrado em {dataset_root}")


def _load_raw_series(cfg: dict, sensor: str) -> pd.Series:
    dataset_id = cfg.get("CLEARML_DATASET_ID")
    dataset_name = cfg.get("CLEARML_DATASET_NAME", "Cabiunas 2025")
    dataset_project = cfg.get("CLEARML_PROJECT_NAME", "TesteMLCab")

    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
    else:
        dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project)

    dataset_root = Path(dataset.get_local_copy())
    raw_path = _resolve_dataset_file(dataset_root, str(cfg["RAW_CSV"]))
    df_raw = pd.read_csv(raw_path, usecols=[str(cfg["TIME_COL"]), sensor])
    idx = pd.to_datetime(df_raw[str(cfg["TIME_COL"])], errors="coerce")
    series = pd.Series(pd.to_numeric(df_raw[sensor], errors="coerce").values, index=idx, name=sensor)
    series = series[~series.index.isna()].sort_index()
    return series[~series.index.duplicated(keep="last")]


def _load_alarm_times(cfg: dict, sensor: str) -> pd.Series:
    dataset_id = cfg.get("CLEARML_DATASET_ID")
    dataset_name = cfg.get("CLEARML_DATASET_NAME", "Cabiunas 2025")
    dataset_project = cfg.get("CLEARML_PROJECT_NAME", "TesteMLCab")

    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
    else:
        dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project)

    dataset_root = Path(dataset.get_local_copy())
    alarm_path = _resolve_dataset_file(dataset_root, str(cfg["ALARM_CSV"]))
    df_alarm = pd.read_csv(alarm_path)
    if "Data da Ocorrência" in df_alarm.columns and "Data da Ocorrencia" not in df_alarm.columns:
        df_alarm["Data da Ocorrencia"] = df_alarm["Data da Ocorrência"]
    if "Tag" in df_alarm.columns:
        df_alarm = df_alarm[df_alarm["Tag"] == sensor]
    return pd.to_datetime(df_alarm["Data da Ocorrencia"], errors="coerce").dropna().sort_values()


def _plot_series_anomalies(series: pd.Series, point_df: pd.DataFrame, out_path: Path, title: str) -> None:
    anom = point_df[point_df["is_anom_point"].astype(int) == 1]
    anom_idx = pd.DatetimeIndex(pd.to_datetime(anom.index, errors="coerce")).intersection(series.index)
    anom_values = series.reindex(anom_idx).dropna()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(series.index, series.values, linewidth=0.8, label="Serie raw")
    if not anom_values.empty:
        ax.scatter(anom_values.index, anom_values.values, s=8, color="red", label="Anomalias")
    ax.set_title(title)
    ax.set_xlabel("tempo")
    ax.set_ylabel(series.name)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_alarm_overlay(series: pd.Series, point_df: pd.DataFrame, alarm_times: pd.Series, out_path: Path, title: str) -> None:
    anom = point_df[point_df["is_anom_point"].astype(int) == 1]
    anom_idx = pd.DatetimeIndex(pd.to_datetime(anom.index, errors="coerce")).intersection(series.index)
    anom_values = series.reindex(anom_idx).dropna()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(series.index, series.values, linewidth=0.8, label="Serie raw")
    if not anom_values.empty:
        ax.scatter(anom_values.index, anom_values.values, s=8, color="red", label="Anomalias")
    for ts in alarm_times:
        ax.axvline(ts, color="green", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("tempo")
    ax.set_ylabel(series.name)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_mae_scores(seq_df: pd.DataFrame, threshold: float | None, out_path: Path, title: str) -> None:
    x = pd.to_datetime(seq_df["seq_start_time"], errors="coerce")
    y = pd.to_numeric(seq_df["mae_seq"], errors="coerce")
    anom = seq_df["is_anom_seq"].astype(int) == 1

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, y, linewidth=0.8, label="MAE sequencia")
    if threshold is not None:
        ax.axhline(threshold, color="orange", linestyle="--", label="threshold")
    if anom.any():
        ax.scatter(x[anom], y[anom], color="red", s=8, label="anomalia seq")
    ax.set_title(title)
    ax.set_xlabel("tempo")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_trials(trials_df: pd.DataFrame, out_path: Path) -> None:
    value_col = "val_loss" if "val_loss" in trials_df.columns else trials_df.select_dtypes("number").columns[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(trials_df) + 1), pd.to_numeric(trials_df[value_col], errors="coerce"), marker="o")
    ax.set_title(f"KerasTuner trials - {value_col}")
    ax.set_xlabel("trial")
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_point_anomalies_daily(point_df: pd.DataFrame, out_path: Path, title: str) -> None:
    daily = point_df["is_anom_point"].astype(int).resample("D").sum()
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.bar(daily.index, daily.values, width=0.9)
    ax.set_title(title)
    ax.set_xlabel("dia")
    ax.set_ylabel("anomalias ponto")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    keys = [
        "threshold",
        "hit_rate",
        "n_alarms",
        "alarms_with_detected_anomaly_in_window",
    ]
    values = {k: row.get(k) for k in keys if pd.notna(row.get(k))}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    lines = ["Resumo da task"]
    lines.extend(f"{k}: {v}" for k, v in values.items())
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_plots(task_id: str, out_dir: Path, include_raw: bool = False) -> None:
    task = Task.get_task(task_id=task_id)
    cfg = _pipeline_params(task)
    sensor_list = cfg.get("SENSOR_LIST") or []
    if isinstance(sensor_list, str):
        sensor_list = json.loads(sensor_list)
    if not sensor_list:
        raise ValueError("Nao foi possivel inferir SENSOR_LIST dos parametros da task.")
    sensor = sensor_list[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    point_path = _copy_artifact(task, f"{sensor}/csv/point_anomalies_all.csv", out_dir)
    seq_path = _copy_artifact(task, f"{sensor}/csv/sequence_scores_all.csv", out_dir)
    trials_path = _copy_artifact(task, f"{sensor}/csv/trials_ranking.csv", out_dir)
    calibration_path = _copy_artifact(task, f"{sensor}/csv/calibration_report.json", out_dir)
    summary_path = _copy_artifact(task, "summary_all_sensors_csv", out_dir)
    _copy_artifact(task, "time_integrity_report_json", out_dir)

    point_df = pd.read_csv(point_path, index_col=0)
    point_df.index = pd.to_datetime(point_df.index, errors="coerce")
    point_df = point_df[~point_df.index.isna()]

    seq_df = pd.read_csv(seq_path)
    trials_df = pd.read_csv(trials_path)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    threshold = calibration.get("threshold")

    _plot_mae_scores(seq_df, threshold, out_dir / "sequence_mae_scores.png", f"{sensor} - MAE por sequencia")
    _plot_trials(trials_df, out_dir / "tuner_trials.png")
    _plot_point_anomalies_daily(point_df, out_dir / "point_anomalies_daily.png", f"{sensor} - anomalias por dia")

    summary = pd.read_csv(summary_path)
    _plot_summary(summary, out_dir / "task_summary.png")

    generated_plots = [
        "sequence_mae_scores.png",
        "tuner_trials.png",
        "point_anomalies_daily.png",
        "task_summary.png",
    ]

    if include_raw:
        series = _load_raw_series(cfg, sensor)
        alarm_times = _load_alarm_times(cfg, sensor)
        _plot_series_anomalies(series, point_df, out_dir / "series_with_anomalies.png", f"{sensor} - serie + anomalias")
        _plot_alarm_overlay(series, point_df, alarm_times, out_dir / "series_alarm_anomaly_overlay.png", f"{sensor} - alarmes + anomalias")
        generated_plots.extend(["series_with_anomalies.png", "series_alarm_anomaly_overlay.png"])

    report = {
        "task_id": task_id,
        "task_name": task.name,
        "status": task.status,
        "sensor": sensor,
        "output_dir": str(out_dir),
        "threshold": threshold,
        "summary": summary.to_dict(orient="records"),
        "generated_plots": generated_plots,
    }
    (out_dir / "plot_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from a ClearML Task without downloading all outputs manually.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-raw", action="store_true", help="Também baixa o dataset raw para graficos de serie temporal.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("task_plots") / args.task_id
    generate_plots(args.task_id, out_dir, include_raw=args.include_raw)


if __name__ == "__main__":
    main()
