from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task


TASK_ID = "56603e6272dd4868abf39db79a1fa505"
SENSOR = None  # Ex.: "T5_AVG_A"; None usa o sensor salvo na task.
START_DATE = "2025-04-17 00:00:00"
END_DATE = "2025-04-21 23:59:59"
OUTPUT_ROOT = "task_zoom_plots"


def _ensure_clearml_config() -> None:
    if os.getenv("CLEARML_CONFIG_FILE"):
        return
    local_config = Path.cwd() / "clearml.conf"
    if local_config.is_file():
        os.environ["CLEARML_CONFIG_FILE"] = str(local_config)
        return
    raise RuntimeError(
        "ClearML nao esta configurado. Rode da raiz do projeto com clearml.conf "
        "ou exporte CLEARML_CONFIG_FILE=/caminho/para/clearml.conf"
    )


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


def _sensor_from_task(cfg: dict) -> str:
    if SENSOR:
        return SENSOR
    sensor_list = cfg.get("SENSOR_LIST") or []
    if isinstance(sensor_list, str):
        sensor_list = json.loads(sensor_list)
    if not sensor_list:
        raise ValueError("Nao foi possivel inferir SENSOR_LIST dos parametros da task.")
    return str(sensor_list[0])


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


def _dataset_root(cfg: dict) -> Path:
    dataset_id = cfg.get("CLEARML_DATASET_ID")
    dataset_name = cfg.get("CLEARML_DATASET_NAME", "Cabiunas 2025")
    dataset_project = cfg.get("CLEARML_PROJECT_NAME", "TesteMLCab")
    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
    else:
        dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project)
    return Path(dataset.get_local_copy())


def _load_raw_series(cfg: dict, sensor: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    root = _dataset_root(cfg)
    raw_path = _resolve_dataset_file(root, str(cfg["RAW_CSV"]))
    time_col = str(cfg["TIME_COL"])

    chunks = []
    for chunk in pd.read_csv(raw_path, usecols=[time_col, sensor], chunksize=250_000):
        idx = pd.to_datetime(chunk[time_col], errors="coerce")
        mask = (idx >= start) & (idx <= end)
        if mask.any():
            selected = chunk.loc[mask, [sensor]].copy()
            selected.index = idx[mask]
            chunks.append(selected)

    if not chunks:
        return pd.Series(dtype=float, name=sensor)

    df = pd.concat(chunks).sort_index()
    series = pd.Series(pd.to_numeric(df[sensor], errors="coerce").values, index=df.index, name=sensor)
    return series[~series.index.duplicated(keep="last")]


def _load_alarm_times(cfg: dict, sensor: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    root = _dataset_root(cfg)
    alarm_path = _resolve_dataset_file(root, str(cfg["ALARM_CSV"]))
    df_alarm = pd.read_csv(alarm_path)
    if "Data da Ocorrência" in df_alarm.columns and "Data da Ocorrencia" not in df_alarm.columns:
        df_alarm["Data da Ocorrencia"] = df_alarm["Data da Ocorrência"]
    if "Tag" in df_alarm.columns:
        df_alarm = df_alarm[df_alarm["Tag"] == sensor]
    times = pd.to_datetime(df_alarm["Data da Ocorrencia"], errors="coerce").dropna().sort_values()
    return times[(times >= start) & (times <= end)]


def _load_point_anomalies(task: Task, sensor: str, out_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    point_path = _copy_artifact(task, f"{sensor}/csv/point_anomalies_all.csv", out_dir)
    chunks = []
    for chunk in pd.read_csv(point_path, index_col=0, chunksize=250_000):
        idx = pd.to_datetime(chunk.index, errors="coerce")
        mask = (idx >= start) & (idx <= end)
        if mask.any():
            selected = chunk.loc[mask].copy()
            selected.index = idx[mask]
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=["is_anom_point"], index=pd.DatetimeIndex([]))
    return pd.concat(chunks).sort_index()


def _plot_zoom(series: pd.Series, point_df: pd.DataFrame, alarms: pd.Series, out_path: Path, sensor: str) -> None:
    anom = point_df[point_df["is_anom_point"].astype(int) == 1]
    anom_idx = pd.DatetimeIndex(pd.to_datetime(anom.index, errors="coerce")).intersection(series.index)
    anom_values = series.reindex(anom_idx).dropna()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(series.index, series.values, color="#1f77b4", linewidth=0.9, label=f"{sensor} raw")
    if not anom_values.empty:
        ax.scatter(anom_values.index, anom_values.values, color="red", s=18, alpha=0.9, label="Anomalias")
    for ts in alarms:
        ax.axvline(ts, color="green", linestyle="--", linewidth=1.0, alpha=0.45)
    if len(alarms):
        ax.scatter(alarms, [series.median()] * len(alarms), marker="x", color="green", s=70, label="Alarmes")

    ax.set_title(f"{sensor} | zoom {START_DATE} a {END_DATE}")
    ax.set_xlabel("tempo")
    ax.set_ylabel(sensor)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _ensure_clearml_config()
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    if end < start:
        raise ValueError("END_DATE precisa ser maior ou igual a START_DATE.")

    task = Task.get_task(task_id=TASK_ID)
    cfg = _pipeline_params(task)
    sensor = _sensor_from_task(cfg)

    out_dir = Path(OUTPUT_ROOT) / TASK_ID / sensor / f"{start:%Y%m%d}_{end:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    series = _load_raw_series(cfg, sensor, start, end)
    if series.empty:
        raise ValueError(f"Nenhum ponto encontrado para {sensor} entre {START_DATE} e {END_DATE}.")

    point_df = _load_point_anomalies(task, sensor, out_dir, start, end)
    alarms = _load_alarm_times(cfg, sensor, start, end)
    out_path = out_dir / f"zoom_{sensor}_{start:%Y%m%d}_{end:%Y%m%d}.png"
    _plot_zoom(series, point_df, alarms, out_path, sensor)

    report = {
        "task_id": TASK_ID,
        "sensor": sensor,
        "start": START_DATE,
        "end": END_DATE,
        "series_points": int(len(series)),
        "anomaly_points": int(point_df["is_anom_point"].astype(int).sum()) if not point_df.empty else 0,
        "alarms": int(len(alarms)),
        "plot": str(out_path),
    }
    (out_dir / "zoom_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
