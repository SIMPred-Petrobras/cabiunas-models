#!/usr/bin/env python3
"""Gera plots duplo-eixo (bruto+MAE+anomalia) de episódios 'far' específicos,
para inspecionar visualmente por que se parecem (duração/magnitude) com
episódios 'near' da falha real — investigação ad-hoc, não roda a pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.transpetro.io import _is_long_format, _pivot_long
from src.cnn1d_ae.plots import plot_signal_mae_anomaly
from src.cnn1d_ae.pipeline import parse_failure_dates

FEATHER_BASE = Path("/home/dvar/transpetro/PROJETO")
OUT_DIR = Path("/tmp/far_episode_inspect")
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    # (modo, equipamento, sensor, episodio_start, episodio_end, label)
    ("uni", "B-6511502A", None, "2022-11-22 10:30", "2022-11-23 02:30", "far_maior_duracao"),
    ("uni", "B-6511502A", None, "2022-08-26 04:45", "2022-08-26 09:20", "far_maior_peak_ratio"),
    ("mult", "B-6511502A", None, "2023-01-02 16:20", "2023-01-03 06:20", "far_maior_peak_ratio_e_duracao"),
    ("uni", "B-4064A", None, "2024-05-10 16:50", "2024-05-10 18:02", "unico_far_relevante"),
    ("mult", "B-4064A", None, "2024-05-15 09:58", "2024-05-15 11:42", "far_maior_peak_ratio_3.6x"),
]


def load_raw(eq: str, sensor: str, feather_path: str, time_col: str) -> pd.Series:
    fp = FEATHER_BASE / feather_path
    df = pd.read_feather(fp)
    if _is_long_format(df):
        df = _pivot_long(df)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        tcol = next(c for c in ("Timestamp", "Data Hora", time_col) if c in df.columns)
        df = df.set_index(pd.to_datetime(df[tcol], errors="coerce")).sort_index()
    return pd.to_numeric(df[sensor], errors="coerce")


def main() -> None:
    for mode, eq, _, ep_start, ep_end, label in CASES:
        base = Path(f"resultados/{'Uni_sensor' if mode == 'uni' else 'Mult_sensor'}/{eq}")
        cfg = json.loads((base / "csv" / "run_config.json").read_text(encoding="utf-8"))
        calib = json.loads((base / "csv" / "calibration_report.json").read_text(encoding="utf-8"))
        sensor = calib.get("sensor") or calib.get("target_sensor")
        threshold = float(calib["threshold"])
        failure_times = parse_failure_dates(cfg.get("FAILURE_DATE", ""))

        raw = load_raw(eq, sensor, cfg["FEATHER_PATH"], cfg.get("TIME_COL", "data_datetime"))

        seq = pd.read_csv(base / "csv" / "sequence_scores_all.csv", usecols=["seq_start_time", "mae_seq"])
        seq["seq_start_time"] = pd.to_datetime(seq["seq_start_time"], errors="coerce")
        mae_series = pd.Series(seq["mae_seq"].values, index=seq["seq_start_time"])

        pt = pd.read_csv(base / "csv" / "point_anomalies_all.csv")
        pt["data_datetime"] = pd.to_datetime(pt.iloc[:, 0], errors="coerce")
        pt = pt.set_index("data_datetime")
        anom_times = pt.index[pt["is_anom_point"] == 1]
        state = pt["operational_state"] if "operational_state" in pt.columns else None

        t0 = pd.Timestamp(ep_start) - pd.Timedelta(hours=12)
        t1 = pd.Timestamp(ep_end) + pd.Timedelta(hours=12)
        out_path = OUT_DIR / f"{mode}_{eq}_{label}.png"
        plot_signal_mae_anomaly(
            raw, mae_series, threshold, anom_times, str(out_path),
            title=f"[{mode}] {eq} | {sensor} | episódio far: {ep_start} -> {ep_end}",
            windows=[(t0, t1, label)], failure_times=failure_times,
            operational_state=state,
        )
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
