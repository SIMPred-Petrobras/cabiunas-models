# =========================
# FILE: src/cnn1d_ae/plots.py
# =========================
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from tensorflow import keras


def _shade_machine_states(ax: plt.Axes, idx: pd.DatetimeIndex, operational_state: pd.Series | None) -> None:
    if operational_state is None or len(idx) == 0:
        return
    st = operational_state.reindex(idx).fillna("on").astype(str)
    is_off = st.isin(["off_curto", "off_longo"])
    is_on = ~is_off

    # Faixa visual no rodape para indicar ligado/desligado de forma explicita.
    ax.fill_between(
        idx,
        0.00,
        0.03,
        where=is_on.values,
        transform=ax.get_xaxis_transform(),
        color="#4caf50",
        alpha=0.35,
        linewidth=0,
    )
    ax.fill_between(
        idx,
        0.00,
        0.03,
        where=is_off.values,
        transform=ax.get_xaxis_transform(),
        color="#ef6c00",
        alpha=0.6,
        linewidth=0,
    )

    if not is_off.any():
        return
    grp = is_off.ne(is_off.shift(fill_value=False)).cumsum()
    runs = pd.DataFrame({"off": is_off.values, "t": idx.values}).groupby(grp)
    for _, g in runs:
        if bool(g["off"].iloc[0]):
            t0 = pd.Timestamp(g["t"].iloc[0])
            t1 = pd.Timestamp(g["t"].iloc[-1])
            ax.axvspan(t0, t1, color="#ef6c00", alpha=0.22)
            ax.axvline(t0, color="#ef6c00", alpha=0.35, linewidth=0.8)
            ax.axvline(t1, color="#ef6c00", alpha=0.35, linewidth=0.8)


def plot_loss(history: keras.callbacks.History, out_path: str) -> None:
    fig = plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hist_mae(train_mae_seq: np.ndarray, threshold: float, out_path: str) -> None:
    fig = plt.figure()
    plt.hist(train_mae_seq, bins=60)
    plt.axvline(threshold, linestyle="--")
    plt.title("Train reconstruction MAE (per-sequence)")
    plt.xlabel("MAE")
    plt.ylabel("count")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_series_with_anomalies(
    series: pd.Series,
    anomalous_times: pd.DatetimeIndex,
    out_path: str,
    title: str,
    operational_state: pd.Series | None = None,
) -> None:
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    # Evita falha em reindex quando a série vem com timestamps duplicados.
    s = s[~s.index.duplicated(keep="last")]

    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    _shade_machine_states(ax, s.index, operational_state)
    plt.plot(s.index, s.values, linewidth=1)
    if len(anomalous_times) > 0:
        # Evita erro de tamanho x/y quando há timestamps repetidos ou fora do índice.
        anom_idx = pd.DatetimeIndex(pd.to_datetime(anomalous_times, errors="coerce"))
        anom_idx = anom_idx.dropna().drop_duplicates().intersection(s.index)
        if len(anom_idx) > 0:
            vals = s.reindex(anom_idx).dropna()
            if len(vals) > 0:
                plt.scatter(vals.index, vals.values, s=10)
    plt.title(f"{title} | faixa verde=ligada, laranja=desligada")
    plt.xlabel("time")
    plt.ylabel("value (raw units)")
    legend_items = [
        Patch(facecolor="#4caf50", alpha=0.35, label="Maquina ligada"),
        Patch(facecolor="#ef6c00", alpha=0.6, label="Maquina desligada"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize="small")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_series_alarm_anomaly_subplots(
    series: pd.Series,
    anomalous_times: pd.DatetimeIndex,
    alarm_times: pd.Series,
    out_path: str,
    title: str,
    operational_state: pd.Series | None = None,
) -> None:
    """
    Subplot legado para auditoria visual:
    - Painel 1: Série + anomalias
    - Painel 2: Série + eventos de alarme + anomalias
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return

    anom_idx = pd.DatetimeIndex(pd.to_datetime(anomalous_times, errors="coerce"))
    anom_idx = anom_idx.dropna().drop_duplicates().intersection(s.index)
    s_anom = s.reindex(anom_idx).dropna() if len(anom_idx) > 0 else pd.Series(dtype=float)

    alarm_idx = pd.to_datetime(alarm_times, errors="coerce")
    alarm_idx = pd.Series(alarm_idx).dropna().sort_values().drop_duplicates()

    # Restringe alarmes ao range da série para não expandir o eixo X além de 2025.
    if len(alarm_idx) and len(s):
        alarm_idx = alarm_idx[(alarm_idx >= s.index.min()) & (alarm_idx <= s.index.max())]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    # Painel 1
    _shade_machine_states(ax1, s.index, operational_state)
    ax1.plot(s.index, s.values, color="blue", linewidth=1, label="Série do sensor")
    if len(s_anom) > 0:
        ax1.scatter(s_anom.index, s_anom.values, color="red", s=14, alpha=0.8, label="Anomalias (ponto)")
    ax1.set_title(f"{title} | Série + Anomalias")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize="small")

    # Painel 2
    _shade_machine_states(ax2, s.index, operational_state)
    ax2.plot(s.index, s.values, color="blue", linewidth=1, alpha=0.85, label="Série do sensor")
    if len(s_anom) > 0:
        ax2.scatter(s_anom.index, s_anom.values, color="red", s=16, alpha=0.85, label="Anomalias (ponto)")
    for t in alarm_idx:
        ax2.axvline(t, color="green", linestyle="--", linewidth=1.1, alpha=0.55)
    if len(alarm_idx) > 0:
        ax2.scatter(alarm_idx.values, [np.nanmedian(s.values)] * len(alarm_idx), marker="x", color="green", s=55, label="Eventos de alarme")

    ax2.set_title("Série + Eventos de Alarme + Anomalias")
    ax2.set_xlabel("time")
    ax2.set_ylabel("value (raw units)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize="small")
    ax1.legend(
        handles=[
            Patch(facecolor="#4caf50", alpha=0.35, label="Maquina ligada"),
            Patch(facecolor="#ef6c00", alpha=0.6, label="Maquina desligada"),
        ]
        + ax1.get_legend_handles_labels()[0],
        labels=["Maquina ligada", "Maquina desligada"] + ax1.get_legend_handles_labels()[1],
        loc="upper right",
        fontsize="small",
    )
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y %H:%M"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
# =========================
# END FILE
