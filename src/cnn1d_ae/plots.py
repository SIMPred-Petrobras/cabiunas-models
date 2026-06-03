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
    plt.plot(history.history.get("val_loss", []), label="val_loss (normal)")
    anom = history.history.get("val_loss_anomaly", [])
    if len(anom) > 0:
        plt.plot(anom, label="val_loss (anomalia)", linestyle="--")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (mse + l2)")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictive_curve(curves: dict, op_points: dict, out_path: str) -> None:
    """Curva preditiva oficial: recall vs FA/dia + lead time vs FA/dia, por horizonte."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for h, curve in sorted(curves.items()):
        if curve is None or len(curve) == 0:
            continue
        axes[0].plot(curve["fa_per_day"], curve["recall"], marker=".", label=f"H={h}h")
        axes[1].plot(curve["fa_per_day"], curve["median_lead_hours"], marker=".", label=f"H={h}h")
        op = op_points.get(h)
        if op:
            axes[0].scatter([op["fa_per_day"]], [op["recall"]], s=60, edgecolor="black", zorder=3)
    axes[0].axvline(1.0, color="gray", ls=":", label="orçamento 1 FA/dia")
    axes[0].set_xlabel("falso-alarmes / dia")
    axes[0].set_ylabel("recall (incidentes genuínos)")
    axes[0].set_title("Curva preditiva: recall vs FA")
    axes[0].set_xlim(0, 10); axes[0].set_ylim(0, 1.02); axes[0].legend()
    axes[1].set_xlabel("falso-alarmes / dia")
    axes[1].set_ylabel("lead time mediano (h)")
    axes[1].set_title("Antecipação vs FA")
    axes[1].set_xlim(0, 10); axes[1].legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_health_index_over_time(
    t_end, health_ewma, op_threshold, incident_times, out_path: str
) -> None:
    """Health-index EWMA ao longo do tempo, com threshold operacional e incidentes."""
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(t_end, health_ewma, lw=0.6, color="steelblue", label="health-index EWMA")
    if op_threshold is not None:
        ax.axhline(op_threshold, color="r", ls="--", label=f"alerta (thr={op_threshold:.3f})")
    if incident_times is not None and len(incident_times):
        for ti in pd.DatetimeIndex(incident_times):
            ax.axvline(ti, color="green", alpha=0.3, lw=0.8)
    ax.set_title("Health-index EWMA | verde = incidente genuíno")
    ax.set_ylabel("EWMA(MAE)")
    ax.set_xlabel("tempo")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_hist_normal_vs_anomaly(
    mse_normal: np.ndarray,
    mse_anomaly: np.ndarray,
    threshold: float,
    out_path: str,
) -> None:
    """Histograma do erro de reconstrucao: normal vs anomalia, com o threshold marcado.
    A separacao entre as duas distribuicoes e o que realmente importa num AE de anomalia."""
    fig = plt.figure(figsize=(9, 4))
    if mse_normal is not None and len(mse_normal) > 0:
        plt.hist(mse_normal, bins=80, alpha=0.6, density=True, label="normal (val)")
    if mse_anomaly is not None and len(mse_anomaly) > 0:
        plt.hist(mse_anomaly, bins=80, alpha=0.6, density=True, label="anomalia (janelas de alarme)")
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4g}")
    plt.legend()
    plt.title("Erro de reconstrucao: normal vs anomalia")
    plt.xlabel("erro por sequencia")
    plt.ylabel("densidade")
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
    if len(alarm_idx) > 0:
        ymin, ymax = float(np.nanmin(s.values)), float(np.nanmax(s.values))
        for t in alarm_idx:
            # ax.plot em coordenadas de dados (sem blended transform) — compatível com ClearML
            ax2.plot([t, t], [ymin, ymax], color="green", linestyle="--", linewidth=1.1, alpha=0.7)
        ax2.scatter(alarm_idx.values, [np.nanmedian(s.values)] * len(alarm_idx),
                    marker="x", color="green", s=55, label=f"Alarmes ({len(alarm_idx)})")

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


def plot_series_with_mae_reconstruction(
    series: pd.Series,
    df_seq_scores: pd.DataFrame,
    threshold: float,
    anomalous_times: pd.DatetimeIndex,
    alarm_times: pd.Series,
    out_path: str,
    title: str,
    operational_state: pd.Series | None = None,
) -> None:
    """Dois painéis diagnóstico por sensor:
    - Painel 1: série temporal + pontos anômalos (vermelho) + eventos de alarme (verde)
    - Painel 2: erro de reconstrução MAE por sequência + threshold + regiões anômalas sombreadas
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return

    # MAE series from df_seq_scores
    mae_s = df_seq_scores["mae_seq"].copy()
    mae_s.index = pd.to_datetime(mae_s.index, errors="coerce")
    mae_s = mae_s.dropna().sort_index()

    alarm_idx = pd.to_datetime(alarm_times, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(alarm_idx) and len(s):
        alarm_idx = alarm_idx[(alarm_idx >= s.index.min()) & (alarm_idx <= s.index.max())]

    anom_idx = pd.DatetimeIndex(pd.to_datetime(anomalous_times, errors="coerce")).dropna().drop_duplicates()
    anom_idx = anom_idx.intersection(s.index)
    s_anom = s.reindex(anom_idx).dropna() if len(anom_idx) > 0 else pd.Series(dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    plt.subplots_adjust(hspace=0.08)

    # --- Painel 1: série + anomalias + alarmes ---
    _shade_machine_states(ax1, s.index, operational_state)
    ax1.plot(s.index, s.values, color="#1565c0", linewidth=0.8, alpha=0.9, label="Sensor")
    if len(s_anom) > 0:
        ax1.scatter(s_anom.index, s_anom.values, color="red", s=12, alpha=0.7, zorder=4, label="Anomalia")
    if len(alarm_idx) > 0:
        ymin1, ymax1 = float(np.nanmin(s.values)), float(np.nanmax(s.values))
        for t in alarm_idx:
            ax1.plot([t, t], [ymin1, ymax1], color="#2e7d32", linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.scatter([], [], color="#2e7d32", marker="|", s=80, label=f"Alarme ({len(alarm_idx)})")
    ax1.set_ylabel("Valor (unidade bruta)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)
    legend_patches = [
        Patch(facecolor="#4caf50", alpha=0.35, label="Ligada"),
        Patch(facecolor="#ef6c00", alpha=0.6, label="Desligada"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0],
               labels=["Ligada", "Desligada"] + ax1.get_legend_handles_labels()[1],
               loc="upper right", fontsize="x-small", ncol=2)

    # --- Painel 2: MAE + threshold ---
    ax2.plot(mae_s.index, mae_s.values, color="#37474f", linewidth=0.7, alpha=0.85, label="MAE reconstrução")
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.4f})")

    # Sombreia regiões anômalas (MAE > threshold)
    anom_mae = mae_s > threshold
    if anom_mae.any():
        ax2.fill_between(mae_s.index, 0, mae_s.values,
                         where=anom_mae.values, color="red", alpha=0.25, label="MAE > threshold")

    if len(alarm_idx) > 0:
        ymin2 = float(max(0, np.nanmin(mae_s.values)))
        ymax2 = float(np.nanmax(mae_s.values))
        for t in alarm_idx:
            ax2.plot([t, t], [ymin2, ymax2], color="#2e7d32", linestyle="--", linewidth=1.0, alpha=0.7)

    ax2.set_ylabel("MAE por sequência")
    ax2.set_xlabel("Tempo")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize="x-small", ncol=2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
# =========================
# END FILE
