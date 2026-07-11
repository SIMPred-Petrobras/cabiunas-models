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


def plot_loss(history: keras.callbacks.History, out_path: str, title: str | None = None) -> None:
    train = list(history.history.get("loss", []))
    val = list(history.history.get("val_loss", []))
    epochs = range(1, len(train) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train, marker="o", markersize=3, linewidth=1.6,
            color="#1f77b4", label="Treino (loss)")
    if val:
        ax.plot(range(1, len(val) + 1), val, marker="s", markersize=3, linewidth=1.6,
                color="#d62728", label="Validação (val_loss)")
        best_ep = int(np.argmin(val)) + 1
        best_val = float(np.min(val))
        ax.axvline(best_ep, color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.scatter([best_ep], [best_val], color="#2ca02c", zorder=5, s=45)
        ax.annotate(f"melhor época={best_ep}\nval={best_val:.4g}",
                    xy=(best_ep, best_val), xytext=(6, 10),
                    textcoords="offset points", fontsize="small", color="#2ca02c")

    header = "Curva de Loss (treino / validação)"
    if title:
        header = f"{header}\n{title}"
    ax.set_title(header)
    ax.set_xlabel("época")
    ax.set_ylabel("MSE (reconstrução)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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


def _clean_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def plot_signal_mae_anomaly(
    raw_series: pd.Series,
    mae_series: pd.Series,
    threshold: float,
    anomalous_times: pd.DatetimeIndex,
    out_path: str,
    title: str,
    windows: list | None = None,
    failure_times: list | None = None,
    alarm_times: pd.Series | None = None,
    operational_state: pd.Series | None = None,
) -> None:
    """Painel duplo-eixo por janela: sinal BRUTO (eixo esq, azul) + ERRO DE
    RECONSTRUÇÃO MAE (eixo dir, laranja, com threshold e área acima realçada) +
    ANOMALIAS (pontos vermelhos sobre o bruto) + falha (linha vermelha) +
    alarmes (linha verde) + estado operacional (faixas no rodapé).

    `windows` = lista de (t0, t1, rótulo); None => série completa (1 painel).
    Um subplot por janela — use os zooms passando janelas ±N dias na falha.
    """
    raw = _clean_series(raw_series)
    mae = _clean_series(mae_series)
    if raw.empty:
        return

    anom_idx = pd.DatetimeIndex(pd.to_datetime(anomalous_times, errors="coerce")).dropna()
    anom_idx = anom_idx.drop_duplicates().intersection(raw.index)
    s_anom = raw.reindex(anom_idx).dropna() if len(anom_idx) else pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    fails = pd.DatetimeIndex(pd.to_datetime(pd.Series(failure_times or []), errors="coerce")).dropna()
    alarms = pd.DatetimeIndex(pd.to_datetime(pd.Series(alarm_times if alarm_times is not None else []),
                                             errors="coerce")).dropna()

    if not windows:
        windows = [(raw.index.min(), raw.index.max(), "período completo")]

    n = len(windows)
    fig, axes = plt.subplots(n, 1, figsize=(15, 4.6 * n), squeeze=False)
    axes = axes[:, 0]

    for ax, (t0, t1, label) in zip(axes, windows):
        t0 = raw.index.min() if t0 is None else pd.Timestamp(t0)
        t1 = raw.index.max() if t1 is None else pd.Timestamp(t1)
        rw = raw[(raw.index >= t0) & (raw.index <= t1)]
        mw = mae[(mae.index >= t0) & (mae.index <= t1)]
        aw = s_anom[(s_anom.index >= t0) & (s_anom.index <= t1)]

        _shade_machine_states(ax, rw.index, operational_state)
        # Eixo esquerdo: sinal bruto
        l_raw, = ax.plot(rw.index, rw.values, color="#1f4e79", linewidth=1.0, zorder=3, label="Sinal bruto")
        ax.set_ylabel("Sinal bruto (unidades reais)", color="#1f4e79")
        ax.tick_params(axis="y", labelcolor="#1f4e79")

        # Eixo direito: MAE (erro de reconstrução)
        ax2 = ax.twinx()
        l_mae, = ax2.plot(mw.index, mw.values, color="#e8820c", linewidth=1.0, alpha=0.85, zorder=2, label="Erro reconstr. (MAE)")
        if np.isfinite(threshold):
            ax2.axhline(threshold, color="#8B4513", linestyle="--", linewidth=1.0, alpha=0.8)
            ax2.fill_between(mw.index, threshold, mw.values, where=(mw.values > threshold),
                             color="#e8820c", alpha=0.25, zorder=1)
        ax2.set_ylabel("MAE (erro de reconstrução)", color="#e8820c")
        ax2.tick_params(axis="y", labelcolor="#e8820c")
        ax2.set_ylim(bottom=0)

        # Anomalias (sobre o bruto)
        if len(aw):
            ax.scatter(aw.index, aw.values, color="red", s=18, zorder=5, label="Anomalia (ponto)")

        # Falhas e alarmes na janela
        for ft in fails[(fails >= t0) & (fails <= t1)]:
            ax.axvline(ft, color="red", linewidth=2.0, alpha=0.9, zorder=4)
        for at in alarms[(alarms >= t0) & (alarms <= t1)]:
            ax.axvline(at, color="green", linestyle="--", linewidth=1.0, alpha=0.55, zorder=4)

        ax.set_title(f"{title} | {label}")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        handles = [
            l_raw, l_mae,
            plt.Line2D([], [], color="red", marker="o", ls="", label="Anomalia (ponto)"),
            plt.Line2D([], [], color="red", lw=2.0, label="Falha documentada"),
            plt.Line2D([], [], color="green", ls="--", lw=1.0, label="Alarme"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize="small", framealpha=0.9)

    axes[-1].set_xlabel("tempo")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_series_failure_zoom(
    series: pd.Series,
    anomalous_times: pd.DatetimeIndex,
    alarm_times: pd.Series,
    failure_times: list,
    out_path: str,
    title: str,
    zoom_days: float = 10.0,
    operational_state: pd.Series | None = None,
) -> None:
    """
    Zoom apresentável em torno de cada data de falha documentada.
    Um painel por falha, recortando a série a ±zoom_days e destacando:
    linha vermelha = falha, x verde = alarmes, pontos vermelhos = anomalias,
    faixas = estado operacional (verde ligada / laranja desligada).
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return

    fails = pd.DatetimeIndex(pd.to_datetime(pd.Series(failure_times), errors="coerce")).dropna()
    fails = fails.drop_duplicates().sort_values()
    if len(fails) == 0:
        return

    anom_idx = pd.DatetimeIndex(pd.to_datetime(anomalous_times, errors="coerce"))
    anom_idx = anom_idx.dropna().drop_duplicates().intersection(s.index)
    # Index datetime mesmo quando vazio — evita comparar RangeIndex(int) >= Timestamp.
    s_anom = (s.reindex(anom_idx).dropna() if len(anom_idx) > 0
              else pd.Series(dtype=float, index=pd.DatetimeIndex([])))

    alarm_idx = pd.to_datetime(alarm_times, errors="coerce")
    alarm_idx = pd.DatetimeIndex(pd.Series(alarm_idx).dropna().sort_values().drop_duplicates())

    win = pd.Timedelta(days=float(zoom_days))
    n = len(fails)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.2 * n), squeeze=False)
    axes = axes[:, 0]

    for ax, ft in zip(axes, fails):
        t0, t1 = ft - win, ft + win
        s_w = s[(s.index >= t0) & (s.index <= t1)]
        _shade_machine_states(ax, s_w.index, operational_state)
        if len(s_w) > 0:
            ax.plot(s_w.index, s_w.values, color="#1f4e79", linewidth=1.1, label="Série do sensor")

        a_w = s_anom[(s_anom.index >= t0) & (s_anom.index <= t1)]
        if len(a_w) > 0:
            ax.scatter(a_w.index, a_w.values, color="red", s=16, alpha=0.85, zorder=4,
                       label="Anomalias (ponto)")

        for at in alarm_idx[(alarm_idx >= t0) & (alarm_idx <= t1)]:
            ax.axvline(at, color="green", linestyle="--", linewidth=1.1, alpha=0.6)

        ax.axvline(ft, color="red", linewidth=2.2, alpha=0.9, label="Falha documentada")
        ax.set_title(f"{title} | Falha em {ft.strftime('%d/%m/%Y %H:%M')} (±{zoom_days:g} dias)")
        ax.set_ylabel("value (raw units)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        handles = [
            plt.Line2D([], [], color="#1f4e79", lw=1.1, label="Série do sensor"),
            plt.Line2D([], [], color="red", lw=2.2, label="Falha documentada"),
            plt.Line2D([], [], color="green", ls="--", lw=1.1, label="Alarme"),
            plt.Line2D([], [], color="red", marker="o", ls="", label="Anomalias (ponto)"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize="small")

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
# =========================
# END FILE
