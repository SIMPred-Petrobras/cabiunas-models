#!/usr/bin/env python3
"""
plot_spike_experiment_offline.py
Gera as figuras do experimento de supressão de spikes SEM acessar o servidor ClearML,
lendo os artefatos já baixados no cache local (~/.clearml/cache).

Cada braço é identificado por impressão digital: reconstrói (recall, threshold_q,
duty_cycle) a partir do arquivo de cache e casa com os valores já medidos na tabela
de resultados. Isso torna a identificação verificável, não um chute por data de arquivo.

Uso:
    PYTHONPATH=. python scripts/plot_spike_experiment_offline.py
"""
from __future__ import annotations

import glob
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "eval_per_sensor_level", os.path.join(_HERE, "eval_per_sensor_level.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

CACHE = os.path.expanduser("~/.clearml/cache/storage_manager/global")
SENSOR = "TC382_03_A"
RAW_CSV = "../dados/sensores_brutos_2025_30s.csv"
OUT_DIR = "spike_suppress_eval_out"
EVAL_START = pd.Timestamp("2025-01-01", tz="UTC")
EVAL_END = pd.Timestamp("2026-01-01", tz="UTC")
EXCL_CONDS = ["UNDER", "LOLO", "OVER", "CFN"]
HALF_LIFE, HORIZON, MAX_DUTY = 4.0, 8.0, 0.25

# (recall, threshold_q, duty_cycle) medidos SEM mask_off — impressão digital de cada braço
FINGERPRINTS = {
    "base":          (0.9166666666666666, 0.7822626262626262, 0.2177444227408245),
    "hampel":        (0.9166666666666666, 0.8326666666666667, 0.1673349097015181),
    "gradmask":      (0.8333333333333334, 0.7822626262626262, 0.2177444227408245),
    "nosent_base":   (0.0833333333333333, 0.7520202020202020, 0.2479809921981198),
    "nosent_hampel": (0.0,                0.9990000000000000, 0.0010052202917423),
}
LABELS = {
    "base":          "Sentinel 500–950 (baseline v10)",
    "hampel":        "Sentinel + Hampel",
    "gradmask":      "Sentinel + grad-mask (treino)",
    "nosent_base":   "Sem sentinel",
    "nosent_hampel": "Sem sentinel + Hampel",
}
ORDER = ["base", "hampel", "gradmask", "nosent_base", "nosent_hampel"]

C_TEMP, C_HEALTH = "#2c3e50", "#1f5c8b"
C_THR, C_ALERT, C_OFF = "#b3541e", "#e8b23a", "#c8cdd2"
C_HIT, C_MISS = "#1b7f4b", "#c0392b"


def read_mae(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
    return df.set_index("seq_start_time")["mae_seq"]


def read_on_mask(path: str) -> pd.Series:
    df = pd.read_csv(path, usecols=["data_datetime", "operational_state"])
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["data_datetime"]).set_index("data_datetime")
    return (df["operational_state"] == "on").resample(ev.SAMPLING_INTERVAL).mean() >= 0.5


def incidents_for(on_mask: pd.Series) -> list:
    alarms = ev.load_alarms_gap(ev.ALARM_CSV_DEFAULT, EXCL_CONDS).get(SENSOR, [])
    alarms = [a for a in alarms if EVAL_START <= a <= EVAL_END]
    if alarms:
        on_at = on_mask.reindex(pd.DatetimeIndex(alarms), method="nearest",
                                tolerance=pd.Timedelta("30min")).fillna(True)
        alarms = [a for a, ok in zip(alarms, on_at.tolist()) if ok]
    return ev.cluster_incidents(alarms)


def evaluate(mae: pd.Series, on_mask: pd.Series, incidents: list, mask_off: bool) -> dict:
    h = ev.ewma_quantile(mae, HALF_LIFE)
    h = h[(h.index >= EVAL_START) & (h.index <= EVAL_END)]
    # astype(bool) é obrigatório: reindex com tolerance devolve dtype object, e "~"
    # sobre object faz complemento de inteiro (~False == -1, truthy), não negação.
    om = on_mask.reindex(h.index, method="nearest",
                         tolerance=pd.Timedelta("6min")).fillna(False).astype(bool)
    if mask_off:
        h = h.where(om, other=0.0)
    res = ev.best_point_for_sensor(h, incidents, HORIZON, max_duty_cycle=MAX_DUTY)
    alert = h >= res["threshold_q"]
    hits, misses = [], []
    for t in incidents:
        w0 = t - pd.Timedelta(hours=HORIZON)
        (hits if alert[(alert.index >= w0) & (alert.index <= t)].any() else misses).append(t)
    return {"health": h, "alert": alert, "on_mask": om, "res": res,
            "incidents": incidents, "hits": hits, "misses": misses}


def identify_arms() -> dict:
    """Casa cada arquivo de cache com um braço pela impressão digital."""
    seq_files = sorted(glob.glob(os.path.join(CACHE, "*.sequence_scores_all.csv")),
                       key=os.path.getmtime, reverse=True)
    pt_files = sorted(glob.glob(os.path.join(CACHE, "*.point_anomalies_all.csv")),
                      key=os.path.getmtime, reverse=True)
    # operational_state é o mesmo em todos (vem de RUNNING_A); usa o 2025-only mais recente
    on_mask = None
    for p in pt_files:
        try:
            m = read_on_mask(p)
            if m.index.min() >= pd.Timestamp("2024-12-01", tz="UTC"):
                on_mask = m
                print(f"  máscara ON/OFF de {os.path.basename(p)[:12]}… "
                      f"({m.index.min().date()} → {m.index.max().date()})")
                break
        except Exception:
            continue
    if on_mask is None:
        raise SystemExit("Não achei point_anomalies 2025-only no cache.")

    incidents = incidents_for(on_mask)
    print(f"  {len(incidents)} incidentes HI/HIHI (equipamento ligado) em 2025")

    found = {}
    for path in seq_files:
        if len(found) == len(FINGERPRINTS):
            break
        try:
            mae = read_mae(path)
        except Exception:
            continue
        if mae.index.min() < pd.Timestamp("2024-12-01", tz="UTC"):
            continue  # série 2024h2 — não é deste experimento
        r = evaluate(mae, on_mask, incidents, mask_off=False)["res"]
        got = (r["recall"], r["threshold_q"], r.get("duty_cycle", np.nan))
        for arm, fp in FINGERPRINTS.items():
            if arm in found:
                continue
            if all(abs(a - b) < 1e-6 for a, b in zip(got, fp)):
                found[arm] = path
                print(f"  ✓ {arm:<14} ← {os.path.basename(path)[:12]}… "
                      f"(recall {got[0]*100:.1f}%, q={got[1]:.4f})")
                break
    missing = [a for a in FINGERPRINTS if a not in found]
    if missing:
        print(f"  [AVISO] não identificados no cache: {missing}")
    return found, on_mask, incidents


def shade(ax, mask, color, alpha, label=None):
    if mask is None:
        return
    mask = mask.astype(bool)
    if not mask.any():
        return
    v, idx = mask.values, mask.index
    e = np.diff(v.astype(np.int8))
    starts, ends = list(idx[1:][e == 1]), list(idx[1:][e == -1])
    if v[0]:
        starts.insert(0, idx[0])
    if v[-1]:
        ends.append(idx[-1])
    for i, (s, t) in enumerate(zip(starts, ends)):
        ax.axvspan(s, t, color=color, alpha=alpha, lw=0, label=label if i == 0 else None)


def panel_common(ax, arm, title, extra=""):
    r = arm["res"]
    for i, t in enumerate(arm["hits"]):
        ax.axvline(t, color=C_HIT, lw=1.3, alpha=.9,
                   label="Alarme HI/HIHI detectado" if i == 0 else None)
    for i, t in enumerate(arm["misses"]):
        ax.axvline(t, color=C_MISS, lw=1.5, ls=":", alpha=.95,
                   label="Alarme HI/HIHI perdido" if i == 0 else None)
    ax.set_title(
        f"{title}{extra}  —  recall {r['recall']*100:.1f}% "
        f"({len(arm['hits'])}/{len(arm['incidents'])})   FA {r['fa_per_day']:.3f}/dia"
        f"   tempo em alerta {r.get('duty_cycle', np.nan):.0%}",
        fontsize=10, loc="left", pad=6)
    ax.grid(alpha=.22, lw=.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def draw_health(ax, arm, title):
    shade(ax, ~arm["on_mask"], C_OFF, .5, "Equipamento desligado")
    shade(ax, arm["alert"], C_ALERT, .4, "Alerta ativo")
    ax.plot(arm["health"].index, arm["health"].values, color=C_HEALTH, lw=.7,
            label="Índice de saúde (EWMA)")
    ax.axhline(arm["res"]["threshold_q"], color=C_THR, ls="--", lw=1.2,
               label=f"Threshold")
    ax.set_ylim(-.03, 1.03)
    ax.set_ylabel("saúde")
    panel_common(ax, arm, title)


def draw_temp(ax, temp, arm, title):
    om = arm["on_mask"].reindex(temp.index, method="nearest",
                                tolerance=pd.Timedelta("6min")).fillna(False).astype(bool)
    off = ~om
    al = arm["alert"].reindex(temp.index, method="nearest",
                              tolerance=pd.Timedelta("6min")).fillna(False).astype(bool)
    shade(ax, off, C_OFF, .5, "Equipamento desligado")
    shade(ax, al, C_ALERT, .45, "Anomalia detectada")
    ax.plot(temp.index, temp.values, color=C_TEMP, lw=.45, label=f"{SENSOR} (°C)")
    ax.set_ylabel("°C")
    panel_common(ax, arm, title)


def save(fig, axes, name, suptitle, xlabel="2025"):
    axes[-1].set_xlabel(xlabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, -.004))
    fig.suptitle(suptitle, fontsize=13, y=.998)
    fig.tight_layout(rect=[0, .022, 1, .986])
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", p, flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1/4] Identificando braços no cache local...")
    found, on_mask, incidents = identify_arms()
    arms_order = [a for a in ORDER if a in found]
    if not arms_order:
        raise SystemExit("Nenhum braço identificado.")

    print("[2/4] Reconstruindo com mask_off (protocolo correto)...")
    arms = {a: evaluate(read_mae(found[a]), on_mask, incidents, mask_off=True)
            for a in arms_order}

    print("[3/4] Figura de índice de saúde...")
    fig, axes = plt.subplots(len(arms_order), 1, figsize=(14, 3.0 * len(arms_order)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, arms_order):
        draw_health(ax, arms[a], LABELS[a])
    save(fig, axes, "fig_spike_exp_series.png",
         f"{SENSOR} — índice de saúde por braço do experimento (2025, HI/HIHI, com mask_off)")

    print("[4/4] Figura de temperatura + anomalias...")
    df = pd.read_csv(RAW_CSV, usecols=["data_datetime", SENSOR])
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], utc=True, errors="coerce")
    temp = df.dropna(subset=["data_datetime"]).set_index("data_datetime")[SENSOR] \
             .resample(ev.SAMPLING_INTERVAL).mean()
    # Limita à cobertura real dos modelos (o CSV vai até dez, os modelos até nov):
    # plotar além disso desenharia "desligado" onde na verdade não há avaliação.
    cov_end = min(EVAL_END, on_mask.index.max())
    temp = temp[(temp.index >= EVAL_START) & (temp.index <= cov_end)]
    print(f"      janela plotada: {temp.index.min().date()} → {temp.index.max().date()}")

    fig, axes = plt.subplots(len(arms_order), 1, figsize=(15, 2.9 * len(arms_order)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, arms_order):
        draw_temp(ax, temp, arms[a], LABELS[a])
    save(fig, axes, "fig_spike_exp_anomalias_temp.png",
         f"{SENSOR} — temperatura e anomalias detectadas por braço (2025)")

    # Confound: sem sentinel, sem vs com mask_off
    if "nosent_base" in found:
        raw = evaluate(read_mae(found["nosent_base"]), on_mask, incidents, mask_off=False)
        fig, axes = plt.subplots(2, 1, figsize=(14, 6.2), sharex=True)
        draw_health(axes[0], raw, "Sem sentinel — avaliado SEM mask_off (confound)")
        draw_health(axes[1], arms["nosent_base"], "Sem sentinel — avaliado COM mask_off (correto)")
        save(fig, axes, "fig_spike_exp_confound.png",
             f"{SENSOR} — o confound: equipamento desligado domina o sinal de saúde")

    # Zoom no mês com mais incidentes
    if incidents:
        month = pd.Series(1, index=pd.DatetimeIndex(incidents)).resample("MS").sum().idxmax()
        z0, z1 = month, month + pd.offsets.MonthEnd(1)
        tz = temp[(temp.index >= z0) & (temp.index <= z1)]
        fig, axes = plt.subplots(len(arms_order), 1, figsize=(14, 2.6 * len(arms_order)),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, a in zip(axes, arms_order):
            draw_temp(ax, tz, arms[a], LABELS[a])
            ax.set_xlim(z0, z1)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        save(fig, axes, "fig_spike_exp_anomalias_zoom.png",
             f"{SENSOR} — zoom no mês com mais incidentes ({month.strftime('%b/%Y')})",
             xlabel=month.strftime("%B de %Y"))


if __name__ == "__main__":
    main()
