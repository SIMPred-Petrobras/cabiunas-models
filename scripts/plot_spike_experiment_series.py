#!/usr/bin/env python3
"""
plot_spike_experiment_series.py
Plota na série temporal o resultado dos braços do experimento de supressão de spikes
(TC382_03_A, 2025). Reaproveita as funções de eval_per_sensor_level.py para que as
curvas e o ponto de operação sejam exatamente os mesmos da tabela de resultados.

Gera duas figuras:
  fig_spike_exp_series.png    — health de cada braço com protocolo correto (--mask_off)
  fig_spike_exp_confound.png  — o confound: mesmo braço sem e com mask_off

Uso:
    PYTHONPATH=. python scripts/plot_spike_experiment_series.py
"""
from __future__ import annotations

import importlib.util
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task

# Importa o avaliador como módulo para reusar suas funções (mesma matemática da tabela)
_SPEC = importlib.util.spec_from_file_location(
    "eval_per_sensor_level",
    os.path.join(os.path.dirname(__file__), "eval_per_sensor_level.py"),
)
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

SENSOR = "TC382_03_A"
EVAL_START = pd.Timestamp("2025-01-01", tz="UTC")
EVAL_END = pd.Timestamp("2026-01-01", tz="UTC")
EXCL_CONDS = ["UNDER", "LOLO", "OVER", "CFN"]
HALF_LIFE = 4.0
HORIZON = 8.0
MAX_DUTY = 0.25
OUT_DIR = "spike_suppress_eval_out"

ARMS = [
    ("base",          "f68e42fde66c4c66b8f4c792b8397cd7", "Sentinel 500–950 (baseline v10)"),
    ("hampel",        "a559ac9a8ffd4ed39fcb0aab8fa8451d", "Sentinel + Hampel"),
    ("gradmask",      "d479efaf35ae44c9a09a7e349e753e7e", "Sentinel + grad-mask (treino)"),
    ("nosent_base",   "810adbe43adf46f3951a0bb53fbe4b74", "Sem sentinel"),
    ("nosent_hampel", "191c8d60ec36421d8e143ea7be38c52c", "Sem sentinel + Hampel"),
]

C_HEALTH = "#1f5c8b"   # azul — sinal de saúde
C_THR    = "#b3541e"   # laranja queimado — threshold
C_HIT    = "#1b7f4b"   # verde — incidente detectado
C_MISS   = "#c0392b"   # vermelho — incidente perdido
C_OFF    = "#c8cdd2"   # cinza — equipamento desligado
C_ALERT  = "#e8b23a"   # âmbar — episódio de alerta


def build_arm(task_id: str, mask_off: bool) -> dict:
    """Reconstrói health + ponto de operação de um braço, igual ao eval."""
    task = Task.get_task(task_id=task_id)
    mae = ev.load_mae_series(task, [SENSOR])[SENSOR]
    running = ev.load_running_masks(task, [SENSOR]).get(SENSOR)

    health = ev.ewma_quantile(mae, HALF_LIFE)
    health = health[(health.index >= EVAL_START) & (health.index <= EVAL_END)]

    on_mask = None
    if running is not None:
        on_mask = running.reindex(health.index, method="nearest",
                                  tolerance=pd.Timedelta("6min")).fillna(False)
        if mask_off:
            health = health.where(on_mask, other=0.0)

    alarms = ev.load_alarms_gap(ev.ALARM_CSV_DEFAULT, EXCL_CONDS).get(SENSOR, [])
    alarms = [a for a in alarms if EVAL_START <= a <= EVAL_END]
    if running is not None and alarms:
        on_at = running.reindex(pd.DatetimeIndex(alarms), method="nearest",
                                tolerance=pd.Timedelta("30min")).fillna(True)
        alarms = [a for a, ok in zip(alarms, on_at.tolist()) if ok]
    incidents = ev.cluster_incidents(alarms)

    res = ev.best_point_for_sensor(health, incidents, HORIZON,
                                   max_duty_cycle=MAX_DUTY)
    alert = health >= res["threshold_q"]

    # Um incidente é "detectado" se houve alerta na janela de H horas antes dele
    hits, misses = [], []
    for t in incidents:
        w0 = t - pd.Timedelta(hours=HORIZON)
        (hits if alert[(alert.index >= w0) & (alert.index <= t)].any() else misses).append(t)

    return {"health": health, "alert": alert, "on_mask": on_mask,
            "incidents": incidents, "hits": hits, "misses": misses, "res": res}


def shade_spans(ax, mask: pd.Series, color: str, alpha: float, label: str | None = None):
    """Sombreia trechos contíguos onde mask é True."""
    if mask is None or not mask.any():
        return
    vals = mask.values.astype(bool)
    idx = mask.index
    edges = np.diff(vals.astype(np.int8))
    starts = list(idx[1:][edges == 1])
    ends = list(idx[1:][edges == -1])
    if vals[0]:
        starts.insert(0, idx[0])
    if vals[-1]:
        ends.append(idx[-1])
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0,
                   label=label if i == 0 else None)


def draw_panel(ax, arm: dict, title: str, show_off: bool = True):
    h, res = arm["health"], arm["res"]

    if show_off and arm["on_mask"] is not None:
        shade_spans(ax, ~arm["on_mask"], C_OFF, 0.55, "Equipamento desligado")
    shade_spans(ax, arm["alert"], C_ALERT, 0.35, "Alerta ativo")

    ax.plot(h.index, h.values, color=C_HEALTH, lw=0.7, label="Índice de saúde (EWMA)")
    ax.axhline(res["threshold_q"], color=C_THR, ls="--", lw=1.2,
               label=f"Threshold q={res['threshold_q']:.3f}")

    for i, t in enumerate(arm["hits"]):
        ax.axvline(t, color=C_HIT, lw=1.4, alpha=0.9,
                   label="Incidente detectado" if i == 0 else None)
    for i, t in enumerate(arm["misses"]):
        ax.axvline(t, color=C_MISS, lw=1.6, ls=":", alpha=0.95,
                   label="Incidente perdido" if i == 0 else None)

    n_inc = len(arm["incidents"])
    ax.set_title(
        f"{title}  —  recall {res['recall']*100:.1f}% "
        f"({len(arm['hits'])}/{n_inc})   FA {res['fa_per_day']:.3f}/dia   "
        f"duty {res.get('duty_cycle', float('nan')):.2f}",
        fontsize=10, loc="left", pad=6,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("saúde")
    ax.grid(alpha=0.25, lw=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Figura 1: os 5 braços com o protocolo correto (mask_off) ----
    print("[1/2] Construindo braços com --mask_off...")
    arms = {}
    for key, tid, label in ARMS:
        print(f"  {key}...")
        arms[key] = build_arm(tid, mask_off=True)

    fig, axes = plt.subplots(len(ARMS), 1, figsize=(14, 3.0 * len(ARMS)), sharex=True)
    for ax, (key, _, label) in zip(axes, ARMS):
        draw_panel(ax, arms[key], label)
    axes[-1].set_xlabel("2025")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"{SENSOR} — experimento de supressão de spikes (2025, HI/HIHI, protocolo com mask_off)",
        fontsize=13, y=0.997,
    )
    fig.tight_layout(rect=[0, 0.022, 1, 0.985])
    p1 = os.path.join(OUT_DIR, "fig_spike_exp_series.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    print(f"  salvo: {p1}")

    # ---- Figura 2: o confound de OFF ----
    print("[2/2] Construindo figura do confound (sem sentinel, sem vs com mask_off)...")
    raw = build_arm("810adbe43adf46f3951a0bb53fbe4b74", mask_off=False)
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6.2), sharex=True)
    draw_panel(axes2[0], raw, "Sem sentinel — avaliado SEM mask_off (confound)")
    draw_panel(axes2[1], arms["nosent_base"], "Sem sentinel — avaliado COM mask_off (correto)")
    axes2[-1].set_xlabel("2025")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
                fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig2.suptitle(
        f"{SENSOR} — o confound: períodos de equipamento desligado dominam o sinal de saúde",
        fontsize=13, y=0.997,
    )
    fig2.tight_layout(rect=[0, 0.04, 1, 0.975])
    p2 = os.path.join(OUT_DIR, "fig_spike_exp_confound.png")
    fig2.savefig(p2, dpi=140, bbox_inches="tight")
    print(f"  salvo: {p2}")


if __name__ == "__main__":
    main()
