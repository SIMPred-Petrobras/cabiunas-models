#!/usr/bin/env python3
"""
plot_baseline_vs_ae_series.py
A comparação que decide o projeto, na série temporal — mesmo estilo da figura da
Frente B: painel de temperatura, pista dedicada do alarme do DCS, e um painel de
health por braço com incidente detectado / perdido / episódio de falso positivo.

  painel 3  AE (autoencoder, controle 3b34a312)      raw 62,0% · FA 0,114/dia
  painel 4  LIMIAR na própria temperatura            raw 81,0% · FA 0,047/dia

Janela FULL jan/2024 → abr/2026, 79 incidentes HI/HIHI com máquina ligada. Um único
ponto de operação por braço, escolhido pelo MESMO `best_point_for_sensor` (horizonte
8 h, sticky 12 h, FA ≤ 1/dia, duty ≤ 0,25) — e verificado contra
`baseline_trivial_vs_ae.csv` antes de desenhar. Aborta se não reproduzir.

O que a figura mostra: o limiar acende nas mesmas regiões quentes onde o alarme
dispara, enquanto o health do AE fica alto em 2024 sem discriminar — porque o
autoencoder aprendeu a reconstruir bem o regime quente e o erro CAI justo quando a
temperatura sobe.

Uso:
    PYTHONPATH=. python scripts/plot_baseline_vs_ae_series.py
"""
from __future__ import annotations

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")
bl = _load("baseline_trivial_vs_ae")
fb = _load("plot_frenteB_series")

SENSOR = "TC382_03_A"
RAW = bl.RAW
OUT = "eval_predictive_out/fig_baseline_vs_ae_TC382_03_A.png"
T0, T1 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")
SETPOINT = 760.0

INK, INK_MUTED, GRID = fb.INK, fb.INK_MUTED, fb.GRID
SERIES, OFF_BAND = fb.SERIES, fb.OFF_BAND
ALERT, THR, HIT, MISS, FP = fb.ALERT, fb.THR, fb.HIT, fb.MISS, fb.FP
SET_C = "#b8792a"


def main() -> None:
    raw = pd.read_csv(RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    raw = raw[(raw.index >= T0) & (raw.index < T1)]
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")
    on_full = running > 0.5

    inc = sw.incidents_on(running, tc03, T0, T1)
    mae = ev.load_mae_series(Task.get_task(task_id=bl.TASK), [SENSOR])[SENSOR]
    mae = mae[(mae.index >= T0) & (mae.index < T1)]
    t_grid = tc03.reindex(mae.index, method="nearest")

    esperado = (pd.read_csv("eval_predictive_out/baseline_trivial_vs_ae.csv")
                .query("janela == 'FULL jan/24→abr/26'").set_index("braco"))

    arms = {}
    for key, score, label in [
        ("ae", mae, "AUTOENCODER — health = rank da EWMA do erro de reconstrução"),
        ("temp", t_grid, "LIMIAR TRIVIAL — health = rank da EWMA da própria temperatura"),
    ]:
        r = bl.best_over_hl(score.dropna(), inc, running)
        hl, q = float(r["hl"]), float(r["threshold_q"])
        h = sw.ewma_on(score.dropna(), hl, running).rank(pct=True)
        exp = esperado.loc[[i for i in esperado.index if i.startswith(key)][0]]
        if not (abs(r["recall_raw"] - exp.recall_raw) < 0.01
                and abs(r["fa_per_day"] - exp.fa_per_day) < 0.01):
            raise SystemExit(f"{key}: ponto não reproduz o baseline — abortando.")
        hits, misses = fb.raw_hits(h, q, inc)
        alert = ev.apply_sticky(h, q, sw.STICKY)
        matched, fps = fb.classify_episodes(alert, inc)
        total_days = (h.index[-1] - h.index[0]).total_seconds() / 86400.0
        assert abs(len(fps) / total_days - r["fa_per_day"]) < 0.005, f"{key}: FA não bate"
        arms[key] = dict(h=h, q=q, hits=hits, misses=misses, matched=matched, fps=fps,
                         label=label, rr=r["recall_raw"], fa=r["fa_per_day"], hl=hl)
        print(f"{key:<5} hl={hl} q={q:.4f}  raw={r['recall_raw']:.1%} fa={r['fa_per_day']:.3f}")

    step = max(1, len(tc03) // 20000)
    temp, on_p = tc03.iloc[::step], on_full.iloc[::step]

    fig, axes = plt.subplots(4, 1, figsize=(13.5, 9.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.9, 0.12, 1, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    fb.shade_off(ax, on_p)
    ax.plot(temp.index, temp.values, lw=0.55, color=INK, alpha=0.85, zorder=2)
    ax.axhline(SETPOINT, color=SET_C, lw=1.1, ls="--", zorder=3)
    ax.annotate(f"setpoint HI {SETPOINT:.0f} °C  —  é ISTO que o alarme do DCS mede",
                xy=(0.004, 0.80), xycoords="axes fraction", fontsize=8, color=SET_C,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
    fb.style(ax)
    ax.set_ylabel("TC382_03_A (°C)", fontsize=9, color=INK)
    ax.set_title("O limiar na própria temperatura supera o autoencoder — "
                 f"janela FULL jan/2024 → abr/2026, {len(inc)} incidentes HI/HIHI, "
                 "um ponto de operação por braço", fontsize=11, color=INK, loc="left")

    lane = axes[1]
    spans = fb.alarm_active_spans(temp.index.min(), temp.index.max())
    for a, b in spans:
        lane.axvspan(a, max(b, a + pd.Timedelta(hours=24)), ymin=0.15, ymax=0.85,
                     color=MISS, alpha=0.9, lw=0)
    lane.set_ylim(0, 1)
    lane.set_yticks([])
    lane.set_ylabel("alarme\nDCS", fontsize=8, color=MISS, rotation=0,
                    ha="right", va="center", labelpad=12)
    for s in ("top", "right", "left"):
        lane.spines[s].set_visible(False)
    lane.spines["bottom"].set_color(GRID)
    lane.tick_params(colors=INK_MUTED, labelsize=8, length=3)

    for ax, key in zip(axes[2:], ["ae", "temp"]):
        a = arms[key]
        fb.shade_off(ax, on_p)
        min_w = pd.Timedelta(hours=24)
        for s0, s1 in a["matched"]:
            ax.axvspan(s0, max(s1, s0 + min_w), color=ALERT, alpha=0.8, lw=0, zorder=1)
        for s0, s1 in a["fps"]:
            ax.axvspan(s0, max(s1, s0 + min_w), color=FP, alpha=0.55, lw=0, zorder=1)
        ax.plot(a["h"].index, a["h"].values, lw=0.6, color=SERIES)
        ax.axhline(a["q"], color=THR, lw=1.1, ls="--")
        for t in a["hits"]:
            ax.axvline(t, color=HIT, lw=1.2, zorder=3)
        for t in a["misses"]:
            ax.axvline(t, color=MISS, lw=1.2, zorder=3)
        fb.style(ax)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("health index", fontsize=9, color=INK)
        vence = key == "temp"
        ax.text(0.005, 0.96,
                f"{a['label']}   —   raw {a['rr']:.1%} ({len(a['hits'])}/{len(inc)})"
                f"  ·  FA {a['fa']:.3f}/dia ({len(a['fps'])} episódios FP)  ·  hl={a['hl']}",
                transform=ax.transAxes, fontsize=8.8, color=INK, va="top",
                bbox=dict(facecolor="#eaf5ee" if vence else "white",
                          edgecolor=HIT if vence else GRID,
                          boxstyle="round,pad=0.28", alpha=0.95))

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[
        Patch(facecolor=MISS, alpha=0.9, label="alarme HI/HIHI ativo no DCS (pista própria)"),
        Line2D([], [], color=HIT, lw=1.4, label="incidente detectado (cruzamento bruto, 8h)"),
        Line2D([], [], color=MISS, lw=1.4, label="incidente perdido"),
        Line2D([], [], color=THR, lw=1.1, ls="--", label="threshold (1 por braço)"),
        Patch(facecolor=ALERT, label="alerta que antecede incidente"),
        Patch(facecolor=FP, alpha=0.55, label="alerta FALSO POSITIVO"),
        Patch(facecolor=OFF_BAND, alpha=0.6, label="equipamento OFF"),
    ], loc="lower right", fontsize=7.5, framealpha=0.95, ncol=3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
