#!/usr/bin/env python3
"""
plot_fp_agosto2025.py
Anatomia dos falsos positivos de agosto/2025 no TC382_03_A (braço b2024).

Agosto tem 6 episódios FP; 5 são pós-partida, mas o de 17–18/08 dura 28h com
404h desde a última partida — operação estável, sem transiente para culpar. O
diagnóstico (ver painel do meio) é REGIME DE CARGA BAIXA: os 6 termopares caem
~150°C juntos e o ΔP do gás combustível (PDI_0317, proxy de carga) cai 70%. O
spread TC03−irmãos não muda (+21,8→+24,0°C), então NÃO há anomalia do sensor —
o AE só nunca viu operação sustentada nesse patamar e marca como anômalo.

Uso:
    PYTHONPATH=. CLEARML_CONFIG_FILE=$(pwd)/clearml.conf python scripts/plot_fp_agosto2025.py
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
pf = _load("plot_frenteB_series")

T0 = pd.Timestamp("2025-08-14", tz="UTC")
T1 = pd.Timestamp("2025-09-01", tz="UTC")
OUT = "eval_predictive_out/fig_fp_agosto2025_TC382_03_A.png"

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
SERIES, OFF_BAND = "#2a78d6", "#d8d7d2"
ALERT, THR, FP = "#f6e3b4", "#b3541e", "#ec835a"
LOAD = "#1baf7a"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def main() -> None:
    cols = ["data_datetime", "RUNNING_A", "TC382_03_A", "954005_624_PDI_0317"] + \
           [f"TC382_0{i}_A" for i in (1, 2, 4, 5, 6)]
    d = pd.read_csv("../dados/sensores_brutos_2025_2026_30s.csv", low_memory=False,
                    usecols=lambda c: c in set(cols))
    d["data_datetime"] = pd.to_datetime(d["data_datetime"], format="ISO8601", utc=True)
    d = d.set_index("data_datetime").apply(pd.to_numeric, errors="coerce").sort_index()
    w = d[(d.index >= T0) & (d.index <= T1)]

    running, tc03, _ = sw.load_raw()
    row = pd.read_csv("eval_predictive_out/fleet_v13_b2024_FULL_hihihi.csv").set_index("sensor").loc["TC382_03_A"]
    mae = ev.load_mae_series(Task.get_task(task_id="1a15c26d994e44febb77f0bec8c2b378"),
                             ["TC382_03_A"])["TC382_03_A"]
    q = float(row["threshold_q"])
    h = sw.ewma_on(mae, float(row["hl"]), running).rank(pct=True)
    inc = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())
    alert = ev.apply_sticky(h, q, sw.STICKY)
    matched, fps = pf.classify_episodes(alert, inc)

    hw = h[(h.index >= T0) & (h.index <= T1)]
    on_w = (w["RUNNING_A"] > 0.5)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    fig.patch.set_facecolor("white")

    def bands(ax):
        blk = (on_w != on_w.shift()).cumsum()
        for _, g in on_w.groupby(blk):
            if not bool(g.iloc[0]):
                ax.axvspan(g.index[0], g.index[-1], color=OFF_BAND, alpha=0.5, lw=0, zorder=0)
        for s0, s1 in fps:
            if s1 >= T0 and s0 <= T1:
                ax.axvspan(max(s0, T0), min(s1, T1), color=FP, alpha=0.45, lw=0, zorder=1)
        for s0, s1 in matched:
            if s1 >= T0 and s0 <= T1:
                ax.axvspan(max(s0, T0), min(s1, T1), color=ALERT, alpha=0.7, lw=0, zorder=1)

    # painel 1 — os 6 termopares juntos
    ax = axes[0]
    bands(ax)
    for c in [f"TC382_0{i}_A" for i in (1, 2, 4, 5, 6)]:
        ax.plot(w.index, w[c].values, lw=0.5, color=INK_MUTED, alpha=0.35, zorder=2)
    ax.plot(w.index, w["TC382_03_A"].values, lw=1.0, color=INK, zorder=3)
    ax.axhline(760, color=INK_MUTED, lw=0.9, ls=":")
    ax.annotate("setpoint HI 760°C", xy=(T0, 760), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color=INK_MUTED)
    style(ax)
    ax.set_ylabel("termopares (°C)", fontsize=9, color=INK)
    ax.set_title("Falsos positivos de agosto/2025 — TC382_03_A (preto) e os 5 irmãos (cinza). "
                 "Os 6 caem juntos: é carga, não sensor.", fontsize=10.5, color=INK, loc="left")

    # painel 2 — proxy de carga
    ax = axes[1]
    bands(ax)
    ax.plot(w.index, w["954005_624_PDI_0317"].values, lw=0.9, color=LOAD, zorder=2)
    style(ax)
    ax.set_ylabel("PDI_0317 (ΔP gás comb.)\nproxy de carga", fontsize=8.5, color=INK)

    # painel 3 — health
    ax = axes[2]
    bands(ax)
    ax.plot(hw.index, hw.values, lw=0.9, color=SERIES, zorder=2)
    ax.axhline(q, color=THR, lw=1.1, ls="--")
    ax.annotate(f"q={q:.3f}", xy=(T1, q), xytext=(-4, 5), textcoords="offset points",
                fontsize=8, color=THR, ha="right")
    style(ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("health index", fontsize=9, color=INK)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(facecolor=FP, alpha=0.45, label="episódio FALSO POSITIVO"),
        Patch(facecolor=ALERT, alpha=0.7, label="alerta que antecede incidente"),
        Patch(facecolor=OFF_BAND, alpha=0.6, label="equipamento OFF"),
        Line2D([], [], color=THR, lw=1.1, ls="--", label="threshold"),
    ], loc="upper left", fontsize=7.5, framealpha=0.95, ncol=2)

    fig.text(0.995, 0.005,
             "17–18/08: 28h de alerta, 404h desde a última partida · 6 termopares −150°C · "
             "PDI_0317 −70% · spread TC03−irmãos inalterado (+21,8→+24,0°C)",
             ha="right", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
