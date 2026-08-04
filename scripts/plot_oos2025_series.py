#!/usr/bin/env python3
"""
plot_oos2025_series.py
Série temporal do resultado OOS 2025 do TC382_03_A (modelo v10, 17/17 raw):
  painel 1 — temperatura do sensor, com OFF sombreado e os 17 incidentes marcados;
  painel 2 — health index (rank da EWMA do MAE, hl=4h) no ponto de operação da
             auditoria (q=0,8397), com regiões de alerta e hit/miss por incidente.

Offline: usa o MAE em cache identificado por impressão digital no sweep de regime
(`10a2393754b01839ef12c014fe358cda`) e valida antes de plotar que o ponto reproduz
a linha OOS da auditoria (recall_raw=100%, FA≈0,082/dia, duty≈0,235) — se não
reproduzir, aborta em vez de plotar a coisa errada.

Uso:
    PYTHONPATH=. python scripts/plot_oos2025_series.py
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

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")

MAE_FILE = os.path.join(sw.CACHE, "10a2393754b01839ef12c014fe358cda.sequence_scores_all.csv")
OUT = "eval_predictive_out/fig_oos2025_TC382_03_A_serie.png"

# linha OOS da auditoria (fleet_audit_2e92c618_OOS_hihihi.csv, TC382_03_A)
HL, Q = 4.0, 0.839655
EXP_RR, EXP_FA, EXP_DUTY = 1.0, 0.081622, 0.234526
T0 = pd.Timestamp("2025-07-01", tz="UTC")

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
SERIES, OFF_BAND = "#2a78d6", "#d8d7d2"
ALERT, THR = "#f6e3b4", "#b3541e"
HIT, MISS = "#0ca30c", "#d03b3b"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def shade_off(ax, on: pd.Series):
    blk = (on != on.shift()).cumsum()
    for _, g in on.groupby(blk):
        if not bool(g.iloc[0]):
            ax.axvspan(g.index[0], g.index[-1], color=OFF_BAND, alpha=0.5, lw=0, zorder=0)


def main() -> None:
    running, tc03, _ = sw.load_raw()
    mae = sw.read_mae(MAE_FILE)
    mae = mae[mae.index >= T0]

    inc = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())
    h = sw.ewma_on(mae, HL, running).rank(pct=True)
    rr, fa, duty = sw.metrics_at(h, Q, inc)
    print(f"Sanidade OOS: recall_raw={rr:.1%} (esp. {EXP_RR:.0%})  "
          f"fa={fa:.3f} (esp. {EXP_FA:.3f})  duty={duty:.3f} (esp. {EXP_DUTY:.3f})")
    if not (abs(rr - EXP_RR) < 0.01 and abs(fa - EXP_FA) < 0.01 and abs(duty - EXP_DUTY) < 0.01):
        raise SystemExit("ponto de operação NÃO reproduz a auditoria — arquivo errado?")

    alert = ev.apply_sticky(h, Q, sw.STICKY)
    raw_s = np.array([t.timestamp() for t in h.index[h >= Q]])
    hs = sw.HORIZON * 3600.0
    hits = [t for t in inc if raw_s.size and
            np.any((raw_s >= t.timestamp() - hs) & (raw_s <= t.timestamp()))]
    misses = [t for t in inc if t not in hits]

    # decimação para plot
    on_full = (running > 0.5)
    step = max(1, len(tc03) // 20000)
    temp = tc03[tc03.index >= T0].iloc[::step]
    on_p = on_full[on_full.index >= T0].iloc[::step]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.15]})
    fig.patch.set_facecolor("white")

    # painel 1 — temperatura
    shade_off(ax1, on_p)
    ax1.plot(temp.index, temp.values, lw=0.6, color=INK, alpha=0.85)
    for t in inc:
        ax1.axvline(t, color=MISS, lw=0.8, alpha=0.55, zorder=2)
    style(ax1)
    ax1.set_ylabel("TC382_03_A (°C)", fontsize=9, color=INK)
    ax1.set_title("OOS jul/2025 → abr/2026 — modelo v10 (2e92c618), ponto da auditoria",
                  fontsize=11, color=INK, loc="left")

    # painel 2 — health index
    shade_off(ax2, on_p)
    # regiões de alerta (pós-sticky)
    blk = (alert != alert.shift()).cumsum()
    for _, g in alert.groupby(blk):
        if bool(g.iloc[0]):
            ax2.axvspan(g.index[0], g.index[-1], color=ALERT, alpha=0.8, lw=0, zorder=1)
    ax2.plot(h.index, h.values, lw=0.7, color=SERIES)
    ax2.axhline(Q, color=THR, lw=1.2, ls="--")
    ax2.annotate(f"q={Q:.3f}", xy=(h.index[-1], Q), xytext=(-4, 5),
                 textcoords="offset points", fontsize=8, color=THR, ha="right")
    for t in hits:
        ax2.axvline(t, color=HIT, lw=1.3, zorder=3)
    for t in misses:
        ax2.axvline(t, color=MISS, lw=1.3, zorder=3)
    style(ax2)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("health index\n(rank EWMA hl=4h)", fontsize=9, color=INK)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Line2D([], [], color=HIT, lw=1.4, label=f"incidente detectado ({len(hits)})"),
        Line2D([], [], color=MISS, lw=1.4, label=f"incidente perdido ({len(misses)})"),
        Line2D([], [], color=THR, lw=1.2, ls="--", label="threshold"),
        Patch(facecolor=ALERT, label="alerta ativo (sticky 12h)"),
        Patch(facecolor=OFF_BAND, alpha=0.6, label="equipamento OFF"),
    ], loc="upper left", fontsize=7.5, framealpha=0.95, ncol=2)

    fig.text(0.995, 0.005,
             f"recall_raw {rr:.0%} ({len(hits)}/{len(inc)})  ·  FA {fa:.3f}/dia  ·  "
             f"duty pós-sticky {duty:.1%}",
             ha="right", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
