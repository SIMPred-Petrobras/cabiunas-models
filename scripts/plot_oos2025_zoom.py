#!/usr/bin/env python3
"""
plot_oos2025_zoom.py
Zoom de ~4 dias em torno de incidentes do OOS 2025 do TC382_03_A (v10): mostra o
mecanismo que o panorama só afirma — o health cruzando o threshold horas antes do
HI/HIHI. Mesmo ponto de operação da auditoria (hl=4h, q=0,8397), mesma sanidade.

Uso:
    PYTHONPATH=. python scripts/plot_oos2025_zoom.py            # 3 incidentes espaçados
    PYTHONPATH=. python scripts/plot_oos2025_zoom.py --n 4 --days 4
"""
from __future__ import annotations

import argparse
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
OUT = "eval_predictive_out/fig_oos2025_TC382_03_A_zoom.png"
HL, Q = 4.0, 0.839655
T0 = pd.Timestamp("2025-07-01", tz="UTC")

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
SERIES, OFF_BAND = "#2a78d6", "#d8d7d2"
ALERT, THR, CRIT = "#f6e3b4", "#b3541e", "#d03b3b"
SETPOINT = 760.0


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=7, length=3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--days", type=float, default=4.0)
    args = p.parse_args()

    running, tc03, _ = sw.load_raw()
    mae = sw.read_mae(MAE_FILE)
    mae = mae[mae.index >= T0]
    inc = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())
    h = sw.ewma_on(mae, HL, running).rank(pct=True)
    rr, fa, duty = sw.metrics_at(h, Q, inc)
    assert abs(rr - 1.0) < 0.01, "não reproduz a auditoria"
    alert = ev.apply_sticky(h, Q, sw.STICKY)

    picks = [inc[i] for i in np.linspace(0, len(inc) - 1, args.n).astype(int)]
    half = pd.Timedelta(days=args.days / 2)
    on = running > 0.5

    fig, axes = plt.subplots(2, args.n, figsize=(4.6 * args.n, 5.4), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("white")
    for j, t in enumerate(picks):
        w0, w1 = t - half, t + half
        ax_t, ax_h = axes[0][j], axes[1][j]

        tw = tc03[(tc03.index >= w0) & (tc03.index <= w1)].dropna()
        ax_t.plot(tw.index, tw.values, lw=0.8, color=INK, alpha=0.9)
        ax_t.axhline(SETPOINT, color=INK_MUTED, lw=0.9, ls=":")
        ax_t.annotate("setpoint HI 760°C", xy=(tw.index[0], SETPOINT), xytext=(2, 4),
                      textcoords="offset points", fontsize=6.5, color=INK_MUTED)
        ax_t.axvline(t, color=CRIT, lw=1.3)
        style(ax_t)
        if j == 0:
            ax_t.set_ylabel("TC382_03_A (°C)", fontsize=8.5, color=INK)
        ax_t.set_title(str(t)[:16] + " UTC", fontsize=9, color=INK)

        hw = h[(h.index >= w0) & (h.index <= w1)]
        aw = alert[(alert.index >= w0) & (alert.index <= w1)]
        blk = (aw != aw.shift()).cumsum()
        for _, g in aw.groupby(blk):
            if bool(g.iloc[0]):
                ax_h.axvspan(g.index[0], g.index[-1], color=ALERT, alpha=0.85, lw=0)
        ax_h.plot(hw.index, hw.values, lw=0.9, color=SERIES)
        ax_h.axhline(Q, color=THR, lw=1.1, ls="--")
        ax_h.axvline(t, color=CRIT, lw=1.3)
        # lead: primeiro cruzamento bruto dentro do horizonte de 8h
        raw = hw.index[(hw >= Q) & (hw.index >= t - pd.Timedelta(hours=sw.HORIZON))
                       & (hw.index <= t)]
        if len(raw):
            lead_h = (t - raw[0]).total_seconds() / 3600.0
            ax_h.annotate(f"alerta {lead_h:.1f}h antes", xy=(raw[0], Q),
                          xytext=(4, -14), textcoords="offset points",
                          fontsize=7.5, color=THR, fontweight="bold")
        style(ax_h)
        ax_h.set_ylim(0, 1.02)
        if j == 0:
            ax_h.set_ylabel("health index", fontsize=8.5, color=INK)
        ax_h.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax_h.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
        for lb in ax_h.get_xticklabels():
            lb.set_rotation(30)
            lb.set_ha("right")

    fig.suptitle("TC382_03_A — zoom nos incidentes OOS (v10, hl=4h, q=0,840): "
                 "alerta amarelo ativo antes do HI/HIHI (linha vermelha)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
