#!/usr/bin/env python3
"""
plot_ngp_agosto.py
Auditoria do NGP_A em AGOSTO, nos dois anos disponíveis — e o achado que ela produz.

  ago/2024 (Interpolados, 30s, 94 valores distintos): NGP é VÁLIDO e concorda
           100,000% com RUNNING_A. Máquina parada quase o mês inteiro (ON 4,1%).
  ago/2025 (record): o export cobre APENAS 01/08 03:00→11:39 — 8h39 de um mês de
           744h. Os 994 registros não são um mês de dados; são uma manhã.

⚠️ O `sensores_filtrados_record_2025.csv` é um export TRUNCADO: quase toda tag traz
~1000 registros concentrados no PRIMEIRO DIA de cada mês (NGP, TC382_03, T5, NPT
idênticos nisso; só jan e mai fogem, com 8 e 23 dias). Não é sensor congelado nem
export por exceção — é limite de linhas na extração. Preencher com ffill produz
absurdo: carrega NGP=88 por dias em que o TC03 está a 30°C.

Consequência: o NGP_A NÃO pode auditar os falsos positivos de agosto/2025 — não há
dado. Para 2025 ele só cobre jan (8 dias) e mai (23 dias).

Uso:
    PYTHONPATH=. python scripts/plot_ngp_agosto.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

CSV_2024 = "../dados/sensores_filtrados_Interpolados_2024.csv"
CSV_2025_NGP = "../dados/sensores_filtrados_record_2025.csv"
CSV_2025_DENSE = "../dados/sensores_brutos_2025_2026_30s.csv"
OUT = "eval_predictive_out/fig_ngp_agosto_2024_2025.png"

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
C_NGP, C_TEMP, BAD, OFF_BAND = "#1baf7a", "#0b0b0b", "#d03b3b", "#d8d7d2"


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
    # ---- agosto/2024 (NGP denso e válido)
    d24 = pd.read_csv(CSV_2024, usecols=["data_datetime", "NGP_A", "RUNNING_A", "TC382_03_A"],
                      low_memory=False)
    d24["data_datetime"] = pd.to_datetime(d24["data_datetime"], errors="coerce", utc=True)
    d24 = d24.dropna(subset=["data_datetime"]).set_index("data_datetime").apply(
        pd.to_numeric, errors="coerce").sort_index()
    a24 = d24[(d24.index >= "2024-08-01") & (d24.index < "2024-09-01")].dropna(subset=["NGP_A"])

    # ---- agosto/2025 (NGP do record + TC03/RUNNING densos do brutos)
    g = pd.read_csv(CSV_2025_NGP, usecols=["data_datetime", "NGP_A"], low_memory=False)
    g["data_datetime"] = pd.to_datetime(g["data_datetime"], errors="coerce", utc=True)
    g = g.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    g["NGP_A"] = pd.to_numeric(g["NGP_A"], errors="coerce")
    g = g.dropna()
    b = pd.read_csv(CSV_2025_DENSE, usecols=["data_datetime", "RUNNING_A", "TC382_03_A"],
                    low_memory=False)
    b["data_datetime"] = pd.to_datetime(b["data_datetime"], format="ISO8601", utc=True)
    b = b.set_index("data_datetime").apply(pd.to_numeric, errors="coerce").sort_index()
    a25 = b[(b.index >= "2025-08-01") & (b.index < "2025-09-01")].copy()
    ngp25 = g[(g.index >= "2025-08-01") & (g.index < "2025-09-01")]["NGP_A"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 6.4),
                             gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("white")

    for col, (label, ngp, temp, on, valid) in enumerate([
        ("agosto/2024 — NGP VÁLIDO (94 valores distintos)",
         a24["NGP_A"], a24["TC382_03_A"], a24["RUNNING_A"] > 0.5, True),
        ("agosto/2025 — NGP só existe em 01/08, 03:00→11:39 (8h39 de 744h)",
         ngp25, a25["TC382_03_A"], a25["RUNNING_A"] > 0.5, False),
    ]):
        ax = axes[0][col]
        shade_off(ax, on)
        ax.plot(ngp.index, ngp.values, lw=1.4, color=C_NGP if valid else BAD)
        ax.axhline(50, color=BAD, lw=1.0, ls="--")
        ax.set_ylim(-5, 105)
        # eixo x igual ao painel de baixo, para a ausência de dado ficar visível
        ax.set_xlim(temp.index.min(), temp.index.max())
        style(ax)
        ax.set_title(label, fontsize=10, color=INK if valid else BAD, loc="left")
        if col == 0:
            ax.set_ylabel("NGP_A (% rotação)", fontsize=9, color=INK)
        ax.annotate("limiar 50", xy=(temp.index.min(), 50), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=BAD)
        if not valid:
            ax.axvspan(ngp.index.max(), temp.index.max(), color=BAD, alpha=0.07, lw=0)
            ax.annotate("SEM DADO — export truncado no 1º dia do mês",
                        xy=(0.55, 0.5), xycoords="axes fraction", fontsize=9.5,
                        color=BAD, ha="center", fontweight="bold")

        ax = axes[1][col]
        shade_off(ax, on)
        ax.plot(temp.index, temp.values, lw=0.6, color=C_TEMP, alpha=0.9)
        style(ax)
        if col == 0:
            ax.set_ylabel("TC382_03_A (°C)", fontsize=9, color=INK)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        frac_on = float(on.mean())
        ax.text(0.01, 0.93, f"faixa cinza = RUNNING_A OFF ({1 - frac_on:.0%} do mês)",
                transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="top")

    fig.suptitle("NGP_A como árbitro de 'equipamento ligado' — por que agosto/2025 não pode ser auditado",
                 fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
