#!/usr/bin/env python3
"""
plot_tc03_por_ano.py
Série do TC382_03_A ano a ano (2022→2026), para responder a olho a pergunta que a
métrica não responde: **em jun–dez/2024 a turbina estava em regime degradado?**

Cada painel traz a faixa min–max por bin de 30 min (não a decimação simples, que
esconderia excursão), a mediana, o setpoint HI de 760 °C, a máscara de desligado e
os onsets HI/HIHI reais do DCS numa pista própria. O rótulo de cada ano mostra a
taxa de alarme POR DIA LIGADO — que é a comparação justa, já que a máquina roda
número muito diferente de dias em cada ano.

Fontes (fusos diferentes, ver memória):
  2022–2023  sensores_filtrados_Interpolados_YYYY.csv   — já em UTC
  2024–2026  sensores_full_2024_2026_30s.csv            — já em UTC (remessa +3h aplicado)

Uso:
    PYTHONPATH=. python scripts/plot_tc03_por_ano.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = "../dados/"
FULL = D + "sensores_full_2024_2026_30s.csv"
ALARM = D + "alarmes_selecionados_turbina_a.csv"
OUT = "eval_predictive_out/fig_tc03_por_ano.png"
SENSOR = "TC382_03_A"
SETPOINT_HI = 760.0
BIN = "30min"

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
SERIES, BAND, OFF_BAND = "#1f4e79", "#a8c4de", "#dcdcd8"
ALARM_C, SET_C = "#c0392b", "#b8792a"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def load_year(year: int) -> pd.DataFrame:
    if year >= 2024:
        d = pd.read_csv(FULL, usecols=["data_datetime", SENSOR, "RUNNING_A"], low_memory=False)
    else:
        d = pd.read_csv(f"{D}sensores_filtrados_Interpolados_{year}.csv",
                        usecols=["data_datetime", SENSOR, "RUNNING_A"], low_memory=False)
    d["data_datetime"] = pd.to_datetime(d["data_datetime"], utc=True, errors="coerce")
    d = d.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    d = d.apply(pd.to_numeric, errors="coerce")
    return d.loc[str(year)]


def main() -> None:
    al = pd.read_csv(ALARM)
    al = al[(al["Tag Alarme"] == SENSOR) & (al["Condição do Alarme"].isin(["HI", "HIHI"]))].copy()
    al["_t"] = pd.to_datetime(al["Data da Ocorrência"], errors="coerce", utc=True)
    al = al.dropna(subset=["_t"])

    years = [2022, 2023, 2024, 2025, 2026]
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 13.5))
    fig.patch.set_facecolor("white")

    for ax, yr in zip(axes, years):
        d = load_year(yr)
        v = d[SENSOR]
        g = v.resample(BIN)
        lo, hi, med = g.min(), g.max(), g.median()
        hot = (v > 500)
        dias_on = float(hot.sum()) * 30 / 86400
        a_yr = al[al["_t"].dt.year == yr]
        # taxa só faz sentido com massa de operação; em 2023 (0,3 dia quente) a
        # divisão explode e produziria um "31 alarmes/dia" sem significado
        taxa = len(a_yr) / dias_on if dias_on >= 5 else float("nan")
        t_med = float(v[hot].median()) if hot.any() else float("nan")

        # faixa de desligado (árbitro físico, imune ao RUNNING_A)
        on_b = hot.resample(BIN).max().fillna(False).astype(bool)
        blk = (on_b != on_b.shift()).cumsum()
        for _, grp in on_b.groupby(blk):
            if not bool(grp.iloc[0]):
                ax.axvspan(grp.index[0], grp.index[-1], color=OFF_BAND, alpha=0.65, lw=0, zorder=0)

        ax.fill_between(lo.index, lo.values, hi.values, color=BAND, lw=0, alpha=0.85, zorder=2)
        ax.plot(med.index, med.values, lw=0.5, color=SERIES, zorder=3)
        ax.axhline(SETPOINT_HI, color=SET_C, lw=1.0, ls="--", zorder=4)

        # pista de alarmes
        for t in a_yr["_t"]:
            ax.plot([t, t], [845, 900], color=ALARM_C, lw=1.3, solid_capstyle="butt", zorder=5)

        ax.set_xlim(pd.Timestamp(f"{yr}-01-01", tz="UTC"), pd.Timestamp(f"{yr+1}-01-01", tz="UTC"))
        ax.set_ylim(-80, 910)
        style(ax)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_ylabel(f"{yr}\n°C", fontsize=10, color=INK, rotation=0,
                      ha="right", va="center", labelpad=26, weight="bold")
        destaque = (yr == 2024)
        txt_taxa = "sem operação" if np.isnan(taxa) else f"{taxa:.2f} alarme por dia ligado"
        txt_t = "—" if np.isnan(t_med) else f"{t_med:.0f} °C"
        ax.text(0.006, 0.94,
                f"{dias_on:.0f} dias quente   ·   {len(a_yr)} alarmes HI/HIHI   ·   {txt_taxa}"
                f"   ·   T mediana em operação: {txt_t}",
                transform=ax.transAxes, fontsize=9, va="top",
                color=ALARM_C if destaque else INK_MUTED,
                weight="bold" if destaque else "normal")
        if yr == 2022:
            ax.annotate(f"setpoint HI {SETPOINT_HI:.0f} °C", xy=(0.60, 0.86),
                        xycoords="axes fraction", fontsize=7.5, color=SET_C)
            ax.annotate("barras vermelhas = alarme HI/HIHI no DCS", xy=(0.60, 0.955),
                        xycoords="axes fraction", fontsize=7.5, color=ALARM_C)
            ax.annotate("faixa cinza = equipamento frio (TC03 < 500 °C)", xy=(0.60, 0.055),
                        xycoords="axes fraction", fontsize=7.5, color=INK_MUTED)

    fig.suptitle("TC382_03_A ano a ano — a faixa azul é o intervalo min–max em janelas de 30 min",
                 fontsize=12.5, color=INK, y=0.997, x=0.5)
    fig.text(0.5, 0.978, "2024 concentra 3,5× mais alarme por dia de operação que 2025; "
                         "jun–dez/2024 chega a 0,68/dia — a máquina alarmava a cada dia e meio",
             fontsize=9.5, color=INK_MUTED, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
