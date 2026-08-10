#!/usr/bin/env python3
"""
audit_running_vs_ngp.py
Auditoria cruzada da definição de "equipamento ligado": `RUNNING_A > 0.5` (usado em
TODAS as camadas do projeto) contra `NGP_A > 50` (rotação do gerador de gás, o
árbitro físico que a metodologia antiga usava).

Motivação: o RUNNING_A foi sinalizado como não-confiável pela Petrobras, e toda a
métrica honesta depende dele (máscara ON, denominador de incidentes, duty). Se as
duas definições divergirem, os números mudam.

Cobertura: o NGP_A existe em `sensores_filtrados_Interpolados_2024.csv`
(jun–dez/2024, 98,3% preenchido, 30s) e esparso em `record_2025` (jan–ago/2025).
NÃO existe de set/2025 em diante — limitação registrada.

⚠️ DEFASAGEM DE FUSO: os exports `sensores_filtrados_record_*.csv` estão em UTC-3,
enquanto `sensores_brutos_*` / `sensores_2024h2_*` estão em UTC. Medido com o TC03 como
sonda: erro absoluto mediano 2,32°C em offset 0 contra 0,09°C em −180min. Cruzar as duas
famílias sem `index -= Timedelta(hours=3)` FABRICA discordâncias (foi o que produziu uma
leitura inicial errada de "o NGP erra em 98% dos casos"). O arquivo Interpolados_2024
traz NGP e RUNNING_A na MESMA linha, então a comparação de 2024 é imune a isso.

Saídas: tabela de concordância + `eval_predictive_out/fig_audit_running_vs_ngp.png`.

Uso:
    PYTHONPATH=. python scripts/audit_running_vs_ngp.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_2024 = "../dados/sensores_filtrados_Interpolados_2024.csv"
CSV_2025 = "../dados/sensores_filtrados_record_2025.csv"
OUT = "eval_predictive_out/fig_audit_running_vs_ngp.png"

NGP_ON = 50.0     # limiar clássico do projeto para "gerador de gás girando"
RUN_ON = 0.5

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
C_NGP, C_RUN, C_TEMP = "#1baf7a", "#2a78d6", "#0b0b0b"
DISAGREE, OFF_BAND = "#d03b3b", "#d8d7d2"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def load(path: str, cols: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=lambda c: c in set(cols), low_memory=False)
    d["data_datetime"] = pd.to_datetime(d["data_datetime"], errors="coerce", utc=True)
    d = d.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    return d.apply(pd.to_numeric, errors="coerce")


def report(d: pd.DataFrame, label: str) -> dict:
    m = d["NGP_A"].notna() & d["RUNNING_A"].notna()
    d = d[m]
    run, ngp = d["RUNNING_A"] > RUN_ON, d["NGP_A"] > NGP_ON
    agree = float((run == ngp).mean())
    disc = run != ngp
    print(f"\n=== {label} — {len(d)} amostras com ambos os sinais ===")
    print(f"  ON por RUNNING_A: {run.mean():.1%} | ON por NGP_A>{NGP_ON:.0f}: {ngp.mean():.1%}")
    print(f"  CONCORDÂNCIA: {agree:.4%}  ({int(disc.sum())} amostras discordantes "
          f"= {disc.sum() * 0.5 / 60:.2f}h)")
    print(pd.crosstab(run.map({True: "RUN=ON", False: "RUN=OFF"}),
                      ngp.map({True: "NGP=ON", False: "NGP=OFF"})).to_string())
    if disc.any():
        sub = d[disc]
        print(f"  nos discordantes: NGP p50={sub['NGP_A'].median():.1f}"
              + (f"  TC03 p50={sub['TC382_03_A'].median():.0f}°C" if "TC382_03_A" in sub else ""))
    print(f"  NGP quando RUN=ON : min={d.loc[run, 'NGP_A'].min():.1f} p1={d.loc[run, 'NGP_A'].quantile(.01):.1f}")
    print(f"  NGP quando RUN=OFF: max={d.loc[~run, 'NGP_A'].max():.1f} p99={d.loc[~run, 'NGP_A'].quantile(.99):.1f}")
    return dict(d=d, run=run, ngp=ngp, agree=agree, disc=disc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    cols = ["data_datetime", "NGP_A", "RUNNING_A", "TC382_03_A"]
    r24 = report(load(CSV_2024, cols), "jun–dez/2024 (Interpolados, 30s)")
    r25 = report(load(CSV_2025, cols), "jan–ago/2025 (record, esparso)")

    d, run, ngp, disc = r24["d"], r24["run"], r24["ngp"], r24["disc"]
    step = max(1, len(d) // 12000)
    ds = d.iloc[::step]

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.5]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(ds.index, ds["NGP_A"].values, lw=0.7, color=C_NGP)
    ax.axhline(NGP_ON, color=DISAGREE, lw=1.0, ls="--")
    ax.annotate(f"limiar NGP>{NGP_ON:.0f}", xy=(ds.index[0], NGP_ON), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color=DISAGREE)
    style(ax)
    ax.set_ylabel("NGP_A (% rotação)", fontsize=9, color=INK)
    ax.set_title(f"Auditoria da definição de ON — RUNNING_A vs NGP_A "
                 f"(jun–dez/2024, {len(d)} amostras, concordância {r24['agree']:.4%})",
                 fontsize=11, color=INK, loc="left")

    ax = axes[1]
    if "TC382_03_A" in ds:
        ax.plot(ds.index, ds["TC382_03_A"].values, lw=0.6, color=C_TEMP, alpha=0.85)
    style(ax)
    ax.set_ylabel("TC382_03_A (°C)", fontsize=9, color=INK)

    # pista de estado: as duas definições lado a lado
    ax = axes[2]
    rs, ns = run.iloc[::step], ngp.iloc[::step]
    for y, s, c, lab in [(0.62, rs, C_RUN, "RUNNING_A>0.5"), (0.18, ns, C_NGP, f"NGP_A>{NGP_ON:.0f}")]:
        blk = (s != s.shift()).cumsum()
        for _, g in s.groupby(blk):
            if bool(g.iloc[0]):
                ax.axhspan(y, y + 0.2, xmin=0, xmax=0, color=c)  # placeholder p/ legenda
                ax.fill_between([g.index[0], g.index[-1]], y, y + 0.2, color=c, lw=0)
        ax.text(-0.005, y + 0.1, lab, transform=ax.get_yaxis_transform(), fontsize=7.5,
                color=c, ha="right", va="center")
    for t in d.index[disc]:
        ax.axvline(t, color=DISAGREE, lw=1.6, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.set_ylabel("estado\nligado", fontsize=8, color=INK, rotation=0, ha="right", va="center", labelpad=44)
    ax.text(0.999, 0.9, f"barras vermelhas = discordância ({int(disc.sum())} amostras)",
            transform=ax.transAxes, fontsize=7.5, color=DISAGREE, ha="right", va="top")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"\nFigura: {args.out}")


if __name__ == "__main__":
    main()
