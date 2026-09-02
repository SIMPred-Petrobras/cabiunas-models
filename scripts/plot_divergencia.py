#!/usr/bin/env python3
"""
plot_divergencia.py
Inspeção visual dos episódios achados por `divergencia_termopares.py`: os 6
termopares do array TC382 crus, e os mesmos centrados pela própria mediana, no
mesmo eixo de tempo.

O painel de baixo é o que importa. Se os seis sobem e descem juntos, é carga —
e o resíduo fica plano. Se um se descola, é divergência local. Se vários se
descolam em direções opostas, é o perfil de temperatura da exaustão que mudou,
que é justamente o que limiar sobre um sensor sozinho não vê.

Uso:
    PYTHONPATH=. python scripts/plot_divergencia.py 2025-08-15 2025-08-28
    PYTHONPATH=. python scripts/plot_divergencia.py 2026-03-30 2026-04-07 --out fig.png
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

GRUPO = [f"TC382_0{i}_A" for i in range(1, 7)]
VALID_LOW, VALID_HIGH = -30.0, 1200.0
# seis matizes bem separados; nenhum par adjacente confundível em cinza
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c9308e", "#8a6d1f"]
COR_OFF = "#e4e7eb"


def _dados() -> str:
    for up in ("..", "../..", "../../.."):
        c = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", up, "dados"))
        if os.path.isdir(c):
            return c
    raise SystemExit("diretório 'dados/' não encontrado.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inicio")
    p.add_argument("fim")
    p.add_argument("--out", default=None)
    return p.parse_args()


def spans_off(on: pd.Series, min_minutes: float = 20.0):
    if on.empty:
        return []
    off = (~on).to_numpy()
    idx = on.index
    corte = np.flatnonzero(off[1:] != off[:-1]) + 1
    out = []
    for a, b in zip(np.concatenate(([0], corte)), np.concatenate((corte, [len(off)]))):
        if not off[a]:
            continue
        ta, tb = idx[a], idx[min(b, len(idx) - 1)]
        if (tb - ta) >= pd.Timedelta(minutes=min_minutes):
            out.append((ta, tb))
    return out


def main() -> None:
    a = parse_args()
    t0 = pd.Timestamp(a.inicio, tz="UTC")
    t1 = pd.Timestamp(a.fim, tz="UTC")
    D = _dados()
    df = pd.read_csv(os.path.join(D, "sensores_2024h2_2025_2026_30s.csv"),
                     usecols=["data_datetime", "RUNNING_A", *GRUPO], low_memory=False)
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    on_all = pd.to_numeric(df["RUNNING_A"], errors="coerce") > 0.5
    T_all = df[GRUPO].apply(pd.to_numeric, errors="coerce")
    T_all = T_all.where((T_all >= VALID_LOW) & (T_all <= VALID_HIGH))

    # centro de cada sensor pela mediana do resíduo em TODA a série ON — a mesma
    # referência do script de detecção, para o recorte não redefinir o zero
    centros = {}
    for s in GRUPO:
        irm = [c for c in GRUPO if c != s]
        r = (T_all[s] - T_all[irm].mean(axis=1)).where(T_all[irm].notna().sum(axis=1) >= 3)
        centros[s] = float(r.where(on_all).median())

    m = (df.index >= t0) & (df.index <= t1)
    T, on = T_all[m], on_all[m]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15.5, 8.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12})
    for x0, x1 in spans_off(on):
        for ax in (ax1, ax2):
            ax.axvspan(x0, x1, color=COR_OFF, lw=0, zorder=0)

    for s, cor in zip(GRUPO, CORES):
        ax1.plot(T.index, T[s], color=cor, lw=0.8, zorder=3, label=s.replace("_A", ""))
    ax1.set_ylabel("temperatura (°C)")
    ax1.legend(loc="upper left", ncol=6, fontsize=9, framealpha=.9)
    ax1.set_title(f"Array TC382 — os seis termopares e o desvio de cada um em relação aos irmãos"
                  f"   ·   {a.inicio} a {a.fim}", loc="left", fontsize=12.5, pad=11)

    for s, cor in zip(GRUPO, CORES):
        irm = [c for c in GRUPO if c != s]
        r = (T[s] - T[irm].mean(axis=1)).where(T[irm].notna().sum(axis=1) >= 3)
        ax2.plot(T.index, (r - centros[s]).where(on), color=cor, lw=1.0, zorder=3)
    ax2.axhline(0, color="#1f2933", lw=0.9, ls="--", zorder=4)
    ax2.set_ylabel("desvio em relação aos irmãos\n(°C, centrado na mediana histórica)")
    ax2.set_xlabel("tempo (UTC)")

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=13))
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#ffffff", lw=0.8, alpha=.9, zorder=1)
        ax.set_axisbelow(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.09)
    out = a.out or f"eval_predictive_out/fig_divergencia_{a.inicio}_{a.fim}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"gravado: {out}")


if __name__ == "__main__":
    main()
