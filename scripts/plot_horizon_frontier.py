#!/usr/bin/env python3
"""
plot_horizon_frontier.py
A pergunta que decide o rumo do projeto, numa figura: **onde o limiar trivial quebra?**

O alarme do DCS é `TC382_03_A > 760 °C`, e em 8 h de antecedência um limiar na própria
temperatura já ganha do autoencoder (81,0% × 62,0%). Se o ML tem valor aqui, ele tem de
aparecer no HORIZONTE LONGO — o limiar só sabe que está quente agora, enquanto um preditor
com carga e contexto poderia segurar a 24 h ou 72 h.

Lê `eval_predictive_out/forecast_crossing_horizon.csv`.

  linha de cima   recall_raw × horizonte, um painel por janela — a fronteira
  linha de baixo  FA/dia × horizonte, mesma escala de x — o custo do recall

⚠️ Cada painel é uma comparação braço-contra-braço DENTRO de um horizonte. A curva subir
com H **não** é o modelo melhorando: a janela de crédito é maior. O que se lê é a
DISTÂNCIA VERTICAL entre um braço supervisionado e o braço trivial, e se ela se abre.

Identidade por cor E por rótulo direto no fim de cada linha — nenhuma série depende só
da cor para ser identificada.

Uso:
    PYTHONPATH=. python scripts/plot_horizon_frontier.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = "eval_predictive_out/forecast_crossing_horizon.csv"
FLOOR = "eval_predictive_out/forecast_crossing_chancefloor.csv"
OUT = "eval_predictive_out/fig_horizon_frontier_TC382_03_A.png"
FLOOR_C = "#c0392b"

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"

# A0 é a BARRA (tracejado escuro), REF é contexto congelado (pontilhado cinza) e os três
# braços supervisionados formam a família categórica.
ARMS = {
    "A0 trivial (limiar de T)": dict(c=INK,       ls="--", lw=2.2, z=5, lab="A0 limiar trivial"),
    "A1 logística":             dict(c="#2a78d6", ls="-",  lw=1.7, z=3, lab="A1 logística"),
    "A2 GBM":                   dict(c="#b3541e", ls="-",  lw=2.4, z=4, lab="A2 GBM"),
    "A3 GBM + AE":              dict(c="#6b3fa0", ls="-",  lw=1.7, z=3, lab="A3 GBM+AE"),
    "REF autoencoder":          dict(c="#8a8886", ls=":",  lw=1.8, z=2, lab="REF autoencoder"),
}


def spread_labels(ax, items, min_gap_frac=0.062):
    """Rótulo direto no fim de cada linha, empurrado na vertical para não colidir.

    Os braços convergem justamente à direita — que é onde os rótulos ficam —, então sem
    isso três deles se sobrepõem e a figura perde a identificação que não depende de cor.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    gap = min_gap_frac * span
    items = sorted(items, key=lambda it: it[0])
    ys = [it[0] for it in items]
    for i in range(1, len(ys)):                       # empurra para cima
        ys[i] = max(ys[i], ys[i - 1] + gap)
    excedente = ys[-1] - (hi - 0.02 * span)
    if excedente > 0:
        ys = [y - excedente for y in ys]
        for i in range(len(ys) - 2, -1, -1):          # e reacomoda para baixo
            ys[i] = min(ys[i], ys[i + 1] - gap)
    for y, (y0, x0, lab, color) in zip(ys, items):
        ax.annotate(lab, xy=(x0, y), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=7.6, color=color, zorder=6,
                    annotation_clip=False)
        if abs(y - y0) > gap * 0.3:      # linha-guia só quando o rótulo saiu do lugar
            ax.plot([x0, x0], [y0, y], color=color, lw=0.6, alpha=0.5,
                    clip_on=False, zorder=5)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def main() -> None:
    df = pd.read_csv(SRC)
    floor = pd.read_csv(FLOOR) if os.path.exists(FLOOR) else None
    janelas = list(dict.fromkeys(df.janela))
    hs = sorted(df.H.unique())

    fig, axes = plt.subplots(2, len(janelas), figsize=(4.0 * len(janelas), 7.4),
                             sharex=True, sharey="row")
    fig.patch.set_facecolor("white")
    if len(janelas) == 1:
        axes = axes.reshape(2, 1)

    for j, wlab in enumerate(janelas):
        d = df[df.janela == wlab]
        n_inc = int(d.n_inc.iloc[0])
        for row, (col, ylab, pct) in enumerate(
                [("recall_raw", "recall_raw", True), ("fa_per_day", "falsos alarmes / dia", False)]):
            ax = axes[row, j]
            # O PISO DO ACASO, onde foi medido. Sem esta faixa a figura mente: a 72 h a
            # curva sobe, mas o chão sobe junto — um modelo com rótulo embaralhado chega
            # a empatar com o limiar. Só o que passa do topo da faixa é habilidade.
            if row == 0 and floor is not None and wlab in set(floor.janela):
                fw = floor[floor.janela == wlab].set_index("H").reindex(hs)
                ax.fill_between(hs, 0, fw.rotulo_embaralhado.values * 100,
                                color=FLOOR_C, alpha=0.09, lw=0, zorder=0)
                ax.plot(hs, fw.rotulo_embaralhado.values * 100, color=FLOOR_C, lw=1.1,
                        ls=(0, (2, 2)), alpha=0.75, zorder=1)
                ax.plot(hs, fw.piso_ruido.values * 100, color=FLOOR_C, lw=0.9,
                        ls=(0, (1, 2.5)), alpha=0.6, zorder=1)
                ax.annotate("piso do acaso\n(rótulo embaralhado)",
                            xy=(hs[0], fw.rotulo_embaralhado.values[0] * 100),
                            xytext=(3, -13), textcoords="offset points",
                            fontsize=6.8, color=FLOOR_C, va="top", zorder=6)
            marcas = []
            for name, st in ARMS.items():
                s = d[d.braco == name].set_index("H")[col].reindex(hs)
                if s.isna().all():
                    continue
                y = s.values * (100 if pct else 1)
                ax.plot(hs, y, color=st["c"], ls=st["ls"], lw=st["lw"], zorder=st["z"],
                        marker="o", ms=4.5, mfc="white", mew=1.4, mec=st["c"])
                marcas.append((y[-1], hs[-1], st["lab"], st["c"]))
            style(ax)
            if j == len(janelas) - 1:          # rótulo direto só no painel da direita
                spread_labels(ax, marcas)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9, color=INK)
            if row == 0:
                ax.set_title(f"{wlab}\n{n_inc} incidentes HI/HIHI", fontsize=9.5,
                             color=INK, loc="left")
            else:
                ax.set_xlabel("horizonte de antecipação (h)", fontsize=9, color=INK)
            ax.set_xticks(hs)
            ax.set_xticklabels([f"{int(h)}h" for h in hs])
            ax.margins(x=0.12)

    fig.suptitle("O limiar trivial não quebra — TC382_03_A, previsão de cruzamento de 760 °C",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.938,
             "A aposta era que o limiar (linha tracejada preta) degradaria no horizonte longo e o modelo "
             "seguraria. Não acontece: as curvas CONVERGEM à direita.\nE o piso do acaso sobe junto — a 72 h "
             "um modelo com o rótulo embaralhado já empata com o limiar, então recall alto ali é aritmética "
             "da janela de crédito, não habilidade.",
             fontsize=8.4, color=INK_MUTED, ha="left", va="top")

    fig.tight_layout(rect=[0, 0, 0.965, 0.925])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
