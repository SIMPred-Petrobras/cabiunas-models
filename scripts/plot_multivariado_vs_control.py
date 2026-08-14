#!/usr/bin/env python3
"""
plot_multivariado_vs_control.py
O plano recall × falso alarme da ablação pareada, com a SETA por semente.

Por que a seta e não dois grupos de pontos: com 3 sementes, comparar nuvens é fraco —
o ruído entre sementes é 10,3pp, da ordem do efeito procurado. Ligando controle→
multivariado dentro da MESMA semente, cada par vira sua própria observação e a leitura
fica direta: seta apontando para a ESQUERDA é menos falso alarme, para CIMA é mais
recall. Três setas concordando valem mais que uma diferença de medianas.

O limiar trivial é a cruz fixa — é a régua do critério pré-registrado.
A faixa cinza é a amplitude das 5 réplicas NÃO semeadas do b2024: efeito menor que
essa faixa não é resultado, é sorteio.

Uso:
    PYTHONPATH=. python scripts/plot_multivariado_vs_control.py
"""
from __future__ import annotations

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
E = "eval_predictive_out"
SRC = f"{E}/multivariado_vs_control.csv"
REPL = f"{E}/replicas_b2024.csv"
OUT = f"{E}/fig_multivariado_vs_control.png"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
C_CTRL, C_MULTI, C_TRI, C_BAND = "#6b7b8c", "#2a78d6", "#b3541e", "#d9d8d4"
SEEDS = [42, 7, 13]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ce = _load("consolida_experimentos")


def main() -> None:
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} não existe — rode scripts/eval_multivariado_vs_control.py")
    df = pd.read_csv(SRC)
    repl = pd.read_csv(REPL) if os.path.exists(REPL) else None

    # `poe_rotulo` só tenta deslocamentos VERTICAIS, e aqui duas sementes caem quase no
    # mesmo ponto (no OOS as três batem em 100% de recall) — os rótulos se fundem.
    # Acrescenta candidatos laterais para esta figura.
    ce._OFFSETS = [(0, 10), (0, -16), (-22, 8), (22, 8), (-24, -10), (24, -10),
                   (0, 24), (0, -30), (-34, 0), (34, 0)]

    janelas = [j for j in df.janela.unique()]
    fig, axes = plt.subplots(1, len(janelas), figsize=(5.0 * len(janelas), 4.9),
                             squeeze=False)
    axes = axes[0]

    # `poe_rotulo` decide colisão em coordenadas de PIXEL (ax.transData). Se for chamado
    # durante o desenho, os limites dos eixos ainda não estão fixados e o transform
    # devolve posições erradas — os rótulos se sobrepõem em silêncio. Por isso acumulamos
    # aqui e só posicionamos depois de um fig.canvas.draw().
    pendentes = []

    for ax, jan in zip(axes, janelas):
        w = df[df.janela == jan].set_index("braco")
        n = int(w.n_inc.iloc[0])

        # faixa das réplicas não semeadas: a régua de "isso é ruído?"
        if repl is not None:
            r = repl[(repl.janela == jan) & (repl.braco.str.contains("réplica|ORIGINAL"))]
            if len(r) >= 2:
                ax.axhspan(r.recall_raw.min() * 100, r.recall_raw.max() * 100,
                           color=C_BAND, alpha=0.55, zorder=0, lw=0)
                # à direita: à esquerda colide com o rótulo do limiar trivial
                ax.annotate(f"amplitude de {len(r)} réplicas idênticas do b2024",
                            xy=(0.98, r.recall_raw.max() * 100),
                            xycoords=("axes fraction", "data"), ha="right",
                            fontsize=6.8, color=MUTED, va="bottom", zorder=1)

        # setas pareadas por semente
        for s in SEEDS:
            c, m = f"ctrl semente {s:02d}", f"multi semente {s:02d}"
            if c not in w.index or m not in w.index:
                continue
            x0, y0 = w.loc[c, "fa_per_day"], w.loc[c, "recall_raw"] * 100
            x1, y1 = w.loc[m, "fa_per_day"], w.loc[m, "recall_raw"] * 100
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1,
                                        alpha=0.75, shrinkA=4, shrinkB=4), zorder=2)
            ax.scatter([x0], [y0], s=34, facecolor="none", edgecolor=C_CTRL,
                       lw=1.6, zorder=3)
            ax.scatter([x1], [y1], s=38, color=C_MULTI, zorder=3)
            pendentes.append((ax, x1, y1, f"s{s}", C_MULTI))

        if "temp (limiar trivial)" in w.index:
            t = w.loc["temp (limiar trivial)"]
            ax.scatter([t.fa_per_day], [t.recall_raw * 100], marker="P", s=125,
                       color=C_TRI, zorder=4)
            pendentes.append((ax, t.fa_per_day, t.recall_raw * 100,
                              "limiar trivial", C_TRI))
            # o canto que o critério pré-registrado exige
            ax.axvline(t.fa_per_day, color=C_TRI, lw=0.8, ls=":", alpha=0.7, zorder=1)
            ax.axhline(t.recall_raw * 100, color=C_TRI, lw=0.8, ls=":", alpha=0.7, zorder=1)

        ax.set_title(f"{jan}  ·  n = {n} incidentes", fontsize=9.5, color=INK, pad=9)
        ax.set_xlabel("falso alarme por dia  ←  melhor", fontsize=8.4, color=MUTED)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(labelsize=7.8, colors=MUTED)

    axes[0].set_ylabel("recall_raw (%)  ↑  melhor", fontsize=8.4, color=MUTED)

    # agora os limites estão fixados: transData é confiável e a de-colisão funciona
    fig.canvas.draw()
    ocupadas_por_ax = {}
    for ax, x, y, txt, cor in pendentes:
        ce.poe_rotulo(ax, x, y, txt, cor, ocupadas_por_ax.setdefault(id(ax), []))

    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=C_CTRL, mew=1.6,
                   label="controle univariado"),
        plt.Line2D([], [], marker="o", ls="", color=C_MULTI, label="AE multivariado"),
        plt.Line2D([], [], marker="P", ls="", color=C_TRI, label="limiar trivial (a régua)"),
        plt.Line2D([], [], color=MUTED, lw=1.1, label="mesma semente, controle → multi"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("AE multivariado × univariado no TC382_03_A — ablação pareada por semente",
                 fontsize=11.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Gravado: {OUT}")


if __name__ == "__main__":
    main()
