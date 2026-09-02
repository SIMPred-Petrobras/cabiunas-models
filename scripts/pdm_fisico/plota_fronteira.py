#!/usr/bin/env python3
"""A fronteira custo x deteccao do nosso detector, contra os pontos publicados
do Francisco e da Lara. Figura para a apresentacao.

O QUE ESTA PLOTADO. Nossa fronteira vem de `_tmp_fronteira_fp.py`: 756
configuracoes (limiar kb/kv x silencio pos-partida x refratario x duracao
minima), e para cada nivel de deteccao toma-se o MENOR custo alcancado. Os
pontos deles sao os 7 candidatos que o Francisco recalculou na regra C no
notebook 10 (secao 11) -- valores MEDIDOS, nao convertidos.

POR QUE NAO PLOTO A FRONTEIRA COMPLETA DELE. A fronteira das 10.368
configuracoes por variante esta publicada em regra A. Converter para regra C
usando a media que ele declarou ("tira entre 0,08 e 0,10 do FP/mes de todo
mundo") daria uma linha estimada, e linha estimada ao lado de linha medida se
le como se fossem a mesma coisa. Ficam so os pontos medidos.

RESSALVA QUE VAI NA FIGURA. Os dois lados sao MINIMOS SELECIONADOS sobre 8
eventos: o minimo de uma varredura e otimista por construcao, e nenhuma
diferenca aqui e estatisticamente significativa (nosso 5/8 contra o ponto da
Lara: p = 0,119 no teste exato de duas taxas de Poisson).
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tinta e reguas -- convencao tipografica do projeto
INK, INK2, MUTED = "#131a20", "#3d4a55", "#6b7885"
RULE, GROUND = "#dde2e6", "#f4f6f7"
# series: slots 1-3 da paleta validada (all-pairs, light: CVD dE 9.2, normal 24.0)
NOSSO, FRAN, LARA = "#2a78d6", "#eb6834", "#1baf7a"

NIVEIS = [4, 5, 6, 7, 8]

# --- pontos medidos do Francisco e da Lara, regra C (notebook 10, secao 11) ---
DELES = [
    ("produção (2 min)",      6, 0.86, 13.3, FRAN),
    ("PCA 30 s, sem filtro",  6, 0.86,  9.2, FRAN),
    ("silêncio 12 h",         6, 0.84,  3.6, FRAN),
    ("silêncio 24 h",         6, 0.77,  4.0, FRAN),
    ("PCA 30 s, alvo 5/8",    5, 0.52,  8.7, FRAN),
    ("silêncio 24 h, 5/8",    5, 0.58,  2.0, FRAN),
    ("ponto da Lara",         5, 0.43, 10.2, LARA),
]


def nossa_fronteira():
    T = pd.read_csv("_tmp_fronteira_fp.csv")
    fp, hm = [], []
    for n in NIVEIS:
        s = T[T.det == n]
        fp.append(s.fp_mes.min() if len(s) else np.nan)
        hm.append(s.h_mes.min() if len(s) else np.nan)
    return np.array(fp), np.array(hm)


def eixo(ax, x_nosso, deles_x, rotulo, xmax):
    # nossa fronteira: linha 2px + marcadores >=8px, anel de superficie
    ax.plot(x_nosso, NIVEIS, color=NOSSO, lw=2.0, zorder=4,
            marker="o", ms=9, mfc=NOSSO, mec="white", mew=2.0,
            label="nosso — 4 sinais físicos (756 configurações)")
    for nome, niv, xv, cor in deles_x:
        ax.plot([xv], [niv], marker="s", ms=9, mfc=cor, mec="white", mew=2.0,
                lw=0, zorder=5)
    ax.set_xlabel(rotulo + "   (menor é melhor)", fontsize=9.5, color=INK2)
    ax.set_ylabel("falhas antecipadas (de 8)", fontsize=9.5, color=INK2)
    ax.set_yticks(NIVEIS)
    ax.set_ylim(3.55, 8.5)
    ax.set_xlim(-xmax * 0.035, xmax)
    ax.grid(axis="both", color=RULE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(labelsize=8.5, colors=INK2)


if __name__ == "__main__":
    fp, hm = nossa_fronteira()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6), facecolor="white")
    for a in (ax1, ax2):
        a.set_facecolor("white")

    eixo(ax1, fp, [(n, niv, x, c) for n, niv, x, _, c in DELES],
         "falsos positivos por mês", 1.02)
    eixo(ax2, hm, [(n, niv, x, c) for n, niv, _, x, c in DELES],
         "horas por mês em alarme falso", 15.2)

    # rotulos diretos, seletivos -- e a regra de relevo para o aqua (contraste < 3:1)
    ax1.annotate("produção\n8/8 · 0,52", (fp[4], 8), textcoords="offset points",
                 xytext=(12, -4), fontsize=8.5, color=INK, fontweight="bold", va="center")
    ax1.annotate("mínimo FP\n5/8 · 0,09", (fp[1], 5), textcoords="offset points",
                 xytext=(14, -18), fontsize=8.5, color=INK, va="center")
    ax1.annotate("zero FP\n4/8", (fp[0], 4), textcoords="offset points",
                 xytext=(10, 4), fontsize=8.5, color=INK, va="center")
    ax1.annotate("Lara\n0,43", (0.43, 5), textcoords="offset points",
                 xytext=(-6, -24), fontsize=8.5, color=INK, ha="center")
    ax2.annotate("Lara\n10,2", (10.2, 5), textcoords="offset points",
                 xytext=(0, -26), fontsize=8.5, color=INK, ha="center")

    # a regiao que a abordagem sem vibracao nao alcanca
    for a in (ax1, ax2):
        a.axhspan(7.5, 8.5, color=MUTED, alpha=0.10, lw=0, zorder=1)
    ax1.text(1.00, 8.34, "nenhuma das 51.840 configurações sem vibração chega aqui",
             fontsize=8, color=MUTED, ha="right", style="italic", zorder=3)

    leg = [
        plt.Line2D([0], [0], color=NOSSO, lw=2.0, marker="o", ms=9, mfc=NOSSO,
                   mec="white", mew=2.0, label="nosso — 4 sinais físicos (fronteira de 756 configs)"),
        plt.Line2D([0], [0], lw=0, marker="s", ms=9, mfc=FRAN, mec="white", mew=2.0,
                   label="Francisco — pontos publicados"),
        plt.Line2D([0], [0], lw=0, marker="s", ms=9, mfc=LARA, mec="white", mew=2.0,
                   label="Lara — fp_first"),
    ]
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.055, 0.925),
               fontsize=9, ncol=3, frameon=False)

    fig.suptitle("A fronteira custo × detecção — canto superior esquerdo é o melhor lugar",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.055, ha="left", y=0.995)
    fig.text(0.055, 0.945,
             "Mesma régua nos dois lados: 8 eventos-alvo, janela de 48 h antes do trip, "
             "episódios agrupados a 2 h, regra C, denominador em tempo de operação.",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.055, 0.022,
             "Os dois lados são mínimos selecionados sobre 8 eventos — otimistas por construção. "
             "Nenhuma diferença aqui é estatisticamente significativa "
             "(nosso 5/8 contra o ponto da Lara: p = 0,119, teste exato de duas taxas de Poisson). "
             "O detector do Diego não entra: a régua dele é ±24 h simétrica e não converte para esta.",
             fontsize=7.8, color=MUTED, ha="left")

    fig.tight_layout(rect=(0.045, 0.055, 0.995, 0.90))
    fig.savefig("fig_fronteira.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print("nossa fronteira (regra C):")
    for n, a, b in zip(NIVEIS, fp, hm):
        print(f"  {n}/8   menor FP/mes = {a:.3f}   menor h/mes = {b:.2f}")
    print("-> fig_fronteira.png")
