#!/usr/bin/env python3
"""
plot_fleet_baseline.py
A comparação AE × limiar trivial em todos os sensores térmicos, num gráfico de haltere:
um par de pontos por sensor, ligados — a distância entre eles É o resultado.

Haltere (dumbbell) porque a pergunta é PAREADA: para cada sensor, quem ganha e por
quanto. Barras agrupadas esconderiam a diferença, que é o que importa aqui.

Painel 1  recall_raw   quanto de cada incidente é detectado
Painel 2  FA/dia       o que se paga por isso  (eixos separados — nunca eixo duplo)

Cada linha traz o `n` de incidentes, porque metade da frota tem amostra pequena demais
para sustentar conclusão e isso precisa estar visível no gráfico, não na legenda.

Uso:
    PYTHONPATH=. python scripts/plot_fleet_baseline.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = "eval_predictive_out/baseline_trivial_fleet.csv"
OUT = "eval_predictive_out/fig_fleet_baseline_TC382.png"

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
AE_C, TRIV_C = "#6b7b8c", "#b3541e"
WIN, LOSE = "#0ca30c", "#d03b3b"


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, axis="x", color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)


def main() -> None:
    df = pd.read_csv(SRC)
    d = df[df.avaliado == True].copy()            # noqa: E712
    d = d.sort_values("n_inc", ascending=True).reset_index(drop=True)
    y = range(len(d))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 1.05 * len(d) + 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    for ax, (ca, ct, lab, pct) in zip(axes, [
            ("ae_recall", "trivial_recall", "recall_raw  (maior é melhor)", True),
            ("ae_fa", "trivial_fa", "falsos alarmes / dia  (menor é melhor)", False)]):
        for i, r in d.iterrows():
            a, t = r[ca] * (100 if pct else 1), r[ct] * (100 if pct else 1)
            melhor_trivial = (t > a) if pct else (t < a)
            ax.plot([a, t], [i, i], color=WIN if melhor_trivial else LOSE,
                    lw=2.0, alpha=0.32, zorder=1, solid_capstyle="round")
            ax.scatter([a], [i], s=62, color=AE_C, zorder=3, edgecolor="white", lw=1.2)
            ax.scatter([t], [i], s=62, color=TRIV_C, zorder=3, edgecolor="white", lw=1.2)
            dpp = t - a
            rot = f"{dpp:+.1f}pp" if pct else f"{dpp:+.3f}"   # FA é da ordem de 0,07
            ax.annotate(rot, xy=(max(a, t), i), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=7.8,
                        color=WIN if melhor_trivial else LOSE)
        # sombreia as linhas de amostra insuficiente: a diferença ali não é resultado
        for i, r in d.iterrows():
            if r.n_inc < 15:
                ax.axhspan(i - 0.45, i + 0.45, color=INK, alpha=0.035, lw=0, zorder=0)
        style(ax)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{r.sensor}\n{int(r.n_inc)} inc · alarme {r.direcao.split()[0].lower()}"
                            + ("" if r.n_inc >= 15 else " · n baixo")
                            for _, r in d.iterrows()], fontsize=8.2, color=INK)
        ax.set_xlabel(lab, fontsize=9, color=INK)
        ax.margins(x=0.20, y=0.10)

    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, color=AE_C, label="autoencoder"),
        Line2D([], [], marker="o", ls="", ms=8, color=TRIV_C,
               label="limiar trivial no próprio sinal"),
        Line2D([], [], color=WIN, lw=2.4, alpha=0.5, label="limiar vence"),
        Line2D([], [], color=LOSE, lw=2.4, alpha=0.5, label="autoencoder vence"),
    ], loc="lower right", fontsize=8, framealpha=0.95, ncol=2)

    fig.suptitle("Onde há amostra, o limiar vence — e o autoencoder custa mais falso alarme "
                 "em TODOS os sete canais",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.945,
             "As duas primeiras linhas (n = 81 e n = 17) são as únicas com amostra que sustenta conclusão, "
             "e nas duas o limiar ganha.\nNas cinco linhas sombreadas n vai de 4 a 10 — um único incidente "
             "vale 10 a 25 pp, então a diferença ali é ruído. O painel da direita não depende de amostra: "
             "a FA do autoencoder é 2 a 4× maior em TODOS os sete.",
             fontsize=8.4, color=INK_MUTED, ha="left", va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
