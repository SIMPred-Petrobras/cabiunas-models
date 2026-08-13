#!/usr/bin/env python3
"""
consolida_experimentos.py
Junta num só lugar tudo que foi medido no TC382_03_A e desenha o plano recall × falso
alarme.

⚠️ A REGRA QUE ORGANIZA A FIGURA: cada painel é UMA janela com UM denominador de
incidentes. Números de janelas diferentes NÃO vão para o mesmo painel, por mais tentador
que seja — comparar 86,2% (58 incidentes) com 62,0% (79 incidentes) foi exatamente o erro
que atrasou esta análise. O `n` aparece no título de cada painel.

Leitura: o canto BOM é embaixo à direita (muito recall, pouco falso alarme). A faixa
vermelha à esquerda é o piso do acaso, onde medido — ponto ali dentro não significa nada.

Uso:
    PYTHONPATH=. python scripts/consolida_experimentos.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

E = "eval_predictive_out"
OUT_FIG = f"{E}/fig_consolidado_experimentos.png"
OUT_CSV = f"{E}/consolidado_experimentos.csv"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
FAM = {                       # família → cor (a identidade também vai no rótulo direto)
    "limiar trivial": "#b3541e",
    "autoencoder": "#6b7b8c",
    "supervisionado": "#2a78d6",
    "fusão": "#6b3fa0",
    "externo": "#1f8a6d",
    "acaso": "#c0392b",
}


def ler(p):
    return pd.read_csv(f"{E}/{p}") if os.path.exists(f"{E}/{p}") else None


# candidatos de posição para o rótulo, em pontos, na ordem de preferência
_OFFSETS = [(0, 10), (0, -16), (0, 22), (0, -28), (0, 34), (0, -40), (0, 46)]


def poe_rotulo(ax, x, y, texto, cor, ocupadas):
    """Escreve o rótulo na primeira posição que não colide com as já usadas.

    Os pontos se amontoam justamente na região interessante (recall alto, FA baixo), então
    sem isso três rótulos se sobrepõem e a figura perde a identificação — que aqui não pode
    depender só da cor, porque há duas famílias com pontos vizinhos.
    """
    inv = ax.transData.transform((x, y))
    larg = 4.6 * len(texto) + 8          # largura estimada do texto em pontos
    for dx, dy in _OFFSETS:
        cx, cy = inv[0] + dx, inv[1] + dy
        caixa = (cx - larg / 2, cy - 6, cx + larg / 2, cy + 6)
        if not any(not (caixa[2] < o[0] or caixa[0] > o[2] or
                        caixa[3] < o[1] or caixa[1] > o[3]) for o in ocupadas):
            ocupadas.append(caixa)
            ax.annotate(texto, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                        ha="center", va="center", fontsize=7.2, color=cor, zorder=5)
            if abs(dy) > 16:             # longe do ponto: liga com uma guia
                ax.annotate("", xy=(x, y), xytext=(dx, dy * 0.55),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="-", color=cor, lw=0.5, alpha=0.5))
            return
    ocupadas.append((inv[0] - larg / 2, inv[1] + 3, inv[0] + larg / 2, inv[1] + 15))
    ax.annotate(texto, xy=(x, y), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=7.2, color=cor, zorder=5)


def main() -> None:
    linhas = []

    def add(painel, n, nome, fam, rec, fa, nota=""):
        if rec is None or pd.isna(rec):
            return
        linhas.append(dict(painel=painel, n_inc=n, experimento=nome, familia=fam,
                           recall=float(rec) * 100, fa=float(fa), nota=nota))

    # ── painel 1: jan/24 → abr/26, 79 incidentes
    P1 = "jan/24→abr/26"
    b = ler("baseline_trivial_vs_ae.csv")
    if b is not None:
        d = b[b.janela.str.startswith("FULL")].set_index("braco")
        add(P1, 79, "limiar trivial", "limiar trivial",
            d.loc["temp (limiar trivial)", "recall_raw"], d.loc["temp (limiar trivial)", "fa_per_day"])
        add(P1, 79, "AE v14-control", "autoencoder",
            d.loc["ae  (autoencoder)", "recall_raw"], d.loc["ae  (autoencoder)", "fa_per_day"])
        add(P1, 79, "só tendência (dT/dt)", "limiar trivial",
            d.loc["slope (só tendência)", "recall_raw"], d.loc["slope (só tendência)", "fa_per_day"])
    c = ler("combine_ae_temp.csv")
    if c is not None:
        d = c[c.janela.str.startswith("FULL")].set_index("fusao")
        for k, lab in [("mean (consenso)", "fusão consenso"), ("max  (OR suave)", "fusão OR")]:
            if k in d.index:
                add(P1, 79, lab, "fusão", d.loc[k, "recall_raw"], d.loc[k, "fa_per_day"])
    h = ler("forecast_crossing_horizon.csv")
    if h is not None:
        d = h[(h.H == 8.0) & (h.janela.str.startswith("FULL"))].set_index("braco")
        for k, lab in [("A2 GBM", "GBM 30 features"), ("A1 logística", "logística")]:
            if k in d.index:
                add(P1, 79, lab, "supervisionado", d.loc[k, "recall_raw"], d.loc[k, "fa_per_day"],
                    "ponto buscado na janela")
    f = ler("forecast_crossing_chancefloor.csv")
    piso = {}
    if f is not None:
        r = f[(f.H == 8.0) & (f.janela.str.startswith("FULL"))]
        if len(r):
            piso[P1] = float(r.rotulo_embaralhado.iloc[0]) * 100
            add(P1, 79, "piso do acaso", "acaso", r.piso_ruido.iloc[0], r.piso_fa.iloc[0])

    # ── painel 2: jun/24 → abr/26, 58 incidentes (onde o melhor AE existe)
    P2 = "jun/24→abr/26"
    r = ler("replicas_b2024.csv")
    if r is not None:
        d = r[r.janela.str.startswith("FULL")].set_index("braco")
        add(P2, 58, "limiar trivial", "limiar trivial",
            d.loc["limiar trivial", "recall_raw"], d.loc["limiar trivial", "fa_per_day"])
        reps = [i for i in d.index if i.startswith("réplica") or i.startswith("b2024")]
        for i in reps:
            melhor = "b2024" in i
            add(P2, 58, "AE b2024 (melhor sorteio)" if melhor else "", "autoencoder",
                d.loc[i, "recall_raw"], d.loc[i, "fa_per_day"],
                "1 de 5 execuções idênticas")

    # ── painel 3: jan/25 → abr/26, 30 incidentes (janela da Transpetro)
    P3 = "jan/25→abr/26"
    t = ler("eval_transpetro_automl.csv")
    if t is not None:
        d = t[(t.conjunto.str.contains("TC382_03_A")) &
              (t.janela == "interseção completa")].set_index("braco")
        mapa = {"AutoML Transpetro (Dense, 36 sensores)": ("AutoML Transpetro", "externo"),
                "AE b2024 (nosso melhor)": ("AE b2024", "autoencoder"),
                "limiar trivial (TC382_03_A)": ("limiar trivial", "limiar trivial"),
                "PISO DO ACASO (5 sementes)": ("piso do acaso", "acaso")}
        for k, (lab, fam) in mapa.items():
            if k in d.index:
                add(P3, 30, lab, fam, d.loc[k, "recall_raw"], d.loc[k, "fa_per_day"])
                if fam == "acaso":
                    piso[P3] = float(d.loc[k, "recall_raw"]) * 100

    # ── painel 4: walk-forward (predições fora de amostra), 58 incidentes
    P4 = "walk-forward jul/24→abr/26"
    w = ler("forecast_crossing_walkforward.csv")
    if w is not None:
        d = w[w.H == 8.0].set_index("braco")
        mapa = {"A0 trivial (limiar de T)": ("limiar trivial", "limiar trivial"),
                "A1 logística": ("logística", "supervisionado"),
                "A2 GBM": ("GBM 30 features", "supervisionado"),
                "A3 GBM + AE": ("GBM + erro do AE", "supervisionado"),
                "REF autoencoder": ("AE v14-control", "autoencoder")}
        for k, (lab, fam) in mapa.items():
            if k in d.index:
                add(P4, 58, lab, fam, d.loc[k, "recall_raw"], d.loc[k, "fa_per_day"])

    # ── painel 5: ponto CONGELADO no treino, 17 incidentes
    P5 = "OOS congelado jul/25→abr/26"
    fz = ler("forecast_crossing_frozen.csv")
    if fz is not None:
        d = fz[fz.H == 8.0].set_index("braco")
        mapa = {"A0 trivial (limiar de T)": ("limiar trivial", "limiar trivial"),
                "A2 GBM": ("GBM 30 features", "supervisionado"),
                "REF autoencoder": ("AE v14-control", "autoencoder")}
        for k, (lab, fam) in mapa.items():
            if k in d.index:
                add(P5, 17, lab, fam, d.loc[k, "recall_raw"], d.loc[k, "fa_per_day"],
                    "threshold fixado no treino")

    df = pd.DataFrame(linhas)
    df.to_csv(OUT_CSV, index=False)

    paineis = [p for p in [P1, P2, P3, P4, P5] if p in set(df.painel)]
    fig, axes = plt.subplots(1, len(paineis), figsize=(4.15 * len(paineis), 5.6),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    if len(paineis) == 1:
        axes = [axes]

    for ax, p in zip(axes, paineis):
        d = df[df.painel == p]
        n = int(d.n_inc.iloc[0])
        if p in piso:
            ax.axvspan(0, piso[p], color=FAM["acaso"], alpha=0.07, lw=0, zorder=0)
            ax.axvline(piso[p], color=FAM["acaso"], lw=1.0, ls=(0, (2, 2)), alpha=0.7, zorder=1)
        for _, r in d.iterrows():
            ax.scatter(r.recall, r.fa, s=78, color=FAM[r.familia], zorder=3,
                       edgecolor="white", lw=1.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"{p}\n{n} incidentes", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("recall_raw (%)", fontsize=9, color=INK)
        ax.set_xlim(0, 108)
        ax.set_ylim(-0.012, 0.26)
        # rótulos só depois dos limites fixados — a de-colisão trabalha em pixels
        ocupadas: list = []
        for _, r in d.sort_values("fa").iterrows():
            if r.experimento:
                poe_rotulo(ax, r.recall, r.fa, r.experimento, FAM[r.familia], ocupadas)
    axes[0].set_ylabel("falsos alarmes / dia", fontsize=9, color=INK)
    axes[0].annotate("melhor\naqui →", xy=(0.90, 0.06), xycoords="axes fraction",
                     fontsize=7.6, color=MUTED, ha="right")

    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[Line2D([], [], marker="o", ls="", ms=7, color=c, label=k)
                             for k, c in FAM.items()],
                    loc="upper left", fontsize=7.4, framealpha=0.95)

    fig.suptitle("Tudo que foi testado no TC382_03_A — recall × custo em falso alarme",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.932,
             "Cada painel é UMA janela com UM denominador — pontos de painéis diferentes não são comparáveis entre si. "
             "O canto bom é embaixo à direita.\nA faixa vermelha é o piso do acaso: recall à esquerda dela não "
             "significa nada. Os cinco pontos cinza do 2º painel são execuções IDÊNTICAS da mesma config.",
             fontsize=8.3, color=MUTED, ha="left", va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.915])
    fig.savefig(OUT_FIG, dpi=135, facecolor="white")
    plt.close(fig)

    pd.set_option("display.width", 200)
    for p in paineis:
        d = df[df.painel == p].sort_values("recall", ascending=False)
        print(f"\n=== {p}  ({int(d.n_inc.iloc[0])} incidentes) ===")
        print(d[["experimento", "familia", "recall", "fa", "nota"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nFigura: {OUT_FIG}\nCSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
