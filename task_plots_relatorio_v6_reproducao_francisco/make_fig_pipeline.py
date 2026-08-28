"""Diagrama esquematico da metodologia do Francisco (politica de producao
DETECTION_POLICY), passo a passo, replicada no nosso v6. Nao depende de
dados -- so ilustra o pipeline."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0ca30c"
CRITICAL = "#d03b3b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
BG = "#f7f6f3"

fig, ax = plt.subplots(figsize=(12, 10.2))
ax.set_xlim(0, 12)
ax.set_ylim(-2.3, 11.0)
ax.axis("off")


def box(x, y, w, h, text, color, fontsize=9.5, text_color="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, wrap=True, linespacing=1.35)


def arrow(x0, y0, x1, y1, color=TEXT_SECONDARY):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.6, color=color)
    ax.add_patch(a)


# --- linha 1: dois ramos de sensores -> sinal --------------------------
box(0.3, 8.1, 3.0, 1.1, "14 sensores de\ntemperatura\n(sensor bruto limpo)", BLUE)
box(8.7, 8.1, 3.0, 1.1, "TI_0305 vs. mediana\ndos 3 mancais irmãos\n(TI_0301/0303/0307)", ORANGE)

arrow(1.8, 8.1, 1.8, 7.35)
arrow(10.2, 8.1, 10.2, 7.35)

box(0.3, 6.3, 3.0, 1.0, "RobustScaler\n(mediana / IQR)", BLUE, fontsize=9)
box(8.7, 6.3, 3.0, 1.0, "spread = TI_0305 -\nmediana(irmãos)", ORANGE, fontsize=9)

arrow(1.8, 6.3, 1.8, 5.55)
arrow(10.2, 6.3, 10.2, 5.55)

box(0.3, 4.5, 3.0, 1.0, "PCA(0,95, svd=full)\nerro de reconstrução\n(estatística Q / SPE)", BLUE, fontsize=9)
box(8.7, 4.5, 3.0, 1.0, "z-robusto\n(mediana / MAD)", ORANGE, fontsize=9)

arrow(1.8, 4.5, 1.8, 3.75)
arrow(10.2, 4.5, 10.2, 3.75)

box(0.3, 2.7, 3.0, 1.0, "EWMA (halflife 1h)\nno score contínuo", BLUE, fontsize=9)
box(8.7, 2.7, 3.0, 1.0, "EWMA (halflife 30min)\nno score contínuo", ORANGE, fontsize=9)

arrow(1.8, 2.7, 1.8, 1.95)
arrow(10.2, 2.7, 10.2, 1.95)

box(0.3, 0.9, 3.0, 1.0, "limiar = 2,0 x p99\ndo baseline suavizado\n+ sustentado 30min", BLUE, fontsize=8.7)
box(8.7, 0.9, 3.0, 1.0, "limiar |z| > 3,0\n+ sustentado 30min", ORANGE, fontsize=9)

# setas convergindo para o AND central
arrow(3.3, 1.4, 5.15, 1.4)
arrow(8.7, 1.4, 6.85, 1.4)

box(5.15, 0.85, 1.7, 1.1, "E\n(confirmacao)", CRITICAL, fontsize=11)

arrow(6.0, 0.85, 6.0, 0.05)

# caixa final de alerta, um pouco abaixo (canto)
box(4.5, -0.85, 3.0, 0.75, "ALERTA CONFIRMADO", GREEN, fontsize=10)

# titulo e legendas por familia
ax.text(1.8, 9.55, "Sinal “temperatura”\n(multivariado)", ha="center", fontsize=10.5,
        fontweight="bold", color=TEXT_SECONDARY)
ax.text(10.2, 9.55, "Sinal “mancal_spread”\n(univariado)", ha="center", fontsize=10.5,
        fontweight="bold", color=TEXT_SECONDARY)

ax.text(6.0, 10.6, "Política de produção (DETECTION_POLICY) — decisão de 18/07/2026",
        ha="center", fontsize=13, fontweight="bold", color="#222")

# nota lateral: baseline
ax.text(0.0, -1.7,
        "Treino de cada mês: últimas 3.000h de operação ELEGÍVEL (não calendário) --\n"
        "excluindo parada, transiente de partida e alguns dias antes de cada evento curado.",
        ha="left", fontsize=9, color=TEXT_SECONDARY, style="italic")

plt.tight_layout()
fig.savefig("fig_pipeline_v6.png", dpi=170, facecolor="white", bbox_inches="tight")
print("ok: fig_pipeline_v6.png")
