import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
NEUTRAL_FILL = "#f4f3f0"
BLUE = "#2a78d6"
BLUE_FILL = "#eaf1fc"
GREEN = "#0ca30c"
GREEN_FILL = "#e8f6e8"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT_PRIMARY})

fig, ax = plt.subplots(figsize=(9.6, 10.2))
ax.set_xlim(0, 10)
ax.set_ylim(17, 100)
ax.axis("off")

def box(y, h, label, sub=None, color=BLUE, fill=BLUE_FILL, w=8.6, x=0.7, fontsize=11.0):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.28",
                        linewidth=1.6, edgecolor=color, facecolor=fill, zorder=3)
    ax.add_patch(b)
    cy = y + h / 2
    if sub:
        n_sub_lines = sub.count("\n") + 1
        title_offset = 1.15 if n_sub_lines > 1 else 0.75
        sub_offset = 0.95 if n_sub_lines > 1 else 0.75
        ax.text(x + w / 2, cy + title_offset, label, ha="center", va="center", fontsize=fontsize, color=TEXT_PRIMARY, zorder=4, fontweight="bold")
        ax.text(x + w / 2, cy - sub_offset, sub, ha="center", va="center", fontsize=8.8, color=TEXT_SECONDARY, zorder=4, linespacing=1.55)
    else:
        ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=fontsize, color=TEXT_PRIMARY, zorder=4, fontweight="bold")
    return y, y + h

def arrow(y_from, y_to, x=5.0):
    a = FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.4, color=TEXT_SECONDARY, zorder=2)
    ax.add_patch(a)

def tag(x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=8.1, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, linewidth=1.1), zorder=5)

y = 98
b1 = box(y - 4.0, 4.0, "Dados brutos", "sensores 30s (alvo + vibração) + catálogo de alarmes 2022-2026",
         color=TEXT_SECONDARY, fill=NEUTRAL_FILL)
y = b1[0] - 2.0
arrow(y + 2.0, y + 0.1)

b2 = box(y - 5.0, 5.0, "Seleção do alvo", "954005_624_PI_0308 (alarme PALL_6240309) —\núnico dos 3 trips de óleo lub. com evento genuíno em operação")
y = b2[0] - 2.0
arrow(y + 2.0, y + 0.1)

b3 = box(y - 5.6, 5.6, "Pré-processamento", "exclusão ±24h ao redor de alarmes · exclusão de gaps longos\nclipping de outliers (p0,1%-p99,9%) · normalização treino-apenas (z-score)")
y = b3[0] - 2.0
arrow(y + 2.0, y + 0.1)

b4 = box(y - 5.0, 5.0, "Features derivadas", "médias/desvios móveis em 4 escalas (6min·1h·4h·24h)\nsobre alvo + 10 canais de vibração")
y = b4[0] - 2.0
arrow(y + 2.0, y + 0.1)

b5 = box(y - 5.6, 5.6, "Grid AutoML (EXP16a)", "dense · ocsvm (grade nu×gamma) · iforest\n× 7 percentis × 6 debounces × 2 grupos = 840 trials",
         color=GREEN, fill=GREEN_FILL)
tag(9.0, (b5[0] + b5[1]) / 2 + 1.5, "explora", GREEN)
y = b5[0] - 2.0
arrow(y + 2.0, y + 0.1)

b6 = box(y - 5.0, 5.0, "Modelo vencedor: IsolationForest", "n_estimators=200 · contamination=0,05 · treinado só com\ndados 'on', pré-OOS, fora de janelas de alarme (~900k pontos)",
         color=GREEN, fill=GREEN_FILL, fontsize=10.6)
y = b6[0] - 2.0
arrow(y + 2.0, y + 0.1)

b7 = box(y - 4.2, 4.2, "Score de anomalia", "isolation score por ponto, sobre alvo + 10 canais de vibração\njuntos (reconstrução/isolamento multivariado, não só o alvo)")
y = b7[0] - 2.0
arrow(y + 2.0, y + 0.1)

b8 = box(y - 4.6, 4.6, "Threshold + debounce", "percentil 99,5 do score em treino · confirma anomalia só após\n24 pontos consecutivos (12min) acima do limiar")
y = b8[0] - 2.0
arrow(y + 2.0, y + 0.1)

b9 = box(y - 4.6, 4.6, "Máscara operacional", "zera anomalia fora do estado 'on' (desligado/transiente) —\nusa RUNNING_A + piso físico do próprio alvo")
y = b9[0] - 2.0
arrow(y + 2.0, y + 0.1)

b10 = box(y - 5.0, 5.0, "Avaliação OOS (EXP16b)", "split temporal em 2025-07-01 · hit_rate / normal_alert_rate\nseed-sweep (5 sementes) confirma variância zero")
y = b10[0] - 2.0
arrow(y + 2.0, y + 0.1)

box(y - 4.6, 4.6, "Pontos anômalos finais", "1 alarme genuíno preditivo (~14h) · 2 sem sinal detectável\n(investigados a fundo — ver Seção 5)",
    color=TEXT_SECONDARY, fill=NEUTRAL_FILL)

legend_elems = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE_FILL, markeredgecolor=BLUE, markersize=13, label="etapa padrão do pipeline AutoML"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=GREEN_FILL, markeredgecolor=GREEN, markersize=13, label="decisão específica do EXP16 (alvo, grid, modelo vencedor)"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=1,
          frameon=False, fontsize=9.2, bbox_transform=ax.transAxes)

fig.suptitle("Pipeline EXP16 — do dado bruto ao ponto anômalo avaliado", fontsize=13.5, y=0.995, color=TEXT_PRIMARY)
fig.subplots_adjust(bottom=0.03, top=0.965)
fig.savefig("fig_pipeline_exp16.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
