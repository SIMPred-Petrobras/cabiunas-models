import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
NEUTRAL = "#e5e4e0"
NEUTRAL_FILL = "#f4f3f0"
BLUE = "#2a78d6"       # componente original do CNN1D-AE
ORANGE = "#eb6834"     # herdado do AutoML (EXP7/EXP10)
GREEN = "#0ca30c"      # novo neste relatorio (EXP15b)
BLUE_FILL = "#eaf1fc"
ORANGE_FILL = "#fdeee7"
GREEN_FILL = "#e8f6e8"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT_PRIMARY})

fig, ax = plt.subplots(figsize=(9.6, 11.0))
ax.set_xlim(0, 10)
ax.set_ylim(3, 100)
ax.axis("off")

def box(y, h, label, sub=None, color=BLUE, fill=BLUE_FILL, w=8.6, x=0.7, fontsize=11.2):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.28",
                         linewidth=1.6, edgecolor=color, facecolor=fill, zorder=3)
    ax.add_patch(b)
    cy = y + h / 2
    if sub:
        n_sub_lines = sub.count("\n") + 1
        title_offset = 1.15 if n_sub_lines > 1 else 0.75
        sub_offset = 0.95 if n_sub_lines > 1 else 0.75
        ax.text(x + w / 2, cy + title_offset, label, ha="center", va="center", fontsize=fontsize, color=TEXT_PRIMARY, zorder=4, fontweight="bold")
        ax.text(x + w / 2, cy - sub_offset, sub, ha="center", va="center", fontsize=9.0, color=TEXT_SECONDARY, zorder=4, linespacing=1.6)
    else:
        ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=fontsize, color=TEXT_PRIMARY, zorder=4, fontweight="bold")
    return y, y + h

def arrow(y_from, y_to, x=5.0):
    a = FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=14,
                          linewidth=1.4, color=TEXT_SECONDARY, zorder=2)
    ax.add_patch(a)

def tag(x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=8.3, color=color, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, linewidth=1.1), zorder=5)

# --- blocos (topo -> base), posicoes em unidades arbitrarias de 100 ---
y = 96
b1 = box(y - 4, 4, "Dados brutos", "sensores 30s (12 canais) + catálogo de alarmes", color=TEXT_SECONDARY, fill=NEUTRAL_FILL)
y = b1[0] - 2.4
arrow(y + 2.4, y + 0.1)

b2 = box(y - 5.6, 5.6, "Pré-processamento", "interpolação curta · exclusão ±24h de alarme · clipping\nnormalização treino-apenas (z-score)", color=BLUE, fill=BLUE_FILL)
tag(9.0, (b2[0] + b2[1]) / 2 + 1.5, "novo\nEXP15b", GREEN)
y = b2[0] - 2.4
arrow(y + 2.4, y + 0.1)

b3 = box(y - 5.0, 5.0, "Normalização restrita ao período 'on'", "NORMALIZE_ON_STATE_ONLY — center/scale só do treino operacional",
         color=GREEN, fill=GREEN_FILL, fontsize=10.6)
y = b3[0] - 2.4
arrow(y + 2.4, y + 0.1)

b4 = box(y - 5.6, 5.6, "Features multiescala + textura", "roll_med/roll_std/trend em 6min·1h·4h·24h\n+ kurtosis/skewness/crest factor", color=ORANGE, fill=ORANGE_FILL)
tag(9.0, (b4[0] + b4[1]) / 2 + 1.5, "herdado\nAutoML EXP7", ORANGE)
y = b4[0] - 2.4
arrow(y + 2.4, y + 0.1)

b5 = box(y - 4.2, 4.2, "Sequências deslizantes", "janela TIME_STEPS×STRIDE → tensor (N, 60, 12)", color=BLUE, fill=BLUE_FILL)
y = b5[0] - 2.4
arrow(y + 2.4, y + 0.1)

b6 = box(y - 5.0, 5.0, "CNN1D-Autoencoder", "KerasTuner (arquitetura) + treino do canal-alvo TC382_03_A", color=BLUE, fill=BLUE_FILL)
y = b6[0] - 2.4
arrow(y + 2.4, y + 0.1)

b7 = box(y - 4.2, 4.2, "Erro de reconstrução (MAE)", "por sequência e por canal — reconstruction_mae_per_seq", color=BLUE, fill=BLUE_FILL)
y = b7[0] - 2.4
arrow(y + 2.4, y + 0.1)

b8 = box(y - 4.6, 4.6, "Threshold (robust_mad)", "mediana + K × 1,4826 × MAD do erro de treino", color=BLUE, fill=BLUE_FILL)
y = b8[0] - 2.4
arrow(y + 2.4, y + 0.1)

# camada de pos-processamento -- caixa envolvente
wrap_top = y + 1.3
b9 = box(y - 5.6, 5.6, "Máscara operacional", "on / off_curto / off_longo / transiente\n+ 2º critério: sensor-alvo abaixo do piso físico", color=ORANGE, fill=ORANGE_FILL)
tag(9.0, (b9[0] + b9[1]) / 2 + 1.5, "herdado\nAutoML EXP10", ORANGE)
y = b9[0] - 1.6
arrow(y + 1.6, y + 0.1)

b10 = box(y - 4.6, 4.6, "Portão de rampa (load_gate)", "bloqueia manobra de carga legítima (ΔT/h alto)", color=ORANGE, fill=ORANGE_FILL)
tag(9.0, (b10[0] + b10[1]) / 2 + 1.5, "herdado\nAutoML EXP10b", ORANGE)
y = b10[0] - 1.6
arrow(y + 1.6, y + 0.1)

b11 = box(y - 4.6, 4.6, "Portão de volatilidade", "bloqueia vibração elevada persistente (não-transiente)", color=ORANGE, fill=ORANGE_FILL)
tag(9.0, (b11[0] + b11[1]) / 2 + 1.5, "herdado\nAutoML EXP10c", ORANGE)
y = b11[0] - 1.6
arrow(y + 1.6, y + 0.1)

b12 = box(y - 4.6, 4.6, "Bloqueio gradual (gate-escape)", "resgata pontos com MAE > threshold×1,5, mesmo com gate ativo", color=BLUE, fill=BLUE_FILL)
tag(9.0, (b12[0] + b12[1]) / 2 + 1.5, "novo\nEXP13", BLUE)
wrap_bot = b12[0] - 0.4
y = b12[0] - 2.4
arrow(y + 2.4, y + 0.1)

b13 = box(y - 4.6, 4.6, "Pontos anômalos finais", "avaliação contra 40 alarmes OOS: hit_rate / normal_alert_rate", color=TEXT_SECONDARY, fill=NEUTRAL_FILL)

# retangulo pontilhado ao redor da camada de pos-processamento
from matplotlib.patches import FancyBboxPatch as FBP
wrap = FBP((0.35, wrap_bot), 9.3, wrap_top - wrap_bot, boxstyle="round,pad=0.1,rounding_size=0.3",
            linewidth=1.1, edgecolor=TEXT_SECONDARY, facecolor="none", linestyle=(0, (5, 3)), zorder=1)
ax.add_patch(wrap)
ax.text(0.55, wrap_top - 0.55, "camada de pós-processamento (não altera o modelo treinado)",
         fontsize=8.6, color=TEXT_SECONDARY, style="italic", ha="left", va="center")

# legenda
legend_elems = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE_FILL, markeredgecolor=BLUE, markersize=13, label="componente original do CNN1D-AE"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE_FILL, markeredgecolor=ORANGE, markersize=13, label="herdado do AutoML (EXP7/EXP10)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=GREEN_FILL, markeredgecolor=GREEN, markersize=13, label="novo neste relatório (EXP15b)"),
]
ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=1,
           frameon=False, fontsize=9.3, bbox_transform=fig.transFigure)

fig.suptitle("Pipeline CNN1D-AE — do dado bruto ao ponto anômalo avaliado", fontsize=13.5, y=0.995, color=TEXT_PRIMARY)
fig.subplots_adjust(bottom=0.10, top=0.97)
fig.savefig("fig_pipeline_esquema.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
