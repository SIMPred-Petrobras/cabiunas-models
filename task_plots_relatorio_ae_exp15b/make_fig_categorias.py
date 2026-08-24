import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "text.color": TEXT_PRIMARY,
})

cats = ["preditivo\n(antes do alarme)", "reativo\n(no alarme ou depois)", "suspeito\n(artefato de janela)", "sem detecção"]
colors = [GOOD, WARNING, SERIOUS, CRITICAL]

task14 = [15, 12, 3, 10]
exp15b = [17, 8, 6, 9]
n = 40

labels = ["task 14\n(anterior)", "EXP15b\n(novo)"]
data = [task14, exp15b]

fig, ax = plt.subplots(figsize=(9.5, 3.2))

GAP = 0.006  # gap fracionario entre segmentos (superficie visivel)
bar_h = 0.52
y_pos = [1, 0]

for y, row in zip(y_pos, data):
    left = 0.0
    total = sum(row)
    for val, color in zip(row, colors):
        frac = val / total
        w = max(frac - GAP, 0)
        ax.barh(y, w, left=left, height=bar_h, color=color, zorder=3)
        if val > 0:
            ax.text(left + w / 2, y, str(val), ha="center", va="center",
                     fontsize=11, color="white", fontweight="bold", zorder=4)
        left += frac

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylim(-0.55, 1.55)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=TEXT_SECONDARY)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlabel(f"fração dos {n} alarmes do período OOS (2025-07-01 em diante)", color=TEXT_SECONDARY)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(handles, cats, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False, fontsize=9)

ax.set_title("Classificação dos 40 alarmes por qualidade de detecção", fontsize=13, color=TEXT_PRIMARY, loc="left", pad=12)

fig.tight_layout()
fig.savefig("fig_categorias_deteccao.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
