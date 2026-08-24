import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
})

# task14: seed principal 75.0%, seeds 43-46
task14_main = 75.0
task14_seeds = [55.0, 50.0, 57.5, 37.5]

# EXP15b: seed principal 77.5%, seeds 43-46
exp15b_main = 77.5
exp15b_seeds = [62.5, 47.5, 75.0, 57.5]

fig, ax = plt.subplots(figsize=(9, 3.6))

y_task14, y_exp15b = 1, 0
jitter = np.array([-0.09, -0.03, 0.03, 0.09])

# faixa min-max (barra fina)
ax.plot([min(task14_seeds), max(task14_seeds)], [y_task14, y_task14], color=BLUE, lw=2, zorder=2, alpha=0.35)
ax.plot([min(exp15b_seeds), max(exp15b_seeds)], [y_exp15b, y_exp15b], color=ORANGE, lw=2, zorder=2, alpha=0.35)

# sementes do sweep (circulos pequenos)
ax.scatter(task14_seeds, y_task14 + jitter, s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8, label="seeds 43–46 (seed-sweep)")
ax.scatter(exp15b_seeds, y_exp15b + jitter, s=46, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.8)

# media do sweep (linha vertical tracejada curta)
ax.scatter([np.mean(task14_seeds)], [y_task14], marker="|", s=900, color=BLUE, zorder=4, linewidth=2)
ax.scatter([np.mean(exp15b_seeds)], [y_exp15b], marker="|", s=900, color=ORANGE, zorder=4, linewidth=2)

# seed principal (losango maior)
ax.scatter([task14_main], [y_task14], marker="D", s=140, color=BLUE, zorder=5, edgecolor=TEXT_PRIMARY, linewidth=1.2, label="seed principal (42)")
ax.scatter([exp15b_main], [y_exp15b], marker="D", s=140, color=ORANGE, zorder=5, edgecolor=TEXT_PRIMARY, linewidth=1.2)

ax.annotate(f"{task14_main:.1f}%", (task14_main, y_task14), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=10, color=TEXT_PRIMARY)
ax.annotate(f"{exp15b_main:.1f}%", (exp15b_main, y_exp15b), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=10, color=TEXT_PRIMARY)
ax.annotate(f"média sweep\n{np.mean(task14_seeds):.1f}%±{np.std(task14_seeds):.1f}pp", (np.mean(task14_seeds), y_task14), textcoords="offset points", xytext=(0, -34), ha="center", fontsize=8.5, color=TEXT_SECONDARY)
ax.annotate(f"média sweep\n{np.mean(exp15b_seeds):.1f}%±{np.std(exp15b_seeds):.1f}pp", (np.mean(exp15b_seeds), y_exp15b), textcoords="offset points", xytext=(0, -34), ha="center", fontsize=8.5, color=TEXT_SECONDARY)

ax.set_yticks([y_exp15b, y_task14])
ax.set_yticklabels(["EXP15b\n(novo)", "task 14\n(anterior)"])
ax.set_ylim(-0.55, 1.55)
ax.set_xlim(30, 85)
ax.set_xlabel("hit_rate")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=TEXT_SECONDARY)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False, fontsize=9.5)
ax.set_title("Variação entre sementes de retreino (seed-sweep)", fontsize=12.5, color=TEXT_PRIMARY, pad=38, loc="left")

fig.tight_layout()
fig.savefig("fig_seed_sweep.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
