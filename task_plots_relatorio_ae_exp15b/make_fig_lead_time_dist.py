import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT_PRIMARY})

OUT = "/tmp/claude-1000/-home-dvar-REPO-CABIUNAS/087299e3-b18f-4ad0-9503-547fc41ebe17/scratchpad"
df14 = pd.read_csv(f"{OUT}/classif_full_task14.csv")
df15b = pd.read_csv(f"{OUT}/classif_full_exp15b.csv")

pred14 = df14.loc[df14["categoria"] == "preditivo", "lead_h"].values
pred15b = df15b.loc[df15b["categoria"] == "preditivo", "lead_h"].values

fig, ax = plt.subplots(figsize=(9, 3.6))

rng = np.random.default_rng(0)
y14 = 1 + rng.uniform(-0.14, 0.14, size=len(pred14))
y15b = 0 + rng.uniform(-0.14, 0.14, size=len(pred15b))

# caixa IQR fina
for y0, data, color in [(1, pred14, BLUE), (0, pred15b, ORANGE)]:
    q1, q3 = np.percentile(data, [25, 75])
    med = np.median(data)
    ax.plot([q1, q3], [y0, y0], color=color, linewidth=6, alpha=0.25, zorder=2, solid_capstyle="round")
    ax.plot([med, med], [y0 - 0.22, y0 + 0.22], color=color, linewidth=2.4, zorder=3)

ax.scatter(pred14, y14, s=44, color=BLUE, zorder=4, edgecolor="white", linewidth=0.8, label=f"task 14 (n={len(pred14)})")
ax.scatter(pred15b, y15b, s=44, color=ORANGE, zorder=4, edgecolor="white", linewidth=0.8, label=f"EXP15b (n={len(pred15b)})")

for y0, data, color in [(1, pred14, BLUE), (0, pred15b, ORANGE)]:
    med = np.median(data)
    mean = np.mean(data)
    ax.annotate(f"média {mean:.1f}h · mediana {med:.1f}h", (max(data) + 0.8, y0),
                 va="center", ha="left", fontsize=9, color=TEXT_SECONDARY)

ax.set_yticks([1, 0])
ax.set_yticklabels(["task 14\n(anterior)", "EXP15b\n(novo)"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlim(-2, 30)
ax.set_xlabel("horas de antecedência (só detecções preditivas)")
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)

ax.set_title("Distribuição da antecedência entre as detecções preditivas", fontsize=12.5, loc="left", color=TEXT_PRIMARY)

fig.tight_layout()
fig.savefig("fig_lead_time_distribuicao.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
