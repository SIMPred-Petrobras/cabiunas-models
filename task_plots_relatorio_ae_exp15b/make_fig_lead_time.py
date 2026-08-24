import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

CAT_COLOR = {"preditivo": GOOD, "reativo": WARNING, "suspeito": SERIOUS}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.6, "text.color": TEXT_PRIMARY})

OUT = "/tmp/claude-1000/-home-dvar-REPO-CABIUNAS/087299e3-b18f-4ad0-9503-547fc41ebe17/scratchpad"
df14 = pd.read_csv(f"{OUT}/classif_full_task14.csv", parse_dates=["alarm_time"])
df15b = pd.read_csv(f"{OUT}/classif_full_exp15b.csv", parse_dates=["alarm_time"])

# ordem cronologica compartilhada (mesmos 40 alarmes nos dois candidatos)
order = df14.sort_values("alarm_time").reset_index(drop=True)
order["row"] = np.arange(len(order))[::-1]  # mais antigo embaixo, mais recente em cima
row_map = dict(zip(order["alarm_time"], order["row"]))

df14["row"] = df14["alarm_time"].map(row_map)
df15b["row"] = df15b["alarm_time"].map(row_map)

# IMPORTANTE: labels precisam estar na MESMA ordem (posicional) que
# order["row"], nao resortidas -- set_yticks/set_yticklabels pareiam as
# duas listas por indice, nao por valor.
labels = order["alarm_time"].dt.strftime("%Y-%m-%d %Hh") + " " + order["tag"]

fig, ax = plt.subplots(figsize=(9.6, 11.5))

# linha do instante do alarme
ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.1, zorder=2)

OFFSET = 0.17
for _, r in df14.iterrows():
    if pd.isna(r["lead_h"]):
        continue
    color = CAT_COLOR[r["categoria"]]
    ax.scatter([r["lead_h"]], [r["row"] + OFFSET], marker="o", s=42, color=color,
                edgecolor=BLUE, linewidth=1.3, zorder=4)

for _, r in df15b.iterrows():
    if pd.isna(r["lead_h"]):
        continue
    color = CAT_COLOR[r["categoria"]]
    ax.scatter([r["lead_h"]], [r["row"] - OFFSET], marker="^", s=42, color=color,
                edgecolor=ORANGE, linewidth=1.3, zorder=4)

# faixa cinza clara pra cada linha alternada (facilita leitura)
for row in order["row"]:
    if row % 2 == 0:
        ax.axhspan(row - 0.5, row + 0.5, color="#f7f6f4", zorder=0)

ax.set_yticks(order["row"])
ax.set_yticklabels(labels, fontsize=8.2)
ax.set_ylim(-1, len(order))
ax.set_xlim(-25, 25)
ax.set_xlabel("horas de antecedência da 1ª detecção (negativo = depois do alarme)")
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)

cat_handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markeredgecolor=c, markersize=9, label=l)
               for l, c in [("preditivo", GOOD), ("reativo", WARNING), ("suspeito", SERIOUS)]]
cand_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=BLUE, markersize=8, markeredgewidth=1.6, label="task 14 (anterior)"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor=ORANGE, markersize=8, markeredgewidth=1.6, label="EXP15b (novo)"),
]
leg1 = ax.legend(handles=cand_handles, loc="upper center", bbox_to_anchor=(0.28, 1.045), ncol=1, frameon=False, fontsize=9, title="candidato (forma)", title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=cat_handles, loc="upper center", bbox_to_anchor=(0.72, 1.045), ncol=1, frameon=False, fontsize=9, title="categoria (cor)", title_fontsize=9)

ax.set_title("Antecedência de detecção, alarme a alarme (40 alarmes OOS, 2025-07-01 em diante)",
              fontsize=12.5, loc="left", pad=42, color=TEXT_PRIMARY)

fig.tight_layout()
fig.savefig("fig_lead_time_por_alarme.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
