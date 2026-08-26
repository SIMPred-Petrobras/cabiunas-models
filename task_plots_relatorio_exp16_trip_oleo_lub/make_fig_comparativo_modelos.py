import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0ca30c"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": TEXT_PRIMARY})

df = pd.read_csv("/home/dvar/.clearml/cache/storage_manager/global/920d085bdc26be530ffd680573d414b6.automl_ranking.csv")

best_per_model = df.loc[df.groupby("model")["composite_score"].idxmax()].set_index("model")
best_per_model = best_per_model.loc[["dense", "ocsvm", "iforest"]]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax = axes[0]
colors = [BLUE, ORANGE, GREEN]
bars = ax.bar(best_per_model.index, best_per_model["normal_alert_rate"] * 100, color=colors, width=0.55, zorder=3)
for b, v in zip(bars, best_per_model["normal_alert_rate"] * 100):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}%", ha="center", fontsize=9.3)
ax.set_ylabel("falso alerta (normal_alert_rate, %)")
ax.set_title("FP do melhor trial de cada modelo\n(hit_rate = 100% nos 3)", fontsize=10.6, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)

ax = axes[1]
mean_hit = df.groupby("model")["hit_rate"].mean().loc[["dense", "ocsvm", "iforest"]] * 100
bars = ax.bar(mean_hit.index, mean_hit.values, color=colors, width=0.55, zorder=3)
for b, v in zip(bars, mean_hit.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9.3)
ax.set_ylabel("hit_rate médio (%)")
ax.set_ylim(0, 108)
ax.set_title("hit_rate médio entre todos os 420 trials\nde cada modelo (todo threshold/debounce)", fontsize=10.6, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)

fig.suptitle("EXP16a — comparação dos 3 modelos do grid AutoML (grupo controle)", fontsize=12.6, y=1.03)
fig.tight_layout()
fig.savefig("fig_comparativo_modelos.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
print(best_per_model[["threshold_percentile", "debounce", "normal_alert_rate", "composite_score"]])
