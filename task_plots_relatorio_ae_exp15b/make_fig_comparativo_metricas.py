import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

candidates = [
    "AutoML\nEXP10c\n(referência)",
    "CNN1D-AE\ntask 13\n(sem escape)",
    "CNN1D-AE\ntask 14\n(gate-escape)",
    "CNN1D-AE\nEXP15b\n(on-state)",
]
hit_rate = [92.5, 60.0, 75.0, 77.5]
fp = [0.35, 0.17, 0.22, 0.30]
n_det = ["37/40", "24/40", "30/40", "31/40"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
bars = ax.bar(candidates, hit_rate, color=BLUE, width=0.6, zorder=3)
for b, lbl, val in zip(bars, n_det, hit_rate):
    ax.text(b.get_x() + b.get_width() / 2, val + 1.8, f"{val:.1f}%\n({lbl})",
            ha="center", va="bottom", fontsize=9.5, color=TEXT_PRIMARY)
ax.set_ylim(0, 105)
ax.set_ylabel("hit_rate (n=40 alarmes)")
ax.set_title("Cobertura de detecção", fontsize=12, color=TEXT_PRIMARY, loc="left")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)

ax = axes[1]
bars = ax.bar(candidates, fp, color=ORANGE, width=0.6, zorder=3)
for b, val in zip(bars, fp):
    ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.2f}%",
            ha="center", va="bottom", fontsize=9.5, color=TEXT_PRIMARY)
ax.set_ylim(0, 0.45)
ax.set_ylabel("normal_alert_rate (falso positivo)")
ax.set_title("Falso positivo", fontsize=12, color=TEXT_PRIMARY, loc="left")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)

for ax in axes:
    ax.tick_params(axis="x", labelsize=8.5, rotation=0)

fig.suptitle("Evolução do candidato: AutoML EXP10c → CNN1D-AE EXP15b", fontsize=13, color=TEXT_PRIMARY, y=1.02)
fig.tight_layout()
fig.savefig("fig_comparativo_metricas.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
