import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
CRITICAL = "#d03b3b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": TEXT_PRIMARY})

root = "/home/dvar/.clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9"
raw = pd.read_csv(f"{root}/sensores_full_2024_2026_30s.csv", usecols=["data_datetime", "TC382_03_A"], parse_dates=["data_datetime"])
raw["TC382_03_A"] = pd.to_numeric(raw["TC382_03_A"], errors="coerce")
raw = raw.set_index("data_datetime")

lo, hi = pd.Timestamp("2026-04-13 06:00:00"), pd.Timestamp("2026-04-15 12:00:00")
raw_w = raw.loc[lo:hi]

seq14 = pd.read_csv(
    "/tmp/claude-1000/-home-dvar-REPO-CABIUNAS/087299e3-b18f-4ad0-9503-547fc41ebe17/scratchpad/exp13_task14/sequence_scores_all.csv",
    parse_dates=["seq_start_time"],
).set_index("seq_start_time")
seq15b = pd.read_csv(
    "/tmp/claude-1000/-home-dvar-REPO-CABIUNAS/087299e3-b18f-4ad0-9503-547fc41ebe17/scratchpad/exp15b_task/sequence_scores_all.csv",
    parse_dates=["seq_start_time"],
).set_index("seq_start_time")

thr14 = 0.2609116733074188
thr15b = 1.089226484298706

seq14_w = seq14.loc[lo:hi, "mae_TC382_03_A"]
seq15b_w = seq15b.loc[lo:hi, "mae_TC382_03_A"]

alarms = [pd.Timestamp("2026-04-14 00:12:47"), pd.Timestamp("2026-04-14 10:07:43"), pd.Timestamp("2026-04-14 12:01:21")]

fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})

ax = axes[0]
ax.plot(raw_w.index, raw_w["TC382_03_A"], color=TEXT_SECONDARY, linewidth=1.3)
for a in alarms:
    ax.axvline(a, color=CRITICAL, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.8)
ax.set_ylabel("TC382_03_A (°C)")
ax.set_title("Temperatura bruta — deriva lenta de ~677°C para ~793°C em ~24h", fontsize=11.5, loc="left", color=TEXT_PRIMARY)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)

ax = axes[1]
ax.plot(seq14_w.index, seq14_w.values, color=BLUE, linewidth=1.3, label="task 14 (candidato anterior)")
ax.plot(seq15b_w.index, seq15b_w.values, color=ORANGE, linewidth=1.3, label="EXP15b (novo candidato)")
ax.axhline(thr14, color=BLUE, linewidth=1.1, linestyle=(0, (2, 2)), alpha=0.6)
ax.axhline(thr15b, color=ORANGE, linewidth=1.1, linestyle=(0, (2, 2)), alpha=0.6)
ax.text(seq14_w.index[3], thr14 + 0.05, f"threshold task 14 = {thr14:.2f}", color=BLUE, fontsize=8.7, va="bottom")
ax.text(seq14_w.index[3], thr15b + 0.05, f"threshold EXP15b = {thr15b:.2f}", color=ORANGE, fontsize=8.7, va="bottom")
for a in alarms:
    ax.axvline(a, color=CRITICAL, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.8)
ax.set_ylabel("MAE de reconstrução (canal TC382_03_A)")
ax.set_title("Erro de reconstrução — só o EXP15b cruza o próprio threshold", fontsize=11.5, loc="left", color=TEXT_PRIMARY)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
fig.autofmt_xdate(rotation=0, ha="center")

# marca os 3 alarmes uma unica vez com anotacao no topo
axes[0].annotate("alarmes HI/HIHI\nde TC382_03_A", xy=(alarms[1], raw_w["TC382_03_A"].max()),
                   xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.6, color=CRITICAL)

fig.suptitle("Episódio 2026-04-14 — deriva lenta resgatada pela normalização on-state-only", fontsize=13, y=1.0, color=TEXT_PRIMARY)
fig.tight_layout()
fig.savefig("fig_caso_0414.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
