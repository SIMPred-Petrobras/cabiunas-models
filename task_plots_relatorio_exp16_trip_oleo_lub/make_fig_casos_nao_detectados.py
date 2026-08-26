import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
CRITICAL = "#d03b3b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.3, "text.color": TEXT_PRIMARY})

root = "/home/dvar/.clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9"
raw = pd.read_csv(f"{root}/sensores_full_2024_2026_30s.csv", usecols=["data_datetime", "954005_624_PI_0308", "RUNNING_A"], low_memory=False)
raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], errors="coerce")
raw["954005_624_PI_0308"] = pd.to_numeric(raw["954005_624_PI_0308"], errors="coerce")
raw["RUNNING_A"] = pd.to_numeric(raw["RUNNING_A"], errors="coerce")
raw = raw.dropna(subset=["data_datetime"]).drop_duplicates(subset=["data_datetime"]).sort_values("data_datetime").set_index("data_datetime")

p50 = raw.loc[raw["RUNNING_A"] == 1, "954005_624_PI_0308"].median()
p95 = raw.loc[raw["RUNNING_A"] == 1, "954005_624_PI_0308"].quantile(0.95)
p99 = raw.loc[raw["RUNNING_A"] == 1, "954005_624_PI_0308"].quantile(0.99)

events = [
    ("2024-03-07 13:55:37", "2024-03-06 12:00:00", "2024-03-07 17:00:00",
     "Evento de 2024-03-07 — sinal do alarme às 13h55, queda física real só ~2h depois"),
    ("2026-02-26 15:34:20", "2026-02-25 15:00:00", "2026-02-26 18:00:00",
     "Evento de 2026-02-26 — pressão plana até o instante exato do trip"),
]

fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), sharey=False)

for ax, (alarm_str, lo_str, hi_str, title) in zip(axes, events):
    alarm = pd.Timestamp(alarm_str)
    lo, hi = pd.Timestamp(lo_str), pd.Timestamp(hi_str)
    w = raw.loc[lo:hi]
    ax.plot(w.index, w["954005_624_PI_0308"], color=TEXT_SECONDARY, linewidth=1.2, zorder=2)
    ax.axhline(p95, color="#9c8a3a", linewidth=1.0, linestyle=(0, (1, 1)), alpha=0.8, zorder=1)
    ax.axhline(p50, color="#9c8a3a", linewidth=0.8, linestyle=(0, (1, 2)), alpha=0.5, zorder=1)
    ax.axvline(alarm, color=CRITICAL, linewidth=1.4, linestyle=(0, (4, 2)), zorder=3, label="alarme PALL_6240309")
    ax.text(w.index[3], p95 + 0.01, "p95 operação normal", fontsize=7.6, color="#9c8a3a")
    ax.set_title(title, fontsize=10.6, loc="left")
    ax.set_ylabel("PI_0308 (unid. brutas)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.legend(loc="lower left", frameon=False, fontsize=8.4)

fig.suptitle("Casos sem sinal detectável — a pressão não se afasta do normal antes do trip",
              fontsize=13.0, y=1.0)
fig.tight_layout()
fig.savefig("fig_casos_nao_detectados.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
