import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
ANOM = "#c0392b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": TEXT_PRIMARY})

root = "/home/dvar/.clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9"
raw = pd.read_csv(f"{root}/sensores_full_2024_2026_30s.csv", usecols=["data_datetime", "954005_624_PI_0308"])
raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], errors="coerce")
raw["954005_624_PI_0308"] = pd.to_numeric(raw["954005_624_PI_0308"], errors="coerce")
raw = raw.dropna(subset=["data_datetime"]).drop_duplicates(subset=["data_datetime"]).sort_values("data_datetime").set_index("data_datetime")

anom = pd.read_csv(
    "/home/dvar/.clearml/cache/storage_manager/global/8c1e94135b90331f65f71083645fd8f5.point_anomalies_all.csv",
    index_col=0, parse_dates=True,
)

# eventos genuinos conhecidos do alarme PALL_6240309 (turbina rodando no momento do trip)
genuinos = {
    "2024-03-07 13:55:37": ("sem sinal detectável", CRITICAL),
    "2025-11-04 06:22:18": ("detectado — 14h de antecedência", GOOD),
    "2026-02-26 15:34:20": ("sem sinal detectável", CRITICAL),
}
oos_start = pd.Timestamp("2025-07-01")

fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.plot(raw.index, raw["954005_624_PI_0308"], color=TEXT_SECONDARY, linewidth=0.5, alpha=0.85, zorder=1)

anom_pts = anom.loc[anom["is_anom_point"] == 1]
anom_series = raw["954005_624_PI_0308"].reindex(anom_pts.index)
ax.scatter(anom_pts.index, anom_series.values, s=5, color=ANOM, zorder=3, label="pontos sinalizados como anômalos")

for ts_str, (label, color) in genuinos.items():
    t = pd.Timestamp(ts_str)
    ax.axvline(t, color=color, linewidth=1.3, linestyle=(0, (4, 2)), alpha=0.9, zorder=2)

ax.axvline(oos_start, color=TEXT_SECONDARY, linewidth=1.1, linestyle=(0, (1, 1)), alpha=0.7, zorder=2)
ax.annotate("início do período\nde teste (OOS)\n2025-07-01", xy=(oos_start, 0.97), xycoords=("data", "axes fraction"),
            xytext=(-70, 0), textcoords="offset points", fontsize=7.8, color=TEXT_SECONDARY, ha="left", va="top")

ax.annotate("2024-03-07\n(treino,\nsem sinal)", xy=(pd.Timestamp("2024-03-07"), 0.97), xycoords=("data", "axes fraction"),
            xytext=(-8, 0), textcoords="offset points", fontsize=7.8, color=CRITICAL, ha="center", va="top")
ax.annotate("2025-11-04\nDETECTADO\n(~14h antes)", xy=(pd.Timestamp("2025-11-04"), 0.97), xycoords=("data", "axes fraction"),
            xytext=(0, 0), textcoords="offset points", fontsize=7.8, color=GOOD, ha="center", va="top", fontweight="bold")
ax.annotate("2026-02-26\n(sem sinal)", xy=(pd.Timestamp("2026-02-26"), 0.97), xycoords=("data", "axes fraction"),
            xytext=(8, 0), textcoords="offset points", fontsize=7.8, color=CRITICAL, ha="center", va="top")

ax.set_ylim(-1.15, 2.35)
ax.set_ylabel("PI_0308 (pressão óleo lub. compressor, unid. brutas)")
ax.set_title("Série completa 2024-2026 — modelo iforest final (EXP16b) e os 3 eventos genuínos conhecidos",
              fontsize=12.2, loc="left", pad=12)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax.legend(loc="lower left", frameon=False, fontsize=9.0)

fig.tight_layout()
fig.savefig("fig_serie_completa.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
