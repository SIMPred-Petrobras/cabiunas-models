import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
ANOM = "#c0392b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": TEXT_PRIMARY})

root = "/home/dvar/.clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9"
VIB = ["TV_351X_A", "TV_352X_A", "TV_353X_A", "TV_354X_A", "TV_355X_A"]
cols = ["data_datetime", "954005_624_PI_0308"] + VIB
raw = pd.read_csv(f"{root}/sensores_full_2024_2026_30s.csv", usecols=cols, low_memory=False)
raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], errors="coerce")
for c in cols[1:]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw.dropna(subset=["data_datetime"]).drop_duplicates(subset=["data_datetime"]).sort_values("data_datetime").set_index("data_datetime")
raw["vib_mean"] = raw[VIB].mean(axis=1)

anom = pd.read_csv(
    "/home/dvar/.clearml/cache/storage_manager/global/8c1e94135b90331f65f71083645fd8f5.point_anomalies_all.csv",
    index_col=0, parse_dates=True,
)

alarm = pd.Timestamp("2025-11-04 06:22:18")
first_anom = pd.Timestamp("2025-11-03 16:04:30")
lo, hi = alarm - pd.Timedelta(hours=30), alarm + pd.Timedelta(hours=10)

raw_w = raw.loc[lo:hi]
anom_w = anom.loc[lo:hi]
anom_pts = anom_w.loc[anom_w["is_anom_point"] == 1]

fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})

ax = axes[0]
ax.plot(raw_w.index, raw_w["954005_624_PI_0308"], color=TEXT_SECONDARY, linewidth=1.1, zorder=1)
anom_vals = raw_w["954005_624_PI_0308"].reindex(anom_pts.index)
ax.scatter(anom_pts.index, anom_vals.values, s=14, color=ANOM, zorder=3, label="pontos sinalizados como anômalos")
ax.axvline(alarm, color=CRITICAL, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2, label="trip PALL_6240309 (06h22)")
ax.axvline(first_anom, color=GOOD, linewidth=1.4, linestyle=(0, (2, 2)), zorder=2, label="primeiro alerta (16h04, dia anterior)")
ax.annotate("", xy=(alarm, 1.44), xytext=(first_anom, 1.44),
            arrowprops=dict(arrowstyle="<->", color=GOOD, linewidth=1.4))
ax.text((first_anom + (alarm - first_anom) / 2), 1.455, "~14,3h de antecedência",
        ha="center", fontsize=9.5, color=GOOD, fontweight="bold")
ax.set_ylabel("PI_0308 (unid. brutas)")
ax.set_title("Alarme de 2025-11-04 — caso genuinamente preditivo", fontsize=12.2, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="lower left", frameon=False, fontsize=8.6)

ax = axes[1]
ax.plot(raw_w.index, raw_w["vib_mean"], color="#7a5c1e", linewidth=1.1)
ax.axvline(alarm, color=CRITICAL, linewidth=1.4, linestyle=(0, (4, 2)))
ax.axvline(first_anom, color=GOOD, linewidth=1.4, linestyle=(0, (2, 2)))
ax.set_ylabel("vibração média\n(5 canais X, unid. brutas)")
ax.set_title("Vibração no mesmo período — contexto usado pelo IsolationForest", fontsize=10.8, loc="left", color=TEXT_SECONDARY)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
fig.autofmt_xdate(rotation=0, ha="center")
fig.suptitle("Caso detectado: antecedência real de ~14 horas", fontsize=13.3, y=1.0)
fig.tight_layout()
fig.savefig("fig_caso_detectado.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
