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
NEUTRAL = "#8a94a6"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.3, "text.color": TEXT_PRIMARY})

root = "/home/dvar/.clearml/cache/storage_manager/datasets/ds_a97ba56ba14840fbb1125c2a82f883c9"
TEMPS = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0325"]
cols = ["data_datetime", "RUNNING_A", "954005_624_PDIT_0305"] + TEMPS
df = pd.read_csv(f"{root}/sensores_full_2024_2026_30s.csv", usecols=cols, low_memory=False)
df["data_datetime"] = pd.to_datetime(df["data_datetime"], errors="coerce")
for c in cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["data_datetime"]).drop_duplicates(subset=["data_datetime"]).sort_values("data_datetime").set_index("data_datetime")
on = df.loc[df["RUNNING_A"] == 1].copy()

trips = [pd.Timestamp("2024-01-09 11:41:30"), pd.Timestamp("2024-03-07 13:55:37"),
         pd.Timestamp("2024-05-31 01:25:42"), pd.Timestamp("2024-06-11 14:31:12"),
         pd.Timestamp("2024-11-26 18:22:51"), pd.Timestamp("2025-11-04 06:22:18"),
         pd.Timestamp("2026-02-26 15:34:20")]
genuine_trips = [pd.Timestamp("2024-03-07 13:55:37"), pd.Timestamp("2025-11-04 06:22:18"), pd.Timestamp("2026-02-26 15:34:20")]

def episodes(mask, gap_hours=6):
    idx = mask[mask].index
    if len(idx) == 0:
        return pd.DataFrame(columns=["inicio", "fim"])
    gaps = idx.to_series().diff().dt.total_seconds().fillna(99999)
    ep_id = (gaps > gap_hours * 3600).cumsum()
    eps = idx.to_series().groupby(ep_id).agg(["min", "max"])
    eps.columns = ["inicio", "fim"]
    return eps

# --- hipotese 1: declinio de temperatura (rolling max - valor >= 1.2C, 5 sensores juntos) ---
WIN = 2400
decline = pd.DataFrame(index=on.index)
for s in TEMPS:
    decline[s] = on[s].rolling(WIN, min_periods=WIN).max() - on[s]
temp_mask = (decline >= 1.2).all(axis=1)
temp_eps = episodes(temp_mask)

# --- hipotese 2: elevacao rara de PDIT_0305 (>= 0.3) ---
pdit_mask = on["954005_624_PDIT_0305"] >= 0.3
pdit_eps = episodes(pdit_mask)

fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.8), sharex=True)

ax = axes[0]
for _, row in temp_eps.iterrows():
    ax.axvspan(row["inicio"], row["fim"] + pd.Timedelta(hours=2), color=NEUTRAL, alpha=0.5, linewidth=0)
for t in genuine_trips:
    ax.axvline(t, color=CRITICAL, linewidth=1.3, linestyle=(0, (4, 2)))
ax.set_yticks([])
ax.set_title(f"Hipótese 1 — declínio de temperatura do óleo/mancais (cinza = {len(temp_eps)} episódios em 2,3 anos; "
             "só 0,8% seguidos de trip) — REFUTADA", fontsize=10.3, loc="left")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)

ax = axes[1]
for _, row in pdit_eps.iterrows():
    ax.axvline(row["inicio"], color=NEUTRAL, linewidth=2.2, alpha=0.8)
for t in genuine_trips:
    ax.axvline(t, color=CRITICAL, linewidth=1.3, linestyle=(0, (4, 2)))
ax.set_yticks([])
ax.set_title(f"Hipótese 2 — elevação rara de PDIT_0305 (cinza = {len(pdit_eps)} episódios; 12,5% seguidos de trip, "
             "2 deles antes de 2026-02-26) — testada no modelo real e também REFUTADA", fontsize=10.3, loc="left")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color=NEUTRAL, lw=4, alpha=0.7, label="episódios candidatos a precursor"),
    Line2D([0], [0], color=CRITICAL, lw=1.5, linestyle=(0, (4, 2)), label="os 3 trips genuínos conhecidos"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False, fontsize=9.2, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Duas hipóteses de precursor testadas e refutadas (EXP16c)", fontsize=13.2, y=1.02)
fig.tight_layout()
fig.savefig("fig_precursores_refutados.png", dpi=200, bbox_inches="tight", facecolor="white")
print("ok")
print("temp_eps:", len(temp_eps), "pdit_eps:", len(pdit_eps))
