"""Visao geral dos ~26 meses avaliados: score de temperatura (max diario,
escala log) com os 11 eventos curados marcados e os alertas confirmados
(E) destacados -- mostra a serie inteira, nao so o zoom por evento."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0ca30c"
CRITICAL = "#d03b3b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"

print("lendo series...")
df = pd.read_csv("series_v6_para_plots.csv.gz", parse_dates=["timestamp"]).set_index("timestamp")
eventos = pd.read_csv("eventos_v6_hit_miss.csv", parse_dates=["evento"])

daily_max_temp = df["temp_score"].resample("D").max()
daily_max_mancal = df["mancal_z"].abs().resample("D").max()
daily_and = df["alert_and_producao"].resample("D").max()

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax = axes[0]
ax.plot(daily_max_temp.index, daily_max_temp.values + 1.0, color=BLUE, linewidth=0.8)
ax.set_yscale("log")
ax.set_ylabel("score temperatura\n(máx. diário, log)")
for _, row in eventos.iterrows():
    detectado = bool(row["detectado_producao_2sinais_AND"])
    cor = GREEN if detectado else CRITICAL
    ax.axvline(row["evento"], color=cor, linewidth=1.1, alpha=0.85, zorder=3)
ax.set_title("Score de temperatura (PCA-Q) — visão geral do período avaliado", fontsize=11, loc="left")

ax = axes[1]
and_days = daily_and.index[daily_and.values > 0]
for d in and_days:
    ax.axvspan(d, d + pd.Timedelta(days=1), color=CRITICAL, alpha=0.6)
for _, row in eventos.iterrows():
    detectado = bool(row["detectado_producao_2sinais_AND"])
    cor = GREEN if detectado else CRITICAL
    marker = "o" if detectado else "x"
    ax.plot(row["evento"], 0.5, marker=marker, color=cor, markersize=9, markeredgewidth=2)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel("alerta\nconfirmado (E)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# legenda manual
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markeredgecolor=GREEN, markersize=9, label="evento detectado (E)"),
    Line2D([0], [0], marker="x", color=CRITICAL, markersize=9, markeredgewidth=2, label="evento não detectado"),
    Line2D([0], [0], color=CRITICAL, alpha=0.6, linewidth=6, label="dia com pelo menos 1 alerta E confirmado"),
]
ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=9, frameon=False)

plt.tight_layout()
fig.savefig("fig_serie_completa.png", dpi=160, facecolor="white", bbox_inches="tight")
print("ok: fig_serie_completa.png")
