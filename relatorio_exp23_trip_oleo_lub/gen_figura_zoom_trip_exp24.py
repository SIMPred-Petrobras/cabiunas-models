"""Segundo plote: N sensores empilhados (linhas), com zoom nas janelas
onde o EXP24 (so filtro de duracao + portao de volatilidade) detectou
algo perto de cada um dos 2 alarmes de TRIP do periodo OOS.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp23_trip_oleo_lub/gen_figura_zoom_trip_exp24.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/33262a70727aa94f9c628d32041e3718.point_anomalies_all.csv"
SENSORS = ["954005_624_PI_0308", "954005_624_PDIT_0305", "TV_351X_A", "TV_354Y_A"]
LABELS = {
    "954005_624_PI_0308": "PI_0308 (pressão óleo, alvo)",
    "954005_624_PDIT_0305": "PDIT_0305 (diferencial pressão óleo)",
    "TV_351X_A": "TV_351X_A (vibração, referência)",
    "TV_354Y_A": "TV_354Y_A (vibração, mancal 354)",
}

EVENTS = [
    ("2025-11-04 06:22:18", "TRIP #1 -- 2025-11-04"),
    ("2026-02-26 15:34:20", "TRIP #2 -- 2026-02-26"),
]

print("lendo point_anomalies_all.csv (EXP24 -- portão seguro)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)

windows = []
for t_str, label in EVENTS:
    t0 = pd.Timestamp(t_str)
    win = df_point.loc[(df_point.index >= t0 - pd.Timedelta(hours=24)) & (df_point.index <= t0 + pd.Timedelta(hours=24))]
    anom = win.index[win["is_anom_point"] == 1]
    if len(anom):
        zoom_start = anom.min() - pd.Timedelta(hours=2)
        zoom_end = anom.max() + pd.Timedelta(hours=2)
    else:
        zoom_start, zoom_end = t0 - pd.Timedelta(hours=3), t0 + pd.Timedelta(hours=3)
    windows.append((label, t0, zoom_start, zoom_end, anom))
    print(f"{label}: zoom {zoom_start} a {zoom_end}  (n_anom={len(anom)})", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo series brutas dos sensores...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime"] + SENSORS)
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
for c in SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()

n_rows = len(SENSORS)
n_cols = len(EVENTS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False)

for col, (label, t_alarm, zoom_start, zoom_end, anom) in enumerate(windows):
    sub_raw = df_raw.loc[zoom_start:zoom_end]
    for row, sensor in enumerate(SENSORS):
        ax = axes[row][col]
        ax.plot(sub_raw.index, sub_raw[sensor], color="#184F95", linewidth=0.8, zorder=1)
        vals = sub_raw[sensor].reindex(anom.intersection(sub_raw.index))
        ax.scatter(vals.index, vals.values, color="#D03B3B", s=14, zorder=3, label="anomalia detectada")
        ax.axvline(t_alarm, color="#0CA30C", linewidth=1.4, linestyle="--", alpha=0.8, label="ocorrência do TRIP")
        ax.set_ylabel(LABELS[sensor], fontsize=8)
        if row == 0:
            ax.set_title(label, fontsize=10)
        if row == 0 and col == 0:
            ax.legend(loc="upper left", fontsize=7, frameon=True)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)

fig.suptitle("EXP24 (duração + volatilidade) -- zoom nos 2 alarmes de TRIP (OOS), 4 sensores", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = os.path.join(OUT, "serie_zoom_trip_exp24.png")
fig.savefig(out_path, dpi=160)
print("\nfigura salva em", out_path)
