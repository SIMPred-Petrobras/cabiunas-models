"""Figura geral: serie bruta do sensor-alvo (954005_624_PI_0308) do EXP16c
(pipeline "pura", sem nenhum portao novo -- 100% hit_rate/2 alarmes,
7,6% FP) com as anomalias detectadas sobrepostas e cada ocorrencia de
alarme de TRIP (PALL_6240309, catalogo inteiro) marcada como linha
vertical tracejada.

Uso:
    python3 /home/dvar/REPO_CABIUNAS/cabiunas-models/relatorio_exp23_trip_oleo_lub/gen_figura_geral_trip.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from clearml import Dataset

OUT = os.path.dirname(os.path.abspath(__file__))
POINT_CSV = "/home/dvar/.clearml/cache/storage_manager/global/152759f2ef15b33fd8c956d41a80ad62.point_anomalies_all.csv"
TARGET_SENSOR = "954005_624_PI_0308"
DATA_END = pd.Timestamp("2026-04-20 23:59:59")

print("lendo point_anomalies_all.csv (EXP23 -- pilha de portões atual)...", flush=True)
df_point = pd.read_csv(POINT_CSV, index_col=0, parse_dates=True)
df_point = df_point.loc[df_point.index <= DATA_END]
anom_times = df_point.index[df_point["is_anom_point"] == 1]
print(f"anomalias detectadas: {len(anom_times)}", flush=True)

root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
raw_path = os.path.join(root, "sensores_full_2024_2026_30s.csv")
print("lendo serie bruta do sensor-alvo...", flush=True)
df_raw = pd.read_csv(raw_path, usecols=["data_datetime", TARGET_SENSOR])
df_raw["data_datetime"] = pd.to_datetime(df_raw["data_datetime"], errors="coerce")
df_raw[TARGET_SENSOR] = pd.to_numeric(df_raw[TARGET_SENSOR], errors="coerce")
df_raw = df_raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
df_raw = df_raw.loc[(df_raw.index >= "2024-01-01") & (df_raw.index <= DATA_END)]

alarm_path = os.path.join(root, "alarmes_selecionados_turbina_a.csv")
alarm = pd.read_csv(alarm_path)
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia").reset_index(drop=True)
trip = alarm.loc[alarm["Tag"] == "PALL_6240309", "Data da Ocorrencia"]
trip = trip.loc[(trip >= df_raw.index.min()) & (trip <= df_raw.index.max())]
print(f"ocorrencias de TRIP (PALL_6240309) no periodo: {len(trip)}", flush=True)

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df_raw.index, df_raw[TARGET_SENSOR], color="#184F95", linewidth=0.3, zorder=1,
        label=f"{TARGET_SENSOR} (bruto)")

vals = df_raw[TARGET_SENSOR].reindex(anom_times.intersection(df_raw.index))
ax.scatter(vals.index, vals.values, color="#D03B3B", s=10, zorder=3,
           label=f"anomalias detectadas (n={len(anom_times)})")

for i, t in enumerate(trip):
    ax.axvline(t, color="#0CA30C", linewidth=1.3, linestyle="--", alpha=0.75, zorder=2,
               label=f"ocorrência de TRIP (n={len(trip)})" if i == 0 else None)

ax.set_xlim(df_raw.index.min(), df_raw.index.max())
ax.set_ylabel(f"{TARGET_SENSOR}")
ax.set_xlabel("tempo")
ax.set_title("EXP23 (com pilha de portões) -- anomalias x alarmes de TRIP (PALL_6240309)")
ax.legend(loc="upper left", fontsize=9, frameon=True)
fig.tight_layout()

out_path = os.path.join(OUT, "serie_geral_trip_exp23.png")
fig.savefig(out_path, dpi=170)
print("\nfigura salva em", out_path)
