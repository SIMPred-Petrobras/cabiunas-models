#!/usr/bin/env python3
"""Quando exatamente o canal 4 acende dentro da janela de 48 h, e por qual tag.

A checagem anterior olhou so as 24 h antes do trip -- mas o canal acende por 24 h
A PARTIR da ativacao, entao uma tag que dispara 40 h antes mantem o canal aceso
de -40 h a -16 h. Janela correta: 72 h.
"""
import os, pandas as pd, numpy as np
from clearml import Dataset
from testa_nossa_decisao import carrega
from src.cnn1d_ae.scoring import apply_refractory, apply_min_duration_filter

TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
UTILIDADE = {"PI_6240319_AL", "PAH_6240319", "PAL_6240315"}   # gas / motor de partida

idx, canais, op, ft = carrega()
root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
a = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
act = a[a["Status"].astype(str).str.startswith("ACT")].dropna(subset=["t"])
act = act[act["Tag Alarme"].isin(TAGS)]

ns = sum(canais[k].astype(int) for k in canais)
v = (ns >= 2)
vf = apply_min_duration_filter(pd.DataFrame({"is_anom_point": v.astype(int)}, index=idx),
                               45.0)["is_anom_point"].astype(bool)
dec = apply_refractory(vf, refractory_minutes=48*60.0)
JAN = pd.Timedelta("48h")

print("O QUE ACENDE O CANAL 4 EM CADA DETECCAO")
print("=" * 112)
print(f"{'trip':>17} {'episodio comeca':>17} | tags que sustentam o canal na janela do episodio")
print("-" * 112)
n_util = n_total = 0
for t in ft:
    t0 = t - JAN
    j = dec.loc[t0:t]; on = j[j.fillna(False)]
    if not len(on):
        print(f"{t:%d/%m/%Y %H:%M} {'(sem alerta)':>17} |")
        continue
    i0, i1 = on.index[0], on.index[-1]
    # tags cuja janela de 24 h cobre algum instante de [i0, i1]
    s = act[(act.t >= i0 - pd.Timedelta("24h")) & (act.t <= i1)]
    if not len(s):
        print(f"{t:%d/%m/%Y %H:%M} {i0:%d/%m %H:%M} | (canal 4 nao acende aqui)")
        continue
    n_total += 1
    so_util = set(s["Tag Alarme"]) <= UTILIDADE
    n_util += so_util
    det = ", ".join(f"{tg}x{n}" for tg, n in s["Tag Alarme"].value_counts().items())
    marca = "   <-- SO utilidade/gas" if so_util else ""
    print(f"{t:%d/%m/%Y %H:%M} {i0:%d/%m %H:%M} | {det}{marca}")
print("-" * 112)
print(f"  deteccoes em que o canal 4 e sustentado SO por alarme de utilidade/gas: "
      f"{n_util} de {n_total}")

print("\n\nO CANAL 4 SEM AS TAGS DE UTILIDADE  (so PDAL selagem + TC382_05)")
print("=" * 112)
from testa_nossa_decisao import canal_alarme, mede
mecanicas = act[~act["Tag Alarme"].isin(UTILIDADE)]
c4b = canal_alarme(idx, mecanicas["t"], 24.0)
rod = (op.astype(str) != "off_longo") & (op.astype(str) != "off")
print(f"  ciclo de trabalho: {100*float((c4b & rod).sum())/int(rod.sum()):.2f}% "
      f"(era 47,62% com as cinco)")
c2 = dict(canais); c2["alarme"] = c4b
ns2 = sum(c2[k].astype(int) for k in c2)
v2 = (ns2 >= 2)
vf2 = apply_min_duration_filter(pd.DataFrame({"is_anom_point": v2.astype(int)}, index=idx),
                                45.0)["is_anom_point"].astype(bool)
dec2 = apply_refractory(vf2, refractory_minutes=48*60.0)
m2, eps2, cls2 = mede(dec2, op, ft)
print(f"  resultado dele SEM as 3 tags de utilidade: "
      f"{m2['falhas_detectadas']}/8  FP/mes {m2['falso_positivo_por_mes']:.2f}  "
      f"inconclusivo {m2['n_episodios_inconclusivo']}")
print(f"     (com as cinco: 8/8, 2,88, 25)")
