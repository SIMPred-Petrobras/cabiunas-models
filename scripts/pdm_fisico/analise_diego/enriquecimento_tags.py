#!/usr/bin/env python3
"""As 5 tags do canal 4 sao INFORMATIVAS ou apenas MOVIMENTADAS?

Estar presente numa janela de 48 h nao significa nada se a tag dispara 12x/mes.
O controle: comparar a taxa de presenca nas janelas dos 8 trips com a taxa em
janelas SORTEADAS na mesma operacao. E o mesmo controle negativo que o Diego
aplicou corretamente no Passo 3 do relatorio dele (6,9x) e omitiu no Passo 2.
"""
import os
import numpy as np, pandas as pd
from clearml import Dataset
from testa_nossa_decisao import carrega

TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
RNG = np.random.default_rng(20260906)
N = 20000
JAN = pd.Timedelta("48h")

idx, canais, op, ft = carrega()
root = Dataset.get(dataset_id="a97ba56ba14840fbb1125c2a82f883c9").get_local_copy()
a = pd.read_csv(os.path.join(root, "alarmes_selecionados_turbina_a.csv"))
a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
act = a[a["Status"].astype(str).str.startswith("ACT")].dropna(subset=["t"])
act = act[(act.t >= idx.min()) & (act.t <= idx.max())]

rod = ~op.astype(str).str.startswith("off")
elegivel = idx[rod.to_numpy()]
elegivel = elegivel[elegivel >= idx.min() + JAN]
cand = np.asarray(elegivel.astype("int64"))
jan_ns = int(JAN.value) // (1000 if elegivel.dtype == "datetime64[us]" else 1)
sorteio = RNG.choice(cand, size=(N, len(ft)))

print("ENRIQUECIMENTO DE CADA TAG NAS JANELAS DE 48 h ANTES DOS TRIPS")
print("=" * 96)
print(f"{'tag':>16} {'/mes':>7} {'nos 8 trips':>13} {'nulo (sorteado)':>17} "
      f"{'enriq.':>9} {'p':>8}")
print("-" * 96)
for tg in TAGS + ["QUALQUER DAS 5"]:
    s = act if tg == "QUALQUER DAS 5" else act[act["Tag Alarme"] == tg]
    if tg == "QUALQUER DAS 5":
        s = act[act["Tag Alarme"].isin(TAGS)]
    tv = np.sort(np.asarray(pd.DatetimeIndex(s.t).astype("int64")))
    if len(tv) == 0:
        continue
    obs = sum(1 for t in ft
              if np.searchsorted(tv, int(pd.Timestamp(t).value)//1000, "right")
                 - np.searchsorted(tv, int((pd.Timestamp(t)-JAN).value)//1000, "left") > 0)
    hi = np.searchsorted(tv, sorteio, "right")
    lo = np.searchsorted(tv, sorteio - jan_ns, "left")
    d = (hi - lo > 0).sum(axis=1)
    esp = d.mean(); p = float((d >= obs).mean())
    taxa = len(s)/((idx.max()-idx.min()).days/30.44)
    print(f"{tg:>16} {taxa:7.1f} {obs:9d}/{len(ft)} {esp:14.2f}/{len(ft)} "
          f"{(obs/esp if esp else np.inf):9.2f}x {p:8.4f}"
          + ("  ***" if p < 0.05 else "  <-- nao informativo"))
print("-" * 96)
print("  referencia: o Passo 3 do relatorio dele mede 6,9x e chama de 'forte' -- corretamente")
