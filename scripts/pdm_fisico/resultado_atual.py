#!/usr/bin/env python3
"""O ponto PUBLICADO, medido sob todas as reguas de uma vez. Fonte unica."""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from plota_estilo_francisco import alarme, paradas_reais_2h, classifica_regra_c

JAN = pd.Timedelta("48h")
TMIN = pd.Timedelta("4h")
al = alarme(); eps = AV.episodios(al)
paradas = paradas_reais_2h(); cls = classifica_regra_c(eps, paradas)
meses = float(mask.sum())*2/60.0/730.0
m = AV.avalia(al, alvo, mask)

nfp = sum(1 for _, _, k, _ in cls if k == "FP")
nnt = sum(1 for _, _, k, _ in cls if k == "NEUTRO")
ntp = sum(1 for _, _, k, _ in cls if k == "TP")
hfp = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")

det_pe = sum(1 for t in alvo if bool(al.loc[t-JAN:t].any()))
det_ini = sum(1 for t in alvo if any(t-JAN <= a <= t for a, _ in eps))
det_band = sum(1 for t in alvo if any(t-JAN <= a <= t-TMIN for a, _ in eps))
l_pe = m["lead_med"]
l_ini = np.mean([ (t-max([a for a,_ in eps if t-JAN <= a <= t])).total_seconds()/3600
                  for t in alvo if any(t-JAN <= a <= t for a,_ in eps)])
l_band = np.mean([ (t-max([a for a,_ in eps if t-JAN <= a <= t-TMIN])).total_seconds()/3600
                   for t in alvo if any(t-JAN <= a <= t-TMIN for a,_ in eps)])

print("PONTO PUBLICADO -- publica_clearml.py")
print("=" * 78)
print(f"  janela avaliada     : {idx[0]:%d/%m/%Y} a {idx[-1]:%d/%m/%Y}")
print(f"  operacao vigiada    : {meses:.2f} meses ({float(mask.sum())*2/60/24:.0f} dias)")
print(f"  episodios de alarme : {len(eps)}  =  {ntp} TP + {nnt} NEUTRO + {nfp} FP")
print()
print(f"{'regua':>34}{'deteccao':>12}{'lead medio':>14}")
print("-" * 78)
print(f"{'de pe na janela (a NOSSA)':>34}{det_pe:9d}/8{l_pe:12.1f} h   <- censurado em 48h")
print(f"{'inicio na janela (a DELES)':>34}{det_ini:9d}/8{l_ini:12.1f} h")
print(f"{'banda acionavel [4h, 48h]':>34}{det_band:9d}/8{l_band:12.1f} h")
print("-" * 78)
print(f"\n{'custo':>34}")
print("-" * 78)
print(f"{'FP/mes (regra C)':>34}{nfp/meses:12.3f}")
print(f"{'FP/mes (bruto, sem regra C)':>34}{(nfp+nnt)/meses:12.3f}")
print(f"{'horas de alarme falso por mes':>34}{hfp/meses:12.1f}")
print(f"{'ciclo de trabalho do alarme':>34}{100*float(al.sum())/float(mask.sum()):11.2f}%")

print("\n\nPOR EVENTO")
print("=" * 78)
print(f"{'trip':>12}{'como aparece':>34}{'na banda?':>14}")
print("-" * 78)
for t in alvo:
    i = [a for a, _ in eps if t-JAN <= a <= t]
    if i:
        a = max(i); b = [y for x, y in eps if x == a][0]
        lead = (t-a).total_seconds()/3600
        s = f"nasce {lead:.1f}h antes, dura {(b-a).total_seconds()/60:.0f}min"
        ok = "SIM" if lead >= 4 else "nao (<4h)"
    else:
        d = [(x, y) for x, y in eps if x <= t and y >= t-JAN]
        s = f"alarme de pe ha {(t-d[0][0]).total_seconds()/3600:.0f}h" if d else "nao detecta"
        ok = "nao"
    print(f"{t:%d/%m/%Y}{s:>34}{ok:>14}")
