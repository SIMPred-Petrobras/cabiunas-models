#!/usr/bin/env python3
"""O teto de permanencia, reavaliado sob a REGRA DE INICIO.

`teto_permanencia.py` refutou o teto: nenhum valor preservava a deteccao, e a
120 h dava 6/8. Mas aquilo foi medido sob a regra "de pe", onde quebrar um
episodio longo so pode PERDER a deteccao -- o alarme ja estava de pe e o teto o
derruba.

Sob a regra de INICIO a logica se inverte: um episodio de 670 h nao conta como
deteccao de jeito nenhum. Quebra-lo com rearme pode CRIAR um inicio dentro da
janela de 48 h, e ai passa a contar. O teto deixa de ser custo e vira mecanismo.

Varre teto x rearme e pontua nas duas reguas, para ver o trade explicito.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from teto_permanencia import teto_com_rearme
from regra_inicio_varredura import avalia_inicio

TETO = [None, 6, 12, 24, 48, 72, 120]
REARME = [1, 3, 6, 12]
KBs, KVs = [1.3, 1.5, 1.7, 2.0], [1.8, 2.2, 2.8]

lin = []
for kb in KBs:
    for kv in KVs:
        ON = partes(kb, kv)
        ns = sum(ON[c].astype(int) for c in SIN)
        v0 = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
        for th in TETO:
            for rh in (REARME if th else [0]):
                v = teto_com_rearme(v0, th, rh) if th else v0
                al = pos(v, ns, REFRAT_H, DUR_MIN, False)
                mi = avalia_inicio(al); mp = AV.avalia(al, alvo, mask)
                lin.append(dict(kb=kb, kv=kv, teto=th or 0, rearme=rh,
                                det_ini=mi["det"], det_pe=mp["det"],
                                fp_mes=round(mi["fp_mes"], 3),
                                h_fp_mes=round(mi["h_fp_mes"], 1),
                                lead=round(mi["lead_med"], 2) if mi["det"] else np.nan,
                                fp_mes_pe=round(mp["fp_mes"], 3),
                                h_fp_mes_pe=round(mp["h_fp_mes"], 1)))
    print(f"  kb={kb} ok", flush=True)

d = pd.DataFrame(lin)
d.to_csv("teto_regra_inicio.csv", index=False)
print(f"\n{len(d)} pontos -> teto_regra_inicio.csv\n")

print("MELHOR POR NUMERO DE DETECCOES SOB A REGRA DE INICIO")
print("=" * 100)
print(f"{'det_ini':>8} {'n':>5} {'FP/mes':>9} {'h/mes':>8} {'lead':>8} "
      f"{'det_pe':>7} {'FP/mes(pe)':>11}  configuracao")
print("-" * 100)
for k in sorted(d.det_ini.unique(), reverse=True):
    s = d[d.det_ini == k].sort_values(["fp_mes", "h_fp_mes"])
    b = s.iloc[0]
    print(f"{k:6d}/8 {len(s):5d} {b.fp_mes:9.3f} {b.h_fp_mes:8.1f} {b.lead:7.1f}h "
          f"{int(b.det_pe):6d}/8 {b.fp_mes_pe:11.3f}  teto={b.teto or '--'}h "
          f"rearme={b.rearme}h k={b.kb}/{b.kv}")

print("\nEFEITO DO TETO NO PONTO DE PRODUCAO (k=1,7/2,2)")
print("=" * 100)
s = d[(d.kb == 1.7) & (d.kv == 2.2)].sort_values(["teto", "rearme"])
print(f"{'teto':>6} {'rearme':>7} | {'det_ini':>8} {'FP/mes':>8} {'h/mes':>8} {'lead':>7} "
      f"| {'det_pe':>7} {'FP/mes':>8} {'h/mes':>8}")
print("-" * 100)
for _, r in s.iterrows():
    print(f"{(str(int(r.teto))+'h' if r.teto else 'sem'):>6} "
          f"{(str(int(r.rearme))+'h' if r.teto else '--'):>7} | "
          f"{r.det_ini:6.0f}/8 {r.fp_mes:8.3f} {r.h_fp_mes:8.1f} "
          f"{r.lead if np.isfinite(r.lead) else 0:6.1f}h | "
          f"{r.det_pe:5.0f}/8 {r.fp_mes_pe:8.3f} {r.h_fp_mes_pe:8.1f}")
