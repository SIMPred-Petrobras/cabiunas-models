#!/usr/bin/env python3
"""Conjuncao vibracao E mancal.

Argumento fisico: uma excursao de vibracao por mudanca de carga NAO vem
acompanhada de aumento da temperatura do metal do mancal sobre o oleo -- o
calor gerado no filme depende da folga e do alinhamento, nao do ponto de
operacao. Exigir as duas coisas ao mesmo tempo deveria matar o falso positivo
de vibracao sem custar as deteccoes, porque nos quatro trips de mancal as duas
sobem juntas (vib z +7 a +17, mancal z +4,0 a +5,3).

E a regra de voto do Francisco aplicada ao par que a fisica indica, em vez de a
familias inteiras de sensores.
"""
import numpy as np, pandas as pd
import detector as D, avalia as A, rolante as RO, familias as F

CORTE = pd.Timestamp("2025-07-01", tz="UTC")
ev_all = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
q = pd.read_parquet("quente.parquet")["q"]
Z = pd.read_parquet("Z_rolante.parquet")
idx = Z.index; te = pd.Series(idx >= CORTE, index=idx); tr = ~te
ev_tr, ev_te = ev_all[ev_all < CORTE], ev_all[ev_all >= CORTE]

linhas = []
for hl in [2.0, 8.0]:
    Zs = Z.ewm(halflife=int(pd.Timedelta(hours=hl) / D.PAS), min_periods=1).mean()
    rv, _ = RO.limiar_rolante(Zs[F.FAM["vibracao"]], q, ev_all, 0.05, guarda_h=24)
    rm, _ = RO.limiar_rolante(Zs[F.FAM["mancal"]], q, ev_all, 0.05, guarda_h=24)
    for cv in [1.5, 2.0, 3.0, 5.0]:
        for cm in [1.0, 1.5, 2.0]:
            for sus in [120, 480]:
                bruto = ((rv > cv) & (rm > cm)).fillna(False)
                al = A.sustenta(bruto, sus) & q
                d = dict(hl=hl, corte_vib=cv, corte_manc=cm, sus=sus)
                for tag, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te)]:
                    am, qm = al[m], (q & m)[m]
                    x = A.avalia(am, ev, qm); x.update(A.permuta(am, qm, x["det"], len(ev)))
                    d.update({f"{tag}_det": f"{x['det']}/{x['n_ev']}", f"{tag}_lead": x["lead_med"],
                              f"{tag}_fp": x["fp_mes"], f"{tag}_h": x["h_fp_mes"], f"{tag}_p": x["p"]})
                linhas.append(d)
t = pd.DataFrame(linhas); t.to_csv("conjuncao.csv", index=False)
print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
