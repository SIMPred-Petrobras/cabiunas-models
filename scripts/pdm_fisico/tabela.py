#!/usr/bin/env python3
"""Tabela completa treino x teste. So os pontos 'mecanica' foram pre-registrados;
o resto e analise secundaria e esta aqui para nao esconder o espaco de busca."""
import numpy as np, pandas as pd
import detector as D, avalia as A, rolante as RO, familias as F

CORTE = pd.Timestamp("2025-07-01", tz="UTC")
ev_all = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
q = pd.read_parquet("quente.parquet")["q"]
Z = pd.read_parquet("Z_rolante.parquet")
idx = Z.index
tr = pd.Series(idx < CORTE, index=idx); te = ~tr
ev_tr, ev_te = ev_all[ev_all < CORTE], ev_all[ev_all >= CORTE]

linhas = []
for hl in [2.0, 8.0]:
    Zf = Z.ewm(halflife=int(pd.Timedelta(hours=hl) / D.PAS), min_periods=1).mean()
    for conj, fams in F.CONJ.items():
        cols = [c for f in fams for c in F.FAM[f] if c in Zf.columns]
        r, _ = RO.limiar_rolante(Zf[cols], q, ev_all, 0.05, guarda_h=24)
        for c in [1.5, 2.0, 3.0, 5.0, 8.0]:
            for sus in [120, 480]:
                al = A.sustenta((r > c).fillna(False), sus) & q
                d = dict(conj=conj, hl=hl, corte=c, sus=sus)
                for tag, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te)]:
                    am, qm = al[m], (q & m)[m]
                    x = A.avalia(am, ev, qm); x.update(A.permuta(am, qm, x["det"], len(ev)))
                    d.update({f"{tag}_det": f"{x['det']}/{x['n_ev']}", f"{tag}_lead": x["lead_med"],
                              f"{tag}_fpmes": x["fp_mes"], f"{tag}_hmes": x["h_fp_mes"],
                              f"{tag}_p": x["p"]})
                linhas.append(d)
t = pd.DataFrame(linhas)
t.to_csv("tabela_tr_te.csv", index=False)
print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
