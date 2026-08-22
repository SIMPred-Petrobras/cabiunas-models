#!/usr/bin/env python3
"""A janela de 48 h e convencao, nunca foi medida (secao 9, item 2 do relatorio
do Francisco). Aqui ela vira parametro: 24 h, 48 h, 7 d e 14 d, com o falso
positivo recontado sob a mesma janela -- alargar a janela nao pode ser de graca,
ela tambem absolve episodios que antes eram falso positivo."""
import numpy as np, pandas as pd
import detector as D, avalia as A, rolante as RO, familias as F

CORTE = pd.Timestamp("2025-07-01", tz="UTC")
ev_all = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
q = pd.read_parquet("quente.parquet")["q"]
Z = pd.read_parquet("Z_rolante.parquet")
idx = Z.index; te = pd.Series(idx >= CORTE, index=idx); tr = ~te
ev_tr, ev_te = ev_all[ev_all < CORTE], ev_all[ev_all >= CORTE]

for conj, hl, c, sus in [("mecanica", 8.0, 3.0, 120), ("maquina", 2.0, 5.0, 120),
                         ("todas", 8.0, 1.5, 120)]:
    cols = [x for f in F.CONJ[conj] for x in F.FAM[f]]
    Zs = Z[cols].ewm(halflife=int(pd.Timedelta(hours=hl) / D.PAS), min_periods=1).mean()
    r, culp = RO.limiar_rolante(Zs, q, ev_all, 0.05, guarda_h=24)
    al = A.sustenta((r > c).fillna(False), sus) & q
    print(f"\n=== {conj} hl={hl} corte={c} sus={sus}")
    for J in [24, 48, 168, 336]:
        out = []
        for tag, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te)]:
            am, qm = al[m], (q & m)[m]
            x = A.avalia(am, ev, qm, janela_h=J)
            x.update(A.permuta(am, qm, x["det"], len(ev), janela_h=J))
            out.append(f"{tag}: {x['det']}/{x['n_ev']} lead={x['lead_med']:.0f}h "
                       f"FP={x['fp_mes']:.2f}/mes {x['h_fp_mes']:.0f}h/mes p={x['p']:.3f}")
        print(f"  janela {J:>3}h -> " + "   |   ".join(out))
    # veredito por evento com janela de 7 dias
    print("  por evento (janela 7 d):")
    for t in ev_all:
        w = al[(idx >= t - pd.Timedelta(days=7)) & (idx < t)]; w = w[w]
        meta = "TESTE" if t >= CORTE else "treino"
        if len(w):
            cc = culp.loc[w.index].value_counts()
            print(f"    {t.strftime('%Y-%m-%d')} [{meta}] lead={(t-w.index[0]).total_seconds()/3600:6.1f}h  {cc.index[0]}")
        else:
            print(f"    {t.strftime('%Y-%m-%d')} [{meta}] -- nao visto")
