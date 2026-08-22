#!/usr/bin/env python3
"""TESTE FORA DA AMOSTRA. Os dois pontos abaixo foram fixados olhando somente
2024-01 a 2025-06; nada aqui foi ajustado depois de ver a segunda metade."""
import numpy as np, pandas as pd
import detector as D, avalia as A, rolante as RO, familias as F

CORTE = pd.Timestamp("2025-07-01", tz="UTC")
PONTOS = {
    "conservador": dict(hl=2.0, corte=8.0, sus=120),
    "sensivel":    dict(hl=8.0, corte=3.0, sus=120),
}
g = pd.read_parquet("grade2min.parquet").drop(columns=["HSX_6240001A"])
ev_all = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
q = pd.read_parquet("quente.parquet")["q"]
Z = pd.read_parquet("Z_rolante.parquet")
cols = [c for f in F.CONJ["mecanica"] for c in F.FAM[f]]
idx = Z.index
tr = pd.Series(idx < CORTE, index=idx); te = ~tr

for nome, P in PONTOS.items():
    Zs = Z[cols].ewm(halflife=int(pd.Timedelta(hours=P["hl"]) / D.PAS), min_periods=1).mean()
    r, culp = RO.limiar_rolante(Zs, q, ev_all, 0.05, guarda_h=24)
    alerta = A.sustenta((r > P["corte"]).fillna(False), P["sus"]) & q
    for lab, m, ev in [("TREINO 2024-01..2025-06", tr, ev_all[ev_all < CORTE]),
                       ("TESTE  2025-07..2026-04", te, ev_all[ev_all >= CORTE])]:
        am, qm = alerta[m], (q & m)[m]
        res = A.avalia(am, ev, qm)
        res.update(A.permuta(am, qm, res["det"], len(ev)))
        print(f"[{nome:11s}] {lab}: {res['det']}/{res['n_ev']}  "
              f"lead={res['lead_med']:.1f}h (min {res['lead_min']:.1f}h)  "
              f"FP={res['fp_mes']:.2f}/mes  {res['h_fp_mes']:.1f}h/mes  "
              f"op={res['horas_op']:.0f}h  nulo={res['nulo']:.2f}  p={res['p']:.4f}")
        print(f"{'':16s}detectados: {res['detectados']}")
    # quem disparou, por evento
    print(f"{'':16s}canal dominante nos alertas de deteccao:")
    for t in ev_all:
        w = alerta[(idx >= t - pd.Timedelta(hours=48)) & (idx < t)]
        w = w[w]
        if len(w):
            c = culp.loc[w.index].value_counts()
            lead = (t - w.index[0]).total_seconds()/3600
            print(f"{'':18s}{t.strftime('%Y-%m-%d')}  lead={lead:5.1f}h  {c.index[0]} ({c.iloc[0]*2/60:.1f}h)")
        else:
            print(f"{'':18s}{t.strftime('%Y-%m-%d')}  -- nao visto")
    print()
