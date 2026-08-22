#!/usr/bin/env python3
"""Varredura do corte da RAZAO -- o botao continuo que faltava.

alpha saturou porque o quantil de Sidak ja esta no topo da janela de referencia;
mexer nele nao muda nada. A razao z/limiar, ao contrario, e continua e vai a
5, 10, 20. Cortar nela e o que leva o detector ate a faixa de falso positivo do
Francisco (0,3 a 0,8 por mes), que e a unica faixa em que os dois numeros podem
ser comparados.

Tudo aqui roda SO na primeira metade (ate 2025-07-01).
"""
import numpy as np, pandas as pd
import detector as D, avalia as A, rolante as RO, familias as F

CORTE = pd.Timestamp("2025-07-01", tz="UTC")
ev_all = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
q = pd.read_parquet("quente.parquet")["q"]
Z = pd.read_parquet("Z_rolante.parquet")

tr = pd.Series(Z.index < CORTE, index=Z.index)
ev = ev_all[ev_all < CORTE]
print(f"treino: {(q & tr).sum()*2/60:.0f} h quentes, {len(ev)} eventos\n", flush=True)

linhas = []
for hl in [2.0, 8.0]:
    Zs = Z.ewm(halflife=int(pd.Timedelta(hours=hl) / D.PAS), min_periods=1).mean()
    for conj, fams in F.CONJ.items():
        cols = [c for f in fams for c in F.FAM[f] if c in Zs.columns]
        r, culp = RO.limiar_rolante(Zs[cols], q, ev_all, 0.05, guarda_h=24)
        for c in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            for sus in [120, 480, 1440]:
                alerta = A.sustenta((r > c).fillna(False), sus) & q & tr
                res = A.avalia(alerta[tr], ev, q & tr)
                res.update(A.permuta(alerta[tr], q & tr, res["det"], len(ev)))
                res.update(conj=conj, ewma_h=hl, corte=c, sust_min=sus, n_canais=len(cols))
                linhas.append(res)
                print(f"{conj:9s} hl{hl:<4} c={c:<4} s{sus:>4} -> {res['det']}/{res['n_ev']} "
                      f"lead={res['lead_med']:.0f}h FP={res['fp_mes']:.2f}/mes "
                      f"{res['h_fp_mes']:.2f}h/mes duty={res['duty']:.4f} "
                      f"nulo={res['nulo']:.2f} p={res['p']:.4f}", flush=True)
        pd.DataFrame(linhas).to_csv("explora2.csv", index=False)
print("fim")
