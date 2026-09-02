#!/usr/bin/env python3
"""Reavalia o ponto de operacao com a janela oficial de 2025-01 a 2026-04.

Motivo. Os 8 eventos do alvo sao TODOS de 2025-2026 (7 em 2025, 1 em 2026); 2024 entra
so como 6,3 meses de operacao sem evento. Incluir 2024 dilui o custo com um regime que
nao e mais o da maquina: a taxa de FP cai (4,45 contra 4,82) mas as horas por mes sobem
(109,3 contra 126,9) -- 2024 tinha episodios longos e raros, 2025-2026 tem curtos e
numerosos, coerente com a deriva ja diagnosticada.

Restringir a 2025+ nao perde nenhum evento e da o numero mais representativo do estado
atual -- e mais conservador, que e a postura certa para reportar.

Este script refaz nessa janela: a varredura completa do ponto de operacao, a curva de
troca, o LOEO e a deriva por campanha. A pergunta e se o ponto recomendado continua
sendo o certo quando a janela muda.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from auto_reset import trunca
import reduz_fp as RF

T0 = pd.Timestamp("2025-01-01", tz="UTC")
KB = [1.2, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0]
KV = [1.6, 2.2, 2.8, 3.5, 4.5]
RS = [0, 12, 24, 36, 48, 72, 120]
DS = [0, 60]
TE = [0, 12]


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    maskf = mascara_pontuacao(df)
    sel = (idx >= T0)
    mask = maskf & sel
    alvo = todas[todas >= T0]
    m2 = mask[sel]
    meses = mask.sum() * 2 / 60 / 730.0
    out = roda(BRACO, df, todas)

    op = df["in_operation"].astype(bool)
    gid = (op & ~op.shift(fill_value=False)).cumsum().where(op)
    camps = []
    for _, sub in df[op].groupby(gid[op]):
        a, b = sub.index[0], sub.index[-1]
        if a < T0: continue
        h = (mask & (idx >= a) & (idx <= b)).sum() * 2 / 60
        if h >= 24: camps.append((a, b, h))
    jw = [(t - pd.Timedelta(hours=48), t) for t in alvo]
    print(f"janela {T0:%Y-%m} a {idx[-1]:%Y-%m}: {len(alvo)} eventos, "
          f"{mask.sum()*2/60:.0f} h ({meses:.1f} meses), {len(camps)} campanhas\n", flush=True)

    def duty(al):
        y = []
        for a, b, h in camps:
            s = (idx >= a) & (idx <= b); eps = A.episodios(al & s)
            fp = [(x, z) for x, z in eps if not any(x <= t1 and z >= t0 for t0, t1 in jw)]
            y.append(100 * sum((z - x).total_seconds()/3600 + 2/60 for x, z in fp) / h)
        return np.array(y)

    L, ALS = [], {}
    for kb in KB:
        for kv in KV:
            b = alerta_2k(out, mask, kb, kv) & mask
            for Rh in RS:
                aR = RF.refratario(b, Rh) if Rh else b
                for D in DS:
                    aD = RF.dur_min(aR, D) if D else aR
                    for te in TE:
                        al = trunca(aD, te) if te else aD
                        x = A.avalia(al[sel], alvo, m2)
                        ALS[(kb, kv, Rh, D, te)] = al
                        L.append(dict(kb=kb, kv=kv, R=Rh, D=D, teto=te, det=x["det"],
                                      eps=x["episodios"], fp=x["fp_mes"], h=x["h_fp_mes"],
                                      lead=x["lead_med"], quais=",".join(x["detectados"])))
        print(f"  k_base={kb} varrido ({len(L)} configs)", flush=True)
    T = pd.DataFrame(L); T.to_csv("janela_2025.csv", index=False)

    print("\n" + "=" * 104)
    print("1) O PONTO RECOMENDADO NESTA JANELA")
    print("=" * 104)
    print(f"{'configuracao':>34} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'duty':>6} "
          f"{'lead':>6} {'p':>8} | {'rho':>7} {'p_rho':>7}")
    for rot, key in [("k=1,7 sem refratario", (1.7, 2.2, 0, 0, 0)),
                     ("k=1,7 + refrat 48h + dur 60min", (1.7, 2.2, 48, 60, 0)),
                     ("+ teto de 12 h", (1.7, 2.2, 48, 60, 12))]:
        al = ALS[key]; x = A.avalia(al[sel], alvo, m2)
        x.update(A.permuta(al[sel], m2, x["det"], len(alvo)))
        y = duty(al); r = stats.spearmanr(np.arange(len(y)), y)
        print(f"{rot:>34} {x['det']:3d}/{len(alvo)} {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {100*x['h_fp_mes']/730:5.1f}% {x['lead_med']:6.1f} "
              f"{x['p']:8.4f} | {r.statistic:+7.3f} {r.pvalue:7.4f}", flush=True)

    print("\n" + "=" * 104)
    print("2) ALGUM OUTRO PONTO DOMINA NESTA JANELA?  (melhor por nivel de deteccao)")
    print("=" * 104)
    print(f"{'det':>6} {'FP/mes':>7} {'h/mes':>7} {'duty':>6} {'lead':>6}  configuracao")
    for d in sorted(T.det.unique(), reverse=True)[:5]:
        S = T[T.det == d].sort_values(["fp", "h"])
        b = S.iloc[0]
        print(f"{int(b.det):4d}/{len(alvo)} {b.fp:7.2f} {b.h:7.1f} {100*b.h/730:5.1f}% "
              f"{b.lead:6.1f}  k={b.kb} k_vib={b.kv} R={int(b.R)}h D={int(b.D)}min "
              f"teto={int(b.teto)}h")

    print("\n" + "=" * 104)
    print("3) CURVA DE TROCA nesta janela")
    print("=" * 104)
    print(f"{'FP/mes <=':>10} {'det':>6} {'FP/mes':>7} {'/ano':>6} {'h/mes':>7} {'duty':>6} "
          f"{'lead':>6} {'precisao':>9}  configuracao")
    vis = set()
    for lim in [3.0, 2.5, 2.0, 1.6, 1.3, 1.0]:
        S = T[T.fp <= lim]
        if not len(S): continue
        b = S.sort_values(["det", "lead"], ascending=[False, False]).iloc[0]
        if (b.det, round(b.fp, 2)) in vis: continue
        vis.add((b.det, round(b.fp, 2)))
        prec = 100*(b.det/meses)/((b.det/meses) + b.fp)
        print(f"{lim:9.1f} {int(b.det):4d}/{len(alvo)} {b.fp:7.2f} {12*b.fp:6.1f} {b.h:7.1f} "
              f"{100*b.h/730:5.1f}% {b.lead:6.1f} {prec:8.1f}%  k={b.kb} k_vib={b.kv} "
              f"R={int(b.R)}h D={int(b.D)}min teto={int(b.teto)}h")

    print("\n" + "=" * 104); print("4) LOEO nesta janela"); print("=" * 104)
    for orc in [2.6, 3.5]:
        ac = 0
        for t in alvo:
            resto = [x for x in alvo if x != t]; m = None
            for key, al in ALS.items():
                x = A.avalia(al[sel], resto, m2)
                if x["fp_mes"] <= orc and (m is None or (x["det"], -x["fp_mes"]) > m[1]):
                    m = (key, (x["det"], -x["fp_mes"]))
            if m is None: continue
            ac += bool(ALS[m[0]].loc[t-pd.Timedelta(hours=48):t].fillna(False).any())
        print(f"  orcamento <= {orc} FP/mes: LOEO {ac}/{len(alvo)}", flush=True)


if __name__ == "__main__":
    main()
