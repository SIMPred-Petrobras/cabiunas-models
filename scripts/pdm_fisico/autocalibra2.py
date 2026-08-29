#!/usr/bin/env python3
"""O limiar autocalibrado que funciona: quantil do CORPO x multiplicador, nao da cauda.

O que a fase anterior mostrou. O percentil alto do residuo em amostra (p99,9, o que
eles usam) NAO transporta para o nosso escore: a razao limiar/atual varia de 0,36x a
3.414x entre meses. A causa e a forma do escore: `max_j(e_j / p99_j)` com `e` ao
QUADRADO. Um desvio de 100 sigma num sensor vira 10.000 no escore, entao entre o
posto 200 e o posto 20 de 20.000 amostras o valor anda tres ordens de grandeza. Um
percentil ali e definido pelo pior artefato da janela de treino, nao pela escala dela.
O escore deles e MEDIA sobre a familia -- a media mata essa cauda, e por isso o
percentil funciona la e nao aqui.

A correcao: tomar o quantil no CORPO da distribuicao (p90/p95), onde ela e estavel,
e multiplicar por uma constante adimensional. Continua sem constante em unidade
fisica e sem alvo: o limiar e "m vezes o p95 do residuo do proprio modelo naquele
mes". Este script varre m e compara a custo igualado com o ponto calibrado.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV, francisco_lara as F
from publica_clearml import reproduz, SIN, HL, BASE, K, KAPPA, H_CUSUM, CARGA

fin, mask, alvo, ON, idx, sel = reproduz()
ref = AV.avalia(fin, alvo, mask)
g = pd.read_parquet("grade2min.parquet")
z = np.load("piso_fisico_cache.npz"); T = np.load("autocalibra_thr.npz")
sp = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
vb = np.full(len(idx), np.nan); vb[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
vb[~np.isfinite(vb)] = np.nan
cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp, "vb": vb}, index=idx)
E = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
op = (g["RUNNING_A"] > 0.5).fillna(False)
reset = ((~mask) | (op & ~op.shift(fill_value=False))).to_numpy()


def roda(thr):
    ONq = {}
    for c in SIN:
        deg = ((E[c] > thr[c]).astype(int).rolling(15, min_periods=15).sum() >= 15)
        x = ((E[c] / thr[c]).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
        S = np.empty(len(x)); a = 0.0
        for i in range(len(x)):
            a = a * CARGA if reset[i] else max(0.0, a + x[i]); S[i] = a
        ONq[c] = (deg | pd.Series(S > H_CUSUM, index=idx)) & mask
    v = pd.Series(sum(ONq[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    return AV.avalia(F.pos(v, idx) & sel, alvo, mask)


print(f"\n{'limiar':38s} {'det':>5s} {'eps':>5s} {'FP/mes':>8s} {'h/mes':>8s} {'lead':>7s}  perdidos")
print(f"{'k*base -- calibrado nos 8 eventos':38s} {ref['det']:>3d}/8 {ref['episodios']:>5d} "
      f"{ref['fp_mes']:>8.2f} {ref['h_fp_mes']:>8.1f} {ref['lead_med']:>7.1f}")
print("-" * 100)
lin = [dict(limiar="k*base (calibrado)", det=f"{ref['det']}/8", eps=ref["episodios"],
            fp_mes=round(ref["fp_mes"],3), h_fp_mes=round(ref["h_fp_mes"],1),
            lead_h=round(ref["lead_med"],1), perdidos="")]
for qb in [90.0, 95.0]:
    for m in [1.5, 2.0, 2.5, 3.0, 4.0]:
        thr = {c: pd.Series(T[f"{c}|{qb}"], index=idx) * m for c in SIN}
        r = roda(thr)
        perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(r["detectados"]))
        nome = f"{m:.1f} x p{qb:.0f} do residuo do mes"
        print(f"{nome:38s} {r['det']:>3d}/8 {r['episodios']:>5d} {r['fp_mes']:>8.2f} "
              f"{r['h_fp_mes']:>8.1f} {r['lead_med']:>7.1f}  {','.join(perd)}")
        lin.append(dict(limiar=nome, det=f"{r['det']}/8", eps=r["episodios"],
                        fp_mes=round(r["fp_mes"],3), h_fp_mes=round(r["h_fp_mes"],1),
                        lead_h=round(r["lead_med"],1), perdidos=",".join(perd)))
pd.DataFrame(lin).to_csv("autocalibra.csv", index=False)
print("\n-> autocalibra.csv")
