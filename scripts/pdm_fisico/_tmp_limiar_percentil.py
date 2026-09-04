"""A HIPOTESE: nao e o sensor, e ONDE se poe o limiar.

Nosso limiar de vb esta em p76 da distribuicao saudavel; o Francisco opera todos
os canais em p99,9. Um sinal em p80 e INVISIVEL para p99,9 e visivel para p76 --
desde que o falso positivo seja controlado por outra coisa (voto >=2 + SUSTAIN +
CUSUM + refratario), nao pelo limiar.

Teste: substitui o nosso limiar de vb por percentis altos e ve o que morre."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import EW, BASE, mask, idx, alvo, reset, pos
from publica_clearml import SIN, K, KAPPA, H_CUSUM, SUSTAIN, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

paradas = paradas_reais_2h(); JAN = pd.Timedelta(hours=48)
jw = [(t-JAN,t) for t in alvo]
base_vb = EW["vb"].where(mask).dropna()

def roda(thr_vb):
    ON = {}
    for c in SIN:
        thr = thr_vb if c == "vb" else BASE[c]*K[c]
        E = EW[c].where(mask)
        deg = ((E>thr).astype(int).rolling(SUSTAIN,min_periods=SUSTAIN).sum()>=SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(),
                             reset)>H_CUSUM, index=idx)
        ON[c] = (deg|cu)&mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns>=2, index=idx)&mask&(ON["sp"]|ON["vb"])
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    eps = AV.episodios(al)
    if not eps: return 0, np.nan, np.nan, []
    m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
    cl = classifica_regra_c(eps, paradas)
    n_fp = sum(1 for a,b,c,l in cl if c=="FP")
    h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
    pegos = [t.strftime("%d/%m") for t,(t0,t1) in zip(alvo,jw)
             if any(a<=t1 and b>=t0 for a,b in eps)]
    return len(pegos), n_fp/meses, h_fp/meses, pegos

thr_atual = BASE["vb"]*K["vb"]
p_atual = 100*(base_vb < thr_atual).mean()
print(f"nosso limiar de vb = {thr_atual:.2f}  ->  percentil {p_atual:.1f} da distribuicao saudavel\n")
print(f"{'limiar de vb':>26} {'valor':>8} {'det':>5} {'FP/mes':>8} {'h/mes':>8}   perde")
det0, fp0, h0, pg0 = roda(thr_atual)
print(f"{'atual (p%.0f)' % p_atual:>26} {thr_atual:8.2f} {det0:4d}/8 {fp0:8.3f} {h0:8.2f}   --")
for q in (90.0, 95.0, 99.0, 99.9):
    thr = float(np.percentile(base_vb, q))
    d, fp, h, pg = roda(thr)
    perde = sorted(set(pg0) - set(pg))
    print(f"{('percentil %.1f' % q):>26} {thr:8.2f} {d:4d}/8 {fp:8.3f} {h:8.2f}   "
          f"{', '.join(perde) if perde else '--'}")
