"""O ganho de 667h -> 1333h e consistente no tempo, ou vem de um periodo so?
Se vier de um trecho unico, e sorte; se for distribuido, e propriedade real."""
import sys
# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import numpy as np, pandas as pd
from scipy import stats
import avalia as A
from ablacao import canonico, mascara_pontuacao, CORTE
from referencia_campanha import alerta_de
from varre_referencia import roda_param

df = canonico()
falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
mask = mascara_pontuacao(df); idx = df.index
jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

res = {}
for fh in [667., 1333.]:
    out = roda_param(df, falhas, 400.0, int(fh*30))
    al = alerta_de(out, mask)
    eps = A.episodios(al)
    fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
    res[fh] = fp

tr_mask = idx < CORTE
h_tr = (mask & tr_mask).sum()*2/60/730; h_te = (mask & ~tr_mask).sum()*2/60/730
print(f"exposicao: treino {h_tr:.1f} meses, teste {h_te:.1f} meses\n")
print(f"{'ref':>6} {'FP treino':>10} {'/mes':>6} {'FP teste':>9} {'/mes':>6}")
for fh, fp in res.items():
    a_tr = sum(1 for a,_ in fp if a < CORTE); a_te = len(fp)-a_tr
    print(f"{fh:6.0f} {a_tr:10d} {a_tr/h_tr:6.2f} {a_te:9d} {a_te/h_te:6.2f}")

print("\nFP por mes, lado a lado:")
mp = idx.to_period("M")
hm = mask.groupby(mp).sum()*2/60; hm = hm[hm>0]
tab = pd.DataFrame(index=hm.index)
for fh, fp in res.items():
    c = pd.Series(0, index=hm.index)
    for a,_ in fp:
        p = pd.Period(a, freq="M")
        if p in c.index: c[p]+=1
    tab[f"{fh:.0f}h"] = c
tab["delta"] = tab["1333h"] - tab["667h"]
print(tab.to_string())
n_pior = (tab["delta"]>0).sum(); n_melhor = (tab["delta"]<0).sum(); n_igual = (tab["delta"]==0).sum()
print(f"\nmeses em que 1333h teve MENOS FP: {n_melhor} | MAIS: {n_pior} | igual: {n_igual}")
bt = stats.binomtest(n_melhor, n_melhor+n_pior, 0.5)
print(f"teste do sinal (H0: 1333h nao e melhor): p={bt.pvalue:.4f}")
