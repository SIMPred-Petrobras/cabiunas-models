"""Distribuicao de DURACAO dos falsos positivos.

Ate agora so contamos episodios e horas totais. Mas 3 alarmes falsos por mes de
10 min cada e uma coisa; 3 de dois dias cada e outra completamente diferente --
e a media esconde isso. Duas perguntas:
  1. o custo em horas vem de muitos alarmes curtos ou de poucos muito longos?
  2. duracao distingue falso positivo de acerto? Se sim, e pos-filtro.
"""
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src")
import numpy as np, pandas as pd
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
K_BASE, K_VIB = 1.3, 2.2
df = canonico()
ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
mask = mascara_pontuacao(df); idx = mask.index
cal_meses = (idx[-1]-idx[0]).total_seconds()/3600/730
out = roda(BRACO, df, ftodas)
def ew(c,hl): return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
S = {"t": DET._sustained(ew("t","1h"), DET.THR_FAM*K_BASE), "p": DET._sustained(ew("p","1h"), DET.THR_FAM*K_BASE),
     "sp": DET._sustained(ew("sp","30min"), DET.THR_SPREAD*K_BASE), "vb": DET._sustained(ew("vb","30min"), 3.0*K_VIB)}
n = sum(s.astype(int) for s in S.values())
al = (n>=2)&mask
eps = A.episodios(al)
jan48 = [(t-pd.Timedelta(hours=48), t) for t in ftodas]
def dur(a,b): return (b-a).total_seconds()/3600+2/60
fp = [(a,b) for a,b in eps if not any((a<=t1) and (b>=t0) for t0,t1 in jan48)]
tp = [(a,b) for a,b in eps if any((a<=t1) and (b>=t0) for t0,t1 in jan48)]
dfp = np.array([dur(a,b) for a,b in fp]); dtp = np.array([dur(a,b) for a,b in tp])

print(f"FALSO POSITIVO: {len(fp)} episodios, {dfp.sum():.0f} h totais "
      f"({dfp.sum()/cal_meses:.0f} h/mes calendario)\n")
print("distribuicao de duracao:")
for lo, hi, rot in [(0,0.5,'< 30 min'),(0.5,1,'30-60 min'),(1,3,'1-3 h'),(3,6,'3-6 h'),
                    (6,12,'6-12 h'),(12,24,'12-24 h'),(24,48,'1-2 dias'),(48,1e9,'> 2 dias')]:
    m = (dfp>=lo)&(dfp<hi)
    if m.sum():
        print(f"  {rot:>10}: {m.sum():3d} episodios ({100*m.sum()/len(dfp):3.0f}%) | "
              f"{dfp[m].sum():6.0f} h ({100*dfp[m].sum()/dfp.sum():3.0f}% das horas)")
print(f"\n  mediana {np.median(dfp):.1f} h | media {dfp.mean():.1f} h | max {dfp.max():.0f} h")
o = np.sort(dfp)[::-1]
for topn in [5,10,20]:
    print(f"  os {topn:2d} episodios mais longos = {100*o[:topn].sum()/dfp.sum():.0f}% de todas as horas em alarme")
print(f"\nACERTOS (janela de evento): {len(tp)} episodios, mediana {np.median(dtp):.1f} h, "
      f"quartis {np.percentile(dtp,25):.1f}-{np.percentile(dtp,75):.1f} h")
print(f"\nduracao distingue? FP mediana {np.median(dfp):.1f} h vs acerto {np.median(dtp):.1f} h")
from scipy import stats
u = stats.mannwhitneyu(dfp, dtp, alternative='two-sided')
print(f"  Mann-Whitney: p={u.pvalue:.3f} "
      f"{'-> distribuicoes diferentes' if u.pvalue<0.05 else '-> nao distingue'}")
print("\n10 episodios de FP mais longos:")
for a,b in sorted(fp, key=lambda x:-dur(*x))[:10]:
    quais = [c for c,s in S.items() if s.loc[a:b].any()]
    print(f"  {a:%Y-%m-%d %H:%M} -> {b:%m-%d %H:%M}  {dur(a,b):6.1f} h  [{','.join(quais)}]")
