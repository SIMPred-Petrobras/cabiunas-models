"""Sensibilidade a tolerancia: quanto o alerta pode ter CESSADO antes do trip e
ainda contar como aviso daquele evento? A regua do projeto (janela de 48h) nao
faz essa distincao -- basta ter havido alerta em algum momento da janela."""
import sys
# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import numpy as np, pandas as pd
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
K_BASE, K_VIB = 1.7, 2.2
df = canonico()
ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
falhas = ftodas[ftodas>="2025-01-01"].reset_index(drop=True)
mask = mascara_pontuacao(df); idx = mask.index
out = roda(BRACO, df, ftodas)
def ew(c,hl): return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
S = {"t": DET._sustained(ew("t","1h"), DET.THR_FAM*K_BASE), "p": DET._sustained(ew("p","1h"), DET.THR_FAM*K_BASE),
     "sp": DET._sustained(ew("sp","30min"), DET.THR_SPREAD*K_BASE), "vb": DET._sustained(ew("vb","30min"), 3.0*K_VIB)}
n = sum(s.astype(int) for s in S.values())
niveis = {"atencao": (n>=1)&mask, "confirmado": (n>=2)&mask}
print("quantos dos 8 eventos tem um EPISODIO que chega ate 'tol' horas antes do trip:")
print(f"{'tol':>6} {'atencao':>10} {'confirmado':>12}")
for tol in [2,6,12,24,48]:
    cel=[]
    for nome,al in niveis.items():
        eps = A.episodios(al); c=0
        for ev in falhas:
            if any(a<ev and (ev-b).total_seconds()/3600<=tol for a,b in eps): c+=1
        cel.append(c)
    print(f"{tol:4d} h {cel[0]:10d}/8 {cel[1]:11d}/8")
print()
print("para contexto -- quanto tempo o CONFIRMADO ficou ligado nas 48h antes de cada evento:")
al = niveis["confirmado"]
for ev in falhas:
    w = al[(al.index>=ev-pd.Timedelta(hours=48))&(al.index<ev)]
    m = mask[(mask.index>=ev-pd.Timedelta(hours=48))&(mask.index<ev)]
    ult = w[w]
    gap = (ev-ult.index[-1]).total_seconds()/3600 if len(ult) else np.nan
    print(f"  {ev:%Y-%m-%d}: ligado {w.sum()*2/60:5.1f} h das {m.sum()*2/60:5.1f} h pontuaveis"
          + (f" | ultimo alerta {gap:5.1f} h antes do trip" if not np.isnan(gap) else " | nenhum alerta"))
