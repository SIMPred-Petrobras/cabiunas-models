"""Limiar x limite de permanencia, as duas alavancas juntas.

Limiar reduz a QUANTIDADE de alarme falso (e custa deteccao).
Limite de permanencia reduz a DURACAO (e custa quase nada).
Sao independentes, entao a grade mostra o que da pra conseguir combinando.
"""
import sys
# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import numpy as np, pandas as pd
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from auto_reset import trunca

K_VIB = 2.2
KS = [1.0, 1.15, 1.3, 1.5, 1.7, 2.0]
LIMS = [None, 24, 12, 6]

df = canonico()
ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
falhas = ftodas[ftodas>="2025-01-01"].reset_index(drop=True)
mask = mascara_pontuacao(df); idx = mask.index
cal = (idx[-1]-idx[0]).total_seconds()/3600/730
jan48 = [(t-pd.Timedelta(hours=48), t) for t in ftodas]
out = roda(BRACO, df, ftodas)
def ew(c,hl): return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
E = {"t":ew("t","1h"),"p":ew("p","1h"),"sp":ew("sp","30min"),"vb":ew("vb","30min")}

print(f"{'k':>5} {'limite':>7} | {'FP/mes':>7} {'h/mes':>7} {'max ep':>7} | "
      f"{'det':>5} {'ativo':>6} {'lead':>7}")
print("-"*66)
linhas=[]
for k in KS:
    S={"t":DET._sustained(E["t"],DET.THR_FAM*k),"p":DET._sustained(E["p"],DET.THR_FAM*k),
       "sp":DET._sustained(E["sp"],DET.THR_SPREAD*k),"vb":DET._sustained(E["vb"],3.0*K_VIB)}
    n=sum(s.astype(int) for s in S.values()); al0=(n>=2)&mask
    for lim in LIMS:
        al=trunca(al0,lim); eps=A.episodios(al)
        d=lambda a,b:(b-a).total_seconds()/3600+2/60
        fp=[(a,b) for a,b in eps if not any((a<=t1) and (b>=t0) for t0,t1 in jan48)]
        h=sum(d(a,b) for a,b in fp); mx=max((d(a,b) for a,b in fp), default=0)
        det=sum(1 for ev in falhas if al[(al.index>=ev-pd.Timedelta(hours=48))&(al.index<ev)].any())
        leads=[];ativo=0
        for ev in falhas:
            c=[(a,b) for a,b in eps if a<ev and (ev-b).total_seconds()/3600<=2.0]
            if c: a,_=max(c,key=lambda x:x[0]); leads.append((ev-a).total_seconds()/3600); ativo+=1
        rot="sem" if lim is None else f"{lim}h"
        print(f"{k:5.2f} {rot:>7} | {len(fp)/cal:7.2f} {h/cal:7.0f} {mx:6.0f}h | "
              f"{det:3d}/8 {ativo:4d}/8 {np.median(leads) if leads else np.nan:6.1f}h", flush=True)
        linhas.append(dict(k=k, limite=lim, fp=len(fp), fp_mes=len(fp)/cal, h_mes=h/cal,
                           max_ep_h=mx, det=det, ativo=ativo,
                           lead_med=np.median(leads) if leads else np.nan))
pd.DataFrame(linhas).to_csv("combina.csv", index=False)
