"""Efeito do limiar sobre ANTECEDENCIA e PERSISTENCIA, nao so sobre deteccao.

Todas as varreduras anteriores mediram 'houve alerta na janela de 48h' -- um
binario que ignora quando o alerta comecou e se ele ainda estava ligado quando
a maquina caiu. Aqui as tres coisas andam juntas, porque a rodada anterior
mostrou que o confirmado e tardio (mediana 2,8 h) e as vezes intermitente.

Hipotese a testar: subir o limiar deve atrasar o inicio (menos antecedencia) e
piorar a persistencia, porque o escore cruza a linha mais tarde. Baixar deve
antecipar, ao custo de falso positivo.
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

K_VIB = 2.2
KS = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4, 3.0]

df = canonico()
ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
falhas = ftodas[ftodas>="2025-01-01"].reset_index(drop=True)
mask = mascara_pontuacao(df); idx = mask.index
cal_meses = (idx[-1]-idx[0]).total_seconds()/3600/730
out = roda(BRACO, df, ftodas)
def ew(c,hl): return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
E = {"t":ew("t","1h"), "p":ew("p","1h"), "sp":ew("sp","30min"), "vb":ew("vb","30min")}
jan48 = [(t-pd.Timedelta(hours=48), t) for t in ftodas]

print(f"{'k':>5} | {'det 48h':>8} {'ativo<=2h':>10} {'lead med':>9} {'lead min':>9} "
      f"| {'FP':>4} {'FP/mes':>7} {'h/mes':>7}")
print("-"*72)
linhas=[]
for k in KS:
    S = {"t": DET._sustained(E["t"], DET.THR_FAM*k), "p": DET._sustained(E["p"], DET.THR_FAM*k),
         "sp": DET._sustained(E["sp"], DET.THR_SPREAD*k), "vb": DET._sustained(E["vb"], 3.0*K_VIB)}
    n = sum(s.astype(int) for s in S.values())
    al = (n>=2)&mask
    eps = A.episodios(al)
    det48 = sum(1 for ev in falhas if al[(al.index>=ev-pd.Timedelta(hours=48))&(al.index<ev)].any())
    leads=[]; ativo=0
    for ev in falhas:
        cand = [(a,b) for a,b in eps if a<ev and (ev-b).total_seconds()/3600<=2.0]
        if cand:
            a,_ = max(cand, key=lambda x:x[0]); leads.append((ev-a).total_seconds()/3600); ativo+=1
    fp = [(a,b) for a,b in eps if not any((a<=t1) and (b>=t0) for t0,t1 in jan48)]
    h = sum((b-a).total_seconds()/3600+2/60 for a,b in fp)
    lm = np.median(leads) if leads else np.nan; ln = min(leads) if leads else np.nan
    print(f"{k:5.2f} | {det48:6d}/8 {ativo:8d}/8 {lm:8.1f}h {ln:8.1f}h "
          f"| {len(fp):4d} {len(fp)/cal_meses:7.2f} {h/cal_meses:7.0f}")
    linhas.append(dict(k=k, det48=det48, ativo2h=ativo, lead_med=lm, lead_min=ln,
                       fp=len(fp), fp_cal=len(fp)/cal_meses, h_cal=h/cal_meses))
pd.DataFrame(linhas).to_csv("limiar_lead.csv", index=False)
