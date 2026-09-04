"""Regra 'vibracao obrigatoria': vb ativo E pelo menos um outro sinal.

Vem da assinatura medida em voto_fp.py: as 4 combinacoes que dominam o falso
positivo (68% dos 84) nao tem vibracao; 70% dos acertos tem. Se isso se
sustentar, e um filtro fisico e nao um ajuste de limiar -- vibracao mede
movimento do rotor, e uma excursao de temperatura/pressao sem contrapartida
mecanica tende a ser variacao de processo, nao degradacao.

RESSALVA declarada antes do resultado: a assinatura foi lida em 10 episodios de
acerto. A regra e escolhida olhando o desfecho, entao precisa de leave-one-out
antes de valer alguma coisa -- e o que se faz na segunda parte.
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

KS = [0.5,0.7,0.85,1.0,1.15,1.3,1.5,1.7,2.0,2.4,3.0]
KVS = [2.2,3.0,4.0,5.5,7.5]

df = canonico()
falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
mask = mascara_pontuacao(df); idx = mask.index
meses = mask.sum()*2/60/730
jan48 = [(t-pd.Timedelta(hours=48), t) for t in falhas]
out = roda(BRACO, df, falhas)
def ew(c,hl): return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
E = {"t":ew("t","1h"), "p":ew("p","1h"), "sp":ew("sp","30min"), "vb":ew("vb","30min")}

def mede(al):
    eps = A.episodios(al)
    fp = [(a,b) for a,b in eps if not any((a<=t1) and (b>=t0) for t0,t1 in jan48)]
    det = [t.strftime("%Y-%m-%d") for t in falhas
           if al[(al.index>=t-pd.Timedelta(hours=48))&(al.index<t)].any()]
    h = sum((b-a).total_seconds()/3600+2/60 for a,b in fp)
    return len(fp), det, h/meses

linhas = []
for k in KS:
    for kv in KVS:
        S = {"t": DET._sustained(E["t"], DET.THR_FAM*k),
             "p": DET._sustained(E["p"], DET.THR_FAM*k),
             "sp": DET._sustained(E["sp"], DET.THR_SPREAD*k),
             "vb": DET._sustained(E["vb"], 3.0*kv)}
        outros = S["t"].astype(int)+S["p"].astype(int)+S["sp"].astype(int)
        al_vib = (S["vb"] & (outros >= 1)) & mask          # regra nova
        al_base = ((outros + S["vb"].astype(int)) >= 2) & mask  # regra atual
        for nome, al in [("vib_obrig", al_vib), ("2de4", al_base)]:
            fp, det, hm = mede(al)
            linhas.append(dict(regra=nome, k=k, kv=kv, fp=fp, det=len(det), h_mes=hm,
                               perdidos=",".join(t.strftime("%Y-%m-%d") for t in falhas
                                                 if t.strftime("%Y-%m-%d") not in det)))
T = pd.DataFrame(linhas); T.to_csv("regra_vib.csv", index=False)

at = T[(T.regra=="2de4")&(T.k==1.3)&(T.kv==5.5)].iloc[0]
print(f"ATUAL (2 de 4, k=1.3, kv=5.5): FP={int(at.fp)} h/mes={at.h_mes:.0f} det={int(at.det)}/9\n")
print("=== melhor ponto de cada regra, exigindo deteccao 8/9 ===")
for r in ["2de4","vib_obrig"]:
    s = T[(T.regra==r)&(T.det>=8)]
    if s.empty: print(f"  {r}: nunca alcanca 8/9"); continue
    b = s.sort_values(["fp","h_mes"]).iloc[0]
    print(f"  {r:10s} k={b.k:.2f} kv={b.kv:.1f} -> FP={int(b.fp)} ({100*b.fp/at.fp-100:+.0f}%), "
          f"{b.h_mes:.0f} h/mes ({100*b.h_mes/at.h_mes-100:+.0f}%)")
print("\n=== melhor de cada regra a 7/9 (caso 8/9 seja caro demais) ===")
for r in ["2de4","vib_obrig"]:
    s = T[(T.regra==r)&(T.det>=7)]
    b = s.sort_values(["fp","h_mes"]).iloc[0]
    print(f"  {r:10s} k={b.k:.2f} kv={b.kv:.1f} -> FP={int(b.fp)}, {b.h_mes:.0f} h/mes, det={int(b.det)}/9")
