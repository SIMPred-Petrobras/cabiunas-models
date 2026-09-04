"""Comparacao 667h vs 1333h a FALSO POSITIVO IGUALADO.

Motivo: alongar a referencia encolhe a escala dos escores (mediana 1.55 ->
0.45), entao comparar no MESMO limiar fixo confunde 'melhorou a discriminacao'
com 'ficou menos sensivel'. Aqui cada comprimento ganha o proprio botao k e a
curva inteira e reportada; a comparacao acontece na mesma coluna de FP.
"""
import sys
# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import numpy as np, pandas as pd
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, mascara_pontuacao
from varre_referencia import roda_param

K_VIB = 5.5
KS = [0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0, 1.3, 1.7, 2.2]

df = canonico()
falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
mask = mascara_pontuacao(df)
meses = mask.sum()*2/60/730
jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

def avalia_k(out, kb):
    idx = out.index
    def ew(c, hl): return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    n = (DET._sustained(ew("t","1h"), DET.THR_FAM*kb).astype(int)
         + DET._sustained(ew("p","1h"), DET.THR_FAM*kb).astype(int)
         + DET._sustained(ew("sp","30min"), DET.THR_SPREAD*kb).astype(int)
         + DET._sustained(ew("vb","30min"), 3.0*K_VIB).astype(int))
    al = (n >= 2) & mask
    eps = A.episodios(al)
    fp = [(a,b) for a,b in eps if not any((a<=t1) and (b>=t0) for t0,t1 in jan48)]
    det = [t.strftime("%Y-%m-%d") for t in falhas
           if al[(al.index>=t-pd.Timedelta(hours=48)) & (al.index<t)].any()]
    h = sum((b-a).total_seconds()/3600+2/60 for a,b in fp)
    return len(fp), len(det), h/meses, det

curvas = {}
for fh in [667., 1333.]:
    out = roda_param(df, falhas, 400.0, int(fh*30))
    print(f"\n=== referencia {fh:.0f}h ===")
    print(f"{'k_base':>7} {'FP':>4} {'FP/mes':>7} {'h/mes':>7} {'det':>5}  perdidos")
    linhas = []
    for kb in KS:
        fp, nd, hm, det = avalia_k(out, kb)
        perd = [t.strftime("%Y-%m-%d") for t in falhas if t.strftime("%Y-%m-%d") not in det]
        print(f"{kb:7.2f} {fp:4d} {fp/meses:7.2f} {hm:7.1f} {nd:3d}/9  {','.join(perd)}", flush=True)
        linhas.append(dict(fit_h=fh, k=kb, fp=fp, fp_mes=fp/meses, h_mes=hm, det=nd))
    curvas[fh] = pd.DataFrame(linhas)

T = pd.concat(curvas.values()); T.to_csv("fp_igualado_ref.csv", index=False)
print("\n=== comparacao a FP igualado (interpolando cada curva) ===")
print(f"{'FP alvo':>8} {'det @667h':>10} {'det @1333h':>11}")
for alvo in [30, 45, 60, 71, 85, 100]:
    linha = []
    for fh in [667., 1333.]:
        c = curvas[fh].copy(); c["d"] = (c.fp - alvo).abs()
        r = c.sort_values("d").iloc[0]
        linha.append(f"{int(r.det)}/9 (k={r.k}, FP={int(r.fp)})")
    print(f"{alvo:8d} {linha[0]:>22} {linha[1]:>23}")
