#!/usr/bin/env python3
"""Escala de tempo do detector: half-life da EWMA x janela de sustentacao.

Lacuna encontrada ao revisar o que ja se tentou: o detector implantado opera em
30 min de sustentacao com EWMA de 1 h. A exploracao anterior (explora2.csv, no
framework de familias fisicas) achou os melhores resultados em sust=1440 min
(24 h) e EWMA de 8 h -- 48x mais lento. Essa regiao nunca foi testada DENTRO do
detector implantado, onde vive o pipeline validado.

Motivacao fisica: degradacao de mancal se desenvolve em dias. 30 min de
sustentacao com EWMA de 1 h esta sintonizado para transiente; o alvo e deriva
lenta. Quatro dos nove eventos sao 'Temp.Mt.Alta Manc.Rad'.

Comparacao a FP IGUALADO -- sem isso, escala mais lenta so parece melhor porque
suaviza mais e alarma menos (a mesma armadilha de dominancia de Pareto de
sempre). Para cada (hl, sust), varre-se k e reporta-se a deteccao no ponto cujo
FP mais se aproxima do FP nativo atual (85 episodios).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO

K_VIB = 5.5
HLS = ["1h", "2h", "4h", "8h", "16h"]          # half-life da EWMA
SUSTS = [30, 120, 360, 720, 1440]              # sustentacao, em minutos
KS = [0.4, 0.55, 0.7, 0.85, 1.0, 1.3, 1.7, 2.2, 3.0]
FP_ALVO = 85                                    # FP nativo do detector atual


def sustenta(acima: pd.Series, minutos: int) -> pd.Series:
    n = max(1, int(minutos / 2))
    return (acima.astype(int).rolling(n, min_periods=n).sum() >= n)


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    meses = mask.sum() * 2 / 60 / 730
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    print("montando 'out' (uma vez) ...", flush=True)
    out = roda(BRACO, df, falhas)
    idx = out.index

    def curva(hl, sust):
        ewt = out["t"].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
        ewp = out["p"].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
        ews = out["sp"].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
        ewv = out["vb"].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
        linhas = []
        for k in KS:
            n = (sustenta(ewt > DET.THR_FAM * k, sust).astype(int)
                 + sustenta(ewp > DET.THR_FAM * k, sust).astype(int)
                 + sustenta(ews > DET.THR_SPREAD * k, sust).astype(int)
                 + sustenta(ewv > 3.0 * K_VIB, sust).astype(int))
            al = (n >= 2) & mask
            eps = A.episodios(al)
            fp = [(a, b) for a, b in eps
                  if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
            det = [t.strftime("%Y-%m-%d") for t in falhas
                   if al[(al.index >= t - pd.Timedelta(hours=48)) & (al.index < t)].any()]
            h = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
            linhas.append(dict(hl=hl, sust=sust, k=k, fp=len(fp), det=len(det),
                                h_mes=h / meses, perdidos=",".join(
                                    t.strftime("%Y-%m-%d") for t in falhas
                                    if t.strftime("%Y-%m-%d") not in det)))
        return pd.DataFrame(linhas)

    todas = []
    print(f"\ndeteccao no ponto de FP mais proximo de {FP_ALVO} episodios "
          f"(atual = hl 1h, sust 30min):\n")
    print(f"{'sust\\hl':>9} " + "".join(f"{h:>13}" for h in HLS))
    for sust in SUSTS:
        celulas = []
        for hl in HLS:
            c = curva(hl, sust)
            todas.append(c)
            c = c.copy(); c["d"] = (c.fp - FP_ALVO).abs()
            r = c.sort_values("d").iloc[0]
            celulas.append(f"{int(r.det)}/9 (fp{int(r.fp)})")
        print(f"{sust:>7}min " + "".join(f"{x:>13}" for x in celulas), flush=True)

    T = pd.concat(todas, ignore_index=True)
    T.to_csv("escala_tempo.csv", index=False)

    print("\n=== melhores pontos globais (deteccao alta com FP <= 85) ===")
    b = T[T.fp <= FP_ALVO].sort_values(["det", "fp"], ascending=[False, True]).head(10)
    print(b[["hl", "sust", "k", "fp", "det", "h_mes", "perdidos"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))


main()
