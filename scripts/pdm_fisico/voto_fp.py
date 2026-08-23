#!/usr/bin/env python3
"""Reducao de FP por duas vias nunca testadas nesta linha de trabalho:

  1. REGRA DE VOTO. O detector exige >=2 dos 4 sinais. Nunca foi questionado.
     Exigir >=3 pede ABRANGENCIA (mecanismos independentes concordando), que e
     diferente de subir o limiar, que pede INTENSIDADE. As duas coisas trocam
     FP por deteccao de formas diferentes, entao a fronteira pode ser melhor.

  2. ASSINATURA DO FALSO POSITIVO. Com 84 episodios de FP contra 8 de acerto,
     o lado do FP e bem amostrado -- da pra perguntar o que distingue um do
     outro (duracao, quais sinais disparam, pico atingido). Se houver
     assinatura, vira pos-filtro.

Comparacao sempre a DETECCAO igualada, com as duas metricas (episodios e horas)
lado a lado -- a rodada da EWMA lenta mostrou que contagem de episodios sozinha
e manipulavel por suavizacao.
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
KS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.4, 3.0, 4.0]
VOTOS = [2, 3, 4]


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    idx = mask.index
    meses = mask.sum() * 2 / 60 / 730
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)

    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    E = {"t": ew("t", "1h"), "p": ew("p", "1h"),
         "sp": ew("sp", "30min"), "vb": ew("vb", "30min")}

    def sinais_ativos(k):
        return {"t": DET._sustained(E["t"], DET.THR_FAM * k),
                "p": DET._sustained(E["p"], DET.THR_FAM * k),
                "sp": DET._sustained(E["sp"], DET.THR_SPREAD * k),
                "vb": DET._sustained(E["vb"], 3.0 * K_VIB)}

    linhas = []
    print(f"\n{'voto':>5} {'k':>5} {'FP':>4} {'h/mes':>7} {'det':>5}  perdidos")
    for v in VOTOS:
        for k in KS:
            S = sinais_ativos(k)
            n = sum(s.astype(int) for s in S.values())
            al = (n >= v) & mask
            eps = A.episodios(al)
            fp = [(a, b) for a, b in eps
                  if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
            det = [t.strftime("%Y-%m-%d") for t in falhas
                   if al[(al.index >= t - pd.Timedelta(hours=48)) & (al.index < t)].any()]
            h = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
            perd = ",".join(t.strftime("%Y-%m-%d") for t in falhas
                            if t.strftime("%Y-%m-%d") not in det)
            linhas.append(dict(voto=v, k=k, fp=len(fp), det=len(det), h_mes=h / meses,
                                perdidos=perd))
            print(f"{v:>5} {k:5.2f} {len(fp):4d} {h/meses:7.1f} {len(det):3d}/9  {perd}",
                  flush=True)

    T = pd.DataFrame(linhas)
    T.to_csv("voto_fp.csv", index=False)

    atual = T[(T.voto == 2) & (T.k == 1.3)].iloc[0]
    print(f"\nATUAL (voto>=2, k=1.3): FP={int(atual.fp)} h/mes={atual.h_mes:.0f} "
          f"det={int(atual.det)}/9")
    print("\n=== melhor ponto de cada regra de voto, a deteccao 8/9 ===")
    for v in VOTOS:
        s = T[(T.voto == v) & (T.det >= 8)]
        if s.empty:
            print(f"  voto>={v}: nunca alcanca 8/9"); continue
        r = s.sort_values(["fp", "h_mes"]).iloc[0]
        print(f"  voto>={v}: k={r.k:.2f} -> FP={int(r.fp)} ({100*r.fp/atual.fp-100:+.0f}%), "
              f"{r.h_mes:.0f} h/mes ({100*r.h_mes/atual.h_mes-100:+.0f}%)")

    # ---------------- assinatura do FP vs acerto
    print("\n=== assinatura: episodios de FP vs episodios que pegaram trip "
          "(voto>=2, k=1.3) ===")
    S = sinais_ativos(1.3)
    n = sum(s.astype(int) for s in S.values())
    al = (n >= 2) & mask
    eps = A.episodios(al)
    def perfil(a, b):
        dur = (b - a).total_seconds() / 3600 + 2 / 60
        quais = tuple(sorted(c for c, s in S.items() if s.loc[a:b].any()))
        pico = int(n.loc[a:b].max())
        return dur, quais, pico
    fp, tp = [], []
    for a, b in eps:
        alvo = tp if any((a <= t1) and (b >= t0) for t0, t1 in jan48) else fp
        alvo.append(perfil(a, b))
    for nome, grupo in [("FALSO POSITIVO", fp), ("pegou trip", tp)]:
        durs = [d for d, _, _ in grupo]; picos = [p for _, _, p in grupo]
        print(f"\n  {nome} (n={len(grupo)}):")
        print(f"    duracao mediana: {np.median(durs):.1f} h   "
              f"(quartis {np.percentile(durs,25):.1f} - {np.percentile(durs,75):.1f})")
        print(f"    pico de sinais simultaneos: mediana {np.median(picos):.0f}, "
              f"max {max(picos)}")
        from collections import Counter
        for combo, c in Counter(q for _, q, _ in grupo).most_common(4):
            print(f"    {'+'.join(combo):20s} {c:3d} ({100*c/len(grupo):.0f}%)")


main()
