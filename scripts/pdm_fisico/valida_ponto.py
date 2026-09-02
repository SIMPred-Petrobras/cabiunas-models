#!/usr/bin/env python3
"""Ceticismo sobre os dois pontos de operacao candidatos, antes de fixar qualquer um.

  INVESTIGACAO  k=1,7  R=48h  D=60min  sem teto  -> 8/9  2,23 FP/mes  54,3 h/mes
  ACAO          k=2,0  R=120h D=0      teto 12h  -> 7/9  1,51 FP/mes   8,5 h/mes

Quatro testes, os mesmos que fizeram o refratario sobreviver e mataram o piso, o escape,
o phi e o CFAR:

  1. PERMUTACAO -- o recall sobrevive ao nulo, ou e cobertura?
  2. PLATO EM R -- o resultado vale numa faixa de R ou so num valor? O refratario e um
     encadeamento guloso e ja mostrou nao-monotonicidade em R=6h e R=72h. Um ponto
     isolado e ruido (foi assim que o piso de 1,6 degC caiu).
  3. SUPRESSAO -- o refratario chega a apagar o alerta de um evento real, e quanto lead
     se perde? R=120h sao CINCO DIAS de silencio apos cada episodio: se uma degradacao
     nova comecar dentro desse prazo, ela fica muda. n=9 nao consegue medir esse risco,
     mas da para medir quantas vezes um evento real caiu dentro da sombra de um episodio
     anterior.
  4. LOEO -- ponto reescolhido dentro da familia, fora do evento testado, sob orcamento.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, mascara_pontuacao
from portoes import K_VIB
from auto_reset import trunca
import piso_fisico as PF
import reduz_fp as RF

KS = [1.7, 2.0, 2.3, 2.6, 3.0]
RS = [0, 12, 24, 36, 48, 72, 96, 120, 168, 240]
DS = [0, 60]
TETOS = [0, 12]


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    d = PF.pre(df, falhas)
    out = PF.sinais(d, idx, 0.0, 0.0)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in [("t", "1h"), ("p", "1h"), ("sp", "30min"), ("vb", "30min")]}

    def bruto(kb):
        T = {"t": DET.THR_FAM*kb, "p": DET.THR_FAM*kb,
             "sp": DET.THR_SPREAD*kb, "vb": 3.0*K_VIB}
        n = sum(DET._sustained(E[c], T[c]).astype(int) for c in E)
        return (n >= 2) & mask

    BR = {k: bruto(k) for k in KS}
    ALS = {}
    for k in KS:
        for Rh in RS:
            alR = RF.refratario(BR[k], Rh)
            for D in DS:
                alD = RF.dur_min(alR, D)
                for te in TETOS:
                    ALS[(k, Rh, D, te)] = trunca(alD, 12) if te else alD
    print(f"configuracoes montadas: {len(ALS)}\n", flush=True)

    CAND = {"INVESTIGACAO (k=1,7 R=48h D=60min)": (1.7, 48, 60, 0),
            "ACAO (k=2,0 R=120h teto=12h)":       (2.0, 120, 0, 12)}

    print("=" * 92); print("1) PERMUTACAO"); print("=" * 92)
    print(f"{'candidato':>36} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} {'p':>8}")
    for rot, key in [("referencia (k=1,7 R=0)", (1.7, 0, 0, 0))] + list(CAND.items()):
        al = ALS[key]; x = A.avalia(al, falhas, mask)
        x.update(A.permuta(al, mask, x["det"], len(falhas)))
        print(f"{rot:>36} {x['det']:4d}/9 {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {x['lead_med']:6.1f} {x['p']:8.4f}", flush=True)

    print("\n" + "=" * 92); print("2) PLATO EM R -- o resultado vale numa faixa?"); print("=" * 92)
    for rot, (k, _, D, te) in CAND.items():
        print(f"\n  {rot}")
        print(f"     {'R':>6} " + " ".join(f"{r:>6}" for r in RS))
        print(f"     {'det':>6} " + " ".join(f"{A.avalia(ALS[(k,r,D,te)],falhas,mask)['det']:>6}" for r in RS))
        print(f"     {'FP/mes':>6} " + " ".join(f"{A.avalia(ALS[(k,r,D,te)],falhas,mask)['fp_mes']:6.2f}" for r in RS))

    print("\n" + "=" * 92); print("3) SUPRESSAO: o refratario apaga alerta de evento real?"); print("=" * 92)
    for rot, (k, Rh, D, te) in CAND.items():
        base_k = ALS[(k, 0, D, te)]; al = ALS[(k, Rh, D, te)]
        perd, dl, sombra = [], [], 0
        for t in falhas:
            w0 = base_k.loc[t-pd.Timedelta(hours=48):t]; w1 = al.loc[t-pd.Timedelta(hours=48):t]
            a0 = w0[w0.fillna(False)]; a1 = w1[w1.fillna(False)]
            if len(a0) and not len(a1): perd.append(f"{t:%d/%m}")
            if len(a0) and len(a1): dl.append(((t-a0.index[0])-(t-a1.index[0])).total_seconds()/3600)
            # o evento caiu dentro da sombra de um episodio anterior?
            ant = [b for a, b in A.episodios(al) if b < t - pd.Timedelta(hours=48)]
            if ant and (t - max(ant)).total_seconds()/3600 < Rh: sombra += 1
        print(f"  {rot}")
        print(f"     eventos que deixam de ser detectados: {len(perd)} {perd}")
        print(f"     lead perdido nos demais: mediana {np.median(dl) if dl else 0:.2f} h, "
              f"maximo {max(dl) if dl else 0:.2f} h")
        print(f"     eventos que caem na sombra de um episodio anterior: {sombra}/9")

    print("\n" + "=" * 92); print("4) LOEO -- ponto reescolhido fora do evento, sob orcamento")
    print("=" * 92)
    def loeo(teto_fp):
        ac, esc = 0, []
        for t in falhas:
            resto = [x for x in falhas if x != t]; m = None
            for key, al in ALS.items():
                x = A.avalia(al, resto, mask)
                if x["fp_mes"] <= teto_fp and (m is None or (x["det"], -x["fp_mes"]) > m[1]):
                    m = (key, (x["det"], -x["fp_mes"]))
            if m is None: continue
            esc.append(m[0])
            ac += bool(ALS[m[0]].loc[t-pd.Timedelta(hours=48):t].fillna(False).any())
        return ac, pd.Series([str(e) for e in esc]).mode().iloc[0] if esc else "-"
    print(f"{'orcamento':>16} {'LOEO':>7}   escolha tipica (k, R, D, teto)")
    for tf in [4.5, 3.0, 2.3, 1.6, 1.2]:
        a, mo = loeo(tf)
        print(f"{'<= '+str(tf)+' FP/mes':>16} {str(a)+'/9':>7}   {mo}", flush=True)


if __name__ == "__main__":
    main()
