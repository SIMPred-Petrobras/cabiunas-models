#!/usr/bin/env python3
"""Duas estatisticas de deteccao nao testadas: GLR multiescala e canais lentos.

O CUSUM rendeu porque INTEGRA evidencia em vez de comparar com um patamar. Duas formas de
integrar que ainda nao testamos, ambas da mesma familia (estatistica de deteccao, que e
onde todo ganho deste projeto apareceu):

GLR MULTIESCALA. O CUSUM exige declarar o tamanho do desvio de interesse (o `kappa`); se
o desvio real for menor, ele nao acumula, se for maior, ele demora. O GLR nao precisa
desse parametro -- maximiza sobre o instante de mudanca:

    G_t = max_j  (C_t - C_{t-j}) / sqrt(j),   C = soma acumulada de z

ou seja, o maior desvio medio padronizado entre TODAS as janelas j terminando em t.
Avaliado numa grade geometrica de j (15 min a 24 h), fica O(7n) e capta degrau curto e
deriva longa com a mesma estatistica.

CANAIS LENTOS. As meia-vidas atuais sao 1 h (t, p) e 30 min (sp, vb). Se integrar e o que
funciona, uma meia-via de 6 h ou 24 h faz parte disso de graca -- sem parametro novo, so
trocando a constante de tempo. E o controle barato para o GLR: se o canal lento sozinho
der o mesmo, a complexidade do GLR nao se justifica.

Comparacao contra o ponto atual (degrau OU CUSUM, 8/8, 1,46 FP/mes, 52,8 h/mes, lead
30,4 h), a episodios igualados, com permutacao e LOEO pelo criterio de MEDIA da
vizinhanca. Janela 2025-01 a 2026-04, 8 eventos.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
import reduz_fp as RF

T0 = pd.Timestamp("2025-01-01", tz="UTC")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
SIN = ["t", "p", "sp", "vb"]
JS = [8, 15, 30, 60, 120, 360, 720]        # janelas do GLR, em amostras de 2 min (16min..24h)
H_GLR = [3.0, 5.0, 8.0, 12.0, 20.0]
HL_LENTO = ["3h", "6h", "12h", "24h"]
K_LENTO = [0.6, 0.8, 1.0, 1.3]             # limiar do canal lento, fracao do limiar atual
SUST_LENTO = [60, 180, 360]                # minutos


def glr(z, reset, js, cap=20.0):
    """G_t = max_j (C_t - C_{t-j})/sqrt(j), com C zerando nas partidas."""
    x = z.fillna(0.0).clip(upper=cap).to_numpy()
    r = reset.to_numpy()
    seg = np.cumsum(r)                      # id do segmento entre resets
    C = np.zeros(len(x))
    acc = 0.0; s0 = seg[0]
    for i in range(len(x)):
        if seg[i] != s0:
            acc = 0.0; s0 = seg[i]
        acc += x[i]; C[i] = acc
    G = np.full(len(x), -np.inf)
    for j in js:
        d = np.full(len(x), -np.inf)
        d[j:] = (C[j:] - C[:-j]) / np.sqrt(j)
        mesmo = np.full(len(x), False); mesmo[j:] = seg[j:] == seg[:-j]
        d = np.where(mesmo, d, -np.inf)
        G = np.maximum(G, d)
    return pd.Series(G, index=z.index)


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0); mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    op = df["in_operation"].astype(bool)
    reset = ((~mask) | (op & ~op.shift(fill_value=False)))
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    EW = {c: DET._sustained(E[c], BASE[c]*K[c]).values for c in SIN}
    Z = {c: (E[c] / (BASE[c]*K[c])).clip(upper=20) for c in SIN}
    mv = mask.values
    print(f"janela {T0:%Y-%m}+: {len(alvo)} eventos\n", flush=True)

    print("calculando GLR ...", flush=True)
    G = {c: glr(Z[c] - 1.0, reset, JS) for c in SIN}   # centrado no limiar: z=limiar -> 0
    print("calculando canais lentos ...", flush=True)
    SL = {}
    for c in SIN:
        for hl in HL_LENTO:
            s = out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
            for kk in K_LENTO:
                for sm in SUST_LENTO:
                    n = max(1, int(pd.Timedelta(minutes=sm) / pd.Timedelta("2min")))
                    SL[(c, hl, kk, sm)] = ((s > BASE[c]*K[c]*kk).astype(int)
                                           .rolling(n, min_periods=n).sum() >= n).values
    print(f"  {len(SL)} canais lentos\n", flush=True)

    def avalia_al(al, rot, **kw):
        y = A.avalia(al[sel], alvo, m2)
        return dict(rot=rot, det=y["det"], eps=y["episodios"], fp=y["fp_mes"],
                    hm=y["h_fp_mes"], lead=y["lead_med"], quais=",".join(y["detectados"]), **kw)

    L = []
    # GLR: sozinho e em OU com o degrau
    for h in H_GLR:
        gg = np.array([(G[c] > h).values for c in SIN])
        ew = np.array([EW[c] for c in SIN])
        for modo, n in [("GLR so", gg.sum(axis=0)), ("degrau OU GLR", (ew | gg).sum(axis=0))]:
            al = RF.dur_min(RF.refratario(pd.Series((n >= 2) & mv, index=idx), 48), 60)
            L.append(avalia_al(al, modo, par=f"h={h}"))
    print("  GLR varrido", flush=True)
    # canais lentos: sozinho e em OU com o degrau
    for hl in HL_LENTO:
        for kk in K_LENTO:
            for sm in SUST_LENTO:
                sl = np.array([SL[(c, hl, kk, sm)] for c in SIN])
                ew = np.array([EW[c] for c in SIN])
                for modo, n in [("lento so", sl.sum(axis=0)), ("degrau OU lento", (ew | sl).sum(axis=0))]:
                    al = RF.dur_min(RF.refratario(pd.Series((n >= 2) & mv, index=idx), 48), 60)
                    L.append(avalia_al(al, modo, par=f"hl={hl} k={kk} sust={sm}min"))
        print(f"  canal lento hl={hl} varrido", flush=True)
    T = pd.DataFrame(L); T.to_csv("glr_lento.csv", index=False)

    print("\n" + "=" * 100)
    print("REFERENCIA (degrau OU CUSUM, o ponto atual): 8/8  25 eps  1,46 FP/mes  "
          "52,8 h/mes  lead 30,4 h")
    print("=" * 100)
    print(f"\n{'modo':>18} {'parametros':>26} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6}")
    S = T[(T.det == 8)].sort_values(["fp", "hm"])
    for _, r in S.head(14).iterrows():
        print(f"{r.rot:>18} {r.par:>26} {int(r.det):4d}/8 {int(r.eps):5d} {r.fp:7.2f} "
              f"{r.hm:7.1f} {r.lead:6.1f}")
    if not len(S):
        print("  nenhuma configuracao com 8/8")
    print("\nmelhor de cada modo (8/8, menor FP):")
    for modo in T.rot.unique():
        g = T[(T.rot == modo) & (T.det == 8)].sort_values("fp")
        if not len(g):
            g2 = T[T.rot == modo].sort_values(["det", "fp"], ascending=[False, True]).iloc[0]
            print(f"  {modo:>18}: melhor e {int(g2.det)}/8 ({g2.par})"); continue
        r = g.iloc[0]
        print(f"  {modo:>18}: 8/8  {r.fp:.2f} FP/mes  {r.hm:6.1f} h/mes  lead {r.lead:5.1f} h  ({r.par})")


if __name__ == "__main__":
    main()
