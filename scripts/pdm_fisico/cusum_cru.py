#!/usr/bin/env python3
"""CUSUM sobre o sinal CRU vs sobre o sinal ja suavizado por EWMA. Erro de desenho?

No ponto atual eu calculo Z = EWMA(sinal)/limiar e rodo o CUSUM em cima. Isso e integrar
DUAS VEZES: o CUSUM ja e uma soma acumulada de excessos, e a EWMA antes dele so acrescenta
o atraso da meia-vida sem filtrar nada que o acumulador nao filtre melhor. O CUSUM
classico opera sobre a observacao bruta padronizada exatamente por isso.

Se a dupla integracao esta custando, o CUSUM sobre o cru deve disparar mais cedo para o
mesmo custo -- e o lead e a metrica onde o CUSUM ganhou.

Bracos (o canal de degrau permanece sobre a EWMA nos dois, so o CUSUM muda):
  EWMA  Z = EWMA(sinal)/limiar   -> CUSUM        (o que esta rodando hoje)
  CRU   Z = sinal/limiar         -> CUSUM        (o classico)
  MISTO cada sinal escolhe                        (nao testado: t,p sao lentos por
        natureza -- erro de reconstrucao PCA -- e sp,vb sao pontuais)

Alerta = por sinal, degrau OU CUSUM; voto >=2; refratario 48 h; duracao minima 60 min.
Janela 2025-01 a 2026-04, 8 eventos. Referencia: 8/8, 1,46 FP/mes, 52,8 h/mes, lead 30,4 h.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
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
KAPPAS = [0.5, 0.75, 1.0]
HS = [10, 20, 40, 80, 160]


def cusum_bool(z, kappa, h, reset):
    x = (z - kappa).fillna(0.0).to_numpy(); r = reset.to_numpy()
    S = np.empty(len(x)); acc = 0.0
    for i in range(len(x)):
        acc = 0.0 if r[i] else max(0.0, acc + x[i]); S[i] = acc
    return S > h


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0); mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    op = df["in_operation"].astype(bool)
    reset = (~mask) | (op & ~op.shift(fill_value=False))
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    ew = np.array([DET._sustained(E[c], BASE[c]*K[c]).values for c in SIN])
    Z_EW = {c: (E[c] / (BASE[c]*K[c])).clip(upper=20) for c in SIN}
    Z_CR = {c: (out[c].where(mask) / (BASE[c]*K[c])).clip(upper=20) for c in SIN}
    mv = mask.values
    print(f"janela {T0:%Y-%m}+: {len(alvo)} eventos\n", flush=True)
    print("dispersao do sinal, cru vs suavizado (desvio padrao em operacao):")
    for c in SIN:
        print(f"  {c:>3}: cru {Z_CR[c].std():6.2f}   EWMA {Z_EW[c].std():6.2f}   "
              f"razao {Z_CR[c].std()/max(Z_EW[c].std(),1e-9):.2f}x")
    print(flush=True)

    CU = {}
    for fonte, Z in [("EWMA", Z_EW), ("CRU", Z_CR)]:
        for c in SIN:
            for ka in KAPPAS:
                for h in HS:
                    CU[(fonte, c, ka, h)] = cusum_bool(Z[c], ka, h, reset)
        print(f"  acumuladores sobre {fonte} prontos", flush=True)

    L = []
    for fonte in ["EWMA", "CRU"]:
        for ka in KAPPAS:
            for h in HS:
                cu = np.array([CU[(fonte, c, ka, h)] for c in SIN])
                al = RF.dur_min(RF.refratario(
                    pd.Series(((ew | cu).sum(axis=0) >= 2) & mv, index=idx), 48), 60)
                y = A.avalia(al[sel], alvo, m2)
                L.append(dict(fonte=fonte, kappa=ka, h=h, det=y["det"], eps=y["episodios"],
                              fp=y["fp_mes"], hm=y["h_fp_mes"], lead=y["lead_med"],
                              quais=",".join(y["detectados"])))
    # misto: CRU para sp,vb (pontuais) e EWMA para t,p (ja lentos)
    for ka in KAPPAS:
        for h in HS:
            cu = np.array([CU[("EWMA" if c in ("t", "p") else "CRU", c, ka, h)] for c in SIN])
            al = RF.dur_min(RF.refratario(
                pd.Series(((ew | cu).sum(axis=0) >= 2) & mv, index=idx), 48), 60)
            y = A.avalia(al[sel], alvo, m2)
            L.append(dict(fonte="MISTO", kappa=ka, h=h, det=y["det"], eps=y["episodios"],
                          fp=y["fp_mes"], hm=y["h_fp_mes"], lead=y["lead_med"],
                          quais=",".join(y["detectados"])))
    T = pd.DataFrame(L); T.to_csv("cusum_cru.csv", index=False)

    print("\n" + "=" * 96)
    print("REFERENCIA (hoje): EWMA kappa=0,75 h=40 -> 8/8  1,46 FP/mes  52,8 h/mes  lead 30,4 h")
    print("=" * 96)
    print(f"\n{'fonte':>7} {'kappa':>6} {'h':>5} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6}")
    S = T[T.det == 8].sort_values(["fp", "hm"])
    for _, r in S.head(16).iterrows():
        marca = "  <-- hoje" if (r.fonte == "EWMA" and r.kappa == 0.75 and r.h == 40) else ""
        print(f"{r.fonte:>7} {r.kappa:6.2f} {int(r.h):5d} {int(r.det):4d}/8 {int(r.eps):5d} "
              f"{r.fp:7.2f} {r.hm:7.1f} {r.lead:6.1f}{marca}")

    print("\nmelhor de cada fonte (8/8, menor FP; empate -> maior lead):")
    for f in ["EWMA", "CRU", "MISTO"]:
        g = T[(T.fonte == f) & (T.det == 8)].sort_values(["fp", "lead"], ascending=[True, False])
        if not len(g):
            g2 = T[T.fonte == f].sort_values("det", ascending=False).iloc[0]
            print(f"  {f:>6}: melhor e {int(g2.det)}/8"); continue
        r = g.iloc[0]
        print(f"  {f:>6}: 8/8  {r.fp:.2f} FP/mes  {r.hm:6.1f} h/mes  lead {r.lead:5.1f} h  "
              f"(kappa={r.kappa} h={int(r.h)})")

    print("\n" + "=" * 96); print("PLATO em h, por fonte (kappa que da o melhor de cada)")
    print("=" * 96)
    for f in ["EWMA", "CRU", "MISTO"]:
        g0 = T[(T.fonte == f) & (T.det == 8)]
        ka = g0.sort_values("fp").iloc[0].kappa if len(g0) else KAPPAS[1]
        g = T[(T.fonte == f) & (T.kappa == ka)].sort_values("h")
        print(f"  {f:>6} (kappa={ka}):  h = " + " ".join(f"{int(v):>5d}" for v in g.h))
        print(f"  {'':>6}  {'':>12} det = " + " ".join(f"{int(v):>5d}" for v in g.det))
        print(f"  {'':>6}  {'':>12} FP  = " + " ".join(f"{v:5.2f}" for v in g.fp))


if __name__ == "__main__":
    main()
