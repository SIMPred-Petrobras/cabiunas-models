#!/usr/bin/env python3
"""Detector de dois niveis: ATENCAO (canal lento) e CONFIRMADO (ponto atual).

De onde vem. O canal lento de meia-vida 24 h da o maior lead da investigacao -- 39,5 h --
mas custa 154,7 h/mes contra 52,8 do ponto atual. Como alarme unico nao serve; como
AVISO ANTECIPADO de baixa severidade pode servir, porque o custo dele nao e da mesma
natureza: uma flag de atencao que fica ligada nao interrompe ninguem, so muda a cor do
painel.

Os dois niveis se medem com reguas DIFERENTES, e isso e o ponto do exercicio:

  CONFIRMADO  e alarme acionavel -> mede-se por episodio: FP/mes, deteccao, lead.
  ATENCAO     nao e acionavel    -> mede-se por: (a) quanto tempo fica ligado, (b) que
              fracao dos episodios de atencao ESCALA para confirmado, (c) quanta
              antecedencia extra da nos eventos que o confirmado ja pega.

O teste que decide: a atencao precede mesmo o confirmado, evento a evento? Se a diferenca
de lead for pequena ou negativa, o nivel extra e ruido com nome bonito.

E a taxa de escalonamento e o numero que diz se a atencao carrega informacao: se episodios
de atencao que NAO escalam forem a regra, a flag e so um limiar frouxo.
"""
from __future__ import annotations
import sys
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
KAPPA, H_CUSUM = 0.75, 40
LENTOS = [("24h", 0.8, 60), ("24h", 1.0, 180), ("12h", 1.0, 60), ("24h", 1.3, 180)]
JAN = pd.Timedelta(hours=48)


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
    op = df["in_operation"].astype(bool)
    reset = (~mask) | (op & ~op.shift(fill_value=False))
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    Z = {c: (E[c] / (BASE[c]*K[c])).clip(upper=20) for c in SIN}
    mv = mask.values
    meses = mask.sum()*2/60/730.0

    # --- nivel CONFIRMADO: degrau OU CUSUM por sinal, voto >=2
    ew = np.array([DET._sustained(E[c], BASE[c]*K[c]).values for c in SIN])
    cu = np.array([cusum_bool(Z[c], KAPPA, H_CUSUM, reset) for c in SIN])
    conf = RF.dur_min(RF.refratario(pd.Series(((ew | cu).sum(axis=0) >= 2) & mv, index=idx), 48), 60)
    xc = A.avalia(conf[sel], alvo, m2)
    print("=" * 100)
    print(f"NIVEL CONFIRMADO (acionavel): {xc['det']}/8  {xc['episodios']} eps  "
          f"{xc['fp_mes']:.2f} FP/mes  {xc['h_fp_mes']:.1f} h/mes  lead {xc['lead_med']:.1f} h")
    print("=" * 100, flush=True)

    for hl, kk, sm in LENTOS:
        n = max(1, int(pd.Timedelta(minutes=sm) / pd.Timedelta("2min")))
        sl = np.array([((out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
                         > BASE[c]*K[c]*kk).astype(int)
                        .rolling(n, min_periods=n).sum() >= n).values for c in SIN])
        aten = pd.Series((sl.sum(axis=0) >= 2) & mv, index=idx)
        aten = RF.dur_min(aten, 60)          # sem refratario: e uma flag, nao um alarme
        xa = A.avalia(aten[sel], alvo, m2)
        eps_a = A.episodios(aten & sel)
        eps_c = A.episodios(conf & sel)
        # escalonamento: episodio de atencao que contem ou e seguido por confirmado em 48 h
        escala = sum(1 for a, b in eps_a
                     if any(x <= b + JAN and z >= a for x, z in eps_c))
        # antecedencia extra por evento
        extras = []
        for t in alvo:
            wa = aten.loc[t-JAN*3:t]; wc = conf.loc[t-JAN*3:t]
            oa = wa[wa.fillna(False)]; oc = wc[wc.fillna(False)]
            if len(oa) and len(oc):
                extras.append(((oc.index[0]-oa.index[0]).total_seconds()/3600))
        print(f"\nATENCAO  meia-vida {hl}, k x{kk}, sustentacao {sm} min")
        print(f"   cobertura: {xa['det']}/8 eventos   lead {xa['lead_med']:.1f} h   "
              f"duty {100*aten[sel].mean():.1f}% do tempo pontuavel")
        print(f"   episodios de atencao: {len(eps_a)} ({len(eps_a)/meses:.2f}/mes)   "
              f"escalam para confirmado: {escala} ({100*escala/max(len(eps_a),1):.0f}%)")
        print(f"   antecedencia EXTRA sobre o confirmado, por evento: "
              f"mediana {np.median(extras) if extras else float('nan'):.1f} h   "
              f"min {min(extras) if extras else float('nan'):.1f}   "
              f"max {max(extras) if extras else float('nan'):.1f}")
        neg = sum(1 for e in extras if e <= 0)
        print(f"   eventos em que a atencao NAO precede o confirmado: {neg}/{len(extras)}",
              flush=True)


if __name__ == "__main__":
    main()
