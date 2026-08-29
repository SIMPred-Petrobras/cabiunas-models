#!/usr/bin/env python3
"""O blackout faz DUAS coisas. Qual delas e a que importa?

O blackout de 6 h entra em dois lugares do detector:
  (a) MASCARA -- aqueles instantes nao sao pontuados nem podem virar alerta;
  (b) RESET DO CUSUM -- `reset = ~mask | partida`, entao durante as 6 h o
      acumulador fica preso em carga residual em vez de somar.

ajuste_dois_regimes.py mostrou que dar ao transiente um modelo proprio calibra o
escore (mediana do vb 8,25 -> 1,61, bem abaixo do limiar 6,60) e ainda assim o
blackout 0 nao cabe no orcamento: piso de 1,66 FP/mes contra teto de 1,15. Se o
escore esta calibrado, a explicacao que sobra e (b) -- sem o blackout o CUSUM
acumula atraves da partida.

Este script separa: pontua TUDO (mascara = est) mas mantem o reset do CUSUM sobre
as 6 h. Se o custo voltar ao normal, o valor do blackout esta todo no reset, e a
mascara podia ser dispensada -- o que devolveria as 309 h a deteccao de graca.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import GRID, SUSTAIN, SIN, HL, BASE, KAPPA, H_CUSUM, CARGA, T0
from blackout_curto import cusum, pos as pos_bl

KB = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]; KV = [1.8, 2.2, 2.8]; ORC = 1.15
g = pd.read_parquet("grade2min.parquet"); idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False); est = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
blk = part.rolling(int(pd.Timedelta("6h")/pd.Timedelta(GRID)), min_periods=1).max().astype(bool)
steady = est & ~blk; sel = idx >= T0
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))
Z = np.load("dois_regimes_escores.npz")
UM = pd.DataFrame({c: Z[f"um_{c}"] for c in SIN}, index=idx)
DO = pd.DataFrame({c: Z[f"dois_{c}"] for c in SIN}, index=idx)


def roda(S, mask, reset, kb, kv):
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}; ON = {}
    for c in SIN:
        thr = BASE[c]*K[c]
        E = S[c].ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean().where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    v = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    return AV.avalia(pos_bl(v), alvo, mask)


m_est, m_std = est & sel, steady & sel
r_curto = ((~m_est) | part).to_numpy()                 # reset so na partida
r_longo = ((~m_std) | part).to_numpy()                 # reset ao longo das 6 h
CEN = [("mascara 6 h + reset 6 h (atual)", UM,  m_std, r_longo),
       ("pontua tudo + reset 6 h",         UM,  m_est, r_longo),
       ("pontua tudo + reset so partida",  UM,  m_est, r_curto),
       ("dois regimes: tudo + reset 6 h",  DO,  m_est, r_longo)]
lin = []
for nome, S, mk, rs in CEN:
    for kb in KB:
        for kv in KV:
            m = roda(S, mk, rs, kb, kv)
            perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
            lin.append(dict(cenario=nome, kb=kb, kv=kv, det=m["det"], eps=m["episodios"],
                            fp_mes=round(m["fp_mes"],3), h_fp_mes=round(m["h_fp_mes"],1),
                            lead=round(m["lead_med"],2) if m["det"] else np.nan,
                            horas_op=round(m["horas_op"],0), perdidos=",".join(perd)))
    print(f"  {nome}", flush=True)
d = pd.DataFrame(lin); d.to_csv("blackout_decompoe.csv", index=False)
print("\n" + "="*98)
print(f"{'cenario':34s} {'h op':>7s} {'8/8 em':>7s} {'melhor no teto':>16s} {'fp':>6s} {'h/mes':>7s} {'lead':>6s}")
print("="*98)
for nome, s in d.groupby("cenario", sort=False):
    a = s[s.fp_mes <= ORC].sort_values(["det","h_fp_mes"], ascending=[False,True])
    ta = f"{a.iloc[0].det}/8 (k {a.iloc[0].kb}/{a.iloc[0].kv})" if len(a) else "   --"
    print(f"{nome:34s} {s.horas_op.iloc[0]:7,.0f} {int((s.det==8).sum()):>4d}/18 {ta:>16s} "
          f"{a.iloc[0].fp_mes if len(a) else float('nan'):>6.2f} "
          f"{a.iloc[0].h_fp_mes if len(a) else float('nan'):>7.1f} "
          f"{a.iloc[0].lead if len(a) else float('nan'):>6.1f}")
print("\n-> blackout_decompoe.csv")
