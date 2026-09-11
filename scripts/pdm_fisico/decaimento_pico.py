#!/usr/bin/env python3
"""Fechar o episodio quando ele decai em relacao ao PROPRIO pico.

Diagnostico: o episodio de 670 h abre com p em 22,20x o limiar e termina com p e
sp em 1,02x. Nenhum limiar absoluto separa isso -- 1,02x ainda esta aceso, e
subir o limiar mata deteccao (kb 1,5->5,0 leva 8/8 a 0/8). Histerese piora,
porque so prolonga alarme.

Aqui a regra e relativa: dentro de um episodio, acompanha o pico corrido de
max_c(E_c/thr_c) e FECHA quando o valor corrente cai abaixo de FRAC x esse pico.
Um sinal que subiu a 22x e voltou a 1x decaiu 95% -- e outra condicao, nao a
mesma. Um sinal que sobe monotonicamente (17/03: vb 1,10->2,50) nunca decai e
nao e afetado.

E a ultima alavanca estrutural nao testada. Se falhar, o episodio longo e
inseparavel da deteccao e a conclusao e que a saida tem de ser divulgada como e.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, EW, pos, mask, idx, alvo
from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV, paradas_reais_2h, classifica_regra_c
from regra_inicio_varredura import avalia_inicio

K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
RAZ = pd.concat([EW[c].where(mask) / (BASE[c] * K[c]) for c in SIN], axis=1)
RAZ.columns = list(SIN)
FORCA = RAZ.max(axis=1).to_numpy()          # forca do episodio: o canal mais aceso


def corta_por_decaimento(voto: pd.Series, frac: float) -> pd.Series:
    """Fecha o episodio quando a forca cai abaixo de frac x o pico ja atingido."""
    v = voto.to_numpy(); out = np.zeros(len(v), dtype=bool)
    pico = 0.0; dentro = False; morto = False
    for i, cond in enumerate(v):
        if not cond:
            dentro = False; morto = False; pico = 0.0
            continue
        f = FORCA[i]
        if not dentro:
            dentro = True; morto = False; pico = f if np.isfinite(f) else 0.0
        if np.isfinite(f):
            pico = max(pico, f)
            if pico > 0 and f < frac * pico:
                morto = True
        out[i] = not morto
    return pd.Series(out, index=voto.index)


ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)
v0 = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
paradas = paradas_reais_2h()
meses = float(mask.sum()) * 2 / 60.0 / 730.0
JAN = pd.Timedelta(hours=48)

print("CORTE POR DECAIMENTO RELATIVO AO PICO DO EPISODIO")
print("=" * 100)
print(f"{'frac':>6} | {'det':>5} {'FP':>4} {'NEUTRO':>7} {'FP/mes':>8} {'h/mes':>8} "
      f"{'lead':>7} | {'det_ini':>8} {'ep 26/02':>9} {'ep 17/03':>9}")
print("-" * 100)
for frac in [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]:
    v = corta_por_decaimento(v0, frac) if frac > 0 else v0
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    nnt = sum(1 for _, _, k, _ in cls if k == "NEUTRO")
    hfc = sum((b - a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP") / meses
    def dur_de(t):
        c = [(a, b) for a, b in eps if a <= t and b >= t - JAN]
        return f"{(c[0][1]-c[0][0]).total_seconds()/3600:.0f}h" if c else "--"
    print(f"{frac:6.2f} | {m['det']:4d}/8 {nfp:4d} {nnt:7d} {nfp/meses:8.3f} {hfc:8.1f} "
          f"{m['lead_med']:6.1f}h | {mi['det']:6d}/8 {dur_de(alvo.iloc[-1]):>9} "
          f"{dur_de(alvo.iloc[1]):>9}")
print("-" * 100)
print("  frac = 0 e o ponto de producao (8/8, 6 FP, 0,517, 7,1, 29,0 · 670h e 195h)")
