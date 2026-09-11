#!/usr/bin/env python3
"""Reescalonamento por MAGNITUDE -- alarme novo quando a intensidade salta.

Diagnostico (diagnostico_tres_faltantes.py): os tres eventos que falham na regua
de inicio tem sinal FORTE na janela de 48 h, e o pos-processamento o descarta:
  27/02  p vai a 268x o limiar -- episodio ja aberto, vb parado em 3,5x o segura
  17/03  t vai a 8,38x         -- idem, vb em 2,5x segura ha 195 h
  29/04  t vai a 3,20x         -- refratario do episodio de 51,8 h antes

A `escalada` que ja existe conta NUMERO de canais simultaneos, nao magnitude --
por isso nunca ajudou (memoria: "escalada e seguro, nao ganho"). Aqui a regra e
de intensidade:

  CORTE   dentro de uma corrida de voto, fecha o episodio e abre um novo quando
          a forca supera M x o minimo desde o inicio do episodio corrente
          ("reintensificacao")
  FURO    um episodio bloqueado por refratario passa se o seu pico de forca
          supera M x o pico do episodio que abriu o bloqueio
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import KB, KV, paradas_reais_2h, classifica_regra_c


from avalia import GAP_EP_H

N_GAP = int(pd.Timedelta(hours=GAP_EP_H) / (idx[1] - idx[0])) + 1


def corta_reintensifica(voto: np.ndarray, forca: np.ndarray, M: float) -> np.ndarray:
    """Quebra o episodio onde a forca supera M x o minimo desde o seu inicio.

    A quebra tem de ser MAIOR que GAP_EP_H, senao `episodios()` funde as duas
    partes de volta e a quebra e invisivel (foi o bug da primeira versao). A
    folga sai do RABO do episodio velho, nao do inicio do novo -- silenciar
    justamente o pico seria o oposto do que se quer."""
    v = np.asarray(voto, dtype=bool).copy()
    if M <= 1:
        return v
    f = np.where(np.isfinite(forca), forca, 0.0)
    piso = np.inf; dentro = False; ini = 0
    for i in range(len(v)):
        if not v[i]:
            dentro = False; piso = np.inf
            continue
        if not dentro:
            dentro = True; piso = f[i]; ini = i; continue
        if piso > 0 and f[i] > M * piso:
            j = max(ini + 1, i - N_GAP)      # nao come o episodio velho inteiro
            v[j:i] = False
            piso = f[i]; ini = i
        else:
            piso = min(piso, f[i])
    return v


def pos_furo(voto, n_sin, forca, refrat_h, dur_min, M):
    """Refratario + duracao minima, com FURO por magnitude."""
    al = pd.Series(False, index=idx)
    bloq, forca_bloq = None, 0.0
    fs = pd.Series(np.where(np.isfinite(forca), forca, 0.0), index=idx)
    for a, b in AV.episodios(voto):
        pico = float(fs.loc[a:b].max())
        if bloq is not None and a <= bloq:
            if not (M > 1 and pico > M * forca_bloq):
                continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=refrat_h)
        forca_bloq = pico
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= dur_min:
            fin.loc[a:b] = True
    return fin & sel


K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
ON = partes(KB, KV); ns = sum(ON[c].astype(int) for c in SIN)
v_base = (pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])).to_numpy()
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0

print("REESCALONAMENTO POR MAGNITUDE  (sobre o religamento frac=0,03, refrat=72h)")
print("=" * 92)
print(f"{'M':>6} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8} {'lead_ini':>9}  quem passa a nascer")
print("-" * 92)
JAN = pd.Timedelta(hours=48)
base_nasce = None
for M in [0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]:
    v = corta_rearma(v_base, F, 0.03)
    if M > 1:
        v = corta_reintensifica(v, F, M)
    al = pos_furo(pd.Series(v, index=idx), ns, F, 72, DUR_MIN, M)
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    nasce = {t.strftime("%d/%m") for t in alvo
             if any(t - JAN <= x <= t for x, _ in eps)}
    if base_nasce is None:
        base_nasce = nasce
    novos = sorted(nasce - base_nasce)
    print(f"{M:6.1f} {m['det']:5d}/8 {mi['det']:7d}/8 {nfp/meses:9.3f} {h/meses:8.1f} "
          f"{(mi['lead_med'] if mi['det'] else 0):8.1f}h  {', '.join(novos) if novos else '--'}")
