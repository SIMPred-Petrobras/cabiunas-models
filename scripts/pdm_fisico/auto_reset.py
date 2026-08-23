#!/usr/bin/env python3
"""Limite de permanencia: um episodio nao pode durar mais de N horas.

Motivacao medida (dur_fp.py): 69% de todas as horas de alarme falso vem de 14
episodios com mais de 2 dias de duracao; o mais longo tem 225 h (9 dias). Nove
dos dez mais longos acendem os quatro sinais ao mesmo tempo -- assinatura de
mudanca de regime, nao de defeito. Enquanto isso, os acertos tem duracao
mediana de 6,1 h e quartil superior de 15 h.

Regra: cada episodio (agrupado com folga de 2 h, como no resto do projeto) e
truncado em N horas a partir do seu inicio. O resto daquele episodio fica
suprimido; so volta a alarmar depois que o sinal cair e subir de novo. Isso
imita um detector que reconhece o novo patamar como referencia e rearma.

O que se mede, e por que os dois:
  - det48   : houve alerta na janela de 48 h (a regua do projeto)
  - ativo2h : o alerta ainda estava ligado ate 2 h antes do trip
O truncamento nao muda o INICIO do episodio, so o fim -- entao det48 e a
antecedencia so caem quando o episodio comecou mais de N horas antes do trip.
E exatamente o risco desta regra, e 2025-03-17 (antecedencia de 89,7 h) deve
ser o primeiro a cair.
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

K_BASE, K_VIB = 1.3, 2.2
NS = [6, 12, 24, 48, 72, None]          # None = sem limite (situacao atual)


def trunca(al: pd.Series, n_h) -> pd.Series:
    """Corta cada episodio em n_h horas a partir do inicio."""
    if n_h is None:
        return al
    novo = pd.Series(False, index=al.index)
    for a, b in A.episodios(al):
        fim = min(b, a + pd.Timedelta(hours=n_h))
        janela = (al.index >= a) & (al.index <= fim)
        novo.loc[janela] = al.loc[janela]
    return novo


def main():
    df = canonico()
    ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    falhas = ftodas[ftodas >= "2025-01-01"].reset_index(drop=True)
    mask = mascara_pontuacao(df); idx = mask.index
    cal_meses = (idx[-1] - idx[0]).total_seconds() / 3600 / 730
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in ftodas]

    out = roda(BRACO, df, ftodas)
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    S = {"t": DET._sustained(ew("t", "1h"), DET.THR_FAM * K_BASE),
         "p": DET._sustained(ew("p", "1h"), DET.THR_FAM * K_BASE),
         "sp": DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * K_BASE),
         "vb": DET._sustained(ew("vb", "30min"), 3.0 * K_VIB)}
    n = sum(s.astype(int) for s in S.values())
    al0 = (n >= 2) & mask

    print(f"{'limite':>8} | {'FP':>4} {'h/mes':>7} {'h total':>8} {'ep>24h':>7} "
          f"| {'det48':>6} {'ativo2h':>8} {'lead med':>9}  perdidos em ativo2h")
    print("-" * 100)
    linhas = []
    for n_h in NS:
        al = trunca(al0, n_h)
        eps = A.episodios(al)
        def dur(a, b): return (b - a).total_seconds() / 3600 + 2 / 60
        fp = [(a, b) for a, b in eps
              if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
        h = sum(dur(a, b) for a, b in fp)
        longos = sum(1 for a, b in fp if dur(a, b) > 24)
        det48 = sum(1 for ev in falhas
                    if al[(al.index >= ev - pd.Timedelta(hours=48)) & (al.index < ev)].any())
        leads, ativo, perd = [], 0, []
        for ev in falhas:
            cand = [(a, b) for a, b in eps if a < ev and (ev - b).total_seconds() / 3600 <= 2.0]
            if cand:
                a, _ = max(cand, key=lambda x: x[0])
                leads.append((ev - a).total_seconds() / 3600); ativo += 1
            else:
                perd.append(ev.strftime("%m-%d"))
        rot = "sem" if n_h is None else f"{n_h} h"
        print(f"{rot:>8} | {len(fp):4d} {h/cal_meses:7.0f} {h:8.0f} {longos:7d} "
              f"| {det48:4d}/8 {ativo:6d}/8 {np.median(leads):8.1f}h  {','.join(perd)}",
              flush=True)
        linhas.append(dict(limite_h=n_h, fp=len(fp), h_mes=h/cal_meses, h_total=h,
                            eps_longos=longos, det48=det48, ativo2h=ativo,
                            lead_med=np.median(leads) if leads else np.nan))
    pd.DataFrame(linhas).to_csv("auto_reset.csv", index=False)


main()
