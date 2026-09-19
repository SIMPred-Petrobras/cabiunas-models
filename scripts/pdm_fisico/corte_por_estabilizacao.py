#!/usr/bin/env python3
"""Encerrar o alarme quando o sinal PARA DE PIORAR -- corta carga quase de graca.

O PROBLEMA, com numero novo. A metrica de carga ([[a-regra-c-nao-limita-duracao]])
revelou episodios TP de 140 h, 195 h e ate 647 h. Sao acertos -- precedem trip --
mas o operador fica com alarme aceso por SEMANAS. Depois que a operacao foi
avisada, manter aceso 27 dias nao entrega informacao nova; entrega fadiga.

A ASSIMETRIA QUE TORNA ISTO BARATO. As reguas que decidem -- banda acionavel e
regua de inicio -- contam o NASCIMENTO do episodio, nao a duracao. Encerrar mais
cedo nao pode custar banda nem inicio. So a regua "de pe" (alarme ativo no
instante do trip) e afetada, e ela e justamente a que as outras equipes nao usam.

O QUE E DIFERENTE do que ja foi testado. `corte_com_rearme.py` encerra por
DECAIMENTO -- o sinal tem de cair. Uma versao anterior sem religamento matava
deteccao ([[episodios-longos-degradacao-ou-travado]]). Aqui o criterio e
ESTABILIZACAO: o sinal nao precisa cair, so parar de subir. Degradacao ativa
empurra o sinal para cima; sinal alto e parado ha horas ja disse o que tinha a
dizer.

E o religamento continua existindo de graca: a escalada por idade do v2 ja
reabre alarme quando a forca cruza 20x dentro de episodio velho.

Uso:  PYTHONPATH=. python corte_por_estabilizacao.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo, sel
from publica_clearml import (SIN, BASE, VOTO_LO, VOTO_HI, REFRAT_V2, DUR_MIN,
                             ESC_IDADE, ESC_ABS, ESC_DUR, TMIN_BANDA)
from margem_no_v2 import A, B, FORCA, decide
from regra_c_com_teto import classifica

JAN = pd.Timedelta(hours=48)
POR_H = 30


def voto_base():
    nA = sum(A[c].astype(int) for c in SIN)
    nB = sum(B[c].astype(int) for c in SIN)
    vA = pd.Series(nA >= VOTO_LO, index=idx) & mask
    vB = pd.Series(nB >= VOTO_HI, index=idx) & mask & (B["sp"] | B["vb"])
    return vA | vB


def encerra_estabilizado(fin: pd.Series, horas: float, ganho_min: float) -> pd.Series:
    """Corta o episodio quando a forca para de subir.

    Percorre cada episodio acompanhando o maximo acumulado da forca. Se, ao longo
    de `horas`, esse maximo nao cresceu pelo menos `ganho_min` (fracao), o alarme
    ja entregou o que tinha: encerra ali. O resto do episodio e apagado."""
    out = fin.copy()
    f = FORCA.fillna(0.0)
    n = int(horas * POR_H)
    for a, b in AV.episodios(fin):
        w = f.loc[a:b]
        if len(w) <= n:
            continue
        pico = w.cummax().to_numpy()
        # o pico de `horas` atras, alinhado
        antes = np.concatenate([np.full(n, pico[0]), pico[:-n]])
        parado = pico <= antes * (1.0 + ganho_min)
        # encerra no primeiro instante em que ja se passaram `horas` sem ganho
        idx_w = w.index
        cand = np.flatnonzero(parado)
        cand = cand[cand >= n]
        if len(cand):
            out.loc[idx_w[cand[0]]:b] = False
    return out


def mede(fin):
    eps = AV.episodios(fin)
    det = AV.avalia(fin, alvo, mask)
    cls = classifica(eps, None)
    mes = det["horas_op"] / 730.0
    h_fp = sum(d for *_, k, d in cls if k == "FP")
    h_ne = sum(d for *_, k, d in cls if k == "NEUTRO")
    h_tp = sum(d for *_, k, d in cls if k == "TP")
    ban = sum(1 for t in alvo if any(t - JAN <= a <= t - pd.Timedelta(hours=TMIN_BANDA)
                                     for a, _ in eps))
    ini = sum(1 for t in alvo if any(t - JAN <= a <= t for a, _ in eps))
    n_fp = sum(1 for *_, k, _ in cls if k == "FP")
    dur = [d for *_, d in cls]
    return dict(banda=ban, inicio=ini, det=det["det"], fp=n_fp / mes,
                carga=(h_fp + h_ne) / mes, h_tp=h_tp / mes, eps=len(eps),
                maior=max(dur) if dur else 0.0)


def linha(nome, fin):
    r = mede(fin)
    print(f"{nome:<30}{r['banda']:>5}/8{r['inicio']:>6}/8{r['det']:>4}/8"
          f"{r['fp']:>8.3f}{r['carga']:>9.1f}{r['h_tp']:>9.1f}"
          f"{r['eps']:>6}{r['maior']:>9.1f}h")
    return r


if __name__ == "__main__":
    base = decide(voto_base())
    print(f"{'':<30}{'banda':>7}{'inicio':>8}{'det':>6}{'FP/mes':>8}"
          f"{'CARGA':>9}{'h TP/mes':>9}{'eps':>6}{'maior ep':>10}")
    print("-" * 96)
    b0 = linha("v2 publicado", base)

    for horas in (6, 12, 24, 48):
        print(f"\n  ---- sem ganho por {horas} h ----")
        for g in (0.0, 0.05, 0.20):
            r = linha(f"ganho minimo {g:.0%}", encerra_estabilizado(base, horas, g))
            perdeu = []
            if r["banda"] < b0["banda"]: perdeu.append("banda")
            if r["inicio"] < b0["inicio"]: perdeu.append("inicio")
            if perdeu:
                print(f"{'':<30}   perde: {', '.join(perdeu)}")
