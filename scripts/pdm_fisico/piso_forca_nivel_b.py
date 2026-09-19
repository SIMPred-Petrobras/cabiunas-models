#!/usr/bin/env python3
"""Piso de forca no NIVEL B -- a pista que a pericia dos FP deixou.

DE ONDE VEM. `pericia_fp_v2.py` retratou os 4 falsos positivos e os 8 acertos no
instante do nascimento. Nenhuma variavel isolada separa os dois grupos, mas a
FORMA DE ENTRADA mostra assimetria:

    entrada       FP   TP
    so nivel A     0    3
    so nivel B     3    2
    A+B            1    4

O nivel B -- o especifico, 2 de 4 em limiar alto com portao mecanico -- concentra
tres dos quatro FP. E a forca no nascimento separa dentro dele:

    FP por B:   170,6  ·   6,2  ·   3,5
    TP por B:    29,2  · 146,8

Ha uma janela entre 6,2 e 29,2 que mata dois FP sem tocar em acerto nenhum.

A LEITURA FISICA sustenta: o nivel B existe para pegar evento FORTE e estreito (e
por isso tem limiar alto e exige canal mecanico). Deixa-lo disparar com forca 3,5
contradiz o proprio papel -- esse regime e do nivel A, que exige mais canais em
troca de limiar baixo.

O RISCO, e e serio: o limiar esta sendo escolhido olhando 4 FP e 8 TP. E
sobreajuste por construcao. Por isso o script varre a vizinhanca inteira em vez
de reportar um ponto, e a leitura util e o PLATO -- se o ganho existe so num
valor, e sorte ([[o-ponto-publicado-e-o-melhor-de-nove]]).

Uso:  PYTHONPATH=. python piso_forca_nivel_b.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import SIN, VOTO_LO, VOTO_HI, TMIN_BANDA
from margem_no_v2 import A, B, FORCA, decide
from regra_c_com_teto import classifica

JAN = pd.Timedelta(hours=48)


def voto(piso_b: float | None):
    """v2, com piso de forca opcional no nivel B.

    O piso olha a forca NO INSTANTE, nao no pico do episodio: a decisao tem de
    ser tomavel em tempo real, sem saber o que vem depois."""
    nA = sum(A[c].astype(int) for c in SIN)
    nB = sum(B[c].astype(int) for c in SIN)
    vA = pd.Series(nA >= VOTO_LO, index=idx) & mask
    vB = pd.Series(nB >= VOTO_HI, index=idx) & mask & (B["sp"] | B["vb"])
    if piso_b is not None:
        vB = vB & (FORCA >= piso_b)
    return vA | vB


def mede(fin):
    eps = AV.episodios(fin)
    det = AV.avalia(fin, alvo, mask)
    cls = classifica(eps, None)
    mes = det["horas_op"] / 730.0
    n_fp = sum(1 for *_, k, _ in cls if k == "FP")
    h_fp = sum(d for *_, k, d in cls if k == "FP")
    h_ne = sum(d for *_, k, d in cls if k == "NEUTRO")
    ban = sum(1 for t in alvo if any(t - JAN <= a <= t - pd.Timedelta(hours=TMIN_BANDA)
                                     for a, _ in eps))
    ini = sum(1 for t in alvo if any(t - JAN <= a <= t for a, _ in eps))
    leads = [(t - max([a for a, _ in eps if t - JAN <= a <= t])).total_seconds() / 3600
             for t in alvo if any(t - JAN <= a <= t for a, _ in eps)]
    return dict(banda=ban, inicio=ini, det=det["det"], n_fp=n_fp, fp=n_fp / mes,
                h=h_fp / mes, carga=(h_fp + h_ne) / mes, eps=len(eps),
                lead=float(np.mean(leads)) if leads else float("nan"))


if __name__ == "__main__":
    b0 = mede(decide(voto(None)))
    print(f"{'piso no nivel B':<20}{'banda':>7}{'inicio':>8}{'det':>6}"
          f"{'FP':>4}{'FP/mes':>9}{'h/mes':>8}{'CARGA':>8}{'lead':>8}{'eps':>6}")
    print("-" * 84)
    print(f"{'sem piso (v2)':<20}{b0['banda']:>5}/8{b0['inicio']:>6}/8{b0['det']:>4}/8"
          f"{b0['n_fp']:>4}{b0['fp']:>9.3f}{b0['h']:>8.1f}{b0['carga']:>8.1f}"
          f"{b0['lead']:>7.1f}h{b0['eps']:>6}")
    for piso in (2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40):
        r = mede(decide(voto(piso)))
        al = []
        if r["banda"] < b0["banda"]: al.append("-banda")
        if r["inicio"] < b0["inicio"]: al.append("-inicio")
        if r["det"] < b0["det"]: al.append("-det")
        gan = " <<< " + f"{b0['n_fp']-r['n_fp']} FP a menos" if r["n_fp"] < b0["n_fp"] and not al else ""
        print(f"{f'forca >= {piso}':<20}{r['banda']:>5}/8{r['inicio']:>6}/8{r['det']:>4}/8"
              f"{r['n_fp']:>4}{r['fp']:>9.3f}{r['h']:>8.1f}{r['carga']:>8.1f}"
              f"{r['lead']:>7.1f}h{r['eps']:>6}"
              + ("  " + ",".join(al) if al else gan))
