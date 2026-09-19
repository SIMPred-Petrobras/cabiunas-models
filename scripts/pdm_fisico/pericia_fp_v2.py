#!/usr/bin/env python3
"""Os quatro falsos positivos do v2: o que os distingue dos acertos?

RESTRICAO DO PROBLEMA. A frente de reduzir CARGA foi descartada: cortar episodio
longo derruba as horas de 48,9 para 7,2 mas triplica o FP contado, e falso
positivo e justamente o que nao pode subir. Entao a unica direcao util e a
oposta: matar FP sem tocar na deteccao.

Sao so QUATRO episodios. Com tao poucos, olhar um a um e mais informativo que
varrer parametro -- e e o que nunca foi feito sobre o v2. A analise equivalente
no v1 achou um padrao forte: 9 dos 15 nasciam a 6,4667 h da partida, cravado, na
borda do blackout ([[a-borda-do-blackout-explica-os-fp]]).

O QUE ESTE SCRIPT PROCURA. Para cada FP e cada TP, o retrato do nascimento:
  * quantos canais votaram, e quais
  * por qual nivel entrou (A sensivel, B especifico, ou ambos)
  * a forca no nascimento e o pico
  * quanto tempo desde a ultima partida
  * se o CUSUM sozinho sustentou (nivel abaixo do limiar) -- os canais nossos
    devem 11-23% do duty so ao CUSUM ([[o-detector-esta-numa-fronteira]])

Um discriminante util precisa separar TODOS os 4 FP de TODOS os 8 TP. Com n=4 e
n=8 qualquer regra encontrada e fragil, entao o criterio e duro: so vale se for
limpo e tiver leitura fisica.

Uso:  PYTHONPATH=. python pericia_fp_v2.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo, op
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, K_LO, VOTO_LO, VOTO_HI
from margem_no_v2 import A, B, FORCA, decide
from regra_c_com_teto import classifica
from blackout_curto import cusum

KH = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
part = op & ~op.shift(fill_value=False)
partidas = idx[part.to_numpy()]
from pos_processamento import EW


def so_cusum(c, k):
    """o canal esta aceso APENAS pelo CUSUM (nivel ja abaixo do limiar)?"""
    thr = BASE[c] * k
    E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    reset = ((~mask) | part).to_numpy()
    cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    return (cu & ~deg) & mask


SOC = {c: so_cusum(c, K_LO[c]) for c in SIN}


def retrato(a, b):
    """o que estava acontecendo no NASCIMENTO do episodio."""
    j = idx[(idx >= a) & (idx <= min(b, a + pd.Timedelta(hours=1)))]
    acesos_A = [c for c in SIN if bool(A[c].loc[j].any())]
    acesos_B = [c for c in SIN if bool(B[c].loc[j].any())]
    nA = len(acesos_A); nB = len(acesos_B)
    nivel = []
    if nA >= VOTO_LO: nivel.append("A")
    if nB >= VOTO_HI and any(c in acesos_B for c in ("sp", "vb")): nivel.append("B")
    ant = partidas[partidas <= a]
    desde = (a - ant[-1]).total_seconds() / 3600 if len(ant) else float("nan")
    cus = [c for c in SIN if bool(SOC[c].loc[j].any())]
    return dict(nA=nA, nB=nB, canaisA="".join(c[0] for c in acesos_A),
                nivel="+".join(nivel) or "-", desde=desde,
                f_ini=float(FORCA.loc[j].max()), f_pico=float(FORCA.loc[a:b].max()),
                cusum="".join(c[0] for c in cus) or "-",
                dur=(b - a).total_seconds() / 3600 + 2 / 60)


if __name__ == "__main__":
    fin = decide(None) if False else None
    from margem_no_v2 import voto_v2
    fin = decide(voto_v2(None, ""))
    cls = classifica(AV.episodios(fin), None)
    print(f"{'tipo':>7} {'nascimento':<18}{'dur':>8}{'nA':>4}{'nB':>4}"
          f"{'canais A':>10}{'nivel':>7}{'f ini':>8}{'f pico':>8}"
          f"{'desde part':>11}{'so CUSUM':>10}")
    print("-" * 96)
    for grupo in ("FP", "TP", "NEUTRO"):
        for a, b, k, d in cls:
            if k != grupo: continue
            r = retrato(a, b)
            print(f"{k:>7} {a:%Y-%m-%d %H:%M}{r['dur']:>7.1f}h{r['nA']:>4}{r['nB']:>4}"
                  f"{r['canaisA']:>10}{r['nivel']:>7}{r['f_ini']:>8.1f}{r['f_pico']:>8.1f}"
                  f"{r['desde']:>10.1f}h{r['cusum']:>10}")
        print()

    print("=" * 96)
    print("DISCRIMINANTES: algum separa TODOS os 4 FP de TODOS os 8 TP?")
    print("=" * 96)
    R = {k: [retrato(a, b) for a, b, kk, _ in cls if kk == k] for k in ("FP", "TP")}
    for campo, rot in (("f_pico", "forca de pico"), ("f_ini", "forca no nascimento"),
                       ("nA", "canais no nivel A"), ("desde", "horas desde a partida"),
                       ("dur", "duracao do episodio")):
        fp = [r[campo] for r in R["FP"]]; tp = [r[campo] for r in R["TP"]]
        sep = max(fp) < min(tp) or min(fp) > max(tp)
        print(f"  {rot:<24} FP: {min(fp):7.1f} a {max(fp):7.1f}   "
              f"TP: {min(tp):7.1f} a {max(tp):7.1f}   "
              f"{'SEPARA' if sep else 'sobrepoe'}")
