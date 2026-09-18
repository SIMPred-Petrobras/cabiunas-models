#!/usr/bin/env python3
"""DRIFT: o normalizador e o ponto frágil, e dá para estabilizá-lo.

O DIAGNOSTICO. Tudo que se mexe no processo de baseline leva o custo de 6,6 para
40-120 h/mes: tamanho da janela (drift_baseline), ensemble de scores (idem),
limpeza das janelas pre-falha (baseline_limpo). Um detector cujo custo varia 10x
com qualquer perturbacao nao esta num otimo -- esta num ponto de sorte.

A HIPOTESE UNIFICADORA. As tres perturbacoes tem uma coisa em comum: mudam o
`recon_p99`, o percentil 99 do erro no proprio baseline, pelo qual TODO score e
dividido. Ele e o normalizador, e e reestimado do zero a cada mes, de uma amostra
que muda de tamanho e de composicao. Se ele treme, o score inteiro reescala e o
duty dos canais vai junto -- e o duty e o que decide o custo.

A CORRECAO PROPOSTA e cirurgica, e diferente do ensemble que ja falhou: manter o
PCA mensal (a adaptacao ao drift, que e necessaria e esta medida) e suavizar
APENAS o normalizador, pela mediana dos ultimos k. O modelo continua acompanhando
a deriva; so a regua de escala para de tremer.

RESULTADO: DIAGNOSTICO CONFIRMADO, CORRECAO REFUTADA.

O normalizador treme muito -- canal t: p99 de 1,355 a 3,318 (2,45x), variacao
mensal mediana de 11,2% e maxima de +91,4%; canal p: 2,38x, mediana 19,0%.

Mas suavizar piora:

    k_norm   banda  inicio  det   FP/mes   h/mes
      1       5/8    6/8    8/8    0,344     6,6   <- atual
      2       3/8    4/8    7/8    0,603    11,7
      3       3/8    4/8    6/8    0,517     8,6
      5       3/8    4/8    6/8    0,431     8,9

O MOTIVO e conceitual: o p99 nao e ruido EM CIMA do modelo -- ele E a escala
daquele PCA. Cada reajuste move a base e o erro nela; o p99 daquele baseline e a
regua certa para aquele PCA. Usar o p99 de um ajuste antigo com um PCA novo
descasa escala e modelo, e o score inteiro sai deslocado. **O par (PCA, p99) e
atomico** -- nao da para adaptar um e estabilizar o outro.

Parte 1 mede a instabilidade. Parte 2 testa a correcao (e mostra por que falha).

Uso:  PYTHONPATH=. python drift_normalizador.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from cabiunas_pdm import config as C, detector as DET
from ablacao import ScorerMax
from drift_baseline import roda, met, DF, STABLE, IX
from pos_processamento import idx

FIT = DET.FIT_POINTS


def ajusta_todos(fit_points: int = FIT):
    """Um ajuste por mes, guardando os scorers e os normalizadores."""
    meses = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    out = []
    for m0 in meses:
        fit = DF.loc[STABLE & (IX < m0), C.SENSOR_TAGS].dropna().tail(fit_points)
        if len(fit) < fit_points // 4:
            out.append(None); continue
        b = DET._spread_mancal(fit)
        out.append(dict(
            m0=m0,
            st=ScorerMax().fit(fit[C.TEMPERATURE_TAGS]),
            sp=ScorerMax().fit(fit[C.PRESSURE_TAGS]),
            med=float(b.median()),
            mad=float((b - b.median()).abs().median() * 1.4826),
        ))
    return meses, out


def monta(meses, aj, k_norm: int):
    """Pontua cada mes com o SEU PCA. k_norm > 1 suaviza so o recon_p99."""
    n = len(IX)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)
    hist_t, hist_p = [], []
    for i, m0 in enumerate(meses):
        a = aj[i]
        if a is not None:
            hist_t.append(a["st"].recon_p99); hist_p.append(a["sp"].recon_p99)
        viv = [x for x in aj[:i + 1] if x is not None]
        if not viv: continue
        cur = viv[-1]
        m1 = meses[i + 1] if i + 1 < len(meses) else IX[-1] + pd.Timedelta("2min")
        s = (IX >= m0) & (IX < m1)
        if not s.any(): continue
        w = DF.loc[s]
        # score BRUTO (sem dividir pelo p99 do proprio bundle)
        rt = cur["st"]._raw_scores(w[C.TEMPERATURE_TAGS])[0]
        rp = cur["sp"]._raw_scores(w[C.PRESSURE_TAGS])[0]
        nt = float(np.median(hist_t[-k_norm:])) if hist_t else cur["st"].recon_p99
        npp = float(np.median(hist_p[-k_norm:])) if hist_p else cur["sp"].recon_p99
        t[s] = rt / nt
        p[s] = rp / npp
        med_sp[s] = cur["med"]; mad_sp[s] = cur["mad"]
    return t, p, med_sp, mad_sp


if __name__ == "__main__":
    meses, aj = ajusta_todos()
    viv = [a for a in aj if a is not None]
    pt = np.array([a["st"].recon_p99 for a in viv])
    pp = np.array([a["sp"].recon_p99 for a in viv])

    print("=" * 104)
    print("1) QUANTO O NORMALIZADOR TREME, mes a mes")
    print("=" * 104)
    print(f"{'mes':>9}{'recon_p99 t':>14}{'var vs anterior':>18}"
          f"{'recon_p99 p':>14}{'var vs anterior':>18}")
    print("-" * 104)
    for i, a in enumerate(viv):
        dt = f"{100*(pt[i]/pt[i-1]-1):+7.1f}%" if i else "      --"
        dp = f"{100*(pp[i]/pp[i-1]-1):+7.1f}%" if i else "      --"
        print(f"{a['m0']:%Y-%m}{pt[i]:>14.4f}{dt:>18}{pp[i]:>14.4f}{dp:>18}")
    print("-" * 104)
    for nome, v in (("t", pt), ("p", pp)):
        rel = np.abs(np.diff(v) / v[:-1])
        print(f"  canal {nome}: p99 vai de {v.min():.3f} a {v.max():.3f} "
              f"({v.max()/v.min():.2f}x)  |  variacao mensal: mediana "
              f"{100*np.median(rel):.1f}%, maxima {100*rel.max():.1f}%")

    print()
    print("=" * 104)
    print("2) SUAVIZAR SO O NORMALIZADOR (PCA continua mensal)")
    print("=" * 104)
    for k in (1, 2, 3, 5):
        t, p, ms, ds = monta(meses, aj, k)
        rot = (f"k_norm = {k}" + ("  <- equivale ao atual" if k == 1 else
                                  "  (mediana dos ultimos " + str(k) + ")"))
        met(roda(t, p, ms, ds), rot)
