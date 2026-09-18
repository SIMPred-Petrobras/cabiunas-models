#!/usr/bin/env python3
"""DRIFT: o tamanho do baseline e o ensemble temporal de bundles.

CONTEXTO. O drift aqui e real e ja tratado: `t` e `p` sao residuo de PCA contra
um baseline, que descola conforme campanha, carga e ambiente andam, e por isso o
retreino mensal e obrigatorio -- congelar leva 8/8 para 6/8 e 7,15 para 108,19
h/mes ([[cadencia-de-retreino-depende-do-sinal]]). Ja sabemos tambem que LIMPAR o
baseline piora dez vezes, porque o `recon_p99` precisa da cauda alta para calibrar
a escala ([[baseline-precisa-de-anormalidade]]).

Duas coisas nunca medidas, e as duas sao o miolo do problema de drift:

1. FIT_POINTS = 20.000 (~28 dias estaveis) esta fixo desde o inicio e nunca foi
   varrido. E o botao bias-variancia do drift: janela curta acompanha a deriva
   mas treme; janela longa e estavel mas fica para tras. Nao ha razao para supor
   que 20.000 seja o ponto certo -- ninguem procurou.

2. O retreino tem VARIANCIA PROPRIA. Dois treinos de configuracao identica
   diferem 20,7 pontos percentuais de recall ([[piso-de-ruido-retreino]]), e
   retreinamos doze vezes por ano. O ensemble temporal -- pontuar o mes com os k
   bundles mais recentes e tomar a mediana do score -- e a defesa classica: ataca
   a variancia do retreino sem abrir mao da adaptacao ao drift.

O ensemble tem um risco na direcao oposta, e por isso precisa ser medido e nao
presumido: bundle mais velho carrega mais drift. Se a deriva for rapida, a
mediana de tres puxa o score para o passado e perde deteccao.

Uso:  PYTHONPATH=. python drift_baseline.py
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import avalia as AV
from cabiunas_pdm import config as C, detector as DET
from ablacao import canonico, ScorerMax
from pos_processamento import mask, idx, alvo, op, sel, cru as cru_pub
from publica_clearml import (SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, K_LO, VOTO_LO,
                             VOTO_HI, REFRAT_V2, DUR_MIN, ESC_IDADE, ESC_ABS,
                             ESC_DUR, HL, TMIN_BANDA)
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

JAN48 = pd.Timedelta(hours=48)
KH = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
N_GAP = int(pd.Timedelta(hours=AV.GAP_EP_H) / pd.Timedelta("2min")) + 1
paradas = paradas_reais_2h()
part = op & ~op.shift(fill_value=False)
reset = ((~mask) | part).to_numpy()

DF = canonico()
STABLE = DF["stable"].astype(bool)
IX = DF.index


def walkforward(fit_points: int, k_ensemble: int = 1):
    """t, p e (med, mad) mes a mes.

    k_ensemble = 1 e o comportamento atual: o mes e pontuado pelo bundle ajustado
    com dado anterior a ele. k > 1 pontua com os k bundles mais recentes e toma a
    MEDIANA do score -- o voto de tres modelos treinados em janelas deslocadas.
    """
    meses = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    n = len(IX)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)
    hist = []          # (scorer_t, scorer_p, med, mad) dos meses ja ajustados
    n_fit = 0
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else IX[-1] + pd.Timedelta("2min")
        fit = DF.loc[STABLE & (IX < m0), C.SENSOR_TAGS].dropna().tail(fit_points)
        if len(fit) >= fit_points // 4:
            b = DET._spread_mancal(fit)
            hist.append((ScorerMax().fit(fit[C.TEMPERATURE_TAGS]),
                         ScorerMax().fit(fit[C.PRESSURE_TAGS]),
                         float(b.median()),
                         float((b - b.median()).abs().median() * 1.4826)))
            n_fit += 1
        if not hist:
            continue
        s = (IX >= m0) & (IX < m1)
        if not s.any():
            continue
        usar = hist[-k_ensemble:]
        w = DF.loc[s]
        st = np.vstack([sc.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
                        for sc, _, _, _ in usar])
        sp = np.vstack([sc.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
                        for _, sc, _, _ in usar])
        t[s] = np.nanmedian(st, axis=0)
        p[s] = np.nanmedian(sp, axis=0)
        med_sp[s] = float(np.median([m for _, _, m, _ in usar]))
        mad_sp[s] = float(np.median([d for _, _, _, d in usar]))
    return t, p, med_sp, mad_sp, n_fit


def roda(t, p, med_sp, mad_sp):
    z = np.load("piso_fisico_cache.npz")
    spv = np.abs((z["b_all"] - med_sp) / mad_sp)
    cru = pd.DataFrame({"t": t, "p": p, "sp": spv,
                        "vb": cru_pub["vb"].to_numpy()}, index=idx)
    EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}

    def canal(c, k):
        thr = BASE[c] * k; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        return (deg | cu) & mask

    A = {c: canal(c, K_LO[c]) for c in SIN}
    B = {c: canal(c, KH[c]) for c in SIN}
    vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= VOTO_LO, index=idx) & mask
    vB = (pd.Series(sum(B[c].astype(int) for c in SIN) >= VOTO_HI, index=idx)
          & mask & (B["sp"] | B["vb"]))
    voto = vA | vB
    F = pd.concat([EW[c].where(mask) / (BASE[c] * KH[c]) for c in SIN], axis=1).max(axis=1)
    f = F.fillna(0.0).to_numpy(); v = voto.to_numpy().copy()
    n_id = int(ESC_IDADE * 30); dentro, ini, ja = False, 0, False
    for i in range(len(v)):
        if not v[i]: dentro, ja = False, False; continue
        ac = f[i] > ESC_ABS
        if not dentro: dentro, ini, ja = True, i, ac; continue
        if ac and not ja and (i - ini) >= n_id:
            v[max(ini + 1, i - N_GAP):i] = False; ini = i
        ja = ac
    voto = pd.Series(v, index=idx)
    al = pd.Series(False, index=idx); bloq = ini_b = None; fortes = []
    for a, b in AV.episodios(voto):
        forte = float(F.loc[a:b].max()) > ESC_ABS
        velho = ini_b is not None and (a - ini_b).total_seconds() / 3600 >= ESC_IDADE
        if bloq is not None and a <= bloq and not (forte and velho): continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_V2); ini_b = a
        if forte: fortes.append((a, b))
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        d = (b - a).total_seconds() / 60 + 2
        if (any(x >= a and y <= b for x, y in fortes) and d >= ESC_DUR) or d >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


def met(fin, nome, extra=""):
    eps = AV.episodios(fin); det = AV.avalia(fin, alvo, mask)
    cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    hfp = sum((b - a).total_seconds() / 3600 for a, b, k, _ in cls if k == "FP")
    mes = det["horas_op"] / 730.0
    leads = [(t - max([a for a, _ in eps if t - JAN48 <= a <= t])).total_seconds() / 3600
             for t in alvo if any(t - JAN48 <= a <= t for a, _ in eps)]
    ban = sum(1 for t in alvo if any(t - JAN48 <= a <= t - pd.Timedelta(hours=TMIN_BANDA)
                                     for a, _ in eps))
    print(f"{nome:<32} banda {ban}/8  inicio {len(leads)}/8  det {det['det']}/8  "
          f"{nfp/mes:6.3f} FP/mes  {hfp/mes:5.1f} h/mes  "
          f"lead {np.mean(leads) if leads else float('nan'):5.1f}h  "
          f"eps {len(eps):3d}{extra}", flush=True)


if __name__ == "__main__":
    print("=" * 112)
    print("1) TAMANHO DO BASELINE -- nunca varrido. 20.000 pontos ~ 28 dias estaveis.")
    print("=" * 112)
    for fp in (8_000, 14_000, 20_000, 30_000, 45_000):
        t, p, ms, ds, nf = walkforward(fp)
        dias = fp * 2 / 60 / 24
        rot = f"{fp:>6} pts (~{dias:.0f} d)" + ("  <- atual" if fp == 20_000 else "")
        met(roda(t, p, ms, ds), rot)

    print()
    print("=" * 112)
    print("2) ENSEMBLE TEMPORAL -- mediana do score dos k bundles mais recentes.")
    print("   Ataca a variancia do retreino (20,7pp entre treinos identicos).")
    print("=" * 112)
    for k in (1, 2, 3):
        t, p, ms, ds, nf = walkforward(20_000, k_ensemble=k)
        rot = f"k = {k} bundle{'s' if k > 1 else ' '}" + ("       <- atual" if k == 1 else "")
        met(roda(t, p, ms, ds), rot)
