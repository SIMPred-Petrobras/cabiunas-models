#!/usr/bin/env python3
"""A MEDIDA DIRETA: o que producao vai ver quando a composicao do baseline mudar.

`drift_vizinhanca_fit.py` varreu o TAMANHO do baseline (18.000 a 22.000) e achou
6,6 a 68,1 h/mes. Mas isso e uma PROXY: em producao o tamanho fica fixo em 20.000
e o que muda e QUAIS 20.000 -- porque o retreino cai numa data e a maquina operou
de um jeito ou de outro nas semanas anteriores.

Aqui a variacao e a real: FIT_POINTS fixo em 20.000, e o DIA DO RETREINO deslocado
dentro do mes (1, 8, 15, 22). Cada deslocamento produz um baseline do mesmo
tamanho e composicao diferente -- exatamente a fonte de variabilidade que a
operacao vai encontrar, e que ninguem controla.

Se a dispersao aqui for parecida com a do tamanho, o numero publicado e mesmo o
melhor caso de uma distribuicao larga. Se for muito menor, o detector e mais
estavel do que a varredura de tamanho sugeriu, e a proxy exagerou.

Uso:  PYTHONPATH=. python drift_composicao.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from cabiunas_pdm import config as C, detector as DET
from ablacao import ScorerMax
from drift_baseline import roda, met, DF, STABLE, IX
from pos_processamento import mask, alvo
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

FIT = DET.FIT_POINTS
_par = paradas_reais_2h(); _J = pd.Timedelta(hours=48)


def walkforward_dia(dia: int):
    """Retreino no dia `dia` de cada mes, com FIT_POINTS fixo."""
    base = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    cortes = [m + pd.Timedelta(days=dia - 1) for m in base]
    cortes = [c for c in cortes if IX[0] < c < IX[-1]]
    n = len(IX)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)
    for i, c0 in enumerate(cortes):
        c1 = cortes[i + 1] if i + 1 < len(cortes) else IX[-1] + pd.Timedelta("2min")
        fit = DF.loc[STABLE & (IX < c0), C.SENSOR_TAGS].dropna().tail(FIT)
        if len(fit) < FIT // 4:
            continue
        s = (IX >= c0) & (IX < c1)
        if not s.any():
            continue
        w = DF.loc[s]
        t[s] = ScorerMax().fit(fit[C.TEMPERATURE_TAGS]).score(
            w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = ScorerMax().fit(fit[C.PRESSURE_TAGS]).score(
            w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        med_sp[s] = float(b.median())
        mad_sp[s] = float((b - b.median()).abs().median() * 1.4826)
    return t, p, med_sp, mad_sp


def colhe(fin):
    eps = AV.episodios(fin); d = AV.avalia(fin, alvo, mask)
    cls = classifica_regra_c(eps, _par)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    hfp = sum((b - a).total_seconds() / 3600 for a, b, k, _ in cls if k == "FP")
    mes = d["horas_op"] / 730.0
    ban = sum(1 for t in alvo if any(t - _J <= a <= t - pd.Timedelta(hours=4)
                                     for a, _ in eps))
    return ban, d["det"], nfp / mes, hfp / mes


if __name__ == "__main__":
    print("COMPOSICAO DO BASELINE: mesmo tamanho (20.000), dia do retreino deslocado")
    print("E a variabilidade que producao encontra, e que ninguem controla.\n")
    R = []
    for dia in (1, 8, 15, 22):
        t, p, ms, ds = walkforward_dia(dia)
        fin = roda(t, p, ms, ds)
        met(fin, f"retreino no dia {dia:>2}" + ("  <- o publicado" if dia == 1 else ""))
        R.append(colhe(fin))
    B = np.array([r[0] for r in R]); D = np.array([r[1] for r in R])
    F = np.array([r[2] for r in R]); H = np.array([r[3] for r in R])
    print("\n" + "=" * 78)
    print(f"  banda   : {B.min()}/8 a {B.max()}/8   (publicado 5/8)")
    print(f"  deteccao: {D.min()}/8 a {D.max()}/8   (publicado 8/8)")
    print(f"  FP/mes  : {F.min():.3f} a {F.max():.3f}   (publicado 0,344)")
    print(f"  h/mes   : {H.min():.1f} a {H.max():.1f}   (publicado 6,6)")
    print(f"\n  para comparar, a varredura de TAMANHO deu: det 6/8 a 8/8, "
          f"h/mes 6,6 a 68,1")
