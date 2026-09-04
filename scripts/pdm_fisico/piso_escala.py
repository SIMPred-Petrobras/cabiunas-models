#!/usr/bin/env python3
"""A deriva vem do MAD da referencia colapsar quando as campanhas ficam longas.

Cadeia de evidencia ate aqui:
  - o custo do detector deriva com parametros fixos (duty 1,66% -> 8,09%), e o regime de
    operacao nao explica pela contagem de partidas (deriva.py);
  - o unico sinal com tendencia e o spread do mancal: mediana do z de 1,08 -> 4,25
    (deriva_origem.py);
  - mas o spread FISICO nao cresceu: 12,2 -> 14,2 -> 12,4 -> 14,1 -> 9,7 degC, e o pior
    semestre para o detector e o de MENOR spread fisico. Os quatro termopares estao
    estaveis (TI_0305 em ~70-73 degC o tempo todo) (spread_fisico.py).

Logo o numerador do z esta estavel e o denominador encolheu. Hipotese do mecanismo:
o MAD e calculado sobre uma janela de N HORAS de operacao, e a diversidade de condicao
dentro dessa janela depende do tamanho das CAMPANHAS, nao das horas. Em 2026-H1 houve 8
partidas e campanha media de 269 h -- uma janela de 400 h cobre ~1,5 campanha, quase
homogenea, MAD minusculo, e tudo fora dela vira anomalia. Em 2025-H1, com campanha media
de 65 h, as mesmas 400 h cobriam ~6 campanhas com condicoes variadas e MAD maior.

E o mesmo mecanismo de escala que ja mediramos em porque_penhasco.py na direcao oposta
(referencia longa demais infla a escala e cega o detector) -- aqui a referencia fica
curta demais em DIVERSIDADE, mesmo tendo as horas certas.

A correcao ja existe no codigo e nunca foi ligada: rolante.z_rolante tem o parametro
`phi`, que aplica um PISO na escala local (s = max(s_local, phi * s_global)). Todas as
nossas rodadas usam phi=0,0. O mesmo piso e aplicado aqui ao normalizador do spread do
mancal, que em ablacao.roda usa o MAD da janela mensal de fit.

Testa-se: phi de 0 a 0,6, medindo deteccao, custo, e -- o que motivou tudo -- a
TENDENCIA do duty por semestre. Um piso que funcione deve achatar a tendencia sem
derrubar deteccao. Se derrubar, a deriva e o preco da sensibilidade, nao um defeito.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A, rolante as RO
from ablacao import canonico, mascara_pontuacao, ScorerMax, CORTE
from ablacao4 import alerta_2k
from portoes import K_BASE, K_VIB
from auto_reset import trunca

PAS = pd.Timedelta("2min")
PHIS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6]


def p_exato(y):
    n = len(y); r0 = stats.spearmanr(np.arange(n), y).statistic
    t = [abs(stats.spearmanr(np.arange(n), [y[i] for i in pm]).statistic)
         for pm in itertools.permutations(range(n))]
    return float(np.mean(np.array(t) >= abs(r0) - 1e-12)), r0


def roda_phi(df, falhas, phi):
    """ablacao.roda com piso de escala em phi, aplicado ao spread e a vibracao."""
    stable = df["stable"].astype(bool)
    idx = df.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    V = df[C.VIBRATION_TAGS].where(stable)
    vib = RO.z_rolante(V, stable, falhas, horas_base=400, guarda_h=24, phi=phi).max(axis=1)

    # escala global do spread, para o piso (mediana das MADs mensais ja vistas)
    b_glob = DET._spread_mancal(df.loc[stable])
    mad_glob = float((b_glob - b_glob.median()).abs().median() * 1.4826)

    out = pd.DataFrame(index=idx, columns=["t", "p", "sp", "vb"], dtype="float64")
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        fit = df.loc[stable & (idx < m0), C.SENSOR_TAGS].dropna().tail(DET.FIT_POINTS)
        if len(fit) < DET.FIT_POINTS // 4:
            continue
        sel = (idx >= m0) & (idx < m1)
        if not sel.any():
            continue
        w = df.loc[sel]
        st = ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
        sp_ = ScorerMax().fit(fit[C.PRESSURE_TAGS])
        out.loc[sel, "t"] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        out.loc[sel, "p"] = sp_.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        med = float(b.median())
        mad = float((b - med).abs().median() * 1.4826)
        mad = max(mad, phi * mad_glob)                      # <-- o piso
        out.loc[sel, "sp"] = ((DET._spread_mancal(w) - med) / mad).abs().to_numpy()
    out["vb"] = vib
    return out


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    sems = [x.index[0] for _, x in pd.Series(idx, index=idx).groupby(pd.Grouper(freq="2QS"))
            if len(x) and (mask & (idx >= x.index[0]) & (idx <= x.index[-1])).sum() * 2 / 60 >= 300]
    jan = [((idx >= a) & (idx <= (sems[i+1] if i+1 < len(sems) else idx[-1] + PAS)))
           for i, a in enumerate(sems)]
    jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    def duty_sem(al):
        d = []
        for sel in jan:
            eps = A.episodios(al & sel)
            fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jw)]
            h = sum((b - a).total_seconds() / 3600 + 2/60 for a, b in fp)
            d.append(100 * h / max((mask & sel).sum() * 2 / 60, 1))
        return d

    print(f"{'phi':>5} {'det':>4} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} {'p':>8} | "
          f"{'duty por semestre (%)':>30}  tendencia")
    linhas = []
    for phi in PHIS:
        out = roda_phi(df, falhas, phi)
        al = alerta_2k(out, mask, K_BASE, K_VIB)
        x = A.avalia(al, falhas, mask); x.update(A.permuta(al, mask, x["det"], len(falhas)))
        d = duty_sem(al); pr, rr = p_exato(d)
        print(f"{phi:5.2f} {x['det']:>2}/9 {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {x['lead_med']:6.1f} {x['p']:8.4f} | "
              + " ".join(f"{v:5.2f}" for v in d) + f"  rho={rr:+.2f} p={pr:.3f}")
        linhas.append(dict(phi=phi, det=x["det"], eps=x["episodios"], fp_mes=x["fp_mes"],
                           h_mes=x["h_fp_mes"], lead=x["lead_med"], p=x["p"], rho=rr, p_rho=pr,
                           duty=",".join(f"{v:.2f}" for v in d),
                           quais=",".join(x["detectados"])))
        alt = trunca(al, 12)
        xt = A.avalia(alt, falhas, mask); dt = duty_sem(alt); pt, rt = p_exato(dt)
        print(f"{'  +teto':>5} {xt['det']:>2}/9 {xt['episodios']:5d} {xt['fp_mes']:7.2f} "
              f"{xt['h_fp_mes']:7.1f} {xt['lead_med']:6.1f} {'':>8} | "
              + " ".join(f"{v:5.2f}" for v in dt) + f"  rho={rt:+.2f} p={pt:.3f}")
        linhas.append(dict(phi=phi, det=xt["det"], eps=xt["episodios"], fp_mes=xt["fp_mes"],
                           h_mes=xt["h_fp_mes"], lead=xt["lead_med"], p=np.nan, rho=rt, p_rho=pt,
                           duty=",".join(f"{v:.2f}" for v in dt), quais="teto12h"))
    pd.DataFrame(linhas).to_csv("piso_escala.csv", index=False)

    print("\n=== campanha media x duty, por semestre (a relacao que motivou o piso) ===")
    op = df["in_operation"].astype(bool)
    part = op & ~op.shift(fill_value=False)
    out0 = roda_phi(df, falhas, 0.0)
    d0 = duty_sem(alerta_2k(out0, mask, K_BASE, K_VIB))
    camp = []
    for i, a in enumerate(sems):
        n = max(int((part & jan[i]).sum()), 1)
        camp.append((op & jan[i]).sum() * 2 / 60 / n)
    r = stats.spearmanr(camp, d0)
    print(f"  campanha media (h): " + " ".join(f"{c:7.1f}" for c in camp))
    print(f"  duty com phi=0 (%): " + " ".join(f"{v:7.2f}" for v in d0))
    print(f"  Spearman campanha x duty: rho={r.statistic:+.2f}  p={r.pvalue:.3f}  (n=5)")


main()
