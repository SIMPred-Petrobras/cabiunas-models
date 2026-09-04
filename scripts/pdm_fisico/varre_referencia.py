#!/usr/bin/env python3
"""Varredura do COMPRIMENTO das duas referencias, de ponta a ponta.

O 400h da referencia rolante foi escolhido por AUC numa rodada anterior --
metrica diferente da que interessa agora (FP + deteccao no detector de 4
sinais). E o FIT_POINTS=20.000 (~667h) dos escores de PCA nunca foi tocado.
Aqui os dois sao varridos, um de cada vez, medindo o que importa.

Tensao esperada (declarada antes do resultado):
  - encurtar = passa-alta mais agressivo, remove deriva comum, exige menos
    historico; mas deixa de cobrir o envelope operacional (o teste de campanha
    mostrou FP dobrando com 12h) e absorve degradacao lenta na propria
    referencia.
  - alongar = referencia estavel e representativa; mas demora a acompanhar
    mudanca legitima de regime e exige historico longo.

Avaliacao assimetrica de sempre: FP tem n=84 (poder real, IC de Poisson);
deteccao tem n=9 (reportada, sem poder para decidir).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
import rolante as RO
from ablacao import canonico, mascara_pontuacao, ScorerMax
from referencia_campanha import alerta_de, K_BASE, K_VIB, ic_poisson

HORAS_BASE_GRID = [50.0, 100.0, 200.0, 400.0, 800.0]     # ref. rolante da vibracao
FIT_HORAS_GRID = [167.0, 333.0, 667.0, 1333.0]           # ref. mensal (PCA/spread)
PAS = pd.Timedelta("2min")


def roda_param(df, falhas, horas_base, fit_pontos):
    """Como roda('max+vib_rol'), mas com os dois comprimentos parametrizados."""
    stable = df["stable"].astype(bool)
    idx = df.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    out = pd.DataFrame(index=idx, columns=["t", "p", "sp", "vb"], dtype="float64")

    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        fit = df.loc[stable & (idx < m0), C.SENSOR_TAGS].dropna().tail(fit_pontos)
        if len(fit) < fit_pontos // 4:
            continue
        sel = (idx >= m0) & (idx < m1)
        if not sel.any():
            continue
        w = df.loc[sel]
        st = ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
        sp = ScorerMax().fit(fit[C.PRESSURE_TAGS])
        out.loc[sel, "t"] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        out.loc[sel, "p"] = sp.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        med, mad = float(b.median()), float((b - b.median()).abs().median() * 1.4826)
        out.loc[sel, "sp"] = ((DET._spread_mancal(w) - med) / mad).abs().to_numpy()

    V = df[C.VIBRATION_TAGS].where(stable)
    Z = RO.z_rolante(V, stable, falhas, horas_base=horas_base, guarda_h=24, phi=0.0)
    out["vb"] = Z.max(axis=1)
    return out


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    meses_op = mask.sum() * 2 / 60.0 / 730.0
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    def mede(alerta):
        eps = A.episodios(alerta)
        fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
        det = [t.strftime("%Y-%m-%d") for t in falhas
               if alerta[(alerta.index >= t - pd.Timedelta(hours=48)) & (alerta.index < t)].any()]
        h = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
        return len(fp), det, h

    linhas = []
    print("=== A. comprimento da referencia ROLANTE (vibracao), PCA fixo em 667h ===")
    print(f"{'horas_base':>11} {'FP':>4} {'FP/mes':>7} {'IC95%':>16} {'h/mes':>7} {'det':>5}  perdidos")
    for hb in HORAS_BASE_GRID:
        out = roda_param(df, falhas, hb, 20_000)
        al = alerta_de(out, mask)
        fp, det, h = mede(al)
        lo, hi = ic_poisson(fp, meses_op)
        perd = [t.strftime("%Y-%m-%d") for t in falhas if t.strftime("%Y-%m-%d") not in det]
        base = "  <- atual" if hb == 400.0 else ""
        print(f"{hb:11.0f} {fp:4d} {fp/meses_op:7.2f} [{lo:5.2f}, {hi:5.2f}] "
              f"{h/meses_op:7.1f} {len(det):3d}/9  {','.join(perd)}{base}", flush=True)
        linhas.append(dict(eixo="rolante_vib", valor_h=hb, fp=fp, fp_mes=fp/meses_op,
                            ic_lo=lo, ic_hi=hi, h_fp_mes=h/meses_op, det=len(det),
                            perdidos=",".join(perd)))

    print("\n=== B. comprimento da referencia MENSAL (PCA/spread), rolante fixa em 400h ===")
    print(f"{'fit_horas':>10} {'amostras':>9} {'FP':>4} {'FP/mes':>7} {'IC95%':>16} {'h/mes':>7} {'det':>5}  perdidos")
    for fh in FIT_HORAS_GRID:
        npts = int(fh * 30)
        out = roda_param(df, falhas, 400.0, npts)
        al = alerta_de(out, mask)
        fp, det, h = mede(al)
        lo, hi = ic_poisson(fp, meses_op)
        perd = [t.strftime("%Y-%m-%d") for t in falhas if t.strftime("%Y-%m-%d") not in det]
        base = "  <- atual" if npts == 20_010 or abs(fh - 667) < 1 else ""
        print(f"{fh:10.0f} {npts:9d} {fp:4d} {fp/meses_op:7.2f} [{lo:5.2f}, {hi:5.2f}] "
              f"{h/meses_op:7.1f} {len(det):3d}/9  {','.join(perd)}{base}", flush=True)
        linhas.append(dict(eixo="mensal_pca", valor_h=fh, fp=fp, fp_mes=fp/meses_op,
                            ic_lo=lo, ic_hi=hi, h_fp_mes=h/meses_op, det=len(det),
                            perdidos=",".join(perd)))

    R = pd.DataFrame(linhas)
    R.to_csv("varre_referencia.csv", index=False)
    print("\ngravado: varre_referencia.csv")


main()
