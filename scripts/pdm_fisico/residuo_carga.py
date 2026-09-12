#!/usr/bin/env python3
"""RESIDUO CONDICIONADO A CARGA -- muda a informacao, nao o parametro.

DIAGNOSTICO QUE MOTIVA (diagnostico_carga.py, 10/09/2026). Os nossos sinais nao
tem condicionamento a ponto de operacao: so existe um portao binario
`T5_AVG_A > 300`, e a PCA e ajustada sobre sensores CRUS -- aprende UM modelo de
normalidade para todas as cargas. Medido por quintil de |dcarga|:

    quintil        t        p       sp       vb
    muito baixa  2,30     1,58     0,79     1,11
    muito alta   3,40    22,90     0,76     1,25

**O canal de pressao infla 14x durante transicao de carga.** Isso significa que o
INSTANTE do nosso alarme e determinado em parte por quando a carga muda, nao por
quando a degradacao atinge um nivel -- que e exatamente o defeito medido em
[[banda-de-acionabilidade]]: leads de 1,3 a 194,9 h contra 3,8-43,2 h do Diego.

NAO adianta filtrar transicao de carga: TP, NEUTRO e FP ficam todos ~15% do
episodio em transicao (3,5x a taxa base), indistinguiveis. Um portao mataria
deteccao junto.

A SAIDA: condicionar. Modelar o valor esperado de cada sensor DADO o ponto de
operacao e rodar a PCA sobre o residuo. Mudanca de carga deixa de gerar residuo;
degradacao em carga baixa deixa de ficar mascarada.

DESENHO
  covariaveis: L = T5_AVG_A centrado/escalado, L^2, dL (1 h), |dL|
  ajuste      : minimos quadrados por familia, na MESMA janela walk-forward da PCA
                (ultimos FIT_POINTS estaveis antes do mes) -- causal, sem vazamento
  residuo     : Y - X@beta, depois RobustScaler -> PCA -> p99 por sensor, como antes
  T5_AVG_A sai do conjunto de temperatura (e a covariavel; o residuo dele seria ~0)
  sp e vb ficam INTACTOS -- o diagnostico mostra que nao sao contaminados
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from cabiunas_pdm import config as C, detector as DET
from ablacao import canonico, ScorerMax

PAS = pd.Timedelta("2min")
CACHE = "piso_fisico_carga_cache.npz"
CARGA = "T5_AVG_A"
TEMP_RES = [c for c in C.TEMPERATURE_TAGS if c != CARGA]   # 13, sem a covariavel


def covariaveis(df: pd.DataFrame) -> pd.DataFrame:
    """[1, L, L^2, dL, |dL|] com L centrado e escalado (L^2 sem escalar explode)."""
    L = df[CARGA].astype("float64")
    Lc = (L - 650.0) / 50.0                       # centro/escala fisicos, nao do fit
    dL = L.diff().rolling(30, min_periods=5).mean()   # 1 h de tendencia
    return pd.DataFrame({"um": 1.0, "L": Lc, "L2": Lc**2,
                         "dL": dL, "adL": dL.abs()}, index=df.index)


def ajusta_residuo(Yf: pd.DataFrame, Xf: pd.DataFrame,
                   Yw: pd.DataFrame, Xw: pd.DataFrame):
    """beta por minimos quadrados no fit; residuo aplicado no fit E na janela."""
    ok = Xf.notna().all(axis=1) & Yf.notna().all(axis=1)
    if int(ok.sum()) < 200:
        return None, None
    A = Xf[ok].to_numpy(); B = Yf[ok].to_numpy()
    beta, *_ = np.linalg.lstsq(A, B, rcond=None)
    rf = pd.DataFrame(B - A @ beta, index=Yf.index[ok], columns=Yf.columns)
    Aw = Xw.to_numpy()
    rw = pd.DataFrame(Yw.to_numpy() - Aw @ beta, index=Yw.index, columns=Yw.columns)
    return rf, rw


def main():
    if os.path.exists(CACHE) and "--refaz" not in sys.argv:
        print(f"{CACHE} ja existe (use --refaz)"); return
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    stable = df["stable"].astype(bool)
    idx = df.index
    cov = covariaveis(df)

    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    n = len(idx)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    r2t = []; r2p = []
    print("walk-forward mensal com residuo condicionado a carga ...", flush=True)
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        fit = df.loc[stable & (idx < m0), C.SENSOR_TAGS + [CARGA]].dropna().tail(DET.FIT_POINTS)
        if len(fit) < DET.FIT_POINTS // 4:
            continue
        sel = (idx >= m0) & (idx < m1)
        if not sel.any():
            continue
        w = df.loc[sel]
        Xf = cov.loc[fit.index]; Xw = cov.loc[w.index]
        for tags, alvo, acc in ((TEMP_RES, t, r2t), (C.PRESSURE_TAGS, p, r2p)):
            rf, rw = ajusta_residuo(fit[tags], Xf, w[tags], Xw)
            if rf is None:
                continue
            # quanto da variancia a carga explica (media entre sensores)
            v0 = fit[tags].loc[rf.index].var().to_numpy()
            v1 = rf.var().to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                acc.append(float(np.nanmean(1.0 - v1 / v0)))
            sc = ScorerMax().fit(rf)
            alvo[sel] = sc.score(rw)["pca_recon"].to_numpy()
        print(f"  {m0:%Y-%m}  R2_carga  t={r2t[-1] if r2t else np.nan:5.3f}  "
              f"p={r2p[-1] if r2p else np.nan:5.3f}", flush=True)

    z = np.load("piso_fisico_cache.npz")
    d = {k: z[k] for k in z.files}
    d["t"] = t; d["p"] = p                      # so t e p mudam
    np.savez_compressed(CACHE, **d)
    print(f"\n-> {CACHE}")
    print(f"   variancia explicada pela carga, media dos meses: "
          f"t = {np.nanmean(r2t):.3f}   p = {np.nanmean(r2p):.3f}")
    print(f"   cobertura: t {np.isfinite(t).sum()/n:.1%}  p {np.isfinite(p).sum()/n:.1%}"
          f"  (original: {np.isfinite(z['t']).sum()/n:.1%})")


if __name__ == "__main__":
    main()
