#!/usr/bin/env python3
"""Ablacao das duas recomendacoes, DENTRO do detector implantado.

O detector do Francisco (src/cabiunas_pdm/detector.py, modelo FP-first) usa
exatamente as duas construcoes que eu questionei:

  1. `MultivariateScorer._raw_scores` faz `recon = mean((Xs - proj)**2)` --
     media sobre os 14 sensores de temperatura e os 12 de pressao.
  2. a vibracao esta em quarentena declarada ("Vibracao fora da politica").

Cada recomendacao vira uma troca de uma peca, com todo o resto identico:
protocolo mensal walk-forward, fit nas ultimas 20.000 amostras estaveis
anteriores ao mes, blackout de 6 h pos-partida, sustentacao de 30 min,
CONFIRMADO = 2 ou mais sinais simultaneos.

Bracos:
  0  base            o detector como esta
  1  +max            media -> maximo por sensor com piso de escala
  2  +vib_mensal     vibracao entra como 4o sinal, referencia mensal (como os outros)
  3  +vib_rolante    idem, mas referencia rolante com banda de guarda
  4  +max+vib_rol    as duas recomendacoes juntas

Braco 2 existe para separar "por a vibracao de volta" de "mudar a referencia":
sem ele, um ganho do braco 3 nao teria causa atribuivel.
"""
from __future__ import annotations
import sys, os
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, scoring as S, detector as DET

import avalia as A, rolante as RO

CORTE = pd.Timestamp("2025-07-01", tz="UTC")
PAS = pd.Timedelta("2min")


# ---------------------------------------------------------------- dados
def canonico():
    g = pd.read_parquet("grade2min.parquet")
    faltam = [t for t in C.SENSOR_TAGS if t not in g.columns]
    if faltam:
        raise SystemExit(f"tags ausentes no cache: {faltam}")
    df = g[C.SENSOR_TAGS].copy()
    op = (g["RUNNING_A"] > 0.5).fillna(False)          # fallback declarado no config
    df["in_operation"] = op
    df["stable"] = op & (g["T5_AVG_A"] > 300)          # queimando de fato
    return df


# ------------------------------------------- recomendacao 1: maximo com piso
class ScorerMax(S.MultivariateScorer):
    """Troca a media sobre sensores pelo maximo do erro por sensor, normalizado
    pelo p99 do proprio sensor no baseline -- com piso.

    O piso e indispensavel: sem ele o sensor mais quieto do baseline recebe
    normalizador minusculo e passa a dominar o maximo o tempo inteiro. Aqui
    nenhum sensor pode ser considerado mais de 1/phi vezes mais sensivel que a
    mediana da familia.
    """
    PHI = 0.10

    def fit(self, baseline: pd.DataFrame) -> "ScorerMax":
        super().fit(baseline)
        X = baseline.dropna()[self.cols]
        Xs = self.scaler.transform(X)
        e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
        p = np.nanpercentile(e, 99, axis=0)
        self.sens_p99_ = np.maximum(p, self.PHI * np.nanmedian(p))
        r, _ = self._raw_scores(X)
        self.recon_p99 = float(np.nanpercentile(r, 99))
        return self

    def _raw_scores(self, df):
        X = df[self.cols]
        mask = X.notna().all(axis=1).to_numpy()
        recon = np.full(len(X), np.nan); maha = np.full(len(X), np.nan)
        if mask.any():
            Xs = self.scaler.transform(X[mask])
            e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
            if getattr(self, "sens_p99_", None) is None:
                recon[mask] = np.mean(e, axis=1)          # durante o fit inicial
            else:
                recon[mask] = np.max(e / self.sens_p99_, axis=1)
            maha[mask] = self.lw.mahalanobis(Xs)
        return recon, maha


# ------------------------------------ recomendacao 2: vibracao de volta
def sinal_vib_mensal(df, stable, m0, m1, fit_idx):
    """z robusto do maximo sobre as 10 sondas, baseline = mesmo fit dos outros."""
    V = df[C.VIBRATION_TAGS]
    base = V.loc[fit_idx]
    med = base.median(); mad = (base - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan)
    z = ((V - med) / mad).abs().max(axis=1)
    return z


def sinal_vib_rolante(df, stable, falhas):
    """z com referencia rolante e banda de guarda de 24 h (calculado uma vez)."""
    V = df[C.VIBRATION_TAGS].where(stable)
    Z = RO.z_rolante(V, stable, falhas, horas_base=400, guarda_h=24, phi=0.0)
    return Z.max(axis=1)


# ---------------------------------------------------------------- walk-forward
def roda(braco: str, df, falhas) -> pd.DataFrame:
    stable = df["stable"].astype(bool)
    idx = df.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    Scorer = ScorerMax if "max" in braco else S.MultivariateScorer
    vib_rol = sinal_vib_rolante(df, stable, falhas) if "vib_rol" in braco else None

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
        st = Scorer().fit(fit[C.TEMPERATURE_TAGS])
        sp = Scorer().fit(fit[C.PRESSURE_TAGS])
        out.loc[sel, "t"] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        out.loc[sel, "p"] = sp.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        med, mad = float(b.median()), float((b - b.median()).abs().median() * 1.4826)
        out.loc[sel, "sp"] = ((DET._spread_mancal(w) - med) / mad).abs().to_numpy()
        if "vib_mensal" in braco:
            out.loc[sel, "vb"] = sinal_vib_mensal(df, stable, m0, m1, fit.index)[sel].to_numpy()
    if vib_rol is not None:
        out["vb"] = vib_rol
    return out


def alerta_de(out, df, braco, thr_vib=3.0):
    """EWMA + limiar + sustentacao + voto, exatamente como no detector."""
    idx = out.index
    mask = mascara_pontuacao(df)
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    s1 = DET._sustained(ew("t", "1h"), DET.THR_FAM)
    s2 = DET._sustained(ew("p", "1h"), DET.THR_FAM)
    s3 = DET._sustained(ew("sp", "30min"), DET.THR_SPREAD)
    n = s1.astype(int) + s2.astype(int) + s3.astype(int)
    if "vib" in braco:
        n = n + DET._sustained(ew("vb", "30min"), thr_vib).astype(int)
    return (n >= 2) & mask


def mascara_pontuacao(df):
    op = df["in_operation"].astype(bool)
    st = df["stable"].astype(bool)
    starts = op & ~op.shift(fill_value=False)
    n_black = int(pd.Timedelta(DET.BLACKOUT) / pd.Timedelta(C.GRID))
    black = starts.rolling(n_black, min_periods=1).max().astype(bool)
    return st & ~black


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    tr = pd.Series(idx < CORTE, index=idx); te = ~tr
    ev_tr, ev_te = falhas[falhas < CORTE], falhas[falhas >= CORTE]
    mask = mascara_pontuacao(df)
    print(f"pontuavel: {mask.sum()*2/60:.0f} h  (treino {(mask&tr).sum()*2/60:.0f} h, "
          f"teste {(mask&te).sum()*2/60:.0f} h)\n", flush=True)

    linhas = []
    for braco in ["base", "max", "vib_mensal", "vib_rol", "max+vib_rol"]:
        out = roda(braco, df, falhas)
        al = alerta_de(out, df, braco)
        d = {"braco": braco}
        for tag, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te)]:
            am, qm = al[m], (mask & m)[m]
            x = A.avalia(am, ev, qm); x.update(A.permuta(am, qm, x["det"], len(ev)))
            d.update({f"{tag}_det": f"{x['det']}/{x['n_ev']}", f"{tag}_lead": x["lead_med"],
                      f"{tag}_fp": x["fp_mes"], f"{tag}_h": x["h_fp_mes"], f"{tag}_p": x["p"]})
            d[f"{tag}_quais"] = ",".join(x["detectados"])
        linhas.append(d)
        print(f"{braco:12s} treino {d['tr_det']} FP={d['tr_fp']:.2f}/mes {d['tr_h']:.1f}h/mes "
              f"p={d['tr_p']:.3f}  |  TESTE {d['te_det']} FP={d['te_fp']:.2f}/mes "
              f"{d['te_h']:.1f}h/mes p={d['te_p']:.3f}", flush=True)
        print(f"{'':12s}   treino: {d['tr_quais']}\n{'':12s}   teste : {d['te_quais']}", flush=True)
    pd.DataFrame(linhas).to_csv("ablacao.csv", index=False)


main()
