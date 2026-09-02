#!/usr/bin/env python3
"""Janela do PCA a CUSTO IGUALADO -- corrige um erro do teste anterior.

janela_pca.py varreu FIT_POINTS de 667 h a 4000 h com `k` FIXO em 1,7/2,2, e concluiu que
667 h e melhor (8/8 contra 4/8 em 3000 h). Isso esta viciado: o `k` foi calibrado COM a
janela de 667 h, e o escore `pca_recon` e normalizado pelo p99 do proprio baseline --
baseline mais largo tem p99 maior, entao o mesmo `k` significa coisa diferente em cada
janela. Comparar a `k` fixo mede sensibilidade, nao a janela. Armadilha de Pareto, a
setima ocorrencia neste projeto.

Aqui cada janela tem `k_base` varrido, e a comparacao e a EPISODIOS IGUALADOS -- alem do
melhor que cada janela alcanca em deteccao.

Importa porque o Francisco e a Lara usam 3000 h no MESMO alvo (parada), e uma discordancia
dessa magnitude precisa ser resolvida com numero, nao com preferencia.
"""
from __future__ import annotations
import sys, os
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET, scoring as S

T0 = pd.Timestamp("2025-01-01", tz="UTC")
PAS = pd.Timedelta("2min")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
SIN = ["t", "p", "sp", "vb"]
JAN = pd.Timedelta(hours=48)
FITS = [20_000, 45_000, 90_000]          # 667 h, 1500 h, 3000 h
KB = [0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.5]
KV = [1.6, 2.2, 3.0]
CACHE = "janela_pca2_sinais.npz"


class ScorerMax(S.MultivariateScorer):
    PHI = 0.10
    def fit(self, baseline):
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
        m = X.notna().all(axis=1).to_numpy()
        recon = np.full(len(X), np.nan); maha = np.full(len(X), np.nan)
        if m.any():
            Xs = self.scaler.transform(X[m])
            e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
            recon[m] = (np.mean(e, axis=1) if getattr(self, "sens_p99_", None) is None
                        else np.max(e / self.sens_p99_, axis=1))
            maha[m] = self.lw.mahalanobis(Xs)
        return recon, maha


def episodios(al, gap_h=2.0):
    v = al.fillna(False).to_numpy()
    d = np.diff(np.r_[0, v.astype(int), 0])
    br = [(al.index[a], al.index[b-1])
          for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    if not br: return []
    out = [list(br[0])]
    for s, e in br[1:]:
        if (s - out[-1][1]) <= pd.Timedelta(hours=gap_h): out[-1][1] = e
        else: out.append([s, e])
    return [tuple(x) for x in out]


def main():
    g = pd.read_parquet("grade2min.parquet"); idx = g.index
    df = g[C.SENSOR_TAGS].copy()
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    stable = op & (g["T5_AVG_A"] > 300)
    part = op & ~op.shift(fill_value=False)
    n_bl = int(pd.Timedelta(DET.BLACKOUT) / pd.Timedelta(C.GRID))
    sel = idx >= T0
    mask = (stable & ~part.rolling(n_bl, min_periods=1).max().astype(bool)) & sel
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    alvo = list(fal[fal >= T0]); jw = [(t - JAN, t) for t in alvo]
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    meses_op = mask.sum()*2/60/730
    z0 = np.load("piso_fisico_cache.npz")
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z0["Xh"] - z0["MED"]) / z0["S"])
    vbz = np.full(len(idx), np.nan)
    vbz[z0["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    reset = ((~mask) | part).to_numpy()
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")

    if os.path.exists(CACHE):
        zz = np.load(CACHE); SIG = {int(k): zz[k] for k in zz.files}
        print(f"cache {CACHE} reaproveitado", flush=True)
    else:
        SIG = {}
        for FP_ in FITS:
            t_, p_, sp_ = (np.full(len(idx), np.nan) for _ in range(3))
            for i, m0 in enumerate(meses):
                m1 = meses[i+1] if i+1 < len(meses) else idx[-1] + PAS
                fit = df.loc[stable & (idx < m0), C.SENSOR_TAGS].dropna().tail(FP_)
                if len(fit) < FP_ // 4: continue
                s_ = (idx >= m0) & (idx < m1)
                if not s_.any(): continue
                w = df.loc[s_]
                st = ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
                sp2 = ScorerMax().fit(fit[C.PRESSURE_TAGS])
                t_[s_] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
                p_[s_] = sp2.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
                b = DET._spread_mancal(fit); med = float(b.median())
                mad = float((b - med).abs().median() * 1.4826)
                sp_[s_] = ((DET._spread_mancal(w) - med) / mad).abs().to_numpy()
            SIG[FP_] = np.vstack([t_, p_, sp_])
            print(f"  FIT_POINTS={FP_:,} ({FP_*2/60:.0f} h) pronto", flush=True)
        np.savez_compressed(CACHE, **{str(k): v for k, v in SIG.items()})

    def mede(out, kb, kv):
        ON = {}
        for c in SIN:
            E = out[c].ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean().where(mask)
            thr = BASE[c]*(kv if c == "vb" else kb); n = DET.SUSTAIN
            deg = ((E > thr).astype(int).rolling(n, min_periods=n).sum() >= n)
            x = (E/thr).clip(upper=20); xx = (x - 0.75).fillna(0.0).to_numpy()
            Sa = np.empty(len(xx)); acc = 0.0
            for i in range(len(xx)):
                acc = acc*0.25 if reset[i] else max(0.0, acc + xx[i]); Sa[i] = acc
            ON[c] = (deg | pd.Series(Sa > 80, index=idx)) & mask
        voto = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
        al = pd.Series(False, index=idx); bloq = None
        for a, b in episodios(voto):
            if bloq is not None and a <= bloq: continue
            al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
        fin = pd.Series(False, index=idx)
        for a, b in episodios(al):
            if (b - a).total_seconds()/60 + 2 >= 120: fin.loc[a:b] = True
        eps = episodios(fin & sel)
        det = [f"{t:%Y-%m-%d}" for t in alvo if any(a <= t and b >= t - JAN for a, b in eps)]
        fpe = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
        h = sum((b-a).total_seconds()/3600 + 2/60 for a, b in fpe)
        leads = []
        for t in alvo:
            w = fin.loc[t-JAN:t-PAS]; o = w[w.fillna(False)]
            if len(o): leads.append((t - o.index[0]).total_seconds()/3600)
        return dict(det=len(det), eps=len(eps), fp=len(fpe)/meses_op, hm=h/meses_op,
                    lead=float(np.mean(leads)) if leads else np.nan,
                    perde=",".join(sorted(set(alvo_s) - set(det))))

    L = []
    for FP_ in FITS:
        a = SIG[FP_]
        out = pd.DataFrame({"t": a[0], "p": a[1], "sp": a[2], "vb": vbz}, index=idx)
        for kb in KB:
            for kv in KV:
                r = mede(out, kb, kv)
                L.append(dict(fit=FP_, horas=FP_*2/60, kb=kb, kv=kv, **r))
        print(f"  {FP_*2/60:.0f} h varrido", flush=True)
    T = pd.DataFrame(L); T.to_csv("janela_pca2.csv", index=False)

    ref = T[(T.fit == 20_000) & (T.kb == 1.7) & (T.kv == 2.2)].iloc[0]
    print("\n" + "=" * 96)
    print(f"MELHOR DE CADA JANELA (referencia: 667 h k=1,7/2,2 -> {ref.det}/8, "
          f"{ref.fp:.2f} FP/mes, {ref.hm:.1f} h/mes)")
    print("=" * 96)
    print(f"{'janela':>8} {'k_base':>7} {'k_vib':>6} {'det':>6} {'eps':>5} {'FP/mes':>7} "
          f"{'h/mes':>7} {'lead':>6}  perde")
    for FP_ in FITS:
        g_ = T[T.fit == FP_].sort_values(["det", "fp"], ascending=[False, True])
        for _, r in g_.head(2).iterrows():
            print(f"{r.horas:7.0f}h {r.kb:7.2f} {r.kv:6.1f} {int(r.det):4d}/8 {int(r.eps):5d} "
                  f"{r.fp:7.2f} {r.hm:7.1f} {r.lead:6.1f}  {r.perde if r.perde else '—'}")
    print("\n--- a EPISODIOS IGUALADOS (%.2f FP/mes) ---" % ref.fp)
    print(f"{'janela':>8} {'k_base':>7} {'k_vib':>6} {'det':>6} {'FP/mes':>7} {'h/mes':>7} {'lead':>6}")
    for FP_ in FITS:
        g_ = T[T.fit == FP_].assign(d=(T[T.fit == FP_].fp - ref.fp).abs())
        g_ = g_.sort_values(["det", "d"], ascending=[False, True])
        r = g_.iloc[0]
        print(f"{r.horas:7.0f}h {r.kb:7.2f} {r.kv:6.1f} {int(r.det):4d}/8 {r.fp:7.2f} "
              f"{r.hm:7.1f} {r.lead:6.1f}")


main()
