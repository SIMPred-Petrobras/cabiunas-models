"""A cadencia de retreino importa? Fechando a divergencia com o relatorio EXP28/29.

O Diego mediu retreino mensal (walk-forward) no detector DELE e viu +53% de FP,
por isso nao usa. NOS usamos walk-forward mensal para t, p e sp. Ou a nossa
cadencia esta custando caro sem a gente saber, ou o efeito e especifico da familia
de modelo dele (OCSVM/iforest) e nao transfere. So medindo.

CADENCIAS: mensal (atual), trimestral, semestral, anual, congelado (ajusta uma vez
antes de T0 e nunca mais). Em todas, `vb` fica intocado -- ele usa a referencia
rolante de 400h com passo de 6h, que e um mecanismo diferente do ajuste mensal.

CONTROLE DE CORRETUDE (obrigatorio antes de ler qualquer numero): a cadencia
mensal reimplementada aqui tem que reproduzir o cache piso_fisico_cache.npz e o
ponto de operacao publicado (8/8, 0,517 FP/mes, 7,15 h/mes pela regra C).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import avalia as AV
from publica_clearml import (GRID, BLACKOUT, SUSTAIN, SIN, HL, BASE, KAPPA,
                             H_CUSUM, REFRAT_H, DUR_MIN, T0)
from blackout_curto import cusum
from autocalibra import TEMP, PRESS, MANCAL, ALVO_SP, N_FIT
from sklearn.preprocessing import RobustScaler as _RS
from sklearn.decomposition import PCA as _PCA

PHI = 0.10


class ScorerMax:
    """Reimplementacao COM o recon_p99 -- a classe original (cabiunas_pdm.scoring,
    pacote perdido) calculava `self.recon_p99 = p99 do escore no proprio baseline`
    dentro do fit, e o `score` publico dividia por ele. O ScorerMax do autocalibra.py
    omite essa divisao; sem ela o escore sai em outra escala e o limiar BASE=2.0
    deixa de significar 'duas vezes o p99 do baseline'."""

    def fit(self, base):
        self.cols = list(base.columns)
        X = base.dropna()
        self.scaler = _RS().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = _PCA(n_components=0.95).fit(Xs)
        e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
        p = np.nanpercentile(e, 99, axis=0)
        self.sens_p99_ = np.maximum(p, PHI * np.nanmedian(p))
        r = np.max(e / self.sens_p99_, axis=1)
        self.recon_p99 = float(np.nanpercentile(r, 99))
        return self

    def score(self, df):
        X = df[self.cols]
        m = X.notna().all(axis=1).to_numpy()
        out = np.full(len(X), np.nan)
        if m.any():
            Xs = self.scaler.transform(X[m])
            e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
            out[m] = np.max(e / self.sens_p99_, axis=1) / self.recon_p99
        return out
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

KB, KV = 1.7, 2.2

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
sel = idx >= T0
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))

n_bl = int(pd.Timedelta(BLACKOUT) / pd.Timedelta(GRID))
blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
mask = (estavel & ~blk) & sel
reset = ((~mask) | part).to_numpy()
paradas = paradas_reais_2h()

# vb vem da referencia rolante -- nao e afetado pela cadencia mensal
z = np.load("piso_fisico_cache.npz")
with np.errstate(invalid="ignore", divide="ignore"):
    Zv = np.abs((z["Xh"] - z["MED"]) / z["S"])
vb = np.full(len(idx), np.nan)
vb[z["hot"]] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
vb[~np.isfinite(vb)] = np.nan

spread = (g[ALVO_SP] - g[MANCAL].median(axis=1)).abs().to_numpy().astype("float64")
meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")


def constroi(cadencia_meses):
    """Refaz t, p e a normalizacao de sp com a cadencia dada.
    cadencia_meses=None -> congelado: um unico ajuste com os dados anteriores a T0."""
    n = len(idx)
    t_ = np.full(n, np.nan); p_ = np.full(n, np.nan)
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)

    if cadencia_meses is None:
        fit = g.loc[estavel & (idx < T0), TEMP + PRESS].dropna().tail(N_FIT)
        st = ScorerMax().fit(fit[TEMP]); sp_ = ScorerMax().fit(fit[PRESS])
        b = (fit[ALVO_SP] - fit[MANCAL].median(axis=1)).abs()
        m_, d_ = float(b.median()), float((b - b.median()).abs().median() * 1.4826)
        t_[:] = st.score(g[TEMP]); p_[:] = sp_.score(g[PRESS])
        med_sp[:] = m_; mad_sp[:] = d_
        return t_, p_, med_sp, mad_sp, 1

    n_fit = 0
    ancora = None   # ultimo modelo ajustado, reaproveitado enquanto nao toca reajustar
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta(GRID)
        selm = (idx >= m0) & (idx < m1)
        if not selm.any():
            continue
        if ancora is None or (i % cadencia_meses == 0):
            fit = g.loc[estavel & (idx < m0), TEMP + PRESS].dropna().tail(N_FIT)
            if len(fit) < N_FIT // 4:
                continue
            st = ScorerMax().fit(fit[TEMP]); sp_ = ScorerMax().fit(fit[PRESS])
            b = (fit[ALVO_SP] - fit[MANCAL].median(axis=1)).abs()
            ancora = (st, sp_, float(b.median()),
                      float((b - b.median()).abs().median() * 1.4826))
            n_fit += 1
        st, sp_, m_, d_ = ancora
        t_[selm] = st.score(g.loc[selm, TEMP])
        p_[selm] = sp_.score(g.loc[selm, PRESS])
        med_sp[selm] = m_; mad_sp[selm] = d_
    return t_, p_, med_sp, mad_sp, n_fit


def pos(voto):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


def roda(t_, p_, med_sp, mad_sp):
    sp_v = np.abs((spread - med_sp) / mad_sp)
    cru = pd.DataFrame({"t": t_, "p": p_, "sp": sp_v, "vb": vb}, index=idx)
    EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}
    K_ = {"t": KB, "p": KB, "sp": KB, "vb": KV}
    ON = {}
    for c in SIN:
        thr = BASE[c] * K_[c]
        E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v)
    eps = AV.episodios(al)
    m = AV.avalia(al, alvo, mask)
    meses_op = m["horas_op"] / 730.0
    cl = classifica_regra_c(eps, paradas)
    n_tp = sum(1 for a, b, c, l in cl if c == "TP")
    n_fp = sum(1 for a, b, c, l in cl if c == "FP")
    n_ne = sum(1 for a, b, c, l in cl if c == "NEUTRO")
    h_fp = sum((b - a).total_seconds() / 3600 for a, b, c, l in cl if c == "FP")
    perd = sorted(set(t.strftime("%d/%m/%Y") for t in alvo) -
                  set(pd.Timestamp(d).strftime("%d/%m/%Y") for d in m["detectados"]))
    return dict(det=m["det"], TP=n_tp, FP=n_fp, NEU=n_ne, eps=len(eps),
                fp_mes=n_fp / meses_op, h_mes=h_fp / meses_op, lead=m["lead_med"],
                perdidos=perd)


if __name__ == "__main__":
    print("CONTROLE DE CORRETUDE: cadencia mensal reimplementada vs cache publicado")
    t1, p1, ms1, md1, nf1 = constroi(1)
    for nome, novo, velho in (("t", t1, z["t"]), ("p", p1, z["p"])):
        ok = np.isfinite(novo) & np.isfinite(velho)
        r = np.corrcoef(novo[ok], velho[ok])[0, 1]
        dif = np.nanmax(np.abs(novo[ok] - velho[ok]))
        print(f"  {nome}: n={ok.sum():,}  corr={r:.6f}  max|dif|={dif:.4g}")
    base = roda(t1, p1, ms1, md1)
    print(f"  detector -> {base['det']}/8, {base['fp_mes']:.3f} FP/mes, "
          f"{base['h_mes']:.2f} h/mes, lead {base['lead']:.1f}h "
          f"(esperado 8/8, 0,517, 7,15, 29,0)\n")

    print("=" * 104)
    print("EFEITO DA CADENCIA DE RETREINO (t, p, sp; vb inalterado)")
    print("=" * 104)
    print(f"{'cadencia':>14} {'ajustes':>8} {'det':>5} {'TP':>4} {'FP':>4} {'NEU':>4} "
          f"{'FP/mes':>8} {'h/mes':>8} {'lead':>7}  perdidos")
    for nome, cad in (("mensal (atual)", 1), ("trimestral", 3), ("semestral", 6),
                      ("anual", 12), ("congelado", None)):
        t_, p_, ms, md, nf = constroi(cad)
        r = roda(t_, p_, ms, md)
        d_fp = "" if nome.startswith("mensal") else f"  ({100*(r['fp_mes']/base['fp_mes']-1):+.0f}% FP)"
        print(f"{nome:>14} {nf:8d} {r['det']:5d} {r['TP']:4d} {r['FP']:4d} {r['NEU']:4d} "
              f"{r['fp_mes']:8.3f} {r['h_mes']:8.2f} {r['lead']:7.1f}  "
              f"{','.join(r['perdidos']) if r['perdidos'] else '--'}{d_fp}")
