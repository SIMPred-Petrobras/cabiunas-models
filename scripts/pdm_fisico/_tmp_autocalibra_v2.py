"""Reteste da autocalibracao por percentil, com o bug de escala do ScorerMax
corrigido (recon_p99 nao estava sendo aplicado -- ver [[cadencia-de-retreino-
depende-do-sinal]] e [[autocalibracao-refutada]], que ficou em suspenso por isso).

Mesma estrutura do autocalibra.py original: PCA reajustado mensalmente em t e p,
residuo em amostra suavizado por EWMA, percentil desse residuo vira o limiar do
mes. sp e vb usam o mesmo mecanismo de sempre (nao passam pelo ScorerMax, nao
foram afetados pelo bug).

CONTROLE DE CORRETUDE: a reconstrucao de t/p tem que bater com o cache
piso_fisico_cache.npz (que ja usa o scorer original, correto) e reproduzir o
ponto publicado quando os limiares fixos (k*base) sao usados.

Avaliado pela REGRA C (a regua atual), nao pela regua antiga que o autocalibra.py
original usava -- para ficar comparavel com todo o resto desta sessao.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import avalia as AV
from publica_clearml import (GRID, BLACKOUT, SUSTAIN, SIN, HL, BASE, K, KAPPA,
                             H_CUSUM, CARGA, REFRAT_H, DUR_MIN, T0)
from blackout_curto import cusum
from autocalibra import TEMP, PRESS, MANCAL, ALVO_SP, N_FIT
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

QS = [80.0, 90.0, 95.0, 99.0, 99.5, 99.9]
PHI = 0.10


class ScorerMax:
    """Corrigido: aplica recon_p99 (p99 do escore no proprio baseline), que o
    autocalibra.py original omite -- essa e a raiz do bug de escala."""
    def fit(self, base):
        self.cols = list(base.columns)
        X = base.dropna()
        self.scaler = RobustScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = PCA(n_components=0.95).fit(Xs)
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

z = np.load("piso_fisico_cache.npz")
with np.errstate(invalid="ignore", divide="ignore"):
    Zv = np.abs((z["Xh"] - z["MED"]) / z["S"])
vb_all = np.full(len(idx), np.nan)
vb_all[z["hot"]] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
vb_all[~np.isfinite(vb_all)] = np.nan
hot = z["hot"]

spread = (g[ALVO_SP] - g[MANCAL].median(axis=1)).abs()
meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
hl_ew = lambda s, h: s.ewm(halflife=pd.Timedelta(h), times=s.index).mean()

t_oos = np.full(len(idx), np.nan); p_oos = np.full(len(idx), np.nan)
THR = {c: {q: pd.Series(np.nan, index=idx) for q in QS} for c in SIN}

print("Ajuste mensal com ScorerMax CORRIGIDO (recon_p99 aplicado)...")
for i, m0 in enumerate(meses):
    m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta(GRID)
    fit = g.loc[estavel & (idx < m0), TEMP + PRESS].dropna().tail(N_FIT)
    if len(fit) < N_FIT // 4:
        continue
    selm = (idx >= m0) & (idx < m1)
    if not selm.any():
        continue
    st = ScorerMax().fit(fit[TEMP]); sp_ = ScorerMax().fit(fit[PRESS])
    t_oos[selm] = st.score(g.loc[selm, TEMP])
    p_oos[selm] = sp_.score(g.loc[selm, PRESS])

    ins = {"t": pd.Series(st.score(fit[TEMP]), index=fit.index),
           "p": pd.Series(sp_.score(fit[PRESS]), index=fit.index)}
    b = spread.loc[fit.index]
    med, mad = float(b.median()), float((b - b.median()).abs().median() * 1.4826)
    ins["sp"] = ((b - med).abs() / mad)
    jf = idx[hot].isin(fit.index)
    ins["vb"] = pd.Series(vb_all[hot][jf], index=idx[hot][jf])
    for c in SIN:
        v = hl_ew(ins[c].sort_index(), HL[c]).to_numpy()
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        for q in QS:
            THR[c][q][selm] = float(np.percentile(v, q))

print("\n" + "=" * 100)
print("CONTROLE DE CORRETUDE")
print("=" * 100)
ok_t = np.corrcoef(pd.Series(t_oos).fillna(0), pd.Series(z["t"]).fillna(0))[0, 1]
ok_p = np.corrcoef(pd.Series(p_oos).fillna(0), pd.Series(z["p"]).fillna(0))[0, 1]
dif_t = np.nanmax(np.abs(t_oos - z["t"]))
dif_p = np.nanmax(np.abs(p_oos - z["p"]))
print(f"  correlacao com cache:  t={ok_t:.6f}  p={ok_p:.6f}")
print(f"  max|diferenca|:        t={dif_t:.3f}  p={dif_p:.3f}  (tem que ser pequeno, nao 1e5+)")

sp_all = np.abs((spread.to_numpy() - z["med_sp"]) / z["mad_sp"])
cru_fix = pd.DataFrame({"t": t_oos, "p": p_oos, "sp": sp_all, "vb": vb_all}, index=idx)
EW_fix = {c: cru_fix[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}


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


def avalia_regra_c(al):
    eps = AV.episodios(al)
    if not eps:
        return dict(det=0, TP=0, FP=0, NEU=0, fp_mes=np.nan, h_mes=np.nan, lead=np.nan, perd=list(alvo))
    m = AV.avalia(al, alvo, mask)
    meses_op = m["horas_op"] / 730.0
    cl = classifica_regra_c(eps, paradas)
    n_tp = sum(1 for a, b, c, l in cl if c == "TP")
    n_fp = sum(1 for a, b, c, l in cl if c == "FP")
    n_ne = sum(1 for a, b, c, l in cl if c == "NEUTRO")
    h_fp = sum((b - a).total_seconds() / 3600 for a, b, c, l in cl if c == "FP")
    perd = sorted(set(t.strftime("%d/%m/%Y") for t in alvo) -
                  set(pd.Timestamp(d).strftime("%d/%m/%Y") for d in m["detectados"]))
    return dict(det=m["det"], TP=n_tp, FP=n_fp, NEU=n_ne, fp_mes=n_fp / meses_op,
                h_mes=h_fp / meses_op, lead=m["lead_med"] if m["det"] else np.nan, perd=perd)


print("\nreferencia -- limiares fixos k*base (ponto publicado), escore recalculado com o scorer corrigido:")
ON_fix = {}
for c in SIN:
    thr = BASE[c] * K[c]
    E = EW_fix[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    ON_fix[c] = (deg | cu) & mask
ns_fix = sum(ON_fix[c].astype(int) for c in SIN)
v_fix = pd.Series(ns_fix >= 2, index=idx) & mask & (ON_fix["sp"] | ON_fix["vb"])
ref = avalia_regra_c(pos(v_fix))
print(f"  k*base (com scorer corrigido) -> {ref['det']}/8  {ref['fp_mes']:.3f} FP/mes  "
      f"{ref['h_mes']:.2f} h/mes  lead {ref['lead']:.1f}h  (esperado 8/8, 0,517, 7,15, 29,0)")

print("\n" + "=" * 100)
print("LIMIAR AUTOCALIBRADO POR PERCENTIL (ScorerMax corrigido) vs k*base -- mediana e faixa")
print("=" * 100)
print(f"{'q':>9s} " + "".join(f"{c:>22s}" for c in SIN))
for q in QS:
    cel = []
    for c in SIN:
        r = (THR[c][q] / (BASE[c] * K[c])).dropna()
        cel.append(f"{r.median():7.2f} [{r.min():5.2f}-{r.max():6.2f}]")
    print(f"{q:>9.3f} " + "".join(f"{x:>22s}" for x in cel))

print("\n" + "=" * 100)
print("DESEMPENHO POR PERCENTIL (regra C)")
print("=" * 100)
print(f"{'limiar':>30s} {'det':>5s} {'FP/mes':>8s} {'h/mes':>8s} {'lead':>7s}  perdidos")
print(f"{'k*base (fixo, calibrado)':>30s} {ref['det']}/8   {ref['fp_mes']:8.3f} {ref['h_mes']:8.2f} "
      f"{ref['lead']:7.1f}  {','.join(ref['perd']) if ref['perd'] else '--'}")
for q in QS:
    ON = {}
    for c in SIN:
        thr = THR[c][q]
        E = EW_fix[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        x = ((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
        S = np.empty(len(x)); a = 0.0
        for k_ in range(len(x)):
            a = a * CARGA if reset[k_] else max(0.0, a + x[k_]); S[k_] = a
        ON[c] = (deg | pd.Series(S > H_CUSUM, index=idx)) & mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    r = avalia_regra_c(pos(v))
    print(f"{'percentil ' + format(q, '.3f'):>30s} {r['det']}/8   {r['fp_mes']:8.3f} {r['h_mes']:8.2f} "
          f"{r['lead']:7.1f}  {','.join(r['perd']) if r['perd'] else '--'}")
