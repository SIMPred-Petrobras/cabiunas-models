"""In-sample (janela de ajuste do PCA) vs out-of-sample (o que o detector ve).
Sao a mesma distribuicao? Se nao, o percentil em amostra nao diz nada sobre o
comportamento fora de amostra -- e a autocalibracao esta calibrando na
distribuicao errada."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from pos_processamento import cru, mask, idx
from publica_clearml import SIN, BASE, K, T0, GRID
from autocalibra import TEMP, PRESS, N_FIT
PHI = 0.10

class ScorerMax:
    def fit(self, base):
        self.cols = list(base.columns); X = base.dropna()
        self.scaler = RobustScaler().fit(X); Xs = self.scaler.transform(X)
        self.pca = PCA(n_components=0.95).fit(Xs)
        e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs)))**2
        p = np.nanpercentile(e, 99, axis=0)
        self.sens_p99_ = np.maximum(p, PHI*np.nanmedian(p))
        self.recon_p99 = float(np.nanpercentile(np.max(e/self.sens_p99_, axis=1), 99))
        return self
    def score(self, df):
        X = df[self.cols]; m = X.notna().all(axis=1).to_numpy()
        out = np.full(len(X), np.nan)
        if m.any():
            Xs = self.scaler.transform(X[m])
            e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs)))**2
            out[m] = np.max(e/self.sens_p99_, axis=1)/self.recon_p99
        return out

g = pd.read_parquet("grade2min.parquet")
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
m0 = pd.Timestamp("2025-07-01", tz="UTC")   # um mes representativo do meio da serie
m1 = pd.Timestamp("2025-08-01", tz="UTC")
fit = g.loc[estavel & (g.index < m0), TEMP+PRESS].dropna().tail(N_FIT)
st = ScorerMax().fit(fit[TEMP]); sp_ = ScorerMax().fit(fit[PRESS])
selm = (idx >= m0) & (idx < m1) & mask

print(f"mes de referencia: {m0:%Y-%m}   ajuste em {len(fit):,} pontos")
print(f"\n{'':>4} {'quantil':>9} {'IN-SAMPLE (ajuste)':>20} {'OUT-OF-SAMPLE (mascarado)':>27} {'razao':>8}")
for c, cols, sc in (("t", TEMP, st), ("p", PRESS, sp_)):
    ins = pd.Series(sc.score(fit[cols])).dropna()
    oos = pd.Series(sc.score(g.loc[selm, cols])).dropna()
    print(f"\n  {c}  (limiar fixo k*base = {BASE[c]*K[c]:.2f})")
    for q in (50, 80, 90, 95, 99, 99.5, 99.9):
        a, b = np.percentile(ins, q), np.percentile(oos, q)
        print(f"{'':>4} {q:>8.1f}% {a:19.3f} {b:26.3f} {b/a:7.2f}x")
    print(f"{'':>4} {'--> o limiar 3,40 cai em:':>32} in-sample p{100*(ins<BASE[c]*K[c]).mean():.2f}"
          f"   out-of-sample p{100*(oos<BASE[c]*K[c]).mean():.2f}")
