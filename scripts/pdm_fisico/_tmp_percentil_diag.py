"""Por que o percentil ALTO -- que e o certo em teoria -- falha na pratica aqui.

A pergunta: se o limiar deve corresponder a uma taxa de falso alarme baixa no
baseline saudavel, o percentil correto e alto (p99, p99,9), nao p80. Mas o teste
mediu p80 -> 7/8 e p99,9 -> 2/8. Contradicao aparente.

Hipotese a testar: o percentil alto e o ESTIMADOR errado, nao o ALVO errado. Um
quantil extremo de uma distribuicao de cauda pesada, estimado por estatistica de
ordem em janela finita (20.000 pontos), e dominado pelo pior artefato daquela
janela -- entao o limiar oscila ordens de grandeza entre meses, e a mediana da
oscilacao nao diz nada sobre o comportamento tipico.

Tres medicoes:
  1. em que percentil da distribuicao em amostra o limiar FIXO (k*base) cai, mes a mes
  2. estabilidade do limiar autocalibrado por percentil (coef. de variacao entre meses)
  3. quantos pontos do baseline saudavel excedem k*base -- a taxa de falso alarme
     implicita do ponto de operacao atual
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from publica_clearml import SIN, HL, BASE, K, T0, GRID
from autocalibra import TEMP, PRESS, MANCAL, ALVO_SP, N_FIT

PHI = 0.10
QS = [80.0, 90.0, 95.0, 99.0, 99.5, 99.9]


class ScorerMax:
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
spread = (g[ALVO_SP] - g[MANCAL].median(axis=1)).abs()
z = np.load("piso_fisico_cache.npz")
with np.errstate(invalid="ignore", divide="ignore"):
    Zv = np.abs((z["Xh"] - z["MED"]) / z["S"])
vb_all = np.full(len(idx), np.nan)
vb_all[z["hot"]] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
vb_all[~np.isfinite(vb_all)] = np.nan
hot = z["hot"]
meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
hl_ew = lambda s, h: s.ewm(halflife=pd.Timedelta(h), times=s.index).mean()

linhas = []          # por (mes, sinal): percentil em que k*base cai, e excedencia
thr_por_q = {c: {q: [] for q in QS} for c in SIN}

for i, m0 in enumerate(meses):
    fit = g.loc[estavel & (idx < m0), TEMP + PRESS].dropna().tail(N_FIT)
    if len(fit) < N_FIT // 4:
        continue
    st = ScorerMax().fit(fit[TEMP]); sp_ = ScorerMax().fit(fit[PRESS])
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
        fixo = BASE[c] * K[c]
        pct = 100.0 * (v < fixo).mean()          # percentil em que o limiar fixo cai
        exced = 100.0 * (v >= fixo).mean()       # % do baseline saudavel acima dele
        linhas.append(dict(mes=m0, sinal=c, pct_do_fixo=pct, exced_pct=exced))
        for q in QS:
            thr_por_q[c][q].append(float(np.percentile(v, q)))

T = pd.DataFrame(linhas)

print("=" * 96)
print("1. EM QUE PERCENTIL DO BASELINE SAUDAVEL O LIMIAR FIXO (k*base) CAI?")
print("=" * 96)
print(f"{'sinal':>6} {'k*base':>8} {'percentil mediano':>18} {'faixa entre meses':>26} "
      f"{'% do baseline acima':>20}")
for c in SIN:
    s = T[T.sinal == c]
    print(f"{c:>6} {BASE[c]*K[c]:8.2f} {s.pct_do_fixo.median():17.2f}% "
          f"{'[' + format(s.pct_do_fixo.min(), '.2f') + ' - ' + format(s.pct_do_fixo.max(), '.2f') + ']':>26} "
          f"{s.exced_pct.median():19.3f}%")

print("\n" + "=" * 96)
print("2. ESTABILIDADE DO ESTIMADOR: quanto o limiar autocalibrado oscila entre meses")
print("=" * 96)
print(f"{'q':>8} " + "".join(f"{c:>21}" for c in SIN))
print(f"{'':>8} " + "".join(f"{'cv    razao max/min':>21}" for c in SIN))
for q in QS:
    cel = []
    for c in SIN:
        a = np.array(thr_por_q[c][q])
        a = a[np.isfinite(a) & (a > 0)]
        cv = a.std() / a.mean()
        cel.append(f"{cv:6.2f} {a.max()/a.min():13.0f}x")
    print(f"{q:>8.3f} " + "".join(f"{x:>21}" for x in cel))

print("\n" + "=" * 96)
print("3. O ESTIMADOR ROBUSTO: e se o percentil alto fosse tomado sobre TODOS os meses")
print("   juntos (amostra 27x maior) em vez de mes a mes?")
print("=" * 96)
print(f"{'q':>8} " + "".join(f"{c:>16}" for c in SIN))
for q in QS:
    cel = []
    for c in SIN:
        a = np.array(thr_por_q[c][q])
        a = a[np.isfinite(a) & (a > 0)]
        # limiar unico = mediana entre meses (estimador robusto do mesmo quantil)
        cel.append(f"{np.median(a)/(BASE[c]*K[c]):15.2f}x")
    print(f"{q:>8.3f} " + "".join(f"{x:>16}" for x in cel))
print("\n(razao para o limiar fixo atual; 1,00x significaria reproduzir o ponto de operacao)")
