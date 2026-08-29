#!/usr/bin/env python3
"""O nosso detector pode ser AUTOCALIBRADO como o do Francisco e da Lara?

Por que este script existe, e por que ele corrige o anterior
------------------------------------------------------------
francisco_lara.py::teste5 testou o limiar por percentil e ele colapsou (1-2/8).
Mas o teste estava errado: eu tomei o percentil sobre o historico de score FORA
DE AMOSTRA -- que contem todas as excursoes reais dos meses passados, inclusive
falhas. O PASSO 8 deles diz outra coisa: percentil do score do PROPRIO BASELINE,
isto e, o residuo do modelo sobre os dados em que ele foi ajustado. Distribuicao
completamente diferente, e por construcao bem comportada.

Aqui o port e fiel: para cada mes, ajusta o PCA nas ultimas 667 h estaveis,
pontua ESSA MESMA janela em amostra, suaviza com o mesmo EWMA e toma o percentil
q. Esse vira o limiar do mes. Idem para sp (z do spread na janela de ajuste) e
vb (z da vibracao contra a referencia rolante, na janela de ajuste).

A pergunta que responde: os `k` = 1,7 / 2,2 -- as duas constantes calibradas no
proprio alvo de 8 eventos -- podem ser trocadas por um percentil, que nao precisa
de alvo nenhum? Se sim, o detector porta para outra maquina sem recalibrar.

Controle de corretude: o escore fora de amostra recalculado aqui tem que bater
com o cache piso_fisico_cache.npz. Se nao bater, a reimplementacao esta errada e
nada abaixo vale.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import avalia as AV
import francisco_lara as F
from publica_clearml import (reproduz, SIN, HL, BASE, K, KAPPA, H_CUSUM, CARGA, T0)

PRE = "954005_624_"
TEMP = [PRE+x for x in ["TI_0301","TI_0303","TI_0305","TI_0307","TI_0315","TI_0317","TI_0325"]] \
     + ["TC382_0%d_A" % i for i in range(1,7)] + ["T5_AVG_A"]
PRESS = [PRE+x for x in ["PDIT_0305","PDI_0301","PDI_0302","PDI_0317","PDI_0338",
                         "PI_0307","PI_0308","PI_0315","PI_0319","PI_0339","PI_0340"]] + ["PI_5134001"]
MANCAL = [PRE+x for x in ["TI_0301","TI_0303","TI_0307"]]
ALVO_SP = PRE + "TI_0305"
N_FIT = 20_000
PHI = 0.10


class ScorerMax:
    """RobustScaler -> PCA(0.95) -> erro por sensor / p99 do proprio sensor, com piso.
    Reimplementacao autocontida do que vinha de cabiunas_pdm.scoring (pacote perdido).
    O piso phi impede que o sensor mais quieto do baseline domine o maximo."""
    def fit(self, base: pd.DataFrame):
        self.cols = list(base.columns)
        X = base.dropna()
        self.scaler = RobustScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = PCA(n_components=0.95).fit(Xs)
        e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
        p = np.nanpercentile(e, 99, axis=0)
        self.sens_p99_ = np.maximum(p, PHI * np.nanmedian(p))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.cols]
        m = X.notna().all(axis=1).to_numpy()
        out = np.full(len(X), np.nan)
        if m.any():
            Xs = self.scaler.transform(X[m])
            e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
            out[m] = np.max(e / self.sens_p99_, axis=1)
        return out


def main():
    fin, mask, alvo, ON, idx, sel = reproduz()
    ref = AV.avalia(fin, alvo, mask)
    g = pd.read_parquet("grade2min.parquet")
    z = np.load("piso_fisico_cache.npz")
    estavel = ((g["RUNNING_A"] > 0.5).fillna(False) & (g["T5_AVG_A"] > 300))
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    part = op & ~op.shift(fill_value=False)
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")

    spread = (g[ALVO_SP] - g[MANCAL].median(axis=1)).abs()
    hot = z["hot"]; MEDv = z["MED"]; Sv = z["S"]
    with np.errstate(invalid="ignore", divide="ignore"):
        Zv = np.abs((z["Xh"] - MEDv) / Sv)
    vb_all = np.full(len(idx), np.nan)
    vb_all[hot] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
    vb_all[~np.isfinite(vb_all)] = np.nan
    sp_all = np.abs((spread.to_numpy() - z["med_sp"]) / z["mad_sp"])

    print("\nAjuste mensal: PCA nas ultimas 667 h estaveis, pontuando a PROPRIA janela.")
    print("Guarda o percentil do residuo em amostra -- o limiar autocalibrado do mes.\n")
    QS = [80.0, 90.0, 95.0, 99.0, 99.5, 99.9]
    THR = {c: {q: pd.Series(np.nan, index=idx) for q in QS} for c in SIN}
    t_oos = np.full(len(idx), np.nan); p_oos = np.full(len(idx), np.nan)
    hl_ew = lambda s, h: s.ewm(halflife=pd.Timedelta(h), times=s.index).mean()

    for i, m0 in enumerate(meses):
        m1 = meses[i+1] if i+1 < len(meses) else idx[-1] + pd.Timedelta("2min")
        fit = g.loc[estavel & (idx < m0), TEMP + PRESS].dropna().tail(N_FIT)
        if len(fit) < N_FIT // 4:
            continue
        selm = ((idx >= m0) & (idx < m1))
        if not selm.any():
            continue
        st = ScorerMax().fit(fit[TEMP]); sp_ = ScorerMax().fit(fit[PRESS])
        t_oos[selm] = st.score(g.loc[selm, TEMP])
        p_oos[selm] = sp_.score(g.loc[selm, PRESS])
        # --- residuo EM AMOSTRA, suavizado com o mesmo EWMA, na ordem do tempo
        ins = {"t": pd.Series(st.score(fit[TEMP]), index=fit.index),
               "p": pd.Series(sp_.score(fit[PRESS]), index=fit.index)}
        b = spread.loc[fit.index]
        med, mad = float(b.median()), float((b - b.median()).abs().median() * 1.4826)
        ins["sp"] = ((b - med).abs() / mad)
        # o .isin do pandas usa hash; o np.isin em datetime com fuso cai em
        # comparacao objeto-a-objeto (408k x 20k) e trava o mes inteiro
        jf = idx[hot].isin(fit.index)
        ins["vb"] = pd.Series(vb_all[hot][jf], index=idx[hot][jf])
        for c in SIN:
            v = hl_ew(ins[c].sort_index(), HL[c]).to_numpy()
            v = v[np.isfinite(v)]
            if len(v) < 100:
                continue
            for q in QS:
                THR[c][q][selm] = float(np.percentile(v, q))

    # ---------------------------------------------------------- controle
    ok_t = np.corrcoef(pd.Series(t_oos).fillna(0), pd.Series(z["t"]).fillna(0))[0,1]
    ok_p = np.corrcoef(pd.Series(p_oos).fillna(0), pd.Series(z["p"]).fillna(0))[0,1]
    print(f"CONTROLE  correlacao com o cache validado:  t {ok_t:.5f}   p {ok_p:.5f}")
    print("(tem que ser ~1,0; abaixo disso a reimplementacao do scorer esta errada)\n")

    cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp_all, "vb": vb_all}, index=idx)
    E = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in HL.items()}
    reset = ((~mask) | part).to_numpy()

    print("Limiar autocalibrado / limiar atual (k*base), por percentil -- mediana e faixa")
    print(f"{'q':>9s} " + "".join(f"{c:>22s}" for c in SIN))
    for q in QS:
        cel = []
        for c in SIN:
            r = (THR[c][q] / (BASE[c]*K[c])).dropna()
            cel.append(f"{r.median():7.2f} [{r.min():5.2f}-{r.max():6.2f}]")
        print(f"{q:>9.3f} " + "".join(f"{x:>22s}" for x in cel))

    def roda(thr_de):
        ONq = {}
        for c in SIN:
            thr = thr_de(c)
            deg = ((E[c] > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
            x = ((E[c] / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
            S = np.empty(len(x)); a = 0.0
            for k_ in range(len(x)):
                a = a*CARGA if reset[k_] else max(0.0, a + x[k_]); S[k_] = a
            ONq[c] = (deg | pd.Series(S > H_CUSUM, index=idx)) & mask
        v = pd.Series(sum(ONq[c].astype(int) for c in SIN) >= 2, index=idx) & mask
        return AV.avalia(F.pos(v, idx) & sel, alvo, mask)

    np.savez_compressed('autocalibra_thr.npz',
                        **{f'{c}|{q}': THR[c][q].to_numpy() for c in SIN for q in QS})
    print('\nlimiares salvos em autocalibra_thr.npz')
    print(f"\n{'limiar':30s} {'det':>5s} {'eps':>5s} {'FP/mes':>8s} {'h/mes':>8s} {'lead':>7s}  perdidos")
    print(f"{'k*base (atual, calibrado)':30s} {ref['det']:>3d}/8 {ref['episodios']:>5d} "
          f"{ref['fp_mes']:>8.2f} {ref['h_fp_mes']:>8.1f} {ref['lead_med']:>7.1f}")
    for q in QS:
        m = roda(lambda c, q=q: THR[c][q])
        perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
        print(f"{'percentil ' + format(q,'.3f') + ' (autocalibrado)':30s} {m['det']:>3d}/8 "
              f"{m['episodios']:>5d} {m['fp_mes']:>8.2f} {m['h_fp_mes']:>8.1f} "
              f"{m['lead_med']:>7.1f}  {','.join(perd)}")
    return THR, E, ref


if __name__ == "__main__":
    main()
