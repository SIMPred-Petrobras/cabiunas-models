#!/usr/bin/env python3
"""Varredura dos 87 sensores da serie consolidada: algum carrega precursor que nao usamos?

Contexto. O detector usa 38 colunas. Existe uma serie consolidada com 87, contendo
grandezas que faltavam e que eu declarei ausentes: rotacao (NGP_A, NPT_A), torque
(TM_TORQUE_A), vazao (FI_0201, FI_0311), vibracao em canal distinto do deslocamento
(VI_0301/02/04/05), posicao axial (ZI_0301/02) e sete temperaturas na faixa de mancal
(TI_0318..0324).

LIMITE DECLARADO ANTES DO RESULTADO: o arquivo cobre 2025-01-01 a 2025-10-31 -- **5 dos
9 eventos**. Os tres mais recentes ficam de fora. Entao isto e uma TRIAGEM, nao uma
validacao: qualquer sensor promissor aqui precisa ser confirmado depois nos eventos de
11/2025, 12/2025 e 02/2026, que exigem outra fonte de dado.

Metodo. Para cada sensor: z robusto causal contra referencia rolante de 400 h em operacao
quente, com guarda de 24 h (mesma construcao do `vb`). Para cada evento, o maximo de |z|
nas 48 h anteriores. Nulo: o mesmo estatistico em 2.000 janelas de 48 h sorteadas dentro
de operacao quente, longe de qualquer evento. Reporta, por sensor, quantos dos 5 eventos
ficam acima do p99 do nulo -- e o p combinado.

Isto responde "existe informacao que estamos jogando fora?", nao "isto melhora o
detector". A segunda pergunta so faz sentido se a primeira der sim.
"""
from __future__ import annotations
import sys, glob, pathlib
import numpy as np, pandas as pd

SRC = glob.glob(str(pathlib.Path.home() /
      ".clearml/cache/storage_manager/datasets/*/serie_consolidada_2025_interpolated_antigo.csv"))[0]
PASSO = "10min"
BASE_H, GUARDA_H = 400.0, 24.0
N_NULO = 2000
RNG = np.random.default_rng(3)
JAN = pd.Timedelta(hours=48)


def z_causal(s, n_base, n_guarda):
    """z robusto contra mediana/MAD das n_base amostras anteriores, pulando n_guarda."""
    ref = s.shift(n_guarda)
    med = ref.rolling(n_base, min_periods=n_base // 4).median()
    mad = (ref - med).abs().rolling(n_base, min_periods=n_base // 4).median() * 1.4826
    return ((s - med) / mad.replace(0, np.nan)).abs()


def main():
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_localize(None)
    print("carregando a serie consolidada (87 colunas, 756 MB) ...", flush=True)
    d = pd.read_csv(SRC, low_memory=False)
    d["t"] = pd.to_datetime(d["data_datetime"], errors="coerce")
    d = d.dropna(subset=["t"]).set_index("t").sort_index()
    run = pd.to_numeric(d["RUNNING_A"], errors="coerce") > 0.5
    t5 = pd.to_numeric(d["T5_AVG_A"], errors="coerce")
    quente = (run & (t5 > 300)).reindex(d.index).fillna(False)
    cols = [c for c in d.columns if c not in ("data_datetime", "RUNNING_A")]
    X = d[cols].apply(pd.to_numeric, errors="coerce").where(quente)
    X = X.resample(PASSO).median()
    hot = X.notna().any(axis=1)
    ev = [t for t in fal if X.index[0] <= t <= X.index[-1]]
    print(f"  {X.shape[1]} sensores, {len(X):,} linhas a {PASSO}, "
          f"{int(hot.sum())} com operacao quente, {len(ev)} eventos cobertos\n", flush=True)

    por_h = int(pd.Timedelta("1h") / pd.Timedelta(PASSO))
    nb, ng = int(BASE_H * por_h), int(GUARDA_H * por_h)
    idx = X.index
    # janelas de nulo: inicios sorteados em operacao quente, longe de eventos
    longe = hot.copy()
    for t in ev:
        longe &= ~((idx >= t - pd.Timedelta(days=10)) & (idx <= t + pd.Timedelta(days=3)))
    cand = idx[longe.values]
    starts = RNG.choice(len(cand), min(N_NULO, len(cand)), replace=False)

    linhas = []
    for j, c in enumerate(X.columns):
        s = X[c]
        if s.notna().sum() < nb:
            continue
        z = z_causal(s, nb, ng)
        picos_ev = []
        for t in ev:
            w = z.loc[t - JAN:t].dropna()
            picos_ev.append(float(w.max()) if len(w) else np.nan)
        nul = []
        for k in starts:
            t = cand[k]
            w = z.loc[t - JAN:t].dropna()
            if len(w) > 5: nul.append(float(w.max()))
        if len(nul) < 200 or not np.isfinite(picos_ev).any():
            continue
        nul = np.array(nul); p99 = np.nanpercentile(nul, 99)
        acima = int(np.nansum(np.array(picos_ev) > p99))
        # p combinado: probabilidade de >= `acima` eventos acima do p99 por acaso
        n_ok = int(np.isfinite(picos_ev).sum())
        from scipy import stats as st
        p_comb = st.binom.sf(acima - 1, n_ok, 0.01) if acima > 0 else 1.0
        linhas.append(dict(sensor=c, n_ev=n_ok, acima_p99=acima, p=p_comb,
                           pico_med=float(np.nanmedian(picos_ev)), p99_nulo=float(p99),
                           razao=float(np.nanmedian(picos_ev) / p99) if p99 > 0 else np.nan))
        if (j + 1) % 15 == 0:
            print(f"  {j+1}/{X.shape[1]} sensores varridos", flush=True)
    T = pd.DataFrame(linhas).sort_values(["acima_p99", "razao"], ascending=[False, False])
    T.to_csv("varre_87.csv", index=False)

    NOSSOS = ["TI_0301","TI_0303","TI_0305","TI_0307","TC382","T5_AVG","TV_35",
              "TI_0315","TI_0317","TI_0325","PI_03","PDI_03","PDIT_03"]
    def usado(c): return any(k in c for k in NOSSOS)
    print("\n" + "=" * 100)
    print(f"SENSORES COM PRECURSOR (pico de |z| em 48 h acima do p99 do nulo), n={len(ev)} eventos")
    print("=" * 100)
    print(f"{'sensor':>30} {'novo?':>7} {'eventos acima':>14} {'p':>10} "
          f"{'pico mediano':>13} {'p99 nulo':>10} {'razao':>7}")
    for _, r in T.head(25).iterrows():
        print(f"{r.sensor:>30} {'NOVO' if not usado(r.sensor) else '-':>7} "
              f"{int(r.acima_p99):8d}/{int(r.n_ev)} {r.p:10.2e} {r.pico_med:13.1f} "
              f"{r.p99_nulo:10.1f} {r.razao:7.2f}")
    print(f"\n  sensores varridos: {len(T)}   com >=3 de {len(ev)} eventos acima do p99: "
          f"{int((T.acima_p99>=3).sum())}   sendo NOVOS: "
          f"{int(((T.acima_p99>=3) & ~T.sensor.map(usado)).sum())}")


if __name__ == "__main__":
    main()
