#!/usr/bin/env python3
"""Estrutura X-Y das sondas de vibracao: a ultima lacuna identificada.

O detector usa `vb = max(z das 10 sondas)` -- trata as 10 como dez medidas independentes.
Mas elas sao CINCO PARES: TV_35{1..5}{X,Y} sao as sondas X e Y do MESMO mancal. A relacao
entre X e Y e a orbita do eixo, e mudanca de forma de orbita e diagnostico classico de
maquina rotativa -- costuma preceder mudanca de amplitude.

Limite: a 2 min de agregacao nao da para reconstruir a orbita real (precisaria da forma
de onda por rotacao). O que SOBRA de recuperavel do par:

    mag  = sqrt(X^2 + Y^2)          magnitude vetorial (melhor que max(X,Y))
    rat  = log(|X| / |Y|)           elipticidade -- direcao dominante do movimento
    ang  = atan2(Y, X)              direcao do deslocamento medio
    cor  = corr movel(X, Y) em 2 h  forma da precessao
    cruz = mag_i / mag_j            razao entre mancais -- onde esta o problema

Triagem, nao detector: para cada grandeza, z robusto causal contra referencia rolante de
400 h com guarda de 24 h (mesma construcao do `vb`), pico de |z| nas 48 h antes de cada
evento, contra o nulo de 2.000 janelas sorteadas em operacao quente longe de eventos.

A pergunta e "existe informacao na estrutura X-Y que o max joga fora?". Se sim, vira
sinal; se nao, fecha a lacuna. Janela 2025-01 a 2026-04, 8 eventos.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd
from scipy import stats as st

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import avalia as A, rolante as RO
from ablacao import canonico, mascara_pontuacao

T0 = pd.Timestamp("2025-01-01", tz="UTC")
JAN = pd.Timedelta(hours=48)
N_NULO = 2000
RNG = np.random.default_rng(11)
PARES = [(f"TV_35{i}X_A", f"TV_35{i}Y_A", f"mancal{i}") for i in range(1, 6)]


def main():
    df = canonico(); idx = df.index
    g = pd.read_parquet("grade2min.parquet")
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df) & (idx >= T0)
    quente = df["stable"].astype(bool)
    alvo = list(todas[todas >= T0])
    print(f"janela {T0:%Y-%m}+: {len(alvo)} eventos, {mask.sum()*2/60:.0f} h pontuaveis\n", flush=True)

    F = {}
    n2h = int(pd.Timedelta("2h") / pd.Timedelta("2min"))
    mags = {}
    for xc, yc, rot in PARES:
        X = pd.to_numeric(g[xc], errors="coerce").where(quente)
        Y = pd.to_numeric(g[yc], errors="coerce").where(quente)
        mag = np.sqrt(X**2 + Y**2)
        mags[rot] = mag
        F[f"{rot}_mag"] = mag
        F[f"{rot}_rat"] = np.log((X.abs() + 1e-3) / (Y.abs() + 1e-3))
        F[f"{rot}_ang"] = np.arctan2(Y, X)
        F[f"{rot}_cor"] = X.rolling(n2h, min_periods=n2h//2).corr(Y)
    for a, b in itertools.combinations(mags, 2):
        F[f"cruz_{a[-1]}{b[-1]}"] = np.log((mags[a] + 1e-3) / (mags[b] + 1e-3))
    # referencia: o proprio vb, para comparar
    V = g[[c for p in PARES for c in p[:2]]].apply(pd.to_numeric, errors="coerce").where(quente)
    F["REF_vb_max"] = RO.z_rolante(V, quente, todas, horas_base=400, guarda_h=24, phi=0.0).max(axis=1)
    print(f"grandezas construidas: {len(F)}", flush=True)

    Z = {}
    for k, s in F.items():
        if k == "REF_vb_max":
            Z[k] = s; continue
        Z[k] = RO.z_rolante(s.to_frame("v"), quente, todas, horas_base=400,
                            guarda_h=24, phi=0.0)["v"]
        print(f"  z de {k}", flush=True)

    longe = mask.copy()
    for t in alvo:
        longe &= ~((idx >= t - pd.Timedelta(days=10)) & (idx <= t + pd.Timedelta(days=3)))
    cand = idx[longe.values]
    starts = RNG.choice(len(cand), min(N_NULO, len(cand)), replace=False)

    linhas = []
    for k, z in Z.items():
        zi = z.where(mask)
        picos = []
        for t in alvo:
            w = zi.loc[t - JAN:t].dropna()
            picos.append(float(w.max()) if len(w) else np.nan)
        nul = []
        for s_ in starts:
            t = cand[s_]
            w = zi.loc[t - JAN:t].dropna()
            if len(w) > 5: nul.append(float(w.max()))
        if len(nul) < 200: continue
        nul = np.array(nul); p99 = np.nanpercentile(nul, 99)
        n_ok = int(np.isfinite(picos).sum())
        acima = int(np.nansum(np.array(picos) > p99))
        p = st.binom.sf(acima - 1, n_ok, 0.01) if acima > 0 else 1.0
        linhas.append(dict(grandeza=k, n_ev=n_ok, acima=acima, p=p,
                           pico_med=float(np.nanmedian(picos)), p99=float(p99),
                           razao=float(np.nanmedian(picos) / p99) if p99 > 0 else np.nan))
    T = pd.DataFrame(linhas).sort_values(["acima", "razao"], ascending=[False, False])
    T.to_csv("orbita.csv", index=False)

    print("\n" + "=" * 92)
    print(f"PRECURSOR NA ESTRUTURA X-Y (pico de |z| em 48 h acima do p99 do nulo), n={len(alvo)}")
    print("=" * 92)
    print(f"{'grandeza':>18} {'eventos acima':>14} {'p':>10} {'pico med':>10} {'p99 nulo':>10} {'razao':>7}")
    for _, r in T.iterrows():
        marca = "  <-- o sinal atual" if r.grandeza == "REF_vb_max" else ""
        print(f"{r.grandeza:>18} {int(r.acima):8d}/{int(r.n_ev)} {r.p:10.2e} {r.pico_med:10.1f} "
              f"{r.p99:10.1f} {r.razao:7.2f}{marca}")
    ref = T[T.grandeza == "REF_vb_max"].iloc[0]
    melhores = T[(T.grandeza != "REF_vb_max") & (T.acima >= ref.acima)]
    print(f"\n  o sinal atual (max das 10 sondas): {int(ref.acima)}/{int(ref.n_ev)} eventos, "
          f"razao {ref.razao:.2f}")
    print(f"  grandezas de orbita que igualam ou superam: {len(melhores)}")
    if len(melhores):
        print("    " + ", ".join(melhores.grandeza))


if __name__ == "__main__":
    main()
