#!/usr/bin/env python3
"""As 47 tags catalogadas e ausentes da grade carregam sinal precursor?

CONTEXTO. Rodamos com 38 tags; o `metadata.csv` cataloga 85. As 47 que faltam
existem no `portalintegridade` (87 colunas, 2022-2025) e incluem justamente o que
o detector nao ve: deslocamento axial, vibracao do COMPRESSOR (o nosso `vb` so
olha as 10 sondas da turbina), temperatura de oleo de dreno por mancal, e o
quarteto P/T/vazao/rotacao que define o ponto termodinamico.

POR QUE ESTE SCRIPT EXISTE, E NAO O TESTE ANTERIOR. Uma primeira tentativa mediu
enriquecimento com z robusto univariado contra mediana GLOBAL e nao achou nada --
mas tambem nao achou nada nos canais que SABEMOS carregar sinal (`TI_0305` deu
1,06x, `TC382_03_A` deu 0,71x). Quando o controle positivo falha, o resultado nao
vale como refutacao: o teste e que estava cego.

Aqui a estrutura e a do detector: baseline MENSAL (walk-forward), z robusto ou
PCA por familia, EWMA, e a mascara de operacao estavel sem blackout. E os canais
atuais entram como CONTROLE POSITIVO -- se eles nao aparecerem, o teste esta
errado de novo e o resultado e descartado.

RESULTADO: O TESTE NAO PODE SER FEITO COM O DADO QUE TEMOS -- e agora se sabe
exatamente o que pedir.

O `portalintegridade` vem em dois formatos, e a diferenca entre eles decide tudo:

    formato        colunas   passo   mudancas/dia   periodo disponivel
    recorded          86      30 s        ~7        2022-01 a 2025-08
    interpolated      86      30 s     ~2.700       2022-01 a 2024-02
    tagsselec. 30s    38      30 s     ~2.850       2022-02 a 2026-04  <- nossa fonte

O `recorded` e exportacao POR EXCECAO com banda morta larga: **sete valores por
dia**, um a cada 3,3 h. Depois do ffill a serie parece completa (99,1% de
cobertura) mas e constante por trechos longos -- o MAD de uma janela de 20.000
pontos da EXATAMENTE ZERO, e todo z-score explode ou vira NaN. Isso vale inclusive
para tags que JA usamos (TI_0305, T5_AVG_A, TV_351X_A) quando lidas dessa fonte, o
que prova que o problema e o formato e nao as tags.

O `interpolated` tem a cadencia real e as 86 colunas -- mas so ate 2024-02, e
**nenhum dos 8 trips do alvo esta coberto** (todos sao de 2025 em diante).

Entao: as 47 tags ausentes NAO sao inuteis e NAO foram refutadas. Elas sao
inacessiveis com o que esta no repositorio. O que falta e um pedido especifico:

    exportar o portalintegridade no formato INTERPOLATED (30 s, 86 colunas)
    de 2025-01 em diante

Custo de descobrir se ajudam: um pedido de exportacao. Zero trabalho de modelagem
ate o dado chegar. Este script fica pronto para rodar quando chegar -- os canais
novos ja estao definidos (deslocamento axial, vibracao do compressor, oleo de
dreno, mancal de escora, desempenho) e os controles positivos tambem.

LICAO: verificar a TAXA DE MUDANCA do dado antes de modelar, nao so a cobertura.
Uma serie 99% "preenchida" por ffill de uma exportacao comprimida tem a densidade
de informacao de sete pontos por dia.

Uso:  PYTHONPATH=. python tags_ausentes.py <parquet_interpolated>
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

PORTAL = sys.argv[1] if len(sys.argv) > 1 else "_portal_grade2min.parquet"
RNG = np.random.default_rng(20260918)
JAN = pd.Timedelta("48h")
FIT = 20_000
N_NULO = 2000

# ── famílias novas, cada uma análoga a um canal que já existe ───────────────
NOVAS = {
    "ax  deslocamento axial": dict(
        tipo="zmax",
        tags=["954005_624_ZI_0301", "954005_624_ZI_0302", "TV_350A_A", "TV_354A_A"]),
    "vc  vibracao compressor": dict(
        tipo="zmax",
        tags=["954005_624_VI_0301", "954005_624_VI_0302",
              "954005_624_VI_0304", "954005_624_VI_0305"]),
    "od  oleo de dreno": dict(
        tipo="spread",
        tags=["954005_624_TI_0320", "954005_624_TI_0321", "954005_624_TI_0322"]),
    "es  mancal de escora": dict(
        tipo="zmax", tags=["954005_624_TI_0318", "954005_624_TI_0319"]),
    "de  desempenho": dict(
        tipo="pca",
        tags=["954005_624_PI_0201", "954005_624_PI_0204", "954005_624_TI_0201",
              "954005_624_TI_0204", "954005_624_FI_0201", "STD_FLOW_A",
              "NCPSR_A", "NGP_A", "NPT_A", "TM_TORQUE_A", "954005_624_PI_0324"]),
}
# ── controles positivos: canais que JA sabemos carregar sinal ───────────────
CONTROLE = {
    "CTRL sp (mancal LNA)": dict(
        tipo="spread_alvo", alvo="954005_624_TI_0305",
        tags=["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]),
    "CTRL t (temperatura)": dict(
        tipo="pca",
        tags=["954005_624_TI_0325", "954005_624_TI_0315", "954005_624_TI_0317",
              "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
              "954005_624_TI_0301", "TC382_01_A", "TC382_02_A", "TC382_03_A",
              "TC382_04_A", "TC382_05_A", "TC382_06_A", "T5_AVG_A"]),
    "CTRL vb (vibr. turbina)": dict(
        tipo="zmax",
        tags=["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
              "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]),
}


def canal(G, est, spec):
    """Constroi o sinal com baseline MENSAL, do jeito que o detector faz."""
    tags = [t for t in spec["tags"] if t in G.columns]
    if spec["tipo"] == "spread_alvo" and spec["alvo"] not in G.columns:
        return None, 0
    if len(tags) < (2 if spec["tipo"] != "zmax" else 1):
        return None, 0
    ix = G.index
    out = np.full(len(ix), np.nan)
    meses = pd.date_range(ix[0].normalize().replace(day=1), ix[-1], freq="MS", tz="UTC")
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else ix[-1] + pd.Timedelta("2min")
        base = G.loc[est & (ix < m0), tags + ([spec["alvo"]] if spec["tipo"] == "spread_alvo" else [])
                     ].dropna().tail(FIT)
        s = (ix >= m0) & (ix < m1)
        if len(base) < FIT // 4 or not s.any():
            continue
        w = G.loc[s]
        if spec["tipo"] == "pca":
            sc = RobustScaler().fit(base[tags])
            pca = PCA(n_components=0.95, svd_solver="full").fit(sc.transform(base[tags]))
            def rec(X):
                Xs = sc.transform(X)
                return np.max((Xs - pca.inverse_transform(pca.transform(Xs))) ** 2, axis=1)
            p99 = float(np.percentile(rec(base[tags]), 99))
            ok = w[tags].notna().all(axis=1).to_numpy()
            v = np.full(int(s.sum()), np.nan)
            if ok.any():
                v[ok] = rec(w[tags][ok]) / max(p99, 1e-12)
            out[s] = v
        elif spec["tipo"] in ("zmax",):
            med = base[tags].median(); mad = (base[tags] - med).abs().median() * 1.4826
            mad = mad.replace(0, np.nan)
            out[s] = ((w[tags] - med) / mad).abs().max(axis=1).to_numpy()
        elif spec["tipo"] == "spread":
            # divergencia de cada mancal contra a mediana dos irmaos, o pior deles
            sp = pd.concat([(base[c] - base[[x for x in tags if x != c]].median(axis=1))
                            for c in tags], axis=1)
            med = sp.median(); mad = (sp - med).abs().median() * 1.4826
            spw = pd.concat([(w[c] - w[[x for x in tags if x != c]].median(axis=1))
                             for c in tags], axis=1)
            spw.columns = sp.columns = range(len(tags))
            out[s] = ((spw - med.to_numpy()) / mad.replace(0, np.nan).to_numpy()
                      ).abs().max(axis=1).to_numpy()
        elif spec["tipo"] == "spread_alvo":
            b = base[spec["alvo"]] - base[tags].median(axis=1)
            med = float(b.median()); mad = float((b - med).abs().median() * 1.4826)
            ww = w[spec["alvo"]] - w[tags].median(axis=1)
            out[s] = np.abs((ww - med) / max(mad, 1e-12)).to_numpy()
    return pd.Series(out, index=ix).ewm(halflife=pd.Timedelta("1h"), times=ix).mean(), len(tags)


if __name__ == "__main__":
    G = pd.read_parquet(PORTAL)
    ix = G.index
    op = (G["RUNNING_A"] > 0.5).fillna(False)
    est = (op & (G["T5_AVG_A"] > 300)).fillna(False)
    part = op & ~op.shift(fill_value=False)
    blk = part.rolling(int(6 * 30), min_periods=1).max().astype(bool)
    mask = (est & ~blk).to_numpy()

    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    trips = [t for t in fal if ix[0] + JAN <= t <= ix[-1]]
    print(f"serie: {ix[0]:%Y-%m-%d} .. {ix[-1]:%Y-%m-%d}   "
          f"{int(mask.sum())} instantes vigiados ({100*mask.mean():.0f}%)")
    print(f"trips cobertos: {len(trips)} de 8 -> "
          + ", ".join(f"{t:%d/%m}" for t in trips) + "\n")

    ti = np.asarray(ix.tz_convert("UTC").tz_localize(None)
                    .astype("datetime64[ns]").astype("int64"))
    jan_ns = int(JAN.value)
    eleg = ti[(ti >= ti[0] + jan_ns) & mask]
    sort = RNG.choice(eleg, size=(N_NULO, len(trips)))
    obs_t = np.asarray([int(pd.Timestamp(t).tz_localize(None).value) for t in trips])

    def picos(z, quando):
        lo = np.searchsorted(ti, quando - jan_ns, "left")
        hi = np.searchsorted(ti, quando, "right")
        return np.array([np.nanmax(z[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])

    print(f"{'canal':<26}{'tags':>5}{'duty':>8}{'pico trips':>12}{'nulo':>8}"
          f"{'razao':>8}{'p':>9}")
    print("-" * 80)
    for grupo in (CONTROLE, NOVAS):
        for nome, spec in grupo.items():
            s, nt = canal(G, est, spec)
            if s is None:
                print(f"{nome:<26}{'--':>5}   tags ausentes"); continue
            z = s.where(pd.Series(mask, index=ix)).to_numpy()
            duty = float(np.nanmean(z[mask] > np.nanpercentile(z[mask], 80)))
            o = float(np.nanmedian(picos(z, obs_t)))
            nul = np.array([np.nanmedian(picos(z, sort[k])) for k in range(N_NULO)])
            mn = float(np.nanmedian(nul))
            p = float(np.nanmean(nul >= o))
            flag = " ***" if p < 0.05 else ("  *" if p < 0.10 else "")
            print(f"{nome:<26}{nt:>5}{100*duty:>7.0f}%{o:>12.2f}{mn:>8.2f}"
                  f"{o/max(mn,1e-9):>7.2f}x{p:>9.4f}{flag}")
        print()
