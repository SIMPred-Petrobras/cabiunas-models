#!/usr/bin/env python3
"""Detector fisico: maximo sobre canais especificos, com correcao de Sidak.

Por que maximo e nao media -- o relatorio do Francisco ja mostrou o mecanismo:
o PDIT_0305 respondia por 95% do erro antes da falha de selagem e mesmo assim
nao havia alerta, porque o score da familia e a media sobre 12 sensores. Um
sensor gritando vira sussurro depois de dividido por 12. A autopsia dos 9
eventos mostra que isso nao e excecao: cada mecanismo aparece em UM canal
(PDI_0302 em 29/04, vib_351 em 11/04, dT mancal-oleo em 09/12).

O preco do maximo e falso positivo: com M canais a p99.9, a chance de algum
cruzar por acaso e ~M x 0.1%. Sidak paga esse preco explicitamente -- cada
canal e cortado em (1-alpha)^(1/M), de modo que a taxa CONJUNTA fique em alpha.
Nao e um limiar escolhido a mao.

Protocolo temporal identico ao do Francisco: baseline movel das ultimas HORAS_BASE
horas de operacao quente estritamente anteriores ao mes avaliado, refeito a cada
mes, excluindo os 7 dias que antecedem cada falha JA OCORRIDA.
"""
from __future__ import annotations
import numpy as np, pandas as pd

MANC = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0305", "954005_624_TI_0307"]
TC = [f"TC382_0{i}_A" for i in range(1, 7)]
OLEO = "954005_624_TI_0325"
PAS = pd.Timedelta("2min")

HORAS_BASE = 400.0
POS_PARTIDA_H = 2.0      # descarta as 2 h seguintes a cada religamento


def mascara_quente(g: pd.DataFrame) -> pd.Series:
    """Operacao quente estavel: maquina ligada, queimando, e fora do transiente
    de partida. Sem isso qualquer alerta logo apos religamento e artefato."""
    on = (g.RUNNING_A > 0.99) & (g["T5_AVG_A"] > 300)
    on = on.fillna(False)
    # tempo desde o ultimo religamento
    grupo = (on != on.shift()).cumsum()
    desde = on.groupby(grupo).cumcount() * PAS
    return on & (desde >= pd.Timedelta(hours=POS_PARTIDA_H))


def canais_nivel(g: pd.DataFrame) -> pd.DataFrame:
    """Lista fixa de grandezas de diagnostico de maquina rotativa. Escolhida
    por fisica, nao por ajuste: cada uma corresponde a um modo de falha
    documentado (mancal, selagem, oleo, combustao, rotor)."""
    m = g[MANC]
    med = m.median(axis=1)
    c = pd.DataFrame(index=g.index)
    # mancal: desvio contra os irmaos e geracao de calor sobre o oleo
    for t in MANC:
        c[f"spread_{t[-4:]}"] = g[t] - med          # cada mancal contra os irmaos
    c["dT_manc_oleo"] = m.max(axis=1) - g[OLEO]
    c["oleo_T"] = g[OLEO]
    # combustao: pattern factor do array de exaustao
    c["T5_spread"] = g[TC].max(axis=1) - g[TC].min(axis=1)
    c["T5_avg"] = g["T5_AVG_A"]
    for t in TC:                                     # cada termopar contra o array
        c[f"t5r_{t[5:7]}"] = g[t] - g[TC].median(axis=1)
    # selagem
    c["selagem"] = g["954005_624_PDIT_0305"]
    # oleo e filtros
    for t in ["954005_624_PI_0307", "954005_624_PI_0308",
              "954005_624_PDI_0301", "954005_624_PDI_0302",
              "954005_624_PDI_0317", "954005_624_PDI_0338"]:
        c[t.split("_", 2)[2]] = g[t]
    # rotor: cada mancal de vibracao, X e Y
    for t in [x for x in g.columns if x.startswith("TV_")]:
        c[f"vib_{t[3:7]}"] = g[t]
    # processo (contexto, nao degradacao -- entram para nao serem confundidos)
    c["gas_0315"] = g["954005_624_PI_0315"]
    return c


def canais_multiescala(niv: pd.DataFrame, escalas=("1h", "4h", "24h")) -> pd.DataFrame:
    """Dinamica do Diego: a mesma grandeza vista em varias escalas de tempo.
    Uma deriva lenta que nunca sai da faixa em nivel aparece como tendencia
    sustentada em 24 h; um transiente aparece em 1 h."""
    out = {}
    for e in escalas:
        n = int(pd.Timedelta(e) / PAS)
        r = niv.rolling(n, min_periods=max(4, n // 3))
        out.update({f"{c}|sd{e}": v for c, v in r.std().items()})
        # tendencia = variacao ao longo da janela (proporcional a inclinacao)
        d = niv.diff(n)
        out.update({f"{c}|tr{e}": v for c, v in d.items()})
    return pd.DataFrame(out, index=niv.index)


def z_causal(X: pd.DataFrame, quente: pd.Series, falhas: pd.Series,
             horas_base=HORAS_BASE, excl_dias=7.0) -> pd.DataFrame:
    """z robusto contra baseline movel refeito mes a mes, so com o passado."""
    idx = X.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    n_base = int(horas_base * 60 / 2)
    ok = quente.copy()
    Z = pd.DataFrame(np.nan, index=idx, columns=X.columns, dtype="float32")
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        elig = ok & (idx < m0)
        for f in falhas[falhas < m0]:
            elig &= ~((idx >= f - pd.Timedelta(days=excl_dias)) & (idx <= f + pd.Timedelta(days=2)))
        base = X[elig].tail(n_base)
        if len(base) < n_base // 4:
            continue
        med = base.median()
        mad = (base - med).abs().median() * 1.4826
        mad = mad.replace(0.0, np.nan).fillna(base.std().replace(0.0, np.nan))
        sel = (idx >= m0) & (idx < m1)
        Z.loc[sel] = ((X[sel] - med) / mad).values.astype("float32")
    return Z.abs()          # bilateral: desvio para qualquer lado e anomalia


def razao_max(Z: pd.DataFrame, quente: pd.Series, falhas: pd.Series,
              alpha: float, horas_base=HORAS_BASE, excl_dias=7.0,
              devolve_canal=False):
    """Corte por canal tal que a taxa de cruzamento CONJUNTA fique em alpha
    (Sidak: p_canal = (1-alpha)^(1/M)), e devolve so o maximo de Z/limiar.

    Devolver a razao em vez do limiar completo evita materializar uma matriz do
    tamanho de Z de novo -- sao 230 canais x 612 mil instantes.
    """
    M = Z.shape[1]
    p = (1.0 - alpha) ** (1.0 / M)
    idx = Z.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    n_base = int(horas_base * 60 / 2)
    razao = pd.Series(np.nan, index=idx, dtype="float32")
    culpado = pd.Series(pd.NA, index=idx, dtype="object")
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        elig = quente & (idx < m0)
        for f in falhas[falhas < m0]:
            elig &= ~((idx >= f - pd.Timedelta(days=excl_dias)) & (idx <= f + pd.Timedelta(days=2)))
        base = Z[elig].tail(n_base)
        if len(base) < n_base // 4:
            continue
        q = base.quantile(p).replace(0.0, np.nan)
        sel = (idx >= m0) & (idx < m1)
        r = Z[sel] / q
        razao.loc[sel] = r.max(axis=1).values.astype("float32")
        if devolve_canal:
            v = r.to_numpy()
            todo_na = np.isnan(v).all(axis=1)
            j = np.where(todo_na, -1, np.nanargmax(np.where(np.isnan(v), -np.inf, v), axis=1))
            culpado.loc[sel] = np.where(todo_na, None, np.asarray(r.columns)[j])
    return (razao, culpado) if devolve_canal else razao
