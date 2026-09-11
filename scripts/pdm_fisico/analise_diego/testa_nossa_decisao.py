#!/usr/bin/env python3
"""A NOSSA camada de decisao sobre os CANAIS DELE.

TESE (06/09/2026): ele acerta a parte dificil (perfil de antecedencia: leads de
3,8 a 43,2 h, todos numa faixa usavel) e erra a parte facil (custo: 2,88 FP/mes).
Nos o contrario. O caminho de maior retorno e juntar os dois.

A camada dele: voto >= N de 4  ->  duracao minima 45 min  ->  refratario 48 h.
Sem portao de canal obrigatorio, sem varredura de refratario, sem duracao maior.

A nossa peca estrutural que ele nao tem e o PORTAO: `voto >= 2 E (canal
qualificado aceso)` -- contagem MAIS um canal obrigatorio, em vez de contagem
pura. E a tabela 4 do relatorio dele diz onde isso deve morder: o canal de
alarme e o segundo voto em 27+26+6 = 59 dos 96 FP (61%). Se o portao exigir
DOIS canais MODELADOS, esses 59 caem.

Limitacao: os artefatos dele sao `is_anom_point`, BINARIOS. As pecas nossas que
precisam de magnitude (CUSUM, religamento, escalada) nao dao para aplicar aqui.
O que da: voto, portao, duracao e refratario.

Usa as funcoes de regua DELE, para a medida ser a mesma.
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cnn1d_ae.scoring import (
    apply_refractory, apply_min_duration_filter, group_alerts_into_episodes,
    classify_episodes_regua, compute_regua_metrics, compute_operational_period_days,
)

ALARM_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
ALARM_WIN_H = 24.0
CAT_ID = "a97ba56ba14840fbb1125c2a82f883c9"
CAT_FILE = "alarmes_selecionados_turbina_a.csv"
# `dataset_francisco_lara/alarmes_francisco_falhas.csv` e dado, nao esta na
# branch. Usamos a NOSSA lista de trips -- sao os mesmos 8 eventos que ele
# publica na Tabela 7 do relatorio de 04/09/2026, e o script confere isso.
FALHAS = "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models/scripts/pdm_fisico/falhas.csv"
CACHE = "canais/merged.parquet"


def canal_alarme(index, alarm_times, window_hours):
    times = np.sort(pd.DatetimeIndex(pd.Series(alarm_times).dropna()).values.astype("datetime64[ns]"))
    t_arr = index.values.astype("datetime64[ns]")
    out = np.zeros(len(t_arr), dtype=bool)
    if len(times) == 0:
        return pd.Series(out, index=index)
    pos = np.searchsorted(times, t_arr, side="right") - 1
    v = pos >= 0
    dt = (t_arr[v] - times[pos[v]]).astype("timedelta64[s]").astype(np.float64)/3600.0
    out[v] = dt <= float(window_hours)
    return pd.Series(out, index=index)


def carrega():
    if os.path.exists(CACHE):
        m = pd.read_parquet(CACHE)
        idx = m.index
        canais = {k: m[k].astype(bool) for k in ("temperatura", "vibracao", "oleo", "alarme")}
        op = m["operational_state"]
        f = pd.read_csv(FALHAS, parse_dates=["evento"])
        ft = pd.Series(pd.DatetimeIndex(f["evento"]).tz_convert(None)).sort_values()
        ft = ft[(ft >= idx.min()) & (ft <= idx.max())].reset_index(drop=True)
        return idx, canais, op, ft
    d = {}
    for nome in ("temperatura", "vibracao", "oleo"):
        x = pd.read_csv(f"canais/{nome}.csv", index_col=0, parse_dates=True, low_memory=False)
        d[nome] = x
    idx = d["temperatura"].index.intersection(d["vibracao"].index).intersection(d["oleo"].index)
    canais = {k: d[k].loc[idx, "is_anom_point"].astype(bool) for k in d}
    op = d["temperatura"].loc[idx, "operational_state"]
    from clearml import Dataset
    root = Dataset.get(dataset_id=CAT_ID).get_local_copy()
    al = pd.read_csv(os.path.join(root, CAT_FILE))
    al["t"] = pd.to_datetime(al["Data da Ocorrência"], errors="coerce")
    al = al[al["Status"].astype(str).str.startswith("ACT")].dropna(subset=["t"])
    canais["alarme"] = canal_alarme(idx, al.loc[al["Tag Alarme"].isin(ALARM_TAGS), "t"], ALARM_WIN_H)
    pd.DataFrame({**{k: v for k, v in canais.items()},
                  "operational_state": op}, index=idx).to_parquet(CACHE)
    f = pd.read_csv(FALHAS, parse_dates=["evento"])
    ft = pd.Series(pd.DatetimeIndex(f["evento"]).tz_convert(None)).sort_values()
    ft = ft[(ft >= idx.min()) & (ft <= idx.max())].reset_index(drop=True)
    return idx, canais, op, ft


def mede(dec, op, ft):
    df = pd.DataFrame({"is_anom_point": dec.astype(int), "operational_state": op})
    dias = compute_operational_period_days(df)
    eps = group_alerts_into_episodes(df["is_anom_point"], merge_gap_minutes=120.0)
    cls = classify_episodes_regua(eps, ft, df["operational_state"], 48.0, 48.0, 2.0)
    return compute_regua_metrics(cls, ft, dias), eps, cls


MODELADOS = ["temperatura", "vibracao", "oleo"]
PORTOES = {
    "nenhum (dele)":        lambda c: pd.Series(True, index=c["temperatura"].index),
    "exige vibracao":       lambda c: c["vibracao"],
    "exige mancal (T ou V)": lambda c: c["temperatura"] | c["vibracao"],
    ">=2 MODELADOS":        lambda c: (sum(c[k].astype(int) for k in MODELADOS) >= 2),
}

if __name__ == "__main__":
    idx, canais, op, ft = carrega()
    print(f"indice comum: {len(idx):,} amostras | {idx.min():%d/%m/%Y} a {idx.max():%d/%m/%Y}")
    print(f"falhas catalogadas: {len(ft)}\n")
    ns = sum(canais[k].astype(int) for k in canais)

    print("CONTROLE -- a camada DELE (voto>=N, dur 45min, refrat 48h)")
    print("=" * 84)
    for nv in (2, 3):
        v = (ns >= nv)
        vf = apply_min_duration_filter(pd.DataFrame({"is_anom_point": v.astype(int)}, index=idx),
                                       45.0)["is_anom_point"].astype(bool)
        dec = apply_refractory(vf, refractory_minutes=48*60.0)
        m, eps, cls = mede(dec, op, ft)
        print(f"  voto>={nv}: {m['falhas_detectadas']}/{m['n_falhas_catalogadas']}  "
              f"FP/mes {m['falso_positivo_por_mes']:.2f}  inconclusivo "
              f"{m['n_episodios_inconclusivo']}  (publicado: 8/8, 2,88, 25)")

    print("\n\nA NOSSA CAMADA SOBRE OS CANAIS DELE")
    print("=" * 110)
    print(f"{'portao':>22} {'voto':>5} {'dur':>6} {'refrat':>7} | {'det':>6} "
          f"{'FP/mes':>8} {'incon':>6} {'eps':>5}")
    print("-" * 110)
    linhas = []
    for nome, g in PORTOES.items():
        gate = g(canais)
        for nv, dur, rf in itertools.product((2, 3), (45.0, 90.0, 120.0, 180.0), (48, 72, 96, 120)):
            v = (ns >= nv) & gate
            vf = apply_min_duration_filter(pd.DataFrame({"is_anom_point": v.astype(int)}, index=idx),
                                           dur)["is_anom_point"].astype(bool)
            dec = apply_refractory(vf, refractory_minutes=rf*60.0)
            m, eps, cls = mede(dec, op, ft)
            linhas.append(dict(portao=nome, voto=nv, dur=dur, refrat=rf,
                               det=m["falhas_detectadas"], fp=m["falso_positivo_por_mes"],
                               incon=m["n_episodios_inconclusivo"], eps=len(eps)))
        print(f"  {nome} concluido", flush=True)
    D = pd.DataFrame(linhas); D.to_csv("nossa_decisao_nos_canais_dele.csv", index=False)

    print("\nMELHOR POR NIVEL DE DETECCAO")
    print("=" * 110)
    for k in sorted(D.det.unique(), reverse=True):
        s = D[D.det == k].sort_values("fp")
        b = s.iloc[0]
        print(f"  {int(k)}/8  menor FP/mes {b.fp:6.2f}  ({len(s):3d} configs)  "
              f"portao={b.portao} voto>={int(b.voto)} dur={int(b.dur)}min refrat={int(b.refrat)}h")
