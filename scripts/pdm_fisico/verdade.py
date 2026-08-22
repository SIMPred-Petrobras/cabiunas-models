#!/usr/bin/env python3
"""Rotulo de falha pela regra do Francisco, com o fuso do arquivo de alarme
PROVADO em vez de assumido.

Regra (secao 4 do RELATORIO_DETECCAO_TC33003A):
  1. transicao RUNNING_A 1->0
  2. a parada durou >= 2 h
  3. havia alarme de nivel entre 1 h antes e 30 min depois
  4. trips a menos de 24 h contam como um evento
"""
import numpy as np, pandas as pd, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.normpath(os.path.join(HERE, "../../../../../../../dados"))
if not os.path.isdir(DADOS):
    DADOS = "/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/dados"

NIVEL = re.compile(r"TRIP|Mt\.?\s?Alta|M\.?\s?Alta|Mt\.?\s?Bx|M\.?\s?Bx|Mt\.?\s?Baixa|M\.?\s?Baixa", re.I)


def carrega_alarmes(offset_h: float) -> pd.DataFrame:
    a = pd.read_csv(os.path.join(DADOS, "alarmes_selecionados_turbina_a.csv"), low_memory=False)
    t = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
    a = a.assign(ts=t.dt.tz_localize("UTC") + pd.Timedelta(hours=offset_h)).dropna(subset=["ts"])
    a = a[a["Status"].astype(str).str.startswith("ACT")]      # so ativacao, nao normalizacao
    a["nivel"] = a["Descrição Alarme"].astype(str).str.contains(NIVEL)
    return a[["ts", "Tag Alarme", "Descrição Alarme", "nivel"]].sort_values("ts")


def paradas(run: pd.Series, min_h=2.0) -> pd.DataFrame:
    """Transicoes ligado->desligado cuja parada durou min_h ou mais."""
    r = run.dropna()
    on = (r > 0.5).to_numpy()
    idx = r.index
    corte = np.flatnonzero(on[1:] != on[:-1]) + 1
    ini = np.concatenate(([0], corte)); fim = np.concatenate((corte, [len(on)]))
    out = []
    for a, b in zip(ini, fim):
        if on[a]:
            continue
        dur = (idx[b - 1] - idx[a]).total_seconds() / 3600.0
        if a > 0 and dur >= min_h:
            out.append(dict(queda=idx[a], religa=idx[b] if b < len(idx) else pd.NaT, dur_h=dur))
    return pd.DataFrame(out)


def main():
    g = pd.read_parquet(os.path.join(HERE, "grade2min.parquet"), columns=["RUNNING_A"])
    par = paradas(g["RUNNING_A"])
    print(f"paradas >= 2 h: {len(par)}")

    # --- prova do fuso: qual offset alinha alarme de nivel com queda da maquina
    print("\noffset_h  paradas_com_alarme_de_nivel_em_[-1h,+30min]")
    melhor, best = None, -1
    for off in [-3, -2, -1, 0, 1, 2, 3]:
        al = carrega_alarmes(off)
        niv = pd.DatetimeIndex(al.loc[al.nivel, "ts"])
        n = 0
        for q in par["queda"]:
            lo = q - pd.Timedelta(hours=1); hi = q + pd.Timedelta(minutes=30)
            if ((niv >= lo) & (niv <= hi)).any():
                n += 1
        flag = ""
        if n > best:
            best, melhor, flag = n, off, ""
        print(f"  {off:+d}h     {n}")
    print(f"\n-> offset escolhido: {melhor:+d} h  ({best} trips)")

    al = carrega_alarmes(melhor)
    niv = pd.DatetimeIndex(al.loc[al.nivel, "ts"])
    trips = []
    for _, row in par.iterrows():
        q = row["queda"]
        lo = q - pd.Timedelta(hours=1); hi = q + pd.Timedelta(minutes=30)
        m = (niv >= lo) & (niv <= hi)
        if m.any():
            tags = al.loc[al.nivel].loc[m, "Descrição Alarme"].unique()[:3]
            trips.append(dict(queda=q, dur_h=row["dur_h"], alarmes=" | ".join(tags)))
    tr = pd.DataFrame(trips)

    # agrupa trips a menos de 24 h
    ev, ult = [], None
    for _, r in tr.iterrows():
        if ult is not None and (r["queda"] - ult) < pd.Timedelta(hours=24):
            ev[-1]["repeticoes"] += 1
            ev[-1]["dur_h"] = max(ev[-1]["dur_h"], r["dur_h"])
        else:
            ev.append(dict(evento=r["queda"], dur_h=r["dur_h"], repeticoes=0, alarmes=r["alarmes"]))
        ult = r["queda"]
    ev = pd.DataFrame(ev)
    ev.to_csv(os.path.join(HERE, "falhas.csv"), index=False)
    print(f"\ntrips: {len(tr)}   eventos apos agrupar 24 h: {len(ev)}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(ev.to_string(index=False))
    print("\npor semestre:")
    print(ev.groupby(ev["evento"].dt.to_period("Q")).size().to_string())


main()
