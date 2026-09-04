#!/usr/bin/env python3
"""Cobertura dos dois detectores no conjunto COMPLETO de protecao de nivel, por subsistema.

A Secao 18 do EXP10c usa 3 tags -- PALL_6240340, TALL_6240325, PALL_6240309 -- todos do
oleo lubrificante, todos do lado BAIXO da protecao (ALL/LL = "Muito Baixa"). A mesma
convencao do lado ALTO (AHH/HH = "Muito Alta") tem mais 4 tags que ficaram de fora:
TAHH_6240305 (73 onsets), TAHH_6240303 (27), PDAHH6240305 (7), TAHH_6240307 (3).

Isso importa porque 7 dos 11 trips da NOSSA verdade estao nesses 4 tags ausentes
(TAHH_6240305 cinco vezes, PDAHH6240305 duas). Os 20 episodios dele medem a cobertura de
um subsistema que NENHUM dos dois modelos observa; mancal e selagem -- que ambos observam
via TI_030*/TV_35* -- nao entram.

Este script refaz a analise com os 7 tags, separando por subsistema, porque as perguntas
sao diferentes:
  oleo    -> nenhum dos dois tem sensor do subsistema. Deteccao seria eco indireto.
  mancal  -> os dois tem sensor direto. Aqui se espera precursor fisico.
  selagem -> pressao diferencial de selo; nos temos as 12 tags de pressao, ele nao.

Regua: a dele (+-24 h, preditivo / reativo / sem deteccao), para comparabilidade com a
Tabela 13. Reporta tambem quantas horas da janela cada detector consegue pontuar -- sem
isso um "sem deteccao" com a maquina fria e lido como erro do modelo, quando e ausencia
de dominio (ver nosso_nos_trips.py).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca
import fp_alarmes as FA

OOS = pd.Timestamp("2025-07-01", tz="UTC")
JAN = pd.Timedelta(hours=24)
SUB = {"PALL_6240340": "oleo", "TALL_6240325": "oleo", "PALL_6240309": "oleo",
       "TAHH_6240305": "mancal", "TAHH_6240303": "mancal", "TAHH_6240307": "mancal",
       "PDAHH6240305": "selagem"}


def declust(ts, gap_min=30):
    ts = sorted(ts); eps = []; ini = fim = ts[0]; n = 1
    for x in ts[1:]:
        if (x - fim).total_seconds() / 60 <= gap_min:
            fim = x; n += 1
        else:
            eps.append((ini, fim, n)); ini = fim = x; n = 1
    eps.append((ini, fim, n)); return eps


def categoria(al, t):
    j = al.loc[t - JAN: t + JAN]
    on = j[j.fillna(False)]
    if not len(on):
        return "sem deteccao", np.nan
    antes = on[on.index < t]
    if len(antes):
        return "preditivo", (t - antes.index[0]).total_seconds() / 3600
    return "reativo", (on.index[0] - t).total_seconds() / 3600


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    g = pd.read_parquet("grade2min.parquet")
    run = (g["RUNNING_A"] > 0.5).fillna(False); t5 = g["T5_AVG_A"]
    cat = FA.catalogo(idx)

    seg = (pd.DataFrame({"on": run, "gr": run.ne(run.shift()).cumsum()}).reset_index()
           .groupby("gr").agg(on=("on", "first"), ini=("ts", "first"), fim=("ts", "last")))
    seg["dur_h"] = (seg.fim - seg.ini).dt.total_seconds() / 3600
    par = seg[(~seg.on) & (seg.dur_h >= 2)]

    sub = cat[cat["Tag Alarme"].isin(SUB) & (cat.t >= OOS)]
    eps = declust(list(sub.t))
    print(f"episodios de nivel de trip no OOS (7 tags, gap 30 min): {len(eps)}\n", flush=True)

    out = roda(BRACO, df, falhas)
    nosso = alerta_2k(out, mask, K_BASE, K_VIB)
    esc = pd.read_parquet("escore_diego_iforest_estatico.parquet")["escore"].reindex(idx)
    lim = np.nanpercentile(esc.where(mask).dropna(), 99.5)
    dele = A.sustenta((esc > lim).where(mask, False).fillna(False), 2) & mask

    linhas = []
    for a, b, n in eps:
        tags = sorted(cat[(cat.t >= a) & (cat.t <= b) & cat["Tag Alarme"].isin(SUB)]["Tag Alarme"].unique())
        subs = sorted({SUB[t] for t in tags})
        j = run.loc[a - pd.Timedelta("30min"): a + pd.Timedelta("30min")]
        estado = "parada" if not j.any() else ("operando" if j.all() else "transicao")
        m = (par.ini >= a - pd.Timedelta("1h")) & (par.ini <= a + pd.Timedelta("30min"))
        cn, ln = categoria(nosso, a); cd, ld = categoria(dele, a)
        linhas.append(dict(ini=a, onsets=n, subsistema="+".join(subs), tags=",".join(tags),
                           estado=estado, t5=float(t5.loc[a-pd.Timedelta("10min"):a+pd.Timedelta("10min")].median()),
                           parada_real=bool(m.any()),
                           dur_h=float(par[m].dur_h.iloc[0]) if m.any() else np.nan,
                           h_mask=float(mask.loc[a-JAN:a+JAN].sum()*2/60),
                           nosso=cn, lead_nosso=ln, dele=cd, lead_dele=ld))
    T = pd.DataFrame(linhas); T.to_csv("trips_completos.csv", index=False)

    print(f"{'#':>3} {'inicio':>16} {'subsist':>9} {'estado':>10} {'T5':>6} {'parada':>8} "
          f"{'mask':>7} {'NOSSO':>13} {'DELE':>13}")
    for i, r in T.iterrows():
        pr = f"{r.dur_h:.0f}h" if r.parada_real else "-"
        print(f"{i+1:3d} {r.ini:%d/%m/%Y %H:%M} {r.subsistema:>9} {r.estado:>10} {r.t5:5.0f}C "
              f"{pr:>8} {r.h_mask:5.1f}h {r.nosso:>13} {r.dele:>13}")

    print("\n" + "=" * 88)
    print("COBERTURA POR SUBSISTEMA (regua dele, +-24 h)")
    print("=" * 88)
    for filtro, rot in [(T.index >= 0, "TODOS os 28 episodios"),
                        (T.parada_real, "so os que sao PARADA REAL >=2h")]:
        S = T[filtro]
        print(f"\n--- {rot}: n={len(S)} ---")
        print(f"{'subsistema':>10} {'n':>4} {'parada':>7} {'op.':>5} | "
              f"{'NOSSO pred/reat/nada':>22} | {'DELE pred/reat/nada':>21}")
        for s in ["oleo", "mancal", "selagem", "mancal+selagem"]:
            X = S[S.subsistema == s]
            if not len(X): continue
            f = lambda col: (int((X[col]=="preditivo").sum()), int((X[col]=="reativo").sum()),
                             int((X[col]=="sem deteccao").sum()))
            print(f"{s:>10} {len(X):4d} {int(X.parada_real.sum()):7d} "
                  f"{int((X.estado!='parada').sum()):5d} | "
                  f"{'%d / %d / %d' % f('nosso'):>22} | {'%d / %d / %d' % f('dele'):>21}")


if __name__ == "__main__":
    main()
