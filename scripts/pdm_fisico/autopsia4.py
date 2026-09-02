#!/usr/bin/env python3
"""Autopsia dos 4 casos em que o stack do Diego detecta e o nosso nao, COM mascara viva.

Sao os unicos casos acionaveis que sairam do cruzamento dos 28 episodios de nivel de
trip: episodios onde a nossa mascara esta ligada em parte da janela de +-24 h -- portanto
tinhamos direito de alarmar -- e mesmo assim ficamos calados enquanto ele marcou algo.

  05/11/2025 02:42  oleo    mascara viva  3,6 h   ele: preditivo
  24/11/2025 22:49  mancal  mascara viva 10,5 h   ele: preditivo
  30/11/2025 20:50  oleo    mascara viva  1,4 h   ele: reativo
  23/01/2026 21:20  oleo    mascara viva 20,4 h   ele: preditivo

A pergunta nao e "erramos?" -- e ONDE a cadeia parou. Sao quatro pontos de falha
distintos e cada um pede uma correcao diferente:

  (a) o sinal nem subiu               -> o fenomeno nao aparece nos nossos 4 sinais
  (b) subiu mas ficou abaixo do k     -> problema de sensibilidade, k resolve
  (c) passou o k mas nao sustentou 30 min -> a janela de sustentacao e longa demais
  (d) sustentou mas so 1 sinal votou  -> o voto >=2 e o gargalo
  (e) tudo aconteceu fora da mascara  -> ausencia de dominio, nada a corrigir

Para cada caso reporta, por sinal: o pico do EWMA dentro da mascara na janela, a razao
pico/limiar, quantos minutos ficou acima do limiar, e o maximo de votos simultaneos.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB

JAN = pd.Timedelta(hours=24)
CASOS = [("05/11/2025 02:42", pd.Timestamp("2025-11-05 02:42", tz="UTC"), "oleo"),
         ("24/11/2025 22:49", pd.Timestamp("2025-11-24 22:49", tz="UTC"), "mancal"),
         ("30/11/2025 20:50", pd.Timestamp("2025-11-30 20:50", tz="UTC"), "oleo"),
         ("23/01/2026 21:20", pd.Timestamp("2026-01-23 21:20", tz="UTC"), "oleo")]
LIM = {"t": ("1h", DET.THR_FAM * K_BASE), "p": ("1h", DET.THR_FAM * K_BASE),
       "sp": ("30min", DET.THR_SPREAD * K_BASE), "vb": ("30min", 3.0 * K_VIB)}


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    g = pd.read_parquet("grade2min.parquet")
    out = roda(BRACO, df, falhas)
    esc = pd.read_parquet("escore_diego_iforest_estatico.parquet")["escore"].reindex(idx)
    lim_d = np.nanpercentile(esc.where(mask).dropna(), 99.5)

    E = {c: out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
         for c, (hl, _) in LIM.items()}
    sust = {c: DET._sustained(E[c], thr) for c, (_, thr) in LIM.items()}
    votos = sum(sust[c].astype(int) for c in LIM)

    for rot, t, subs in CASOS:
        a, b = t - JAN, t + JAN
        m = mask.loc[a:b]
        h = m.sum() * 2 / 60
        print("=" * 92)
        print(f"{rot}  ({subs})   mascara viva {h:.1f} h de 48 h")
        if h == 0:
            print("   (sem dominio, nada a analisar)\n"); continue
        janela_op = m[m].index
        print(f"   trechos pontuaveis: {janela_op[0]:%d/%m %H:%M} a {janela_op[-1]:%d/%m %H:%M}")
        w = g.loc[a:b]
        print(f"   T5 na janela: min {w['T5_AVG_A'].min():.0f}C  max {w['T5_AVG_A'].max():.0f}C   "
              f"RUNNING liga/desliga: {int((w['RUNNING_A']>0.5).astype(int).diff().abs().sum())}x")

        print(f"\n   {'sinal':>6} {'pico EWMA':>10} {'limiar':>8} {'pico/lim':>9} "
              f"{'min acima':>10} {'sustentou?':>11}")
        for c, (hl, thr) in LIM.items():
            s = E[c].loc[a:b].dropna()
            if not len(s):
                print(f"   {c:>6} {'sem dado':>10}"); continue
            acima = int((s > thr).sum()) * 2
            print(f"   {c:>6} {s.max():10.2f} {thr:8.2f} {s.max()/thr:9.2f} {acima:8d} min "
                  f"{'SIM' if sust[c].loc[a:b].any() else 'nao':>11}")
        vmax = int(votos.loc[a:b].max())
        print(f"\n   maximo de sinais sustentados simultaneamente: {vmax} (precisa de 2)")
        sd = esc.loc[a:b].where(mask.loc[a:b])
        print(f"   escore dele: pico {sd.max():.4f}  limiar {lim_d:.4f}  "
              f"pico/limiar {sd.max()/lim_d:.2f}  minutos acima: {int((sd>lim_d).sum())*2}")

        # diagnostico
        picos = {c: E[c].loc[a:b].max() / LIM[c][1] for c in LIM if E[c].loc[a:b].notna().any()}
        top = sorted(picos.items(), key=lambda kv: -kv[1])[:2]
        if vmax >= 2:
            causa = "(d) dois sinais sustentaram mas o episodio nao entrou na janela avaliada"
        elif vmax == 1:
            causa = f"(d) VOTO: so {top[0][0]} sustentou; o 2o mais alto foi {top[1][0]} a {top[1][1]:.2f}x do limiar"
        elif max(picos.values()) >= 1.0:
            causa = f"(c) SUSTENTACAO: {top[0][0]} passou o limiar ({top[0][1]:.2f}x) mas nao ficou 30 min"
        elif max(picos.values()) >= 0.75:
            causa = f"(b) SENSIBILIDADE: {top[0][0]} chegou a {top[0][1]:.2f}x do limiar"
        else:
            causa = f"(a) SEM SINAL: o maior foi {top[0][0]} a {top[0][1]:.2f}x do limiar"
        print(f"   -> {causa}\n")


if __name__ == "__main__":
    main()
