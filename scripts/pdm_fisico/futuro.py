#!/usr/bin/env python3
"""O detector daqui vai continuar valendo? Quatro medicoes, nenhuma opiniao.

A pergunta nasceu de uma premissa que precisa ser checada antes de respondida: "o nosso
resultado e melhor". Os dois detectores respondem alvos diferentes (o do Diego preve
alarme de temperatura; o daqui preve parada de maquina), entao 'melhor' so tem sentido
no mesmo alvo, na mesma janela, com o mesmo denominador. Item 1 faz essa comparacao.

Depois, tres coisas que decidem se o numero se sustenta no futuro:

  2. n EFETIVO. Sao 9 eventos, mas nao 9 mecanismos. Se seis deles sao o mesmo alarme
     de mancal, o detector foi validado contra ~3 modos de falha, nao 9 -- e a chance
     de um modo novo aparecer sem cobertura e muito maior do que 7/9 sugere.
  3. ESTACIONARIEDADE. As horas de alarme por mes de operacao sao 80 no treino e 140 no
     teste. Se isso for tendencia e nao ruido, o custo operacional do detector cresce
     sozinho com o tempo, sem ninguem mexer em nada.
  4. INTERVALO DE CONFIANCA. 7/9 com IC de Wilson diz quanto do numero e informacao e
     quanto e tamanho de amostra.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

OOS_DIEGO = pd.Timestamp("2025-07-01", tz="UTC")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def main():
    df = canonico()
    falhas_df = pd.read_csv("falhas.csv", parse_dates=["evento"])
    falhas = falhas_df["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask, K_BASE, K_VIB)
    teto = trunca(base, 12)

    # ---------------- 1. mesmo alvo, mesma janela, mesmo denominador
    print("\n" + "=" * 78)
    print("1. 'O NOSSO E MELHOR?' -- so vale no mesmo alvo/janela/denominador")
    print("=" * 78)
    ev_oos = falhas[falhas >= OOS_DIEGO]
    m_oos = mask & (idx >= OOS_DIEGO)
    for nome, al in [("4 sinais (sem teto)", base), ("4 sinais + teto 12h", teto)]:
        x = A.avalia(al[idx >= OOS_DIEGO], ev_oos, m_oos[idx >= OOS_DIEGO])
        print(f"  {nome:22s}: {x['det']}/{len(ev_oos)} paradas reais no OOS do Diego "
              f"(2025-07 -> 2026-04), {x['h_fp_mes']:.1f} h alarme/mes")
    print("  EXP10c (Diego)        : 2/2 paradas reais na mesma janela, ~1,7 h alarme/mes*")
    print("   * 0,35% de normal_alert_rate x ~6.031 h de operacao / 10 meses. Alvo dele e")
    print("     alarme de temperatura, nao parada -- os 2/2 sao subproduto (secao 18 dele).")
    print("\n  => nos dois casos n<=3. Nenhum dos dois numeros distingue 100% de 60%.")

    # ---------------- 2. n efetivo
    print("\n" + "=" * 78)
    print("2. n EFETIVO: 9 eventos, quantos mecanismos?")
    print("=" * 78)
    falhas_df["mec"] = falhas_df["alarmes"].str.split(r"\s*\|\s*").str[0].str.strip()
    cnt = falhas_df["mec"].value_counts()
    for m, c in cnt.items():
        print(f"  {c}x  {m}")
    print(f"\n  eventos = 9   mecanismos distintos = {len(cnt)}")
    print(f"  o mecanismo dominante responde por {cnt.iloc[0]}/9 dos eventos")

    # ---------------- 3. estacionariedade do custo
    print("\n" + "=" * 78)
    print("3. ESTACIONARIEDADE: o custo do detector cresce sozinho?")
    print("=" * 78)
    jan = [(t - pd.Timedelta(hours=48), t) for t in falhas]
    for nome, al in [("sem teto", base), ("teto 12h", teto)]:
        print(f"\n  --- {nome}")
        print(f"  {'semestre':>10} {'h operacao':>11} {'episodios FP':>13} {'FP/mes op':>10} "
              f"{'h alarme/mes':>13} {'h/episodio':>11}")
        linhas = []
        for per, g in pd.Series(idx, index=idx).groupby(pd.Grouper(freq="2QS")):
            if len(g) == 0:
                continue
            sel = (idx >= g.index[0]) & (idx <= g.index[-1])
            qm = (mask & sel)
            ho = qm.sum() * 2 / 60
            if ho < 300:
                continue
            eps = A.episodios(al & sel)
            fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan)]
            hfp = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
            mes = ho / 730
            print(f"  {str(g.index[0].date()):>10} {ho:11.0f} {len(fp):13d} {len(fp)/mes:10.2f} "
                  f"{hfp/mes:13.1f} {hfp/max(len(fp),1):11.1f}")
            linhas.append((g.index[0], len(fp)/mes, hfp/mes, hfp/max(len(fp),1)))
        if len(linhas) >= 4:
            t = np.arange(len(linhas))
            for j, rot in [(1, "FP/mes"), (2, "h alarme/mes"), (3, "h por episodio")]:
                y = np.array([l[j] for l in linhas])
                r = stats.spearmanr(t, y)
                print(f"    tendencia de {rot:15s}: rho={r.statistic:+.2f}  p={r.pvalue:.3f}")

    # ---------------- 4. IC
    print("\n" + "=" * 78)
    print("4. INTERVALO DE CONFIANCA (Wilson 95%)")
    print("=" * 78)
    for k, n, rot in [(7, 9, "LOEO, 9 eventos (2024-2026)"),
                      (7, 8, "LOEO, 8 eventos de 2025-2026"),
                      (8, 9, "ponto de operacao fixo, serie toda"),
                      (3, 3, "so a janela OOS do Diego"),
                      (2, 2, "EXP10c do Diego nas 2 paradas reais")]:
        lo, hi = wilson(k, n)
        print(f"  {rot:38s} {k}/{n} = {100*k/n:5.1f}%   IC95 [{100*lo:4.1f}%, {100*hi:5.1f}%]")
    print("\n  largura do IC de 7/9: %.0f pontos percentuais." % (100*(wilson(7,9)[1]-wilson(7,9)[0])))


main()
