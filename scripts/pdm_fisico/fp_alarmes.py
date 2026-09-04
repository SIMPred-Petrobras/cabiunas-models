#!/usr/bin/env python3
"""Os falsos positivos do nosso detector coincidem com alarme registrado?

Motivo. Os 88 episodios do detector sao classificados contra UM alvo: as 9 paradas reais.
Tudo que nao cai numa janela de 48 h antes de parada vira "falso positivo" por definicao.
Mas o catalogo do SCADA tem 47 tags e 3.757 onsets -- se uma parte desses episodios cai em
cima de alarme registrado, eles nao sao falso alarme: sao anomalia real que simplesmente
nao terminou em parada. E a mesma analise que o EXP10 do Diego fez do lado dele.

O CONTROLE E O PONTO CENTRAL. Sao ~2,4 alarmes por dia no catalogo inteiro. Um episodio de
12 h tem chance alta de conter alarme por puro acaso. Sem nulo, "60% dos FP tem alarme" nao
significa nada. O nulo aqui: mantem os alarmes onde estao e RESSORTEIA os episodios --
mesma quantidade, mesma duracao, inicio uniforme sobre os instantes pontuaveis. Responde
"dados N episodios desta duracao dentro da operacao, quantos conteriam alarme por acaso?".

Reporta tambem o enriquecimento POR TAG, que e a parte acionavel: se os FP concentram
TAH_6240305 (temperatura alta de mancal) ou os TV_* (vibracao), o detector esta pegando
evento de mancal real que nao virou parada -- exatamente os sinais que ele monitora.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

CACHE = pathlib.Path.home() / ".clearml/cache/storage_manager/datasets"
ALARMES = CACHE / "ds_d4c284df665e465d8492afd368837c8f/alarmes_selecionados_turbina_a.csv"
JAN_DET = 48.0
N_NULO = 2000
RNG = np.random.default_rng(0)


def catalogo(idx):
    a = pd.read_csv(ALARMES, low_memory=False)
    a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce").dt.tz_localize("UTC")
    a = a[a.t.notna() & a["Status"].astype(str).str.startswith("ACT")]
    a = a[(a.t >= idx[0]) & (a.t <= idx[-1])]
    return a.sort_values("t").reset_index(drop=True)


def conta(eps, ts):
    """Quantos episodios contem ao menos um alarme, e o total de alarmes contidos."""
    tv = ts.values
    com, tot = 0, 0
    for a, b in eps:
        n = int(((tv >= np.datetime64(a)) & (tv <= np.datetime64(b))).sum())
        com += n > 0
        tot += n
    return com, tot


def nulo(duracoes, pontuaveis, ts, n=N_NULO):
    """Reamostra episodios de mesma duracao, inicio uniforme sobre instantes pontuaveis."""
    tv = ts.values.astype("datetime64[ns]")
    pv = pontuaveis.values.astype("datetime64[ns]")
    durs = np.array([np.timedelta64(int(d.total_seconds() * 1e9), "ns") for d in duracoes])
    coms, tots = np.empty(n), np.empty(n)
    for i in range(n):
        s = pv[RNG.integers(0, len(pv), len(durs))]
        e = s + durs
        ii = np.searchsorted(tv, s, "left")
        jj = np.searchsorted(tv, e, "right")
        c = jj - ii
        coms[i] = (c > 0).sum(); tots[i] = c.sum()
    return coms, tots


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    al = pd.read_csv(ALARMES, low_memory=False)
    cat = catalogo(idx)
    print(f"catalogo no periodo da serie: {len(cat)} onsets ACT, "
          f"{cat['Tag Alarme'].nunique()} tags, {len(cat)/((idx[-1]-idx[0]).days):.2f}/dia\n")

    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask, K_BASE, K_VIB)
    pontuaveis = pd.Series(idx[mask.values], name="t")
    jw = [(t - pd.Timedelta(hours=JAN_DET), t) for t in falhas]

    for rot, alerta in [("sem teto", base), ("com teto de 12 h", trunca(base, 12))]:
        eps = A.episodios(alerta)
        tp = [(a, b) for a, b in eps if any(a <= t1 and b >= t0 for t0, t1 in jw)]
        fp = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
        print("=" * 88)
        print(f"{rot}: {len(eps)} episodios = {len(tp)} perto de parada + {len(fp)} 'falsos positivos'")
        h = sum((b - a).total_seconds() / 3600 for a, b in fp)
        print(f"  horas de FP: {h:.0f}   duracao mediana: "
              f"{np.median([(b-a).total_seconds()/3600 for a,b in fp]):.1f} h")

        com, tot = conta(fp, cat.t)
        durs = [b - a for a, b in fp]
        cn, tn = nulo(durs, pontuaveis, cat.t)
        p_com = (cn >= com).mean()
        print(f"\n  FP com ao menos um alarme dentro: {com}/{len(fp)} = {100*com/len(fp):.1f}%")
        print(f"     nulo (mesma duracao, posicao sorteada): "
              f"{cn.mean():.1f} +- {cn.std():.1f}  ({100*cn.mean()/len(fp):.1f}%)")
        print(f"     p = {p_com:.4f}")
        print(f"  total de alarmes contidos nos FP: {tot}")
        print(f"     nulo: {tn.mean():.1f} +- {tn.std():.1f}   p = {(tn>=tot).mean():.4f}")

        # enriquecimento por tag
        dentro = []
        for a, b in fp:
            m = (cat.t >= a) & (cat.t <= b)
            dentro.append(cat[m])
        D = pd.concat(dentro) if dentro else cat.iloc[0:0]
        obs = D["Tag Alarme"].value_counts()
        # nulo por tag: esperado proporcional a densidade global x horas de FP
        span_h = (idx[-1] - idx[0]).total_seconds() / 3600
        esp = cat["Tag Alarme"].value_counts() * (h / span_h)
        tab = pd.DataFrame({"observado": obs, "esperado_ingenuo": esp}).fillna(0)
        tab = tab[tab.observado >= 3].assign(razao=lambda d: d.observado / d.esperado_ingenuo.replace(0, np.nan))
        print(f"\n  tags mais presentes nos FP (>=3 ocorrencias), ordenado por razao obs/esp:")
        print(tab.sort_values("razao", ascending=False).head(12).round(2).to_string())
        print()


if __name__ == "__main__":
    main()
