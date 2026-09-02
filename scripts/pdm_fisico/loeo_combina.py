#!/usr/bin/env python3
"""LOEO das tres familias (nosso / dele / combinacao) no orcamento de EPISODIOS.

Por que refazer. A primeira versao deste teste orcou o custo em HORAS de alarme, e com
isso deu 8/9 para o detector do Diego contra 6/9 para o nosso -- resultado invertido em
relacao ao duelo. O motivo e que hora e a moeda barata dele: o ponto que ganhou (p99,5,
sustentacao 2 min) custa 5,95 h/mes mas 337 EPISODIOS (17,67 por mes). Orcar em horas
autoriza 337 interrupcoes desde que cada uma seja curta. O operador atende episodio, nao
hora. Este script refaz o LOEO com o orcamento na moeda certa, e tambem com as duas
restricoes simultaneas.

Otimizacao necessaria: a versao ingenua recalcula A.avalia para cada configuracao a cada
evento retirado (~3.200 varreduras da serie) e estoura o tempo. Aqui os episodios de
cada configuracao sao extraidos UMA vez; a cada dobra so se recontam quais episodios sao
FP em relacao ao conjunto de eventos reduzido.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_VIB
from auto_reset import trunca

PAS = pd.Timedelta("2min")
JAN_CONF = 6.0
JAN_DET = 48.0


def dil(al, h):
    n = int(pd.Timedelta(hours=h) / PAS)
    return al.astype(int).rolling(2 * n + 1, min_periods=1, center=True).max().astype(bool)


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    esc = pd.read_parquet("escore_diego_iforest_estatico.parquet")["escore"].reindex(idx)
    out = roda(BRACO, df, falhas)
    meses_op = mask.sum() * 2 / 60 / 730.0

    NOSSO, DELE, COMB = {}, {}, {}
    for k in [1.3, 1.7, 2.1, 2.5, 3.0, 3.5, 4.0, 5.0]:
        an = alerta_2k(out, mask, k, K_VIB)
        NOSSO[f"k={k}"] = an
        NOSSO[f"k={k}t"] = trunca(an, 12)
    for pct in [99.0, 99.5, 99.9, 99.95, 99.99]:
        lim = np.nanpercentile(esc.where(mask).dropna(), pct)
        ac = (esc > lim).where(mask, False).fillna(False)
        for s in [2, 10, 30, 60]:
            ad = A.sustenta(ac, s) & mask
            DELE[f"p{pct}/{s}"] = ad
            for kn, aln in NOSSO.items():
                COMB[f"CONF-d {kn} p{pct}/{s}"] = ad & dil(aln, JAN_CONF)
                COMB[f"INTER {kn} p{pct}/{s}"] = aln & ad
    FAM = {"nosso": NOSSO, "dele": DELE, "combinacao": COMB}
    print(f"configuracoes: " + ", ".join(f"{f}={len(a)}" for f, a in FAM.items()), flush=True)

    # episodios e instantes de alerta, extraidos uma unica vez por configuracao
    EPS, PRIM = {}, {}
    for f, als in FAM.items():
        for nome, al in als.items():
            eps = A.episodios(al)
            EPS[(f, nome)] = eps
            # primeiro alerta dentro de [t-48h, t] para cada evento, pre-calculado
            PRIM[(f, nome)] = {t: any(a <= t and b >= t - pd.Timedelta(hours=JAN_DET)
                                      for a, b in eps) for t in falhas}
        print(f"  {f}: episodios extraidos", flush=True)

    def custo(chave, evs):
        eps = EPS[chave]
        jw = [(t - pd.Timedelta(hours=JAN_DET), t) for t in evs]
        fp = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
        h = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
        return len(fp) / meses_op, h / meses_op

    def det(chave, evs):
        return sum(PRIM[chave][t] for t in evs)

    def loeo(f, teto_ep, teto_h=1e9):
        ac, esc_ = 0, []
        for t in falhas:
            resto = [x for x in falhas if x != t]
            melhor = None
            for nome in FAM[f]:
                ch = (f, nome)
                fp, h = custo(ch, resto)
                if fp <= teto_ep and h <= teto_h:
                    chave_ord = (det(ch, resto), -fp)
                    if melhor is None or chave_ord > melhor[1]:
                        melhor = (nome, chave_ord)
            if melhor is None:
                continue
            esc_.append(melhor[0])
            ac += PRIM[(f, melhor[0])][t]
        moda = pd.Series(esc_).mode().iloc[0] if esc_ else "-"
        return ac, moda

    print("\n" + "=" * 90)
    print("LOEO no orcamento de EPISODIOS (a moeda que o operador atende)")
    print("=" * 90)
    print(f"{'orcamento':>22} {'nosso':>8} {'dele':>8} {'combinacao':>12}   ponto tipico")
    for te in [4.5, 6.0, 10.0, 20.0]:
        r, pt = {}, {}
        for f in FAM:
            r[f], pt[f] = loeo(f, te)
        print(f"{'<= ' + str(te) + ' FP/mes':>22} {str(r['nosso']) + '/9':>8} "
              f"{str(r['dele']) + '/9':>8} {str(r['combinacao']) + '/9':>12}   "
              f"nosso={pt['nosso']} | dele={pt['dele']} | comb={pt['combinacao']}")

    print("\n" + "=" * 90)
    print("LOEO com as DUAS restricoes -- episodios E horas (o cenario de operacao real)")
    print("=" * 90)
    for te, th in [(4.5, 30.0), (4.5, 10.0), (4.5, 5.0), (6.0, 30.0)]:
        r = {}
        for f in FAM:
            r[f], _ = loeo(f, te, th)
        print(f"  <= {te:4.1f} FP/mes  E  <= {th:5.1f} h/mes :  nosso {r['nosso']}/9   "
              f"dele {r['dele']}/9   combinacao {r['combinacao']}/9")


if __name__ == "__main__":
    main()
