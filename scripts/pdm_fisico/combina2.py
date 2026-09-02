#!/usr/bin/env python3
"""Combinar os dois detectores: uniao, intersecao e confirmacao -- a CUSTO IGUALADO.

O que ja se sabia (ensemble.py): a uniao dos dois da 9/9, o unico 9/9 da investigacao.
Mas o p de permutacao PIORA de 0,00035 (nosso sozinho, 8/9) para 0,0136 (uniao, 9/9),
porque a uniao cobre 334 episodios contra 88. Recall comprado com cobertura nao e
deteccao -- e o mesmo mecanismo que fez o 8/9 do stack dele nas nossas paradas ser
p=0,085 a 439 episodios.

A pergunta certa nao e "a uniao acerta mais?" (acerta, por construcao), e sim:

  IGUALANDO O CUSTO, alguma combinacao bate o melhor detector sozinho?

Tres formas de combinar, todas medidas no mesmo custo:

  UNIAO         alerta se qualquer um dos dois disparar. Aumenta recall e custo.
  INTERSECAO    alerta so quando os dois concordam no mesmo instante. Corta custo
                agressivamente -- e o braco que nunca foi testado. Faz sentido a
                priori porque os custos dos dois sao de tipos diferentes: o nosso e
                caro em HORAS (106,7 h/mes) e barato em episodios (88); o dele e o
                inverso (5,8 h/mes, 329 episodios). A intersecao poderia herdar o
                lado bom de cada um.
  CONFIRMACAO   alerta nosso que tenha alerta dele em +-JAN h em volta (e vice-versa).
                Intersecao com tolerancia temporal, porque exigir simultaneidade ao
                minuto e severo demais para dois detectores com dinamicas diferentes.

Protocolo: varre o k do nosso e o (percentil, sustentacao) do dele, monta as tres
combinacoes, e so entao compara no custo do ponto de operacao atual (4,35 FP/mes).
Permutacao rodada apenas nos candidatos que sobrevivem ao filtro de custo -- ela e cara
e nao faz sentido em ponto que ja perdeu por custo.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

PAS = pd.Timedelta("2min")
KS_NOSSO = [1.3, 1.7, 2.1, 2.5, 3.0]
PCTS = [99.0, 99.5, 99.9]
SUSTS = [2, 10, 30]
JAN_CONF = 6.0          # horas de tolerancia da confirmacao


def dilata(al, horas):
    """Alarme dilatado +-horas, para a regra de confirmacao."""
    n = int(pd.Timedelta(hours=horas) / PAS)
    return al.astype(int).rolling(2 * n + 1, min_periods=1, center=True).max().astype(bool)


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    esc = pd.read_parquet("escore_diego_iforest_estatico.parquet")["escore"].reindex(idx)

    out = roda(BRACO, df, falhas)
    nossos = {k: alerta_2k(out, mask, k, K_VIB) for k in KS_NOSSO}
    nossos.update({f"{k}t": trunca(nossos[k], 12) for k in KS_NOSSO})

    deles = {}
    for pct in PCTS:
        lim = np.nanpercentile(esc.where(mask).dropna(), pct)
        acima = (esc > lim).where(mask, False).fillna(False)
        for s in SUSTS:
            deles[(pct, s)] = A.sustenta(acima, s) & mask

    def mede(al, nome, com_p=False):
        x = A.avalia(al, falhas, mask)
        d = dict(nome=nome, det=x["det"], eps=x["episodios"], fp=x["fp_mes"],
                 h=x["h_fp_mes"], lead=x["lead_med"], quais=",".join(x["detectados"]))
        if com_p:
            d["p"] = A.permuta(al, mask, x["det"], len(falhas))["p"]
        return d

    linhas = []
    print("varrendo combinacoes ...", flush=True)
    for kn, aln in nossos.items():
        for (pct, s), ald in deles.items():
            base = f"nosso k={kn} x dele p{pct}/{s}min"
            linhas.append(dict(op="UNIAO", kn=kn, pct=pct, sust=s,
                               **mede(aln | ald, "UNIAO " + base)))
            linhas.append(dict(op="INTERSECAO", kn=kn, pct=pct, sust=s,
                               **mede(aln & ald, "INTER " + base)))
            linhas.append(dict(op="CONF nosso", kn=kn, pct=pct, sust=s,
                               **mede(aln & dilata(ald, JAN_CONF), "CONF-n " + base)))
            linhas.append(dict(op="CONF dele", kn=kn, pct=pct, sust=s,
                               **mede(ald & dilata(aln, JAN_CONF), "CONF-d " + base)))
    T = pd.DataFrame(linhas)

    # referencias sozinhas
    ref = [dict(op="SOZINHO", kn=k, pct=np.nan, sust=np.nan, **mede(nossos[k], f"nosso k={k}"))
           for k in nossos]
    ref += [dict(op="SOZINHO", kn=np.nan, pct=p_, sust=s_, **mede(deles[(p_, s_)], f"dele p{p_}/{s_}min"))
            for (p_, s_) in deles]
    T = pd.concat([T, pd.DataFrame(ref)], ignore_index=True)
    T.to_csv("combina2.csv", index=False)

    ALVO = 4.35
    print("\n" + "=" * 104)
    print(f"A CUSTO IGUALADO ({ALVO:.2f} FP/mes, o do ponto de operacao atual): o melhor de cada familia")
    print("=" * 104)
    cand = T[(T.fp <= ALVO * 1.25) & (T.fp >= ALVO * 0.75)].copy()
    print(f"candidatos dentro de +-25% do custo alvo: {len(cand)}\n")
    print(f"{'familia':12s} {'configuracao':46s} {'det':>6} {'eps':>6} {'FP/mes':>7} "
          f"{'h/mes':>7} {'lead':>6} {'p':>9}")
    melhores = []
    for op in ["SOZINHO", "UNIAO", "INTERSECAO", "CONF nosso", "CONF dele"]:
        g = cand[cand.op == op].sort_values(["det", "h"], ascending=[False, True])
        if not len(g):
            print(f"{op:12s} {'-- nenhum ponto dentro da faixa de custo --':46s}")
            continue
        for _, r in g.head(2).iterrows():
            al = None
            melhores.append((op, r))
    # permutacao so nos finalistas
    for op, r in melhores:
        kn, pct, s = r.kn, r.pct, r.sust
        if op == "SOZINHO":
            al = nossos[kn] if isinstance(kn, str) or not pd.isna(kn) else deles[(pct, int(s))]
        else:
            aln, ald = nossos[kn], deles[(pct, int(s))]
            al = {"UNIAO": aln | ald, "INTERSECAO": aln & ald,
                  "CONF nosso": aln & dilata(ald, JAN_CONF),
                  "CONF dele": ald & dilata(aln, JAN_CONF)}[op]
        x = A.avalia(al, falhas, mask)
        p = A.permuta(al, mask, x["det"], len(falhas))["p"]
        print(f"{op:12s} {r.nome[:46]:46s} {int(r.det):4d}/9 {int(r.eps):6d} {r.fp:7.2f} "
              f"{r.h:7.1f} {r.lead:6.1f} {p:9.4f}")
    print("\nCSV: combina2.csv")


if __name__ == "__main__":
    main()
