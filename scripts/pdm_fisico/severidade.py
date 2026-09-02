#!/usr/bin/env python3
"""Alarme em DOIS NIVEIS: da para separar o que merece acao do que merece registro?

POR QUE ESTE E O EIXO QUE SOBROU. O custo do detector bateu num piso duro de 1,033
FP/mes com 8/8 por tres caminhos independentes -- geometria (nao_decaimento.py),
percentil (autocalibra.py) e pos-processamento (pos_processamento.py, 720 pontos).
Nenhum deles desce dai. Mas todos supunham a mesma entrega: um alarme BINARIO. Se os
episodios puderem ser ORDENADOS por gravidade, a entrega muda de natureza -- um nivel
ALTA que a operacao atende e um nivel BAIXA que so vira registro. O 8/8 tem que caber
inteiro na ALTA; o ganho e quantos falsos positivos caem para BAIXA.

Isso nao muda o detector: muda o que ele entrega. E a unica melhoria possivel que nao
depende de sinal novo nem de rotulo novo.

O RISCO, QUE AQUI E O PROBLEMA CENTRAL. Sao 23 episodios e 8 positivos. Escolher um
corte de gravidade olhando para os 23 e depois reportar quantos FP ele derruba e
sobreajuste pura e simplesmente -- foi exatamente assim que o voto entre sondas deu 8/9
em amostra e 3/9 no LOEO. Entao aqui NADA e reportado em amostra sem o par honesto:
para cada evento, o corte e escolhido nos OUTROS SETE e aplicado a ele. O numero que
vale e o LOEO; o numero em amostra so aparece do lado dele para mostrar o tamanho do
sobreajuste.

CANDIDATOS A GRAVIDADE (todos adimensionais, nenhum depende de unidade da tag):
  n_max      -- maior numero de sinais simultaneos no episodio (a moeda do voto)
  margem     -- pico de max_c E_c/limiar_c: quanto o pior sinal passou do limite
  soma       -- pico da soma_c max(0, E_c/limiar_c - 1): quanto TODOS passaram juntos
  cusum      -- pico do maior CUSUM acumulado: evidencia integrada, o canal lento
  mancal     -- pico de max(E_sp/thr, E_vb/thr): so o subsistema do mancal
  largura    -- horas ate o episodio atingir o seu n_max (rapido = transiente?)
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import (SIN, BASE, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN)
from blackout_curto import cusum
from pos_processamento import partes, pos, mask, idx, alvo, EW, reset

KB, KV = 1.7, 2.2
K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
JAN = pd.Timedelta(hours=48)
FEATS = ["n_max", "margem", "soma", "cusum", "mancal", "largura"]


def monta():
    ON = partes(KB, KV)
    n_sin = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(n_sin >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, n_sin, REFRAT_H, DUR_MIN, False)

    R = {c: (EW[c].where(mask) / (BASE[c] * K[c])) for c in SIN}          # razao ao limiar
    CU = {c: pd.Series(cusum(((R[c]).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset), index=idx) for c in SIN}
    soma = sum((R[c] - 1).clip(lower=0).fillna(0) for c in SIN)
    marg = pd.concat([R[c] for c in SIN], axis=1).max(axis=1)
    manc = pd.concat([R["sp"], R["vb"]], axis=1).max(axis=1)
    cumx = pd.concat([CU[c] for c in SIN], axis=1).max(axis=1)

    jw = [(t - JAN, t) for t in alvo]
    lin = []
    for a, b in AV.episodios(al):
        w = slice(a, b)
        alvo_ = [t for t0, t1 in jw for t in [t1] if a <= t1 and b >= t0]
        nm = n_sin.loc[w]
        pico = nm.idxmax()
        lin.append(dict(
            ini=a, fim=b, dur=(b - a).total_seconds() / 3600,
            positivo=len(alvo_) > 0,
            n_max=int(nm.max()),
            margem=float(marg.loc[w].max()),
            soma=float(soma.loc[w].max()),
            cusum=float(cumx.loc[w].max()),
            mancal=float(manc.loc[w].max()),
            largura=float((pico - a).total_seconds() / 3600),
        ))
    return al, pd.DataFrame(lin)


def melhor_corte(T, f):
    """Menor numero de FP na ALTA que ainda retem TODOS os positivos de T."""
    pos_ = T[T.positivo][f]
    if not len(pos_):
        return -np.inf, 0
    c = pos_.min()                              # o corte mais alto que retem todos
    return c, int((T[~T.positivo][f] >= c).sum())


def loeo(T, f):
    """Para cada positivo: corte escolhido nos OUTROS positivos, aplicado a ele."""
    P = T[T.positivo]
    retidos, fps = 0, []
    for i in P.index:
        c, _ = melhor_corte(T.drop(index=i), f)
        retidos += int(T.loc[i, f] >= c)
        fps.append(int((T[~T.positivo][f] >= c).sum()))
    return retidos, len(P), float(np.mean(fps))


if __name__ == "__main__":
    al, T = monta()
    b = AV.avalia(al, alvo, mask)
    meses = b["horas_op"] / 730.0
    print(f"controle: {b['det']}/8, {b['fp_mes']:.2f} FP/mes, {b['h_fp_mes']:.1f} h/mes "
          f"(esperado 8/8, 1,03, 38,7)")
    print(f"  {len(T)} episodios, {int(T.positivo.sum())} positivos, "
          f"{int((~T.positivo).sum())} falsos positivos em {meses:.1f} meses de operacao\n")
    T.to_csv("severidade_episodios.csv", index=False)

    print("=" * 100)
    print("SEPARACAO POR CANDIDATO: em amostra (otimista) contra LOEO (honesto)")
    print("=" * 100)
    print(f"{'gravidade':>10} | {'em amostra: FP na ALTA':>24} {'taxa':>9} | "
          f"{'LOEO: positivos retidos':>24} {'FP medio':>9} {'taxa':>9}")
    n_fp = int((~T.positivo).sum())
    for f in FEATS:
        c, fp_in = melhor_corte(T, f)
        ret, tot, fp_lo = loeo(T, f)
        print(f"{f:>10} | {fp_in:>7d} de {n_fp:<14d} {fp_in/meses:8.2f}/mes | "
              f"{ret:>7d} de {tot:<14d} {fp_lo:8.1f} {fp_lo/meses:8.2f}/mes")

    print("\n" + "=" * 100)
    print("OS EPISODIOS, ORDENADOS PELA MELHOR GRAVIDADE")
    print("=" * 100)
    ret_por_f = {f: loeo(T, f) for f in FEATS}
    # a melhor e a que retem mais positivos no LOEO; empate desfeito por menos FP
    f = min(FEATS, key=lambda x: (-ret_por_f[x][0], ret_por_f[x][2]))
    print(f"(melhor no LOEO: {f})\n")
    S = T.sort_values(f, ascending=False)
    print(f"{'inicio':>17} {'dur':>7} {'tipo':>10} " +
          "".join(f"{x:>9}" for x in FEATS))
    for _, r in S.iterrows():
        print(f"{r.ini:%d/%m/%Y %H:%M} {r.dur:6.1f}h "
              f"{'POSITIVO' if r.positivo else 'falso pos.':>10} " +
              "".join(f"{r[x]:9.2f}" for x in FEATS))

    ret, tot, fp_lo = ret_por_f[f]
    print(f"\n  veredito ({f}): LOEO retem {ret}/{tot} positivos na ALTA a "
          f"{fp_lo/meses:.2f} FP/mes")
    print(f"  contra o alarme binario de hoje: 8/8 a {b['fp_mes']:.2f} FP/mes")
    if ret < tot:
        print(f"  -> a ALTA perde {tot-ret} positivo(s) no LOEO: NAO adotar como porta;"
              f" serve como ORDENACAO, nao como corte")
    print("\n-> severidade_episodios.csv")
