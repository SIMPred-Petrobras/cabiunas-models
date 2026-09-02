#!/usr/bin/env python3
"""Quanto o NOSSO 1,03 FP/mes se mexe so trocando a convencao de medida.

A HIPOTESE (do Thallys, 30/08/2026): o "5/8 a 0,43 FP/mes" da Lara pode estar na metrica
do resultado ANTIGO do Francisco, nao na regua que usamos hoje. Se for isso, os dois
numeros nunca foram comparaveis e nao ha diferenca nenhuma a explicar.

COMO SE TESTA ISSO SEM O CODIGO DELES. Nao se testa o numero deles -- testa-se a
SENSIBILIDADE do nosso. Pego o nosso alarme, exatamente o mesmo, e reconto o custo sob
cada convencao plausivel. Se alguma delas levar o nosso 1,03 para perto de 0,43, entao a
diferenca entre os dois detectores pode ser inteiramente de contabilidade, e comparar os
dois numeros como estao seria erro nosso, nao deles.

O que muda de convencao para convencao (nada disto toca no detector):

  denominador -- mes de operacao QUENTE (RUNNING_A e T5>300, o nosso), mes de operacao
                 crua (so RUNNING_A), mes de relogio (calendario corrido). O nosso e o
                 mais severo dos tres: divide por menos horas, entao infla a taxa.
                 A serie v6 deles vinha com a coluna `scored` toda em 1, inclusive nas
                 linhas com RUNNING_A=0 -- isto e, o denominador de relogio.
  agrupamento -- gap que funde dois alarmes no mesmo episodio (2 h no nosso). Gap maior
                 = menos episodios = menos falso positivo, sem mudar uma amostra do
                 alarme.
  janela      -- horas antes do evento em que um alarme conta como deteccao em vez de
                 falso positivo (48 h no nosso). Janela maior reclassifica episodios de
                 falso positivo para deteccao.

CONTROLE. A linha "nossa regua publicada" tem que devolver 8/8 e 1,033 FP/mes.

RESSALVA. Isto NAO reconstroi o metodo deles e nao prova o que eles fizeram -- so mede
o tamanho do efeito de convencao no nosso proprio numero. Para comparar de verdade
continua sendo preciso passar a serie deles pela nossa regua.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from pos_processamento import partes, pos, mask, idx, alvo, g

KB, KV = 1.7, 2.2
GAPS_H = [2, 6, 12, 24]
JANELAS_H = [48, 72, 96]


def alarme():
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    return pos(v, ns, REFRAT_H, DUR_MIN, False)


def conta(al, horas_den, gap_h, janela_h):
    """FP por mes de 730 h, com gap de agrupamento e janela de deteccao livres."""
    eps = AV.episodios(al, gap_h=gap_h)
    jan = [(t - pd.Timedelta(hours=janela_h), t) for t in alvo]
    det = sum(1 for t0, t1 in jan
              if any(a <= t1 and b >= t0 for a, b in eps))
    fp = sum(1 for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jan))
    meses = horas_den / 730.0
    return det, len(eps), fp, fp / meses


if __name__ == "__main__":
    al = alarme()
    sel = idx >= alvo.min().normalize().replace(day=1)          # so para o relogio

    op_cru = (g["RUNNING_A"] > 0.5).fillna(False)
    h_quente = float(mask.sum()) * 2 / 60                        # nosso: RUNNING_A e T5>300
    h_op = float((op_cru & (idx >= idx[mask][0])).sum()) * 2 / 60
    span = (idx[mask][-1] - idx[mask][0]).total_seconds() / 3600
    print(f"denominadores disponiveis na mesma janela de dados:")
    print(f"  operacao QUENTE (RUNNING_A e T5>300) : {h_quente:8,.0f} h = "
          f"{h_quente/730:5.1f} meses   <- o nosso")
    print(f"  operacao crua   (so RUNNING_A)       : {h_op:8,.0f} h = "
          f"{h_op/730:5.1f} meses")
    print(f"  relogio         (calendario corrido) : {span:8,.0f} h = "
          f"{span/730:5.1f} meses")

    b = conta(al, h_quente, 2.0, 48.0)
    print(f"\ncontrole (nossa regua publicada): {b[0]}/8, {b[3]:.3f} FP/mes  "
          f"(esperado 8/8, 1,033)")

    print("\n" + "=" * 92)
    print("O MESMO ALARME, RECONTADO SOB CADA CONVENCAO")
    print("=" * 92)
    print(f"{'denominador':>16} {'gap':>5} {'janela':>7} | {'det':>5} {'eps':>5} "
          f"{'FP':>4} {'FP/mes':>8}")
    lin = []
    for nome, h in (("quente (nosso)", h_quente), ("operacao crua", h_op),
                    ("relogio", span)):
        for gap in GAPS_H:
            for jh in JANELAS_H:
                d, e, f, r = conta(al, h, float(gap), float(jh))
                mark = "  <- nossa regua" if (nome.startswith("quente") and gap == 2
                                              and jh == 48) else ""
                print(f"{nome:>16} {gap:>4}h {jh:>6}h | {d:>3}/8 {e:>5} {f:>4} "
                      f"{r:>8.3f}{mark}")
                lin.append(dict(denominador=nome, gap_h=gap, janela_h=jh,
                                det=d, eps=e, fp=f, fp_mes=round(r, 3)))
    T = pd.DataFrame(lin); T.to_csv("denominador.csv", index=False)

    print("\n" + "=" * 92)
    print("VEREDITO")
    print("=" * 92)
    nosso = T[(T.denominador == "quente (nosso)") & (T.gap_h == 2) & (T.janela_h == 48)]
    r0 = float(nosso.fp_mes.iloc[0])
    oito = T[T.det == 8]
    print(f"  nossa regua                        : 8/8 a {r0:.3f} FP/mes")
    print(f"  mesmo alarme, convencao mais branda: {oito.det.iloc[oito.fp_mes.argmin()]}/8 "
          f"a {oito.fp_mes.min():.3f} FP/mes")
    print(f"  faixa do MESMO alarme so trocando convencao: "
          f"{T.fp_mes.min():.3f} a {T.fp_mes.max():.3f} FP/mes "
          f"({T.fp_mes.max()/T.fp_mes.min():.1f}x)")
    alvo_ref = 0.43
    perto = T[(T.fp_mes - alvo_ref).abs() < 0.08]
    print(f"\n  convencoes em que o NOSSO alarme mede perto de {alvo_ref} FP/mes: "
          f"{len(perto)}")
    if len(perto):
        print(perto[["denominador", "gap_h", "janela_h", "det", "fp", "fp_mes"]]
              .to_string(index=False))
        print(f"\n  -> nestas convencoes o nosso detector mede "
              f"{perto.det.max()}/8 a ~{alvo_ref} FP/mes, contra 5/8 a 0,43 do numero")
        print(f"     citado. Se a comparacao estiver sendo feita entre reguas diferentes,")
        print(f"     ela nao mede detector nenhum -- mede contabilidade.")
    else:
        print(f"  -> nenhuma convencao aproxima o nosso numero de {alvo_ref}; a diferenca")
        print(f"     nao e de denominador e precisa de outra explicacao.")
    print("\n-> denominador.csv")
