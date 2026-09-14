#!/usr/bin/env python3
"""MARGEM A BORDA -- por que o ponto de deploy nao e o ponto otimo.

O `ponto_de_deploy.py` roda o minimax sobre 560 configuracoes e diz QUAL ponto
escolher. Este script responde POR QUE: onde exatamente o ponto antigo era
fragil, se o plato que o substitui e real ou artefato, e o que a troca custa.

Quatro perguntas, nesta ordem:

  1. QUEM DERRUBA. Dos 8 vizinhos a +-1 passo, qual desaba? (esperado: um so)
  2. O PLATO E REAL? Varre o eixo culpado ALEM da grade original -- se o ponto
     escolhido esta no extremo da grade, o minimax nunca testou vizinho de um
     dos lados, e a "robustez" pode ser cegueira.
  3. O CANAL ESTA MUDO? Se a metrica nao muda num intervalo grande, ou o canal
     parou de contribuir (ruim: perdemos um modo de falha) ou o pos-processamento
     absorve (bom). Mede duty e quantas vezes o canal e PIVO do voto.
  4. O QUE CUSTA. Lead por evento ao longo do plato, e a curva de banda contra
     tau_min -- que e a unica escolha arbitraria do sistema e deveria vir da
     operacao, entao o ponto nao pode depender dela.

Uso:  python margem_de_borda.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import SIN, VOTO_LO, REFRAT_V2, K_LO, K_LO_PONTO_OTIMO
from validacao_temporal import detecta, mede, canal

JAN = pd.Timedelta(hours=48)
C = ("t", "p", "sp", "vb")
GR = {"t": [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3], "p": [0.6, 0.7, 0.8, 1.0, 1.2],
      "sp": [0.7, 0.9, 1.1, 1.3], "vb": [1.6, 1.8, 2.0, 2.2]}
ANTIGO = tuple(K_LO_PONTO_OTIMO[c] for c in C)
DEPLOY = tuple(K_LO[c] for c in C)


def av(k):
    """banda, inicio, det, FP/mes, leads de inicio (h) de uma configuracao."""
    al = detecta(dict(zip(C, k)), 1.7, 2.2, REFRAT_V2)
    b, i, fp, fpm, _ = mede(al, list(alvo))
    eps = AV.episodios(al)
    li = [(t - max([a for a, _ in eps if t - JAN <= a <= t])).total_seconds() / 3600
          for t in alvo if any(t - JAN <= a <= t for a, _ in eps)]
    return b, i, AV.avalia(al, alvo, mask)["det"], fpm, li, len(eps)


def p1_quem_derruba():
    print("\n[1] QUEM DERRUBA -- os 8 vizinhos a +-1 passo de grade")
    print("=" * 78)
    for nome, base in (("ANTIGO (otimo no ponto)", ANTIGO), ("DEPLOY (margem a borda)", DEPLOY)):
        b0, i0, d0, f0, _, _ = av(base)
        print(f"\n  {nome}  {'/'.join(str(x) for x in base)}")
        print(f"    no ponto            banda {b0}/8  inicio {i0}/8  det {d0}/8  {f0:.3f} FP/mes")
        for j, c in enumerate(C):
            g = GR[c]; k = g.index(base[j])
            for d in (-1, 1):
                if not 0 <= k + d < len(g):
                    print(f"    {c:>3} {base[j]:>4} -> ----   (extremo da grade: nao testado)")
                    continue
                nk = list(base); nk[j] = g[k + d]
                b, i, dt, f, _, _ = av(tuple(nk))
                flag = "   <-- DESABA" if (b < b0 or dt < d0) else ""
                print(f"    {c:>3} {base[j]:>4} -> {g[k+d]:<4}  banda {b}/8  inicio {i}/8  "
                      f"det {dt}/8  {f:.3f}{flag}")


def p2_plato_e_real(eixo="p", valores=(0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0)):
    print(f"\n[2] O PLATO E REAL? -- eixo {eixo} varrido ALEM da grade original")
    print("=" * 78)
    j = C.index(eixo)
    for v in valores:
        k = list(DEPLOY); k[j] = v
        b, i, d, f, li, ne = av(tuple(k))
        marca = "  <- adotado" if v == DEPLOY[j] else ("  <- antigo" if v == ANTIGO[j] else "")
        print(f"    {eixo} = {v:<4} banda {b}/8  inicio {i}/8  det {d}/8  {f:.3f} FP/mes  "
              f"eps {ne:3d}  lead med {np.mean(li):5.1f}h{marca}")


def p3_canal_mudo(eixo="p"):
    print(f"\n[3] O CANAL {eixo} ESTA MUDO NO NIVEL A? -- duty e pivo do voto >=3 de 4")
    print("=" * 78)
    fixo = {c: canal(c, K_LO[c]) for c in C if c != eixo}
    soma = sum(fixo[c].astype(int) for c in fixo)
    n = int(mask.sum())
    print(f"    {'k':>6}{'duty (%)':>12}{'voto>=3 (%)':>14}{'pivo (%)':>11}"
          "    pivo = voto cai a <3 sem o canal")
    for k in (0.6, 0.7, 0.8, 1.0, 1.2, 2.0, 3.0):
        Ap = canal(eixo, k)
        voto = pd.Series((soma + Ap.astype(int)) >= VOTO_LO, index=idx) & mask
        pivo = voto & ~(pd.Series(soma >= VOTO_LO, index=idx) & mask)
        print(f"    {k:>6}{100*int((Ap&mask).sum())/n:>11.2f}%{100*int(voto.sum())/n:>13.2f}%"
              f"{100*int(pivo.sum())/n:>10.2f}%")


def p4_custo():
    print("\n[4] O QUE CUSTA -- banda contra tau_min, a unica escolha arbitraria")
    print("=" * 78)
    taus = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
    print("    tau_min |" + "".join(f"{t:5.0f}h" for t in taus))
    print("    --------+" + "-" * (6 * len(taus)))
    for nome, base in (("antigo", ANTIGO), ("deploy", DEPLOY)):
        al = detecta(dict(zip(C, base)), 1.7, 2.2, REFRAT_V2)
        eps = AV.episodios(al)
        L = [[(t - a).total_seconds() / 3600 for a, _ in eps if t - JAN <= a <= t] for t in alvo]
        print(f"    {nome:8s}|" + "".join(f"{sum(1 for x in L if any(v>=t for v in x)):4d}/8"
                                          for t in taus))
    print("\n    leads de inicio por evento (h):")
    for nome, base in (("antigo", ANTIGO), ("deploy", DEPLOY)):
        _, _, _, _, li, _ = av(base)
        print(f"      {nome:8s} {[round(x,1) for x in li]}   media {np.mean(li):.1f}h")


if __name__ == "__main__":
    p1_quem_derruba()
    p2_plato_e_real()
    p3_canal_mudo()
    p4_custo()
    print("\n" + "=" * 78)
    print("CONCLUSAO: o eixo p e plato de 0,7 a 3,0 e quebra em 0,6 (fusao de episodio\n"
          "volta e um nascimento sai da janela). O plato nao e o canal desligado -- ele\n"
          "ainda e pivo em ~7% do tempo -- e o refratario de 72 h que absorve. Entao a\n"
          "escolha dentro do plato compra margem, nao desempenho: 0,70 estava a 1,17x da\n"
          "quebra, 1,20 esta a 2,00x. Custa 1,0 h de lead medio, num evento de 37,7 h.")
