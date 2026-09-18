#!/usr/bin/env python3
"""O pico em FIT_POINTS = 20.000 e um plato estreito ou pura sorte?

Duas hipoteses ja foram testadas e refutadas para explicar por que 20.000 e tao
melhor que 8.000, 14.000, 30.000 e 45.000 (6 a 10x menos horas de alarme falso):

  1. instabilidade do estimador de escala -> `drift_desacopla.py`: amostra maior
     deu escala 500x MAIS instavel, nao menos. Premissa caiu.
  2. tempo de coerencia do regime -> `drift_coerencia.py`: o baseline nao e
     homogeneo em tamanho nenhum, e nao ha minimo em 20.000.

Sobra a terceira, e e a mais incomoda: **e coincidencia**. Dois treinos de
configuracao IDENTICA ja diferem 20,7 pontos percentuais de recall
([[piso-de-ruido-retreino]]); se essa e a variancia do proprio ato de treinar, a
diferenca entre tamanhos de janela pode estar inteira dentro dela.

O TESTE QUE SEPARA as duas leituras: perturbar POUCO. Se 20.000 e um plato local,
19.000 e 21.000 -- 5% de diferenca, meio dia de operacao a mais ou a menos --
devolvem resultado parecido. Se e sorte, uma perturbacao minuscula ja derruba.

Uso:  PYTHONPATH=. python drift_vizinhanca_fit.py
"""
from __future__ import annotations
import numpy as np
from drift_baseline import walkforward, roda, met
import avalia as AV
from pos_processamento import mask, alvo
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
import pandas as pd
_par = paradas_reais_2h(); _J = pd.Timedelta(hours=48)


def colhe(fin):
    eps = AV.episodios(fin); d = AV.avalia(fin, alvo, mask)
    cls = classifica_regra_c(eps, _par)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    hfp = sum((b - a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    mes = d["horas_op"]/730.0
    ban = sum(1 for t in alvo if any(t-_J <= a <= t-pd.Timedelta(hours=4) for a, _ in eps))
    return ban, d["det"], nfp/mes, hfp/mes

if __name__ == "__main__":
    print("VIZINHANCA IMEDIATA DE FIT_POINTS = 20.000 (+-5% e +-10%)")
    print("Plato -> resultados parecidos. Sorte -> desaba com pouco.\n")
    COLHIDO = []
    for fp in (18_000, 18_500, 19_000, 19_500, 20_000, 20_500, 21_000, 21_500, 22_000):
        t, p, ms, ds, _ = walkforward(fp)
        d = 100 * (fp / 20_000 - 1)
        rot = f"{fp:>6} pts ({d:+.0f}%)" + ("  <- atual" if fp == 20_000 else "")
        fin = roda(t, p, ms, ds)
        met(fin, rot)
        COLHIDO.append(colhe(fin))

    # ── a barra de erro que o numero publicado nao tem ──
    B = np.array([c[0] for c in COLHIDO]); D = np.array([c[1] for c in COLHIDO])
    F = np.array([c[2] for c in COLHIDO]); H = np.array([c[3] for c in COLHIDO])
    print("\n" + "=" * 78)
    print("DISTRIBUICAO sobre calibracoes EQUIVALENTES (+-10% no baseline)")
    print("=" * 78)
    print(f"  banda   : mediana {np.median(B):.0f}/8   faixa {B.min()}/8 a {B.max()}/8"
          f"   (publicado: 5/8)")
    print(f"  deteccao: mediana {np.median(D):.0f}/8   faixa {D.min()}/8 a {D.max()}/8"
          f"   (publicado: 8/8)")
    print(f"  FP/mes  : mediana {np.median(F):.3f}  faixa {F.min():.3f} a {F.max():.3f}"
          f"   (publicado: 0,344)")
    print(f"  h/mes   : mediana {np.median(H):.1f}   faixa {H.min():.1f} a {H.max():.1f}"
          f"   (publicado: 6,6)")
    print(f"\n  o ponto publicado e o MELHOR de {len(B)} equivalentes em "
          f"{int((H <= H.min() + 1e-9).sum())} de {len(H)} -- nao a mediana.")
