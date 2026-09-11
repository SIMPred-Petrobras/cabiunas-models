#!/usr/bin/env python3
"""Deteccao em funcao da BANDA DE ACIONABILIDADE [tau_min, tau_max].

Os dois eventos que perdemos na regua de inicio nao sao deteccoes ausentes: sao
deteccoes CEDO DEMAIS para a janela de 48 h.
  17/03  precursor comeca 194,9 h antes
  29/04  precursor comeca  51,8 h antes -- perde a janela por 3,8 h

E o inverso vale do outro lado: uma deteccao a 1,3 h ou 2,8 h do trip conta pela
regua mas nao da tempo de agir. A regua de 48 h sem piso e cega para os dois
erros.

Compara os dois detectores nas mesmas bandas. Leads de inicio:
  nosso  (ponto com religamento + escalada)
  Diego  (Tabela 7 do relatorio de 04/09/2026)
"""
import numpy as np, pandas as pd

NOSSO = {"27/02/2025": 1.3, "17/03/2025": 194.9, "07/04/2025": 5.0,
         "11/04/2025": 2.8, "29/04/2025": 51.8, "04/11/2025": 8.8,
         "09/12/2025": 23.6, "26/02/2026": 20.2}
DIEGO = {"27/02/2025": 33.8, "17/03/2025": 31.2, "07/04/2025": 43.2,
         "11/04/2025": 19.8, "29/04/2025": 36.7, "04/11/2025": 13.7,
         "09/12/2025": 8.4, "26/02/2026": 3.8}

def conta(leads, lo, hi):
    return sum(1 for v in leads.values() if lo <= v <= hi)

print("DETECCOES POR BANDA [tau_min, tau_max]")
print("=" * 88)
TMAX = [48, 56, 72, 96, 168, 240]
for lo in [0, 2, 4, 6, 8, 12]:
    print(f"\n  tau_min = {lo} h" + ("   (sem piso -- a regua de hoje)" if lo == 0 else ""))
    print(f"    {'tau_max':>9} | {'nos':>6} {'Diego':>7}")
    for hi in TMAX:
        n, d = conta(NOSSO, lo, hi), conta(DIEGO, lo, hi)
        m = "  <-- nos ganhamos" if n > d else ("  <-- ele ganha" if d > n else "")
        print(f"    {hi:8d}h | {n:5d}/8 {d:6d}/8{m}")

print("\n\nONDE CADA UM VIVE  (distribuicao dos leads de inicio)")
print("=" * 88)
n = np.array(sorted(NOSSO.values())); d = np.array(sorted(DIEGO.values()))
print(f"  nos  : {', '.join(f'{v:.1f}' for v in n)}")
print(f"  Diego: {', '.join(f'{v:.1f}' for v in d)}")
print(f"\n  nossa mediana {np.median(n):.1f} h   dele {np.median(d):.1f} h")
print(f"  nossos leads abaixo de 4 h : {int((n < 4).sum())} de 8   dele {int((d < 4).sum())} de 8")
print(f"  nossos leads acima de 48 h : {int((n > 48).sum())} de 8   dele {int((d > 48).sum())} de 8")
print("\n  -> os nossos erros estao nas DUAS pontas; os dele, no meio.")
print("     ele nao tem nenhum lead longo e so um curto; nos temos 3 curtos e 2 longos.")
