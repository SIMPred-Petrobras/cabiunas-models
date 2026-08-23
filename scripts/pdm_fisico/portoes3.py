#!/usr/bin/env python3
"""O portao de volatilidade bate um portao BURRO de mesma cobertura?

portoes2.py mostrou o numero que decide: a 0,15 um o portao bloqueia 14,2% do tempo
pontuavel e remove so 5,9% das horas de alarme. Se o portao estivesse mirando falso
positivo, removeria MAIS que a sua propria cobertura, nao menos. Mas 'menos que a
cobertura' ainda pode ser melhor que nada -- precisa de um nulo.

Nulo correto: deslocar circularmente a serie de volatilidade. Preserva a duracao e a
autocorrelacao dos blocos bloqueados (um portao real bloqueia trechos contiguos, nao
pontos soltos) e destroi so a relacao temporal com o alerta. Se o portao verdadeiro
nao remove mais horas que os deslocados, ele nao esta mirando nada -- e um cortador
de tempo com um nome bonito.

Mede-se tambem quantos dos 9 eventos o deslocado mata, para nao trocar deteccao por
horas sem perceber.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import indice_volatilidade, K_BASE, K_VIB

N_DESL = 200
LIMIARES = [0.12, 0.15, 0.18, 0.22]


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    vol = indice_volatilidade(g).reindex(idx)

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask, K_BASE, K_VIB)
    b0 = A.avalia(base, falhas, mask)
    print(f"\nbase: {b0['det']}/9 eventos, {b0['episodios']} episodios, "
          f"{b0['h_fp_mes']:.1f} h de alarme/mes\n")

    rng = np.random.default_rng(0)
    n = len(idx)
    v = vol.to_numpy()
    print(f"{'limiar':>7} {'cob%':>6} | {'h/mes real':>11} {'det':>4} | "
          f"{'h/mes nulo (media)':>19} {'IC95 do nulo':>18} {'det nulo':>9} | {'p':>6}")
    linhas = []
    for lim in LIMIARES:
        blq = (vol > lim).fillna(False)
        cob = 100 * (blq & mask).sum() / mask.sum()
        real = A.avalia(base & ~blq, falhas, mask)

        hs, ds = [], []
        for _ in range(N_DESL):
            k = int(rng.integers(1, n))
            bd = pd.Series(np.roll(v, k) > lim, index=idx).fillna(False)
            x = A.avalia(base & ~bd, falhas, mask)
            hs.append(x["h_fp_mes"]); ds.append(x["det"])
        hs = np.array(hs); ds = np.array(ds)
        # p: fracao dos deslocados que removem TANTA hora quanto o real (ou mais)
        p = float((hs <= real["h_fp_mes"]).mean())
        print(f"{lim:7.2f} {cob:6.1f} | {real['h_fp_mes']:11.1f} {real['det']:4d} | "
              f"{hs.mean():19.1f} {'[%.1f, %.1f]'%(np.percentile(hs,2.5),np.percentile(hs,97.5)):>18} "
              f"{ds.mean():9.2f} | {p:6.3f}")
        linhas.append(dict(limiar=lim, cobertura=cob, h_real=real["h_fp_mes"], det_real=real["det"],
                           eps_real=real["episodios"], h_nulo=hs.mean(),
                           h_nulo_lo=np.percentile(hs, 2.5), h_nulo_hi=np.percentile(hs, 97.5),
                           det_nulo=ds.mean(), p=p))
    pd.DataFrame(linhas).to_csv("portoes3.csv", index=False)
    print("\np = fracao dos 200 deslocamentos que removem tantas horas quanto o portao real.")
    print("p alto => um portao aleatorio de mesma cobertura faz igual ou melhor.")


main()
