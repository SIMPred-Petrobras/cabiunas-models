#!/usr/bin/env python3
"""O evento 2025-11-04 e invisivel por construcao? Varredura do blackout pos-partida.

Motivo. O EXP10c do Diego detecta o trip de 2025-11-04 com 11,4 h de antecedencia; o
nosso LOEO perde exatamente esse evento. Antes de atribuir isso a features (multi-
escala/textura), ha uma explicacao estrutural mais simples a descartar:

  - a campanha que terminou nesse trip comecou em 2025-11-03 15:04 e durou 15,3 h;
  - a deteccao do Diego cai em 2025-11-03 18:58, ou seja 3,9 h apos a partida;
  - a nossa mascara de pontuacao apaga as 6 h seguintes a cada religamento (BLACKOUT).

Se for isso, o nosso detector nao pode alarmar ali por design, e o problema nao e o
modelo -- e o mesmo tipo de limite estrutural que o Diego documenta na secao 12.4 dele
(deteccao so aparece quando o estado volta a "on"). Simetrico, e igualmente invisivel
para quem so olha a metrica agregada.

O blackout existe por um motivo medido: 48% dos nossos falsos positivos comecam nas
primeiras 30 h apos religamento (fp_rajadas.py). Encurtar tem preco. Este script mede
o preco em vez de supor.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, CORTE
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

HORAS = [0, 1, 2, 3, 4, 6, 8, 12]
ALVO = pd.Timestamp("2025-11-04 06:22", tz="UTC")


def mascara(df, black_h):
    op = df["in_operation"].astype(bool)
    st = df["stable"].astype(bool)
    if black_h <= 0:
        return st
    starts = op & ~op.shift(fill_value=False)
    n = int(pd.Timedelta(hours=black_h) / pd.Timedelta(C.GRID))
    black = starts.rolling(n, min_periods=1).max().astype(bool)
    return st & ~black


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index

    # quanto da campanha pre-trip sobra pontuavel em cada blackout
    st = df["stable"].astype(bool)
    print("=== campanha 2025-11-03 15:04 -> 2025-11-04 06:22 (15,3 h) ===")
    for h in HORAS:
        m = mascara(df, h)
        jan = m.loc["2025-11-03 15:00":"2025-11-04 06:22"]
        print(f"  blackout {h:>2} h -> {jan.sum()*2/60:5.1f} h pontuaveis dentro da campanha")

    print("\nmontando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)

    print(f"\n=== varredura do blackout (serie inteira, 9 eventos) ===")
    print(f"{'blackout':>9} {'h pontuav':>10} | {'det':>4} {'11-04?':>7} {'lead':>6} "
          f"{'eps':>5} {'FP/mes':>7} {'h/mes':>7} | {'com teto 12h: h/mes':>20} {'det':>4}")
    linhas = []
    for h in HORAS:
        m = mascara(df, h)
        al = alerta_2k(out, m, K_BASE, K_VIB)
        x = A.avalia(al, falhas, m); x.update(A.permuta(al, m, x["det"], len(falhas)))
        pega = "2025-11-04" in x["detectados"]
        lead = np.nan
        if pega:
            d = al.loc[ALVO - pd.Timedelta(hours=48):ALVO]
            d = d[d.fillna(False)]
            lead = (ALVO - d.index[0]).total_seconds() / 3600 if len(d) else np.nan
        alt = trunca(al, 12)
        xt = A.avalia(alt, falhas, m)
        print(f"{h:>7} h {m.sum()*2/60:10.0f} | {x['det']:4d} {'SIM' if pega else 'nao':>7} "
              f"{lead:6.1f} {x['episodios']:5d} {x['fp_mes']:7.2f} {x['h_fp_mes']:7.1f} | "
              f"{xt['h_fp_mes']:20.1f} {xt['det']:4d}")
        linhas.append(dict(blackout_h=h, h_pontuavel=m.sum()*2/60, det=x["det"],
                           pega_1104=pega, lead_1104=lead, eps=x["episodios"],
                           fp_mes=x["fp_mes"], h_mes=x["h_fp_mes"], p=x["p"],
                           h_mes_teto12=xt["h_fp_mes"], det_teto12=xt["det"],
                           quais=",".join(x["detectados"])))
    t = pd.DataFrame(linhas)
    t.to_csv("blackout.csv", index=False)
    print("\ndetectados por blackout:")
    for _, r in t.iterrows():
        print(f"  {r.blackout_h:>2.0f} h: {r.quais}")


main()
