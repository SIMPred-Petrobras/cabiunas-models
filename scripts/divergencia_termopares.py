#!/usr/bin/env python3
"""
divergencia_termopares.py
Procura, no dado cru e sem modelo nenhum, episódios em que um termopar do array
TC382 se afasta de forma SUSTENTADA da média dos outros cinco.

A motivação: os alarmes HI/HIHI são cruzamentos de limiar sobre o próprio sensor,
então o alvo é função direta da entrada e uma EWMA da temperatura já os prevê tão
bem quanto o autoencoder. O resíduo contra os irmãos mede outra coisa — divergência
local — que limiar nenhum sobre um sensor sozinho consegue ver. Este script
pergunta se essa outra coisa **existe** na série, antes de treinar qualquer modelo
para procurá-la.

Método (tudo robusto, para não ser arrastado por transiente):
  1. resíduo_i = sensor_i − média(irmãos válidos), só com máquina ligada;
     validade = leitura dentro de [-30, 1200] (rejeita o -40.5 do termopar aberto).
  2. cada sensor tem offset próprio (posição na exaustão), então centra-se pela
     MEDIANA do próprio resíduo e escala-se pelo MAD.
  3. suaviza-se com mediana móvel de 1h — degradação de termopar é deriva lenta,
     não pico.
  4. episódio = |z| acima de Z_MIN por pelo menos DUR_MIN horas contínuas de
     máquina ligada.
  5. para cada episódio, verifica-se se há alarme HI/HIHI do array dentro da
     janela — se houver poucos, o resíduo está achando algo que os alarmes não
     registram, que é exatamente a hipótese.

Uso:
    PYTHONPATH=. python scripts/divergencia_termopares.py
    PYTHONPATH=. python scripts/divergencia_termopares.py --z 6 --dur 12
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

GRUPO = [f"TC382_0{i}_A" for i in range(1, 7)]
VALID_LOW, VALID_HIGH = -30.0, 1200.0
DT = pd.Timedelta("30s")


def _dados() -> str:
    for up in ("..", "../..", "../../.."):
        c = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", up, "dados"))
        if os.path.isdir(c):
            return c
    raise SystemExit("diretório 'dados/' não encontrado.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", type=float, default=5.0, help="limiar em MADs")
    p.add_argument("--dur", type=float, default=6.0, help="duração mínima em horas ON")
    p.add_argument("--suaviza_h", type=float, default=1.0)
    p.add_argument("--min_temp", type=float, default=500.0,
                   help="média do array abaixo disso = turbina fria; RUNNING_A fica ligado\n"
                        "em periodos frios, e ali o residuo vira constante e dispara falso.\n"
                        "Mesmo criterio do filtro de fantasma do protocolo de auditoria.")
    p.add_argument("--out", default="eval_predictive_out/divergencia_termopares.csv")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    D = _dados()
    df = pd.read_csv(os.path.join(D, "sensores_2024h2_2025_2026_30s.csv"),
                     usecols=["data_datetime", "RUNNING_A", *GRUPO], low_memory=False)
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    on = pd.to_numeric(df["RUNNING_A"], errors="coerce") > 0.5
    T = df[GRUPO].apply(pd.to_numeric, errors="coerce")
    T = T.where((T >= VALID_LOW) & (T <= VALID_HIGH))

    alarmes = pd.read_csv(os.path.join(D, "alarmes_selecionados_turbina_a.csv"))
    col_t = next(c for c in alarmes.columns if "time" in c.lower() or "data" in c.lower())
    col_c = next((c for c in alarmes.columns if "cond" in c.lower()), None)
    alarmes[col_t] = pd.to_datetime(alarmes[col_t], utc=True, errors="coerce")
    if col_c is not None:
        alarmes = alarmes[alarmes[col_c].astype(str).str.upper().isin(["HI", "HIHI"])]
    # epoch em segundos: comparar Timestamp tz-aware com datetime64 naive estoura
    t_alarme = np.sort(alarmes[col_t].dropna().map(lambda t: t.timestamp()).to_numpy())

    # RUNNING_A fica ligado com a turbina fria (ex.: 19-23/ago/2025, seis sensores
    # a ~30C e flag em 1). Sem este corte o residuo vira constante e o detector
    # acha "divergencia sustentada" que e so maquina parada mal rotulada.
    quente = T.mean(axis=1) >= a.min_temp
    n_fantasma = int((on & ~quente).sum())
    print(f"amostras com RUNNING_A ligado mas array abaixo de {a.min_temp:.0f}C: "
          f"{n_fantasma} ({n_fantasma / max(int(on.sum()), 1):.2%} do tempo ON) — descartadas\n")
    on = on & quente

    jan = int(pd.Timedelta(hours=a.suaviza_h) / DT)
    linhas = []
    print(f"limiar {a.z} MADs sustentado por {a.dur}h de máquina ligada\n")
    for s in GRUPO:
        irm = [c for c in GRUPO if c != s]
        n_ok = T[irm].notna().sum(axis=1)
        res = (T[s] - T[irm].mean(axis=1)).where(n_ok >= 3)
        res = res.where(on)
        med = float(res.median())
        mad = float((res - med).abs().median()) * 1.4826
        z = ((res - med) / max(mad, 1e-9)).rolling(jan, min_periods=jan // 2).median()

        fora = (z.abs() >= a.z).fillna(False).to_numpy()
        idx = z.index
        corte = np.flatnonzero(fora[1:] != fora[:-1]) + 1
        ini = np.concatenate(([0], corte))
        fim = np.concatenate((corte, [len(fora)]))
        eps = []
        for i0, i1 in zip(ini, fim):
            if not fora[i0]:
                continue
            t0, t1 = idx[i0], idx[min(i1, len(idx) - 1)]
            horas_on = float(on.loc[t0:t1].sum()) * DT.total_seconds() / 3600.0
            if horas_on >= a.dur:
                eps.append((t0, t1, horas_on, float(z.loc[t0:t1].abs().max()),
                            float(res.loc[t0:t1].median() - med)))
        com_alarme = 0
        for t0, t1, *_ in eps:
            if t_alarme.size and np.any((t_alarme >= t0.timestamp()) &
                                        (t_alarme <= t1.timestamp())):
                com_alarme += 1
        print(f"{s}: mediana do resíduo={med:7.2f}°C  MAD={mad:5.2f}  "
              f"episódios={len(eps):>3}  com alarme HI/HIHI dentro={com_alarme}")
        for t0, t1, h, zmax, desvio in eps:
            linhas.append(dict(sensor=s, inicio=t0, fim=t1, horas_on=round(h, 1),
                               z_max=round(zmax, 1), desvio_C=round(desvio, 2),
                               tem_alarme=bool(t_alarme.size and np.any(
                                   (t_alarme >= t0.timestamp()) &
                                   (t_alarme <= t1.timestamp())))))

    if not linhas:
        print("\nNenhum episódio de divergência sustentada nesse limiar.")
        return
    out = pd.DataFrame(linhas).sort_values("horas_on", ascending=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\ntotal de episódios: {len(out)}  "
          f"| com alarme HI/HIHI dentro: {int(out.tem_alarme.sum())} "
          f"({out.tem_alarme.mean():.0%})")
    print(f"horas ON em divergência: {out.horas_on.sum():.0f}h "
          f"| desvio mediano: {out.desvio_C.abs().median():.1f}°C")
    print("\n10 maiores episódios:")
    print(out.head(10).to_string(index=False))
    print(f"\ngravado: {a.out}")


if __name__ == "__main__":
    main()
