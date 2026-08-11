#!/usr/bin/env python3
"""
diag_textura_por_periodo.py
Testa a leitura visual de `fig_tc03_por_ano.png`: jun–dez/2024 (onde o modelo falha,
recall 21%) seria mais "serrilhado" que jan–abr/2026 (onde ele acerta, recall ~94%),
apesar de os dois rodarem no MESMO nível térmico (711 °C × 707 °C).

Se a textura explicar a diferença, ela é a pista do canal que falta. Se NÃO explicar,
a hipótese de regime cai por completo e o problema é outro — e é melhor saber agora
do que depois de treinar mais um modelo.

Todas as medidas são calculadas SÓ com máquina quente (TC03 > 500 °C, árbitro físico,
imune ao RUNNING_A) — sem isso o OFF vira confound, como já aconteceu antes.

Uso:
    PYTHONPATH=. python scripts/diag_textura_por_periodo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FULL = "../dados/sensores_full_2024_2026_30s.csv"
OLD = "../dados/sensores_filtrados_Interpolados_{}.csv"
SENSOR = "TC382_03_A"
SETPOINT = 760.0
HOT = 500.0

PERIODOS = [
    ("2022 inteiro",      "2022-01-01", "2023-01-01", 2022),
    ("2024 jan–mai",      "2024-01-01", "2024-06-01", 2024),
    ("2024 jun–dez  ⟵ modelo FALHA", "2024-06-01", "2025-01-01", 2024),
    ("2025 jan–jun",      "2025-01-01", "2025-07-01", 2024),
    ("2025 jul–dez",      "2025-07-01", "2026-01-01", 2024),
    ("2026 jan–abr  ⟵ modelo ACERTA", "2026-01-01", "2026-05-01", 2024),
]


def load(which: int) -> pd.Series:
    path = FULL if which >= 2024 else OLD.format(which)
    d = pd.read_csv(path, usecols=["data_datetime", SENSOR], low_memory=False)
    d["data_datetime"] = pd.to_datetime(d["data_datetime"], utc=True, errors="coerce")
    d = d.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    return pd.to_numeric(d[SENSOR], errors="coerce")


def metrics(v: pd.Series) -> dict:
    hot = v > HOT
    h = v.where(hot)
    n_on = int(hot.sum())
    if n_on < 2880:                      # < 1 dia quente: não mede textura
        return {}
    dias_on = n_on * 30 / 86400

    # derivada em °C/h sobre passo nativo de 30 s, só dentro de trechos quentes
    dt_h = 30.0 / 3600.0
    dv = h.diff() / dt_h
    dv = dv[h.notna() & h.shift().notna()]

    # volatilidade de curto prazo: desvio dentro de janelas de 1 h
    std1h = h.resample("1h").std()
    rng30 = h.resample("30min").max() - h.resample("30min").min()

    # cruzamentos do setpoint por dia ligado (histerese de 5 °C p/ não contar chatter)
    up = (h > SETPOINT + 2.5)
    dn = (h < SETPOINT - 2.5)
    state = pd.Series(np.where(up, 1.0, np.where(dn, 0.0, np.nan)), index=h.index).ffill()
    cross = int((state.diff() == 1).sum())

    return dict(
        dias_on=dias_on,
        T_p50=float(h.median()),
        T_p99=float(h.quantile(0.99)),
        frac_acima_setpoint=float((h > SETPOINT).sum() / n_on),
        cruz_por_dia=cross / dias_on,
        dTdt_p50=float(dv.abs().median()),
        dTdt_p90=float(dv.abs().quantile(0.90)),
        std1h_p50=float(std1h.median()),
        rng30_p50=float(rng30.median()),
    )


def main() -> None:
    cache: dict[int, pd.Series] = {}
    rows = []
    for lab, a, b, src in PERIODOS:
        if src not in cache:
            cache[src] = load(src)
        v = cache[src]
        m = metrics(v[(v.index >= pd.Timestamp(a, tz="UTC")) & (v.index < pd.Timestamp(b, tz="UTC"))])
        if not m:
            print(f"{lab}: sem operação suficiente")
            continue
        rows.append(dict(periodo=lab, **m))

    df = pd.DataFrame(rows).set_index("periodo")
    pd.set_option("display.width", 200)
    print("\n=== nível térmico ===")
    print(df[["dias_on", "T_p50", "T_p99", "frac_acima_setpoint", "cruz_por_dia"]]
          .rename(columns={"dias_on": "dias", "T_p50": "T p50", "T_p99": "T p99",
                           "frac_acima_setpoint": "% acima 760", "cruz_por_dia": "cruz/dia"})
          .to_string(float_format=lambda x: f"{x:,.2f}"))
    print("\n=== TEXTURA (o que a figura sugeriu) ===")
    print(df[["dTdt_p50", "dTdt_p90", "std1h_p50", "rng30_p50"]]
          .rename(columns={"dTdt_p50": "|dT/dt| p50 C/h", "dTdt_p90": "|dT/dt| p90 C/h",
                           "std1h_p50": "std em 1h (C)", "rng30_p50": "amplitude 30min (C)"})
          .to_string(float_format=lambda x: f"{x:,.2f}"))

    fal = "2024 jun–dez  ⟵ modelo FALHA"
    ok = "2026 jan–abr  ⟵ modelo ACERTA"
    if fal in df.index and ok in df.index:
        print(f"\n=== o teste: {fal.split()[0]} jun–dez  vs  2026 jan–abr ===")
        for c, lab in [("T_p50", "nível (T p50)"), ("std1h_p50", "volatilidade 1h"),
                       ("rng30_p50", "amplitude 30min"), ("dTdt_p90", "|dT/dt| p90"),
                       ("frac_acima_setpoint", "% acima do setpoint"),
                       ("cruz_por_dia", "cruzamentos/dia")]:
            a_, b_ = df.loc[fal, c], df.loc[ok, c]
            razao = a_ / b_ if b_ else float("nan")
            print(f"  {lab:<22} 2024H2={a_:9.3f}   2026={b_:9.3f}   razão={razao:5.2f}x")
        print("\n  Leitura: razão perto de 1 = textura NÃO distingue os dois periodos,")
        print("  e a hipotese de regime nao explica a falha de 2024.")


if __name__ == "__main__":
    main()
