#!/usr/bin/env python3
"""
diagnose_running_proxy.py
Responde: o `RUNNING_A` está descartando indevidamente eventos de alarme dos sensores
novos (pressão / temperatura de mancal)?

Motivação: `validate_pressure_labels.py` marcou 13 das 27 tags candidatas como
`mascarado_RUNNING_A` — a curva está viva em todos os onsets e mesmo assim o evento
sai da conta. A leitura inicial foi "o RUNNING_A é o gargalo". Este script existe
para testar essa leitura em vez de assumi-la, usando o T5_AVG_A como árbitro físico
independente: turbina rodando ⇒ T5 na casa das centenas de °C; turbina parada ⇒ ~30°C.

Três medições:
  1. Os blocos OFF do RUNNING_A são desligamentos reais? (T5 mediano por bloco)
  2. Os onsets descartados caem em OFF real ou em piscada do flag?
  3. Corrigir os OFF falsos recuperaria quantos incidentes?

Resultado (2025-01→2026-04, ver memória `running-proxy-nao-e-gargalo`): o RUNNING_A
está essencialmente correto. Os 52 onsets descartados estão em OFF genuíno (T5 ≈ 30°C,
ZERO com T5>500), e o OFF falso soma 0,1h em 16 meses.

Uso:
    PYTHONPATH=. python scripts/diagnose_running_proxy.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

RAW_CSV = "../dados/sensores_brutos_2025_2026_30s.csv"
ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
VALIDATION_CSV = "eval_pressure_out/label_validation.csv"

HOT_C = 500.0        # T5 acima disto = turbina inequivocamente em operação
SAMPLE_H = 0.5 / 60  # 30s em horas


def load(raw_csv: str) -> pd.DataFrame:
    d = pd.read_csv(raw_csv, low_memory=False)
    d["data_datetime"] = pd.to_datetime(d["data_datetime"], format="ISO8601", utc=True)
    return d.set_index("data_datetime").apply(pd.to_numeric, errors="coerce").sort_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_csv", default=RAW_CSV)
    p.add_argument("--alarm_csv", default=ALARM_CSV)
    p.add_argument("--validation_csv", default=VALIDATION_CSV)
    p.add_argument("--hot_c", type=float, default=HOT_C)
    args = p.parse_args()

    d = load(args.raw_csv)
    on = d["RUNNING_A"] > 0.5
    t5 = d["T5_AVG_A"]
    blk = (on != on.shift()).cumsum()
    info = pd.DataFrame({"on": on, "t5": t5, "blk": blk})
    g = info.groupby("blk").agg(on=("on", "first"), n=("on", "size"), t5=("t5", "median"))
    g["horas"] = g["n"] * SAMPLE_H
    off = g[~g["on"]]

    print(f"RUNNING_A: ON={on.mean():.1%} | {len(g) - 1} transições | {len(off)} blocos OFF\n")
    print("[1/3] Os blocos OFF são desligamentos reais? (árbitro: T5_AVG_A)")
    print(f"      {'faixa':<12}{'n':>5}{'T5 mediano':>13}{'blocos T5>' + str(int(args.hot_c)):>18}")
    for lab, m in [("<30min", off["horas"] < 0.5),
                   ("30min–2h", (off["horas"] >= 0.5) & (off["horas"] < 2)),
                   ("≥2h", off["horas"] >= 2)]:
        s = off[m]
        if not len(s):
            continue
        print(f"      {lab:<12}{len(s):>5}{s['t5'].median():>12.1f}°C"
              f"{int((s['t5'] > args.hot_c).sum()):>18}")

    alarms = pd.read_csv(args.alarm_csv)
    alarms["_t"] = pd.to_datetime(alarms["Data da Ocorrência"], errors="coerce", utc=True)
    alarms = alarms[alarms["Condição do Alarme"].astype(str).str.upper() != "OK"]
    alarms = alarms[(alarms["_t"] >= d.index.min()) & (alarms["_t"] <= d.index.max())]
    idx = d.index.get_indexer(alarms["_t"], method="nearest")

    print("\n[2/3] Onde caem os onsets das tags marcadas 'mascarado_RUNNING_A'?")
    try:
        val = pd.read_csv(args.validation_csv)
        tags = val.loc[val["veredito"] == "mascarado_RUNNING_A", "tag"].tolist()
    except FileNotFoundError:
        print(f"      [skip] rode validate_pressure_labels.py antes ({args.validation_csv})")
        tags = []
    if tags:
        sub = alarms[alarms["Tag Alarme"].isin(tags)]
        sidx = d.index.get_indexer(sub["_t"], method="nearest")
        in_off = ~on.iloc[sidx].values
        hot = (t5.iloc[sidx].values > args.hot_c)
        print(f"      {len(sub)} onsets | {int(in_off.sum())} em OFF | "
              f"{int(np.nansum(hot))} com T5>{int(args.hot_c)}°C | "
              f"T5 mediano {np.nanmedian(t5.iloc[sidx].values):.1f}°C")
        print("      → OFF genuíno (turbina fria) se T5>limiar for ~0: o descarte está CORRETO.")

    print("\n[3/3] Quanto se recupera corrigindo os OFF falsos (T5 quente com flag OFF)?")
    falso = g[(~g["on"]) & (g["t5"] > args.hot_c)].index
    m_falso = info["blk"].isin(falso)
    rec = alarms[m_falso.values[idx]]
    print(f"      {len(falso)} blocos | {int(m_falso.sum())} amostras = "
          f"{m_falso.sum() * SAMPLE_H:.1f}h ({m_falso.mean():.4%} do período)")
    print(f"      ON passaria de {on.mean():.2%} para {(on | m_falso).mean():.2%}")
    print(f"      onsets recuperados: {len(rec)}")
    if len(rec):
        print(rec.groupby("Tag Alarme").size().to_string().replace("\n", "\n        "))
        print("\n      ⚠️ São alarmes do INSTANTE do trip (o flag cai por poucas amostras "
              "enquanto a turbina ainda está quente). Alarme no trip é consequência, não "
              "previsão — continua fora de escopo por decisão de metodologia.")


if __name__ == "__main__":
    main()
