#!/usr/bin/env python3
"""
baseline_trivial_fleet.py
Estende a todos os sensores térmicos a pergunta que derrubou o autoencoder no
TC382_03_A: **o AE ganha de um limiar no próprio sinal?**

No TC382_03_A a resposta foi não (81,0% × 62,0%, FA 2,4× menor). Aqui verificamos se
isso é uma peculiaridade daquele sensor ou a regra da frota.

TRÊS CUIDADOS QUE MUDAM O RESULTADO, e que não existiam no script de um sensor só:

1. DIREÇÃO DO LIMIAR. `TC382_03_A` é o único sensor cujo alarme é superaquecimento
   (HI 84 + HIHI 71). Nos outros o histórico é `UNDER` (63–64 cada) — termopar caindo
   para a sentinela. Um limiar "valor alto" não detectaria nada disso. A direção é
   escolhida pela COMPOSIÇÃO DOS ALARMES do sensor, nunca pelo resultado — senão
   estaríamos gastando um grau de liberdade escondido. A direção oposta entra como
   braço de controle, para mostrar que a escolha não é que está fazendo o trabalho.

2. ÁRBITRO DE MÁQUINA LIGADA. `sw.incidents_on` filtra pelo valor do PRÓPRIO sensor
   (≥500 °C). Para alarmes UNDER isso apagaria 100% dos incidentes, porque o sensor
   quebrado lê −40,5 °C justamente durante o evento. Aqui o árbitro é o MÁXIMO entre os
   7 canais térmicos: um termopar quebrado não derruba a leitura dos outros seis.

3. SEM MASCARAR SENTINELA. Para UNDER, a queda até −40,5 °C É o sinal. Mascarar
   apagaria o alvo.

Saída: `eval_predictive_out/baseline_trivial_fleet.csv` (+ task do ClearML com `--clearml`).

Uso:
    PYTHONPATH=. python scripts/baseline_trivial_fleet.py --clearml
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from clearml import Task

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")

RAW = "../dados/sensores_full_2024_2026_30s.csv"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
TASK = "3b34a312aa234aae9ac1f5c1f922791f"   # controle: cobre os 7 canais térmicos
OUT = "eval_predictive_out/baseline_trivial_fleet.csv"

SENSORS = ["T5_AVG_A"] + [f"TC382_0{i}_A" for i in range(1, 7)]
HL_GRID = [0.5, 1.0, 2.0, 4.0]
HORIZON, STICKY, FA_BUDGET = 8.0, 12.0, 1.0
MAX_DUTY, MAX_STICKY = 0.35, 0.25
HOT = 500.0
EXCL = ["CFN"]                      # mantém UNDER/LOLO: são a maioria fora do TC03
T0 = pd.Timestamp("2024-01-01", tz="UTC")
T1 = pd.Timestamp("2026-05-01", tz="UTC")


def alarm_direction(sensor: str, incidentes: list) -> tuple[int, dict]:
    """+1 se o alarme do sensor é de valor ALTO, −1 se é de valor BAIXO.

    ⚠️ A composição é medida sobre os INCIDENTES QUE ENTRAM NA AVALIAÇÃO, não sobre todos
    os alarmes do sensor. A primeira versão deste script usava o histórico inteiro e errava
    a direção em 2 dos 7 canais: o `UNDER` acontece majoritariamente com a máquina parada,
    então o filtro de máquina ligada descarta quase todos, e o que sobra é HI. Contar o
    histórico completo descreve uma população que não é a avaliada.

    Continua sendo escolha a priori — depende só do rótulo do incidente, nunca da métrica.
    Prova de que não seleciona por desempenho: no TC382_01_A a regra aponta BAIXO, que é o
    braço PIOR (14,3% contra 71,4% do ALTO).
    """
    df, _, cond, tag = ev._parse_alarm_df(ALARM)
    d = df[(df[tag] == sensor) & (df["_time"] >= T0) & (df["_time"] < T1)]
    d = d[~d[cond].astype(str).str.upper().isin(["CFN", "OK"])]
    rotulos = []
    for t in incidentes:
        w = d[(d["_time"] >= t - pd.Timedelta(minutes=1)) & (d["_time"] <= t + pd.Timedelta(hours=4))]
        if len(w):
            rotulos.append(str(w[cond].iloc[0]).upper())
    c = pd.Series(rotulos).value_counts().to_dict() if rotulos else {}
    alto = c.get("HI", 0) + c.get("HIHI", 0)
    baixo = c.get("UNDER", 0) + c.get("LOLO", 0) + c.get("LO", 0)
    return (1 if alto >= baixo else -1), {"HI+HIHI": alto, "UNDER+LOLO": baixo}


def incidents_machine_on(sensor: str, maquina_on: pd.Series) -> list:
    """Incidentes do sensor, filtrados por MÁQUINA ligada — não pelo valor do sensor."""
    alarms = ev.load_alarms_gap(ALARM, exclude_conditions=EXCL).get(sensor, [])
    raw = [a for a in alarms if T0 <= a <= T1]
    inc = ev.cluster_incidents(raw, gap_hours=ev.GAP_HOURS)
    if not inc:
        return []
    on = maquina_on.reindex(pd.DatetimeIndex(inc), method="nearest")
    return [t for t, o in zip(inc, on.values) if bool(o)]


def best_over_hl(score: pd.Series, inc: list, running: pd.Series) -> dict:
    best = None
    for hl in HL_GRID:
        h = sw.ewma_on(score, hl, running).rank(pct=True)
        if h.empty:
            continue
        r = ev.best_point_for_sensor(h, inc, horizon_hours=HORIZON, sticky_hours=STICKY,
                                     fa_budget=FA_BUDGET, n_thresholds=120,
                                     max_duty_cycle=MAX_DUTY, max_sticky_duty=MAX_STICKY)
        r["hl"] = hl
        key = (r.get("recall_raw") or 0.0, -(r.get("fa_per_day") or 9e9))
        if best is None or key > (best.get("recall_raw") or 0.0,
                                  -(best.get("fa_per_day") or 9e9)):
            best = r
    return best or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearml", action="store_true")
    args = ap.parse_args()

    print("[1/3] lendo o CSV bruto...", flush=True)
    raw = pd.read_csv(RAW, usecols=["data_datetime", "RUNNING_A"] + SENSORS, low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    raw = raw[(raw.index >= T0) & (raw.index < T1)]
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    vals = {s: pd.to_numeric(raw[s], errors="coerce") for s in SENSORS}

    # árbitro robusto: basta UM canal térmico quente para a máquina estar operando
    maquina_quente = pd.concat(vals.values(), axis=1).max(axis=1) > HOT
    maquina_on = maquina_quente & (running > 0.5)
    print(f"      máquina ON+quente em {maquina_on.mean():.1%} do período")

    print("[2/3] baixando curvas do AE...", flush=True)
    mae_all = ev.load_mae_series(Task.get_task(task_id=TASK), SENSORS)

    rows = []
    for s in SENSORS:
        if s not in mae_all:
            print(f"\n{s}: sem curva do AE — pulado")
            continue
        inc = incidents_machine_on(s, maquina_on)
        sinal, comp = alarm_direction(s, inc)
        mae = mae_all[s]
        mae = mae[(mae.index >= T0) & (mae.index < T1)]
        v = vals[s].reindex(mae.index, method="nearest")
        dir_lab = "ALTO (HI/HIHI)" if sinal > 0 else "BAIXO (UNDER/LOLO)"
        print(f"\n=== {s} — {len(inc)} incidentes com máquina ON · alarme predominante: "
              f"{dir_lab} ({comp['HI+HIHI']} altos / {comp['UNDER+LOLO']} baixos) ===")
        if len(inc) < 3:
            print("      amostra insuficiente (<3) — não avaliado")
            rows.append(dict(sensor=s, n_inc=len(inc), direcao=dir_lab, avaliado=False))
            continue

        bracos = {"AE (autoencoder)": mae,
                  "limiar trivial (direção do alarme)": v * sinal,
                  "limiar trivial (direção OPOSTA)": v * -sinal}
        print(f"  {'braço':<36}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'lead h':>9}{'hl':>6}")
        res = {}
        for name, sc in bracos.items():
            r = best_over_hl(sc.dropna(), inc, running)
            res[name] = r
            rr = r.get("recall_raw")
            print(f"  {name:<36}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                  f"{r.get('fa_per_day', float('nan')):>10.3f}"
                  f"{r.get('duty_sticky', float('nan')):>8.2f}"
                  f"{r.get('median_lead_hours', float('nan')):>9.1f}"
                  f"{str(r.get('hl')):>6}", flush=True)
        ae = res["AE (autoencoder)"]
        tv = res["limiar trivial (direção do alarme)"]
        op = res["limiar trivial (direção OPOSTA)"]
        # grava as duas direções com NOME ABSOLUTO (alto/baixo), não relativo à escolha —
        # assim uma correção futura na regra de direção não exige refazer os ajustes
        alto, baixo = (tv, op) if sinal > 0 else (op, tv)
        rows.append(dict(
            sensor=s, n_inc=len(inc), direcao=dir_lab, avaliado=True,
            n_inc_alto=comp["HI+HIHI"], n_inc_baixo=comp["UNDER+LOLO"],
            ae_recall=ae.get("recall_raw"), ae_fa=ae.get("fa_per_day"),
            trivial_recall=tv.get("recall_raw"), trivial_fa=tv.get("fa_per_day"),
            oposto_recall=op.get("recall_raw"), oposto_fa=op.get("fa_per_day"),
            trivial_alto_recall=alto.get("recall_raw"), trivial_alto_fa=alto.get("fa_per_day"),
            trivial_baixo_recall=baixo.get("recall_raw"), trivial_baixo_fa=baixo.get("fa_per_day"),
            delta_pp=((tv.get("recall_raw") or 0) - (ae.get("recall_raw") or 0)) * 100))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[3/3] Gravado: {OUT}")

    d = df[df.avaliado == True]  # noqa: E712
    print("\n=== VEREDITO POR SENSOR — o limiar trivial ganha do AE? ===")
    print(f"  {'sensor':<14}{'n':>5}{'AE':>9}{'trivial':>10}{'Δpp':>8}   veredito")
    for _, r in d.iterrows():
        dpp = r.delta_pp
        v = "LIMIAR GANHA" if dpp > 10 else ("empate" if dpp >= -10 else "AE GANHA")
        print(f"  {r.sensor:<14}{r.n_inc:>5}{r.ae_recall*100:>8.1f}%{r.trivial_recall*100:>9.1f}%"
              f"{dpp:>8.1f}   {v}")
    print("\n  (margem de 10pp = ruído de semente medido no projeto)")

    if args.clearml:
        t = Task.init(project_name="TesteMLCab", task_name="baseline-trivial-fleet",
                      output_uri=True, reuse_last_task_id=False)
        t.connect({"sensores": SENSORS, "ae_task": TASK, "horizonte_h": HORIZON,
                   "sticky_h": STICKY, "fa_budget": FA_BUDGET, "max_duty": MAX_DUTY,
                   "max_sticky_duty": MAX_STICKY, "hl_grid": HL_GRID,
                   "excluidas": EXCL, "janela": f"{T0.date()}→{T1.date()}"},
                  name="protocolo")
        t.upload_artifact("baseline_trivial_fleet", df)
        t.get_logger().report_table(title="baseline trivial x AE", series="frota",
                                    table_plot=df)
        print(f"[clearml] task {t.id}")


if __name__ == "__main__":
    main()
