#!/usr/bin/env python3
"""
sweep_incident_definition.py
Varre a DEFINIÇÃO DE INCIDENTE: um `OK` fecha a ocorrência, mas um rearme logo em
seguida é a MESMA ocorrência — só vira incidente novo depois de um intervalo de
silêncio. Varre esse intervalo.

Motivação (11/08/2026): a metodologia hoje é assimétrica.

    falso alarme     -> conta EPISÓDIO   (sticky 12h + debounce 8h)
    incidente perdido-> conta ALARME     (cluster de apenas 4h)

Um alerta ligado 3 dias é 1 FP; uma degradação que dispara o DCS 11 vezes em 15
dias é 11 perdas. Em jun–dez/2024 a turbina alarmou 0,68 vez por dia ligado (7,5×
a taxa de 2025) — os "28 incidentes" são 3 episódios de máquina marginal.

⚠️ ARMADILHA: escolher o intervalo que MAXIMIZA o recall é escolher o denominador
para o número ficar bonito. O critério legítimo é (a) onde a contagem ESTABILIZA —
platô indica a escala natural do evento — e (b) validação com quem operou. O recall
é reportado junto só para mostrar a sensibilidade, NÃO para ser otimizado.

Uso:
    PYTHONPATH=. python scripts/sweep_incident_definition.py
"""
from __future__ import annotations

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
on = _load("sweep_onset_rules_offline")

SENSOR = "TC382_03_A"
RAW = "../dados/sensores_full_2024_2026_30s.csv"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
TASK_CONTROL = "3b34a312aa234aae9ac1f5c1f922791f"     # braço vencedor atual
FLEET = "eval_predictive_out/fleet_v14_control_{}.csv"
OUT = "eval_predictive_out/incident_definition_sweep.csv"

EXCL = {"UNDER", "CFN", "LOLO", "OVER"}
QUIET_GRID = [0.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 168.0]
WINDOWS = [("2024 jun–dez", "2024-06-01", "2025-01-01", "BACKCAST_2024_hihihi"),
           ("2024 jan–mai", "2024-01-01", "2024-06-01", "BACKCAST_2024H1_hihihi"),
           ("OOS jul/25+", "2025-07-01", "2026-05-01", "OOS_hihihi"),
           ("FULL", "2024-01-01", "2026-05-01", "FULL_hihihi")]


def cluster_ok_aware_gap(alarm_csv: str, sensor: str, quiet_hours: float) -> list:
    """Incidentes com `OK` fechando a ocorrência + intervalo mínimo de silêncio.

    Um onset abre incidente novo só se passou >= quiet_hours desde o OK que fechou
    o anterior. Rearme dentro do silêncio é a MESMA ocorrência (reabre, não conta).
    quiet_hours=0 reproduz o OK-aware puro já existente em eval_per_sensor_level.
    """
    df, _, cond_col, tag_col = ev._parse_alarm_df(alarm_csv)
    d = df[df[tag_col] == sensor].sort_values("_time")
    quiet = pd.Timedelta(hours=quiet_hours)
    incidents, open_onset, last_ok = [], None, None
    for _, r in d.iterrows():
        cond = str(r[cond_col]).upper() if cond_col else "CFN"
        t = r["_time"]
        if cond == "OK":
            if open_onset is not None:
                last_ok = t
                open_onset = None
            continue
        if cond in EXCL:
            continue
        if open_onset is not None:
            continue                                  # já dentro de uma ocorrência
        if last_ok is not None and (t - last_ok) < quiet:
            open_onset = incidents[-1] if incidents else t   # rearme: mesma ocorrência
            continue
        incidents.append(t)
        open_onset = t
    return incidents


def filter_on(inc: list, running: pd.Series, tc03: pd.Series, t0, t1) -> list:
    inc = [t for t in inc if t0 <= t <= t1]
    if not inc:
        return []
    ok = running.reindex(pd.DatetimeIndex(inc), method="nearest") > 0.5
    inc = [t for t, o in zip(inc, ok.values) if o]
    if not inc:
        return []
    v = tc03.reindex(pd.DatetimeIndex(inc), method="nearest")
    return [t for t, val in zip(inc, v.values) if pd.isna(val) or val >= 500.0]


def main() -> None:
    raw = pd.read_csv(RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    mae = ev.load_mae_series(Task.get_task(task_id=TASK_CONTROL), [SENSOR])[SENSOR]

    rows = []
    for wlab, a, b, fleet_key in WINDOWS:
        t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        # ponto de operação AUDITADO desta janela — não re-otimizar por definição,
        # senão o sweep mistura dois efeitos
        fr = pd.read_csv(FLEET.format(fleet_key)).set_index("sensor").loc[SENSOR]
        hl, q = float(fr["hl"]), float(fr["threshold_q"])
        m = mae[(mae.index >= t0) & (mae.index < t1)]
        h = sw.ewma_on(m, hl, running).rank(pct=True)
        raw_s = np.array([t.timestamp() for t in h.index[h >= q]])
        hs = 8 * 3600.0

        base = sw.incidents_on(running, tc03, t0, t1)          # regra atual (gap 4h)
        n_hit_base = sum(1 for t in base if raw_s.size and
                         np.any((raw_s >= t.timestamp() - hs) & (raw_s <= t.timestamp())))
        print(f"\n=== {wlab} ===   (hl={hl}, q={q:.4f})")
        print(f"  regra ATUAL (gap 4h, sem OK): {len(base):>3} incidentes  "
              f"recall_raw {n_hit_base/max(len(base),1)*100:5.1f}%")
        print(f"  {'silêncio':>9} {'incidentes':>11} {'detectados':>11} {'recall_raw':>11}")
        for qh in QUIET_GRID:
            inc = filter_on(cluster_ok_aware_gap(ALARM, SENSOR, qh), running, tc03, t0, t1)
            n_hit = sum(1 for t in inc if raw_s.size and
                        np.any((raw_s >= t.timestamp() - hs) & (raw_s <= t.timestamp())))
            rr = n_hit / len(inc) if inc else float("nan")
            rows.append(dict(janela=wlab, quiet_h=qh, n_inc=len(inc), n_hit=n_hit,
                             recall_raw=rr, hl=hl, threshold_q=q,
                             n_inc_atual=len(base), recall_atual=n_hit_base/max(len(base),1)))
            print(f"  {qh:>7.0f}h {len(inc):>11} {n_hit:>11} {rr*100:>10.1f}%")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGravado: {OUT}")
    print("\n⚠️  Escolher o intervalo pelo MAIOR recall é escolher o denominador. "
          "Usar o platô da contagem + validação de quem operou.")


if __name__ == "__main__":
    main()
