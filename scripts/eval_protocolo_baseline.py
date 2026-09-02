#!/usr/bin/env python3
"""
eval_protocolo_baseline.py
Protocolo de auditoria com duas correções que mudam a leitura de todos os braços:

ITEM 1 — BASELINE PERMANENTE
    Inclui o sinal cru do próprio sensor como um "braço" (sem modelo nenhum:
    temperatura → janela de 48 min → EWMA → rank → sticky). Se o autoencoder não
    supera isso, o número dele não significa detecção — significa que o alvo é
    função direta da entrada. O baseline passa a sair em toda tabela.

ITEM 2 — CONTAMINAÇÃO DO RECALL POR GAP_HOURS
    `cluster_incidents` agrupa alarmes separados por menos de GAP_HOURS num único
    incidente. Com GAP_HOURS=4h e horizonte=8h, dois incidentes distintos podem
    estar a 5h um do outro: um único cruzamento de threshold cai dentro da janela
    de 8h dos DOIS e conta como dois acertos. 22% dos intervalos entre alarmes
    deste sensor estão abaixo de 8h, então o recall publicado está inflado.
    Aqui o mesmo ponto de operação é medido com GAP_HOURS ∈ {4, 8, 12} para
    quantificar o quanto disso era artefato.

O cálculo caro (sticky + episódios, 120 thresholds) NÃO depende de GAP_HOURS —
só a lista de incidentes muda. Por isso ele é feito uma vez por (braço, cenário,
hl, q) e reaproveitado nos três agrupamentos, em vez de três varreduras.

Uso:
    PYTHONPATH=. python scripts/eval_protocolo_baseline.py
    PYTHONPATH=. python scripts/eval_protocolo_baseline.py --sensor T5_AVG_A
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")

HL_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
GAPS = [4.0, 8.0, 12.0]          # 4.0 = o valor atual, contaminado
N_Q = 120

ARMS = {                          # rótulo -> task_id ClearML ("" = baseline)
    "baseline_temp": "",
    "ctrl (4320)": "a78df1cf1c6e4f43ac9e1303034d71eb",
    "excl_12h (720)": "e82d06a623644c09a3e0c71ee5f28b2d",
    "excl_24h (1440)": "0fdeb5318361420e904b7994a65e3593",
}
GRADE_TASK = "0fdeb5318361420e904b7994a65e3593"   # grade temporal de referência


def _resolve_dados() -> str:
    for up in ("..", "../..", "../../.."):
        cand = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("diretório 'dados/' não encontrado.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--arm", action="append", default=[],
                   help="label=task_id (repetível); estende/sobrepõe ARMS")
    p.add_argument("--only", default=None,
                   help="lista separada por vírgula dos rótulos a avaliar")
    p.add_argument("--out", default=None)
    return p.parse_args()


def incidentes_por_gap(running, serie, t_lo, t_hi, gap_h: float) -> list:
    """Mesmo filtro do protocolo (máquina ON, fantasma <500°C), agrupando com o
    gap pedido em vez do global do módulo."""
    alarms = ev.load_alarms_gap(sw.ALARM_CSV, exclude_conditions=sw.EXCL).get(sw.SENSOR, [])
    raw = [a for a in alarms if t_lo <= a <= t_hi]
    inc = ev.cluster_incidents(raw, gap_hours=gap_h)
    if not inc:
        return []
    on = running.reindex(pd.DatetimeIndex(inc), method="nearest") > 0.5
    inc = [t for t, o in zip(inc, on.values) if o]
    if inc:
        v = serie.reindex(pd.DatetimeIndex(inc), method="nearest")
        inc = [t for t, val in zip(inc, v.values) if pd.isna(val) or val >= 500.0]
    return inc


def main() -> None:
    args = parse_args()
    S = args.sensor
    sw.SENSOR = S
    dados = _resolve_dados()
    sw.RAW_CSV = os.path.join(dados, "sensores_2024h2_2025_2026_30s.csv")
    sw.ALARM_CSV = os.path.join(dados, "alarmes_selecionados_turbina_a.csv")
    ev.ALARM_CSV_DEFAULT = sw.ALARM_CSV

    from clearml import Task

    running, _, t5 = sw.load_raw()
    raw = pd.read_csv(sw.RAW_CSV, usecols=["data_datetime", S], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    serie = pd.to_numeric(
        raw.dropna(subset=["data_datetime"]).set_index("data_datetime")[S],
        errors="coerce").sort_index()
    t5s = t5.ewm(halflife=int(pd.Timedelta(hours=24) / pd.Timedelta("30s"))).mean()

    grade = sw.read_mae(Task.get_task(task_id=GRADE_TASK)
                        .artifacts[f"{S}_csv_sequence_scores_all.csv"].get_local_copy()).index

    def score_de(tid: str) -> pd.Series:
        if tid == "":
            b = serie.reindex(grade, method="nearest")
            b = b.where(b >= 500).ffill().dropna()
            return b.rolling(window=10, min_periods=1).mean()   # 48 min, igual ao AE
        return sw.read_mae(Task.get_task(task_id=tid)
                           .artifacts[f"{S}_csv_sequence_scores_all.csv"].get_local_copy())

    arms = dict(ARMS)
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm espera label=task_id, recebi {spec!r}")
        k, v = spec.split("=", 1)
        arms[k.strip()] = v.strip()
    if args.only:
        keep = [x.strip() for x in args.only.split(",")]
        faltam = [k for k in keep if k not in arms]
        if faltam:
            raise SystemExit(f"rótulos desconhecidos: {faltam}; disponíveis: {list(arms)}")
        arms = {k: arms[k] for k in keep}

    hs = sw.HORIZON * 3600.0
    rows = []
    for lab, tid in arms.items():
        score = score_de(tid)
        for cen, t0, t1 in [("FULL", None, None),
                            ("BACKCAST_2024", *sw.BACKCAST),
                            ("OOS", *sw.OOS)]:
            s = score
            if t0 is not None:
                s = s[s.index >= t0]
            if t1 is not None:
                s = s[s.index < t1]
            incs = {g: incidentes_por_gap(running, serie, s.index.min(), s.index.max(), g)
                    for g in GAPS}
            dias = (s.index[-1] - s.index[0]).total_seconds() / 86400.0
            melhor = {}
            for hl in HL_GRID:
                health = sw.health_global(s, hl, running, t5s)
                if health.empty:
                    continue
                for q in np.linspace(0.50, 0.999, N_Q):
                    if float((health >= q).mean()) > sw.MAX_DUTY:
                        continue
                    alert = ev.apply_sticky(health, q, sw.STICKY)
                    duty = float(alert.mean())
                    if duty > sw.MAX_STICKY:
                        continue
                    eps = ev.detect_episodes_gap(alert)
                    cruz = np.array([t.timestamp() for t in health.index[health >= q]])
                    ep_s = np.array([(a.timestamp(), b.timestamp()) for a, b in eps]) \
                        if eps else np.empty((0, 2))
                    for g in GAPS:                       # <- só isto muda por gap
                        inc_s = np.array([t.timestamp() for t in incs[g]])
                        if inc_s.size == 0:
                            continue
                        n_hit = sum(1 for ti in inc_s if cruz.size
                                    and np.any((cruz >= ti - hs) & (cruz <= ti)))
                        n_fp = sum(1 for (a, b) in ep_s
                                   if not np.any((inc_s - hs <= b) & (inc_s >= a)))
                        rec = n_hit / len(inc_s)
                        fa = n_fp / max(dias, 1.0)
                        if fa > sw.FA_BUDGET:
                            continue
                        k = (lab, cen, g)
                        cand = (rec, -n_fp)
                        if k not in melhor or cand > melhor[k][0]:
                            melhor[k] = (cand, dict(
                                braco=lab, cenario=cen, gap_h=g, hl=hl, q=round(float(q), 4),
                                inc=len(inc_s), recall=rec, fp=n_fp, fa_per_day=fa,
                                duty_sticky=duty, dias=round(dias)))
            for _, d in melhor.values():
                rows.append(d)
            print(f"[ok] {lab:16s} {cen:14s} "
                  + "  ".join(f"gap{int(g)}h: inc={len(incs[g])}" for g in GAPS))

    df = pd.DataFrame(rows).sort_values(["cenario", "gap_h", "braco"])
    out = args.out or f"eval_predictive_out/protocolo_baseline_{S}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    for cen in ["FULL", "BACKCAST_2024", "OOS"]:
        for g in GAPS:
            sub = df[(df.cenario == cen) & (df.gap_h == g)]
            if sub.empty:
                continue
            print(f"\n=== {cen}  GAP_HOURS={g:g}h  incidentes={sub.inc.iloc[0]}  "
                  f"dias={sub.dias.iloc[0]}")
            print(sub[["braco", "hl", "q", "recall", "fp", "duty_sticky"]].to_string(
                index=False, formatters={"recall": lambda v: f"{v:.1%}",
                                         "duty_sticky": lambda v: f"{v:.3f}"}))
    print(f"\ngravado: {out}")


if __name__ == "__main__":
    main()
