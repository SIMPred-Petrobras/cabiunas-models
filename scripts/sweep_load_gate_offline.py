#!/usr/bin/env python3
"""
sweep_load_gate_offline.py
Portão de TRANSITÓRIO DE CARGA na camada de decisão do TC382_03_A — sem retreinar.

Achado que motiva (medido em sweep_onset_rules_offline.py, braço b2024):
  * o PICO do score no falso positivo é tão alto quanto no evento real
    (AUC 0,44 — 0,9893 no FP contra 0,9713 no TP). Por isso toda regra de
    AMPLITUDE (histerese, confirmação k-de-N) troca recall por FA na razão 1:1.
  * mas na janela [-6h,+3h] do onset o FP está em RAMPA e o TP não:
    |dT5/dt| mediano 172 °C/h (FP) contra 21 °C/h (TP), ~8×; e nível do T5
    610 °C (FP) contra 679 °C (TP). O evento real acontece com a máquina
    estável e quente; o falso positivo acontece manobrando.

Hipótese: suprimir o DISPARO durante manobra de carga remove FP a custo de lead
ZERO (a rampa é conhecida no instante do disparo, não exige esperar).

    t5s  = T5_AVG_A suavizado (half-life 2h)
    ramp = |d t5s/dt|                 °C/h
    gate = ramp.rolling("6h").max()   trailing — só olha o passado
    sig  = (h >= q) & (gate < R) & (t5s >= L)

`T5_AVG_A` como proxy de carga (e não `PDI_0317`) porque existe na janela FULL;
o PDI só cobre 2025+. Suavização de 2h é lenta o bastante para não deixar a
própria excursão do evento se registrar como "rampa" (os TPs ficam em 21 °C/h).

⚠️ FACA DE DOIS GUMES: uma falha real que se manifeste DURANTE manobra é
suprimida pelo portão. Por isso o relatório lista, caso a caso, quais dos TPs
caem acima do R escolhido — esse número pesa mais que o ganho agregado de FA.

Critério de promoção (fixado ANTES de rodar): FA cai ≥30% (0,103 → ≤0,072) com
perda de recall_raw ≤5pp no FULL, E o OOS 2025 mantém 17/17. Se a troca voltar a
ser ~1:1, registrar como refutado — não ajustar o critério depois.

Uso:
    PYTHONPATH=. python scripts/sweep_load_gate_offline.py
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
on = _load("sweep_onset_rules_offline")   # sticky_bool / evaluate vetorizados

SENSOR = on.SENSOR
TASK_B2024 = on.TASK_B2024
FLEET_CSV = on.FLEET_CSV
OUT_CSV = "eval_predictive_out/load_gate_TC382_03_A.csv"

RAMP_HL_H = 2.0        # suavização do proxy de carga
GATE_WIN = "6h"        # janela trailing do máximo de rampa
R_GRID = [40.0, 60.0, 80.0, 120.0, 160.0, 240.0, 320.0, np.inf]
L_GRID = [0.0, 600.0, 640.0]

SCENARIOS = [("FULL", None, None),
             ("BACKCAST_2024", *sw.BACKCAST),
             ("OOS", *sw.OOS)]


def ramp_signal(t5: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Máximo trailing de |dT5/dt| (°C/h) sobre o T5 suavizado — causal.

    A EWMA roda na grade nativa de 30s (é O(n) e o half-life de 2h precisa dela),
    mas a derivada e o rolling máximo são feitos na grade de 5 min do health: um
    rolling temporal de 6h sobre 2M pontos a 30s custa ~1,4G operações e domina o
    tempo total, sem mudar o resultado — o portão só é lido nos instantes do health.
    """
    t5s = t5.ewm(halflife=int(pd.Timedelta(hours=RAMP_HL_H) / pd.Timedelta("30s"))).mean()
    t5g = t5s.resample(ev.SAMPLING_INTERVAL).last().ffill()
    dt_h = t5g.index.to_series().diff().dt.total_seconds() / 3600.0
    ramp = (t5g.diff() / dt_h).abs()
    return ramp.rolling(GATE_WIN, min_periods=1).max(), t5g


def gated_signal(h: pd.Series, q: float, g: pd.Series, lv: pd.Series,
                 r_max: float, l_min: float) -> pd.Series:
    ok = (g < r_max) | g.isna()
    if l_min > 0:
        ok &= (lv >= l_min) | lv.isna()
    return (h >= q) & ok


def main() -> None:
    running, tc03, t5 = sw.load_raw()
    gate, t5s = ramp_signal(t5)

    row = pd.read_csv(FLEET_CSV).set_index("sensor").loc[SENSOR]
    hl, q = float(row["hl"]), float(row["threshold_q"])
    mae = ev.load_mae_series(Task.get_task(task_id=TASK_B2024), [SENSOR])[SENSOR]
    h_full = sw.ewma_on(mae, hl, running).rank(pct=True)
    # alinhar o portão ao health UMA vez (o sweep tem 24 pontos × 3 cenários)
    g_full = gate.reindex(h_full.index, method="nearest")
    lv_full = t5s.reindex(h_full.index, method="nearest")

    # ---- sanidade: o braço base tem de reproduzir a auditoria antes de comparar
    inc_full = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())
    days_full = (h_full.index[-1] - h_full.index[0]).total_seconds() / 86400.0
    base = on.evaluate(h_full >= q, inc_full, days_full)
    print(f"[sanidade] base FULL: recall_raw={base['recall_raw']:.3f} "
          f"(esp {float(row['recall_raw']):.3f})  fa={base['fa_per_day']:.3f} "
          f"(esp {float(row['fa_per_day']):.3f})  duty={base['duty_sticky']:.3f} "
          f"(esp {float(row['duty_sticky']):.3f})")
    if not (abs(base["recall_raw"] - float(row["recall_raw"])) < 0.01
            and abs(base["fa_per_day"] - float(row["fa_per_day"])) < 0.01
            and abs(base["duty_sticky"] - float(row["duty_sticky"])) < 0.01):
        raise SystemExit("ponto de operacao nao reproduz a auditoria — abortando.")

    # ---- quanto cada TP "manobra": o custo do portão, incidente a incidente
    print(f"\n[risco] rampa no onset dos {len(inc_full)} incidentes reais "
          f"(máx de |dT5/dt| em [-6h, t])")
    g_inc = []
    for t in inc_full:
        w = gate[(gate.index >= t - pd.Timedelta(hours=6)) & (gate.index <= t)]
        g_inc.append(float(np.nanmax(w.values)) if len(w) else np.nan)
    g_inc = np.array(g_inc, dtype=float)
    for r in R_GRID[:-1]:
        n = int(np.nansum(g_inc >= r))
        print(f"  R={r:>5.0f} °C/h  → {n:>2}/{len(inc_full)} incidentes ficariam "
              f"em zona de supressão ({n / len(inc_full):.0%})")
    print(f"  distribuição da rampa nos incidentes: p50={np.nanmedian(g_inc):.0f} "
          f"p90={np.nanpercentile(g_inc, 90):.0f} máx={np.nanmax(g_inc):.0f} °C/h")

    # ---- sweep
    rows = []
    for label, t0, t1 in SCENARIOS:
        m = mae
        if t0 is not None:
            m = m[m.index >= t0]
        if t1 is not None:
            m = m[m.index < t1]
        sl = (h_full.index >= m.index.min()) & (h_full.index <= m.index.max())
        h, g, lv = h_full[sl], g_full[sl], lv_full[sl]
        inc = sw.incidents_on(running, tc03, m.index.min(), m.index.max())
        days = (h.index[-1] - h.index[0]).total_seconds() / 86400.0
        print(f"\n=== {label} — {len(inc)} incidentes ON, {days:.0f} dias ===")
        for l_min in L_GRID:
            for r_max in R_GRID:
                sig = gated_signal(h, q, g, lv, r_max, l_min)
                res = on.evaluate(sig, inc, days)
                rows.append(dict(cenario=label, R=r_max, L=l_min, dias=days, **res))
                tag = "base " if (np.isinf(r_max) and l_min == 0) else "     "
                print(f"  {tag}R={'inf' if np.isinf(r_max) else f'{r_max:.0f}':>4} "
                      f"L={l_min:>3.0f}  recall={res['recall_raw']:.1%} "
                      f"({res['n_hit']}/{res['n_inc']})  fa={res['fa_per_day']:.3f}  "
                      f"duty={res['duty_sticky']:.3f}  FP={res['n_fp']:>3}  "
                      f"lead={res['lead_med_h']:.1f}h")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nGravado: {OUT_CSV}")

    # ---- veredito contra o critério fixado antes de rodar
    b = df[(df.cenario == "FULL") & np.isinf(df.R) & (df.L == 0)].iloc[0]
    oos_b = df[(df.cenario == "OOS") & np.isinf(df.R) & (df.L == 0)].iloc[0]
    cand = df[(df.cenario == "FULL")
              & (df.fa_per_day <= 0.70 * b.fa_per_day)
              & (df.recall_raw >= b.recall_raw - 0.05)]
    print(f"\n[veredito] critério: FA ≤ {0.70 * b.fa_per_day:.3f} e "
          f"recall_raw ≥ {b.recall_raw - 0.05:.3f} no FULL, mantendo "
          f"{int(oos_b.n_hit)}/{int(oos_b.n_inc)} no OOS")
    if cand.empty:
        print("  NENHUM ponto passa no FULL → portão de rampa REFUTADO.")
        return
    for _, c in cand.iterrows():
        o = df[(df.cenario == "OOS") & (df.R == c.R) & (df.L == c.L)].iloc[0]
        ok = o.n_hit >= oos_b.n_hit
        print(f"  R={c.R:.0f} L={c.L:.0f}: FULL {c.recall_raw:.1%} fa={c.fa_per_day:.3f} "
              f"| OOS {int(o.n_hit)}/{int(o.n_inc)} → {'PASSA' if ok else 'reprova no OOS'}")


if __name__ == "__main__":
    main()
