#!/usr/bin/env python3
"""
sweep_regime_band_offline.py
Testa, SEM retreinar e sem servidor ClearML, se um ponto de operação POR REGIME
melhora o TC382_03_A — em particular o backcast 2024 (recall_raw 21,4% na janela
de 8h, contra 100% no OOS 2025), que é a fraqueza conhecida do modelo v10.

Hipótese: o "normal" de 2024 opera em outro nível térmico; um rank GLOBAL da EWMA
do erro deixa o threshold calibrado para o regime dominante. Se o rank for
calculado DENTRO de bandas de regime (nível de carga), o mesmo erro relativo
vira o mesmo percentil em qualquer regime.

Braços (todos pós-processamento do MESMO MAE em cache — nada de treino):
  global    — rank global da EWMA (protocolo atual; reproduz a linha da auditoria)
  band2_t5 / band3_t5
            — rank dentro de bandas de nível do T5_AVG_A suavizado (half-life 24h).
              O sinal de banda é LENTO de propósito: uma excursão de horas não muda
              a banda, senão o próprio evento se auto-normalizaria. T5 (média do
              escape) como proxy de carga, e não o próprio TC03, pelo mesmo motivo.
  rolling30 — rank contra a distribuição móvel dos últimos 30 dias (causal).
              Equivale a um baseline adaptativo: imune a drift lento de regime.

Protocolo idêntico ao da auditoria (eval_v9_sentinel500.py, linhas da tabela
fleet_audit_2e92c618_*): HL_GRID {0.5,1,2,4}, horizonte 8h, sticky 12h,
FA≤1/dia, duty bruto ≤0.35, duty pós-sticky ≤0.25, incidentes HI/HIHI com
máquina ON e filtro de fantasma (<500°C). Cenários FULL / BACKCAST_2024 / OOS.

O arquivo de MAE do v10/TC03 é identificado no cache por IMPRESSÃO DIGITAL
(reproduzir recall_raw/FA/duty no ponto de operação gravado na auditoria), como
em plot_spike_experiment_offline.py — sem chute por data de arquivo.

Uso:
    PYTHONPATH=. python scripts/sweep_regime_band_offline.py
"""
from __future__ import annotations

import glob
import importlib.util
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "eval_per_sensor_level", os.path.join(_HERE, "eval_per_sensor_level.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

CACHE = os.path.expanduser("~/.clearml/cache/storage_manager/global")
RAW_CSV = "../dados/sensores_2024h2_2025_2026_30s.csv"
ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
SENSOR = "TC382_03_A"
OUT_CSV = "eval_predictive_out/regime_band_sweep_TC382_03_A.csv"

HL_GRID = [0.5, 1.0, 2.0, 4.0]
HORIZON, STICKY, FA_BUDGET = 8.0, 12.0, 1.0
MAX_DUTY, MAX_STICKY = 0.35, 0.25
EXCL = ["UNDER", "CFN", "LOLO", "OVER"]

# fleet_audit_2e92c618_FULL_hihihi.csv, linha TC382_03_A — a impressão digital
FP_HL, FP_Q = 0.5, 0.936101
FP_RECALL_RAW, FP_FA, FP_DUTY = 46 / 58, 0.145584, 0.244922

BACKCAST = (pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
OOS = (pd.Timestamp("2025-07-01", tz="UTC"), None)


def hl_pts(hl_hours: float) -> int:
    return max(1, int(round(pd.Timedelta(hours=hl_hours) / pd.Timedelta(ev.SAMPLING_INTERVAL))))


def load_raw() -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")
    t5 = pd.to_numeric(raw["T5_AVG_A"], errors="coerce")
    return running, tc03, t5


def read_mae(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
    return df.set_index("seq_start_time")["mae_seq"]


def ewma_on(mae: pd.Series, hl: float, running: pd.Series) -> pd.Series:
    ew = mae.ewm(halflife=hl_pts(hl)).mean()
    on = running.reindex(ew.index, method="nearest") > 0.5
    return ew.where(on).dropna()


def metrics_at(health: pd.Series, q: float, incidents: list) -> tuple[float, float, float]:
    """(recall_raw, fa/dia, duty_sticky) num ponto fixo — réplica do best_point."""
    total_days = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s = np.array([t.timestamp() for t in incidents])
    alert = ev.apply_sticky(health, q, STICKY)
    duty_sticky = float(alert.mean())
    episodes = ev.detect_episodes_gap(alert)
    raw_s = np.array([t.timestamp() for t in health.index[health >= q]])
    hs = HORIZON * 3600.0
    n_raw = sum(1 for ti in inc_s
                if raw_s.size and np.any((raw_s >= ti - hs) & (raw_s <= ti)))
    n_fp = sum(1 for (s0, s1) in episodes
               if not (np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp()))
                       if inc_s.size else False))
    return (n_raw / len(inc_s) if inc_s.size else float("nan"),
            n_fp / max(total_days, 1.0), duty_sticky)


def incidents_on(running: pd.Series, tc03: pd.Series,
                 t_lo, t_hi) -> list:
    alarms = ev.load_alarms_gap(ALARM_CSV, exclude_conditions=EXCL).get(SENSOR, [])
    raw = [a for a in alarms if t_lo <= a <= t_hi]
    inc = ev.cluster_incidents(raw, gap_hours=ev.GAP_HOURS)
    if not inc:
        return []
    on = running.reindex(pd.DatetimeIndex(inc), method="nearest") > 0.5
    inc = [t for t, o in zip(inc, on.values) if o]
    if inc:
        v = tc03.reindex(pd.DatetimeIndex(inc), method="nearest")
        inc = [t for t, val in zip(inc, v.values) if pd.isna(val) or val >= 500.0]
    return inc


def fingerprint(files: list[str], running: pd.Series, incidents: list) -> str:
    print(f"[1/3] Impressão digital em {len(files)} arquivos de cache "
          f"(alvo: recall_raw={FP_RECALL_RAW:.4f} fa={FP_FA:.3f} duty={FP_DUTY:.3f})")
    hits = []
    for f in files:
        try:
            mae = read_mae(f)
        except Exception:
            continue
        if mae.index.min() > pd.Timestamp("2024-07-01", tz="UTC"):
            continue
        h = ewma_on(mae, FP_HL, running).rank(pct=True)
        rr, fa, duty = metrics_at(h, FP_Q, incidents)
        if (abs(rr - FP_RECALL_RAW) < 0.01 and abs(fa - FP_FA) < 0.01
                and abs(duty - FP_DUTY) < 0.01):
            hits.append(f)
            print(f"  MATCH {os.path.basename(f)[:36]}  rr={rr:.4f} fa={fa:.3f} duty={duty:.3f}")
    if len(hits) != 1:
        raise SystemExit(f"esperava exatamente 1 match, achei {len(hits)} — "
                         "impressão digital ambígua, não vou chutar.")
    return hits[0]


# ---------- braços ----------

def health_global(mae, hl, running, t5s):
    return ewma_on(mae, hl, running).rank(pct=True)


def make_health_banded(n_bands: int):
    def fn(mae, hl, running, t5s):
        ew = ewma_on(mae, hl, running)
        regime = t5s.reindex(ew.index, method="nearest")
        try:
            bands = pd.qcut(regime, n_bands, duplicates="drop")
        except ValueError:
            return ew.rank(pct=True)
        return ew.groupby(bands, observed=True).rank(pct=True)
    return fn


def health_rolling(mae, hl, running, t5s, days: int = 30):
    ew = ewma_on(mae, hl, running)
    r = ew.rolling(f"{days}D", min_periods=hl_pts(24.0)).rank(pct=True)
    return r.dropna()


ARMS = {
    "global": health_global,
    "band2_t5": make_health_banded(2),
    "band3_t5": make_health_banded(3),
    "rolling30": health_rolling,
}


def best_over_hl(mae, incidents, running, t5s, health_fn) -> dict:
    if not incidents:
        return {}
    best = None
    for hl in HL_GRID:
        h = health_fn(mae, hl, running, t5s)
        if h.empty:
            continue
        r = ev.best_point_for_sensor(h, incidents, horizon_hours=HORIZON,
                                     sticky_hours=STICKY, fa_budget=FA_BUDGET,
                                     n_thresholds=120, max_duty_cycle=MAX_DUTY,
                                     max_sticky_duty=MAX_STICKY)
        r["hl"] = hl
        key = (r["recall"], r.get("median_lead_hours", 0.0), -r["fa_per_day"])
        if best is None or key > (best["recall"], best.get("median_lead_hours", 0.0),
                                  -best["fa_per_day"]):
            best = r
    return best or {}


def main() -> None:
    running, tc03, t5 = load_raw()
    # sinal de regime: T5 suavizado com half-life de 24h — lento por construção
    t5s = t5.ewm(halflife=int(pd.Timedelta(hours=24) / pd.Timedelta("30s"))).mean()

    files = sorted(glob.glob(os.path.join(CACHE, "*.sequence_scores_all.csv")))
    inc_full_probe = None  # incidentes p/ fingerprint: janela típica do full
    probe_mae = read_mae(files[0])
    # janela do fingerprint = janela FULL da auditoria (a do próprio arquivo)
    t_lo = pd.Timestamp("2024-06-01", tz="UTC")
    t_hi = pd.Timestamp("2026-04-30 23:59:59", tz="UTC")
    inc_full = incidents_on(running, tc03, t_lo, t_hi)
    print(f"Incidentes HI/HIHI ON (filtro fantasma) na janela FULL: {len(inc_full)}")

    path = fingerprint(files, running, inc_full)

    mae = read_mae(path)
    scenarios = [
        ("FULL", None, None),
        ("BACKCAST_2024", *BACKCAST),
        ("OOS", *OOS),
    ]
    print(f"\n[2/3] Sweep: {len(ARMS)} braços × {len(scenarios)} cenários")
    rows = []
    for label, t0, t1 in scenarios:
        m = mae
        if t0 is not None:
            m = m[m.index >= t0]
        if t1 is not None:
            m = m[m.index < t1]
        inc = incidents_on(running, tc03, m.index.min(), m.index.max())
        for arm, fn in ARMS.items():
            r = best_over_hl(m, inc, running, t5s, fn)
            rows.append({
                "cenario": label, "braco": arm, "inc_on": len(inc),
                "recall": r.get("recall"), "recall_raw": r.get("recall_raw"),
                "fa_per_day": r.get("fa_per_day"), "duty_sticky": r.get("duty_sticky"),
                "hl": r.get("hl"), "threshold_q": r.get("threshold_q"),
                "lead_med_h": r.get("median_lead_hours"),
            })
            rr = r.get("recall_raw")
            print(f"  {label:<14} {arm:<10} inc={len(inc):>2}  "
                  f"recall_raw={'—' if rr is None else f'{rr:.1%}'}  "
                  f"fa={r.get('fa_per_day', float('nan')):.3f}  "
                  f"duty={r.get('duty_sticky', float('nan')):.3f}  hl={r.get('hl')}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[3/3] Sanidade: braço 'global' deve reproduzir a auditoria "
          f"(FULL rr≈{FP_RECALL_RAW:.1%}, BACKCAST rr≈21.4%, OOS rr≈100%).")
    print(f"Gravado em {OUT_CSV}")


if __name__ == "__main__":
    main()
