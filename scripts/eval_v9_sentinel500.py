"""Avaliação da task 2c9ccb1d (per-sensor dense, SENTINEL 500, TRAIN_END=2025-07-01)
com a métrica de produção (eval_per_sensor_level), replicando close_fleet_off_excl.py.

Diferenças vs baseline:
  - task 2c9ccb1d (janela MAE 2025-01 → 2026-04, 5 min)
  - ON via RUNNING_A > 0.5 (CSV 2025-2026), não NGP_A > 50
  - 7 sensores (T5 + TC382_01..06)
  - 4 cenários: FULL/OOS (corte 2025-07-01) × allcond/hihi-only

Uso:
  PYTHONPATH=. python scripts/eval_v9_sentinel500.py
"""
import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E

TASK_ID = "2c9ccb1d309c4537a2f02ae1239663d9"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
RAWCSV = "../dados/sensores_brutos_2025_2026_30s.csv"
HORIZON, STICKY, FA_BUDGET = 8.0, 12.0, 1.0
HL_GRID = [0.5, 1.0, 2.0, 4.0]
MAX_DUTY = 0.35
OOS_START = pd.Timestamp("2025-07-01", tz="UTC")
SENSORS7 = ["T5_AVG_A", "TC382_01_A", "TC382_02_A", "TC382_03_A",
            "TC382_04_A", "TC382_05_A", "TC382_06_A"]


def running_series():
    # low_memory=False + seleção posterior: usecols quebra com chunks de dtype misto
    cols = ["data_datetime", "RUNNING_A"] + SENSORS7
    raw = pd.read_csv(RAWCSV, low_memory=False)
    raw = raw[[c for c in cols if c in raw.columns]]
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    vals = raw[[c for c in SENSORS7 if c in raw.columns]].apply(pd.to_numeric, errors="coerce")
    return running, vals


def best_over_hl(mae: pd.Series, inc, running: pd.Series) -> dict:
    if not inc:
        return dict(recall=float("nan"), fa_per_day=float("nan"),
                    n_incidents=0, n_hit=0, hl=float("nan"), threshold_q=float("nan"))
    best = None
    for hl in HL_GRID:
        hl_pts = max(1, int(round(pd.Timedelta(hours=hl) / pd.Timedelta(E.SAMPLING_INTERVAL))))
        ew = mae.ewm(halflife=hl_pts).mean()
        on_h = running.reindex(ew.index, method="nearest") > 0.5
        h = ew.where(on_h).dropna().rank(pct=True)
        if h.empty:
            continue
        r = E.best_point_for_sensor(h, inc, horizon_hours=HORIZON,
                                    sticky_hours=STICKY, fa_budget=FA_BUDGET,
                                    n_thresholds=120, max_duty_cycle=MAX_DUTY,
                                    max_sticky_duty=0.25)
        r["hl"] = hl
        key = (r["recall"], r.get("median_lead_hours", 0.0), -r["fa_per_day"])
        if best is None or key > (best["recall"], best.get("median_lead_hours", 0.0), -best["fa_per_day"]):
            best = r
    return best or dict(recall=float("nan"), fa_per_day=float("nan"),
                        n_incidents=len(inc), n_hit=0, hl=float("nan"),
                        threshold_q=float("nan"))


def health_of(mae: pd.Series, hl: float, running: pd.Series) -> pd.Series:
    hl_pts = max(1, int(round(pd.Timedelta(hours=hl) / pd.Timedelta(E.SAMPLING_INTERVAL))))
    ew = mae.ewm(halflife=hl_pts).mean()
    on_h = running.reindex(ew.index, method="nearest") > 0.5
    return ew.where(on_h).dropna().rank(pct=True)


def leads_at(h: pd.Series, inc, q: float, horizon_hours: float):
    """(recall, lead mediano em horas) no ponto de operação fixo, horizonte dado."""
    al = E.apply_sticky(h, q, STICKY)
    als = np.array([t.timestamp() for t in h.index[al]])
    hs = horizon_hours * 3600.0
    leads = []
    for t in inc:
        ti = t.timestamp()
        w = als[(als >= ti - hs) & (als <= ti)] if als.size else np.array([])
        if w.size:
            leads.append((ti - w.min()) / 3600.0)
    rec = len(leads) / len(inc) if inc else float("nan")
    med = float(np.median(leads)) if leads else float("nan")
    return rec, med


def evaluate(mae_all, alarms_gap, running: pd.Series, t0=None, t1=None,
             sensor_vals: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for sensor in SENSORS7:
        mae = mae_all.get(sensor)
        if mae is None or mae.empty:
            continue
        if t0 is not None:
            mae = mae[mae.index >= t0]
        if t1 is not None:
            mae = mae[mae.index < t1]
        if mae.empty:
            continue
        t_lo, t_hi = mae.index.min(), mae.index.max()
        raw_al = [t for t in alarms_gap.get(sensor, []) if t_lo <= t <= t_hi]
        inc_raw = E.cluster_incidents(raw_al, gap_hours=E.GAP_HOURS)
        if inc_raw:
            on_inc = running.reindex(pd.DatetimeIndex(inc_raw), method="nearest") > 0.5
            inc_on = [t for t, o in zip(inc_raw, on_inc.values) if o]
        else:
            inc_on = []
        # Alarme-fantasma: HI/HIHI com sensor lendo <500°C no onset apesar de RUNNING=ON
        # (contradição física — cf. detalhe_alucinacoes_*.csv) → fora do denominador
        n_ghost = 0
        if inc_on and sensor_vals is not None and sensor in sensor_vals.columns:
            v_at = sensor_vals[sensor].reindex(pd.DatetimeIndex(inc_on), method="nearest")
            keep = v_at.isna() | (v_at >= 500.0)
            n_ghost = int((~keep).sum())
            inc_on = [t for t, k in zip(inc_on, keep.values) if k]

        r = best_over_hl(mae, inc_on, running)
        # lead nos horizontes largos (24/72h): mede a antecipação real do mesmo
        # ponto de operação, sem a saturação do horizonte de 8h
        rec24 = lead24 = rec72 = lead72 = float("nan")
        if inc_on and pd.notna(r.get("hl")):
            h = health_of(mae, r["hl"], running)
            rec24, lead24 = leads_at(h, inc_on, r["threshold_q"], 24.0)
            rec72, lead72 = leads_at(h, inc_on, r["threshold_q"], 72.0)
        rows.append({
            "sensor": sensor,
            "inc_all": len(inc_raw), "inc_on": len(inc_on),
            "inc_off": len(inc_raw) - len(inc_on) - n_ghost, "inc_ghost": n_ghost,
            "recall": r.get("recall"), "recall_raw": r.get("recall_raw"),
            "fa_per_day": r.get("fa_per_day"),
            "duty_sticky": r.get("duty_sticky"),
            "hl": r.get("hl"), "threshold_q": r.get("threshold_q"),
            "lead_med_h": r.get("median_lead_hours"),
            "lead24_med_h": lead24, "lead72_med_h": lead72,
            "recall_24h": rec24, "recall_72h": rec72,
        })
    return pd.DataFrame(rows)


def main():
    global RAWCSV
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default=TASK_ID)
    ap.add_argument("--rawcsv", default=RAWCSV,
                    help="CSV com data_datetime + RUNNING_A cobrindo a janela do MAE")
    ap.add_argument("--tag", default="v9_sentinel500", help="prefixo dos CSVs de saida")
    ap.add_argument("--backcast_2024", action="store_true",
                    help="adiciona cenario BACKCAST_2024 (janela 2024, HI/HIHI-only)")
    ap.add_argument("--extra_2024", action="store_true",
                    help="adiciona 2024_H1 (jan-mai, so existe a partir da remessa de "
                         "10/08/2026) e 2024_FULL (ano inteiro). ⚠️ 2024_H1 e 2024_FULL "
                         "tem DENOMINADOR diferente do BACKCAST_2024 classico (jun-dez): "
                         "nao comparar os percentuais entre si, so entre bracos no MESMO "
                         "recorte — ver memoria confound-janela-de-dado")
    args = ap.parse_args()
    RAWCSV = args.rawcsv

    running, sensor_vals = running_series()
    print(f"RUNNING_A: {len(running)} pts, ON={float((running > 0.5).mean()) * 100:.1f}%")

    task = Task.get_task(task_id=args.task_id)
    mae_all = E.load_mae_series(task, SENSORS7)

    alarms_all = E.load_alarms_gap(ALARM)
    alarms_hihi = E.load_alarms_gap(ALARM, exclude_conditions=["UNDER", "CFN", "LOLO", "OVER"])

    scenarios = [
        ("FULL_allcond", alarms_all, None, None),
        ("FULL_hihihi", alarms_hihi, None, None),
        ("OOS_allcond", alarms_all, OOS_START, None),
        ("OOS_hihihi", alarms_hihi, OOS_START, None),
    ]
    if args.backcast_2024:
        scenarios.append(("BACKCAST_2024_hihihi", alarms_hihi,
                          pd.Timestamp("2024-06-01", tz="UTC"),
                          pd.Timestamp("2025-01-01", tz="UTC")))
        scenarios.append(("BACKCAST_2024_allcond", alarms_all,
                          pd.Timestamp("2024-06-01", tz="UTC"),
                          pd.Timestamp("2025-01-01", tz="UTC")))
    if args.extra_2024:
        scenarios.append(("BACKCAST_2024H1_hihihi", alarms_hihi,
                          pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2024-06-01", tz="UTC")))
        scenarios.append(("BACKCAST_2024FULL_hihihi", alarms_hihi,
                          pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2025-01-01", tz="UTC")))
    pd.set_option("display.width", 160, "display.max_columns", 20)
    for label, alarms, t0, t1 in scenarios:
        print(f"\n===== {label} =====")
        df = evaluate(mae_all, alarms, running, t0=t0, t1=t1, sensor_vals=sensor_vals)
        if df.empty:
            print("(sem dados na janela)")
            continue
        out = df.copy()
        out["recall"] = df["recall"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "  -")
        out["fa_per_day"] = df["fa_per_day"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "  -")
        print(out.to_string(index=False))
        m = df[df.inc_on > 0]
        if len(m):
            print(f"macro recall (inc_on>0): {m.recall.mean()*100:.1f}%  ({len(m)} sensores)")
        path = f"eval_predictive_out/fleet_{args.tag}_{label}.csv"
        df.to_csv(path, index=False)
        print(f"csv: {path}")


if __name__ == "__main__":
    main()
