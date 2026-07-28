"""Testa se um voto OR entre os 6 detectores de resíduo por canal TC382_0X_A
(já treinados, task v11_residual_cnn1d) serve de proxy de early-warning pros
incidentes reais de T5_AVG_A — que não tem resíduo próprio (degenerado, T5 é a
média dos 6 canais).

Como só TC382_03_A tem amostra suficiente pra calibrar um threshold por incidente,
usamos um quantil COMPARTILHADO entre os 6 canais (mesma régua pra todos) em vez de
calibração individual — e varremos esse quantil pra achar o melhor ponto pro voto OR.

Uso:
    PYTHONPATH=. python scripts/ensemble_t5_via_tc382.py \
        --task_id 11978d260dbf4301838fff35452bf97f \
        --eval_start 2026-01-01 --sticky_hours 12 --horizon 8
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from clearml import Task

from scripts.eval_per_sensor_level import (
    ALARM_CSV_DEFAULT,
    load_mae_series,
    ewma_quantile,
    apply_sticky,
    detect_episodes_gap,
    load_alarms_gap,
    cluster_incidents,
)

TC_SENSORS = ["TC382_01_A", "TC382_02_A", "TC382_03_A",
              "TC382_04_A", "TC382_05_A", "TC382_06_A"]


def evaluate_ensemble(
    healths: dict,
    incidents: list,
    horizon_hours: float,
    sticky_hours: float,
    q: float,
    fa_budget: float,
) -> dict:
    """OR-combina alertas dos 6 canais no mesmo quantil q e mede recall/FA contra
    os incidentes de T5."""
    # índice comum: união de todos os índices, health ausente vira 0 (sem alerta)
    common_idx = healths[TC_SENSORS[0]].index
    for s in TC_SENSORS[1:]:
        common_idx = common_idx.union(healths[s].index)
    common_idx = common_idx.sort_values()

    combined_alert = pd.Series(False, index=common_idx)
    for s in TC_SENSORS:
        h = healths[s].reindex(common_idx)
        alert = apply_sticky(h.fillna(0.0), q, sticky_hours)
        combined_alert = combined_alert | alert.fillna(False)

    total_days = (common_idx[-1] - common_idx[0]).total_seconds() / 86400.0
    horizon_sec = horizon_hours * 3600.0
    inc_s = np.array([t.timestamp() for t in incidents])
    alert_s = np.array([t.timestamp() for t in common_idx[combined_alert]])

    episodes = detect_episodes_gap(combined_alert)

    n_hit = 0
    leads = []
    for ti in inc_s:
        w = alert_s[(alert_s >= ti - horizon_sec) & (alert_s <= ti)] if alert_s.size else np.array([])
        if w.size:
            n_hit += 1
            leads.append((ti - w.min()) / 3600.0)

    n_fp = sum(
        1 for (s0, s1) in episodes
        if not (np.any((inc_s - horizon_sec <= s1.timestamp()) & (inc_s >= s0.timestamp())) if inc_s.size else False)
    )
    fa = n_fp / max(total_days, 1.0)
    recall = n_hit / len(incidents) if incidents else 0.0
    return {
        "q": q, "recall": recall, "fa_per_day": fa, "n_hit": n_hit,
        "n_incidents": len(incidents), "median_lead_h": float(np.median(leads)) if leads else 0.0,
        "duty": float(combined_alert.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", required=True)
    p.add_argument("--alarm_csv", default=ALARM_CSV_DEFAULT)
    p.add_argument("--half_life", type=float, default=4.0)
    p.add_argument("--sticky_hours", type=float, default=12.0)
    p.add_argument("--horizon", type=float, default=8.0)
    p.add_argument("--fa_budget", type=float, default=1.0)
    p.add_argument("--eval_start", default=None)
    p.add_argument("--exclude_conditions", nargs="*", default=["UNDER", "OVER", "LOLO", "CFN"])
    p.add_argument("--n_thresholds", type=int, default=60)
    p.add_argument("--max_duty", type=float, default=0.25,
                   help="teto de tempo-em-alerta (pós-sticky) do voto OR combinado — "
                        "sem isso a busca converge pro piso q=0.5 (alerta quase sempre "
                        "ligado), que engana a métrica de FA-por-episódio")
    args = p.parse_args()

    print(f"[1/4] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)

    print(f"[2/4] Baixando sequence_scores dos 6 canais TC382...")
    mae_dict = load_mae_series(task, TC_SENSORS)
    missing = [s for s in TC_SENSORS if s not in mae_dict]
    if missing:
        print(f"  [ERRO] canais ausentes: {missing}")
        return

    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    healths = {}
    for s in TC_SENSORS:
        mae = mae_dict[s]
        if t0 is not None:
            mae = mae[mae.index >= t0]
        healths[s] = ewma_quantile(mae, args.half_life)

    print(f"[3/4] Carregando incidentes de T5_AVG_A (HI/HIHI-only)...")
    raw_alarms = load_alarms_gap(args.alarm_csv, args.exclude_conditions)
    incidents = cluster_incidents(raw_alarms.get("T5_AVG_A", []))
    if t0 is not None:
        incidents = [i for i in incidents if pd.Timestamp(i, tz="UTC" if pd.Timestamp(i).tz is None else None) >= t0]
    print(f"  {len(incidents)} incidentes de T5_AVG_A na janela avaliada")

    print(f"[4/4] Varrendo quantil compartilhado entre os 6 canais...")
    rows = []
    for q in np.linspace(0.50, 0.999, args.n_thresholds):
        r = evaluate_ensemble(healths, incidents, args.horizon, args.sticky_hours, q, args.fa_budget)
        rows.append(r)

    df = pd.DataFrame(rows)
    within_budget = df[(df["fa_per_day"] <= args.fa_budget) & (df["duty"] <= args.max_duty)]
    if within_budget.empty:
        print(f"  Nenhum ponto dentro do orçamento de FA e duty<={args.max_duty}.")
        return
    best = within_budget.sort_values(["recall", "fa_per_day"], ascending=[False, True]).iloc[0]
    print(f"\n=== Melhor ponto do ENSEMBLE (voto OR, 6 canais TC382) ===")
    print(f"  q={best['q']:.3f}  recall={best['recall']:.1%}  FA/dia={best['fa_per_day']:.3f}  "
          f"duty={best['duty']:.2f}  lead_mediano={best['median_lead_h']:.1f}h  "
          f"({int(best['n_hit'])}/{int(best['n_incidents'])} incidentes)")

    out_path = "eval_predictive_out/ensemble_t5_via_tc382_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"  Sweep completo salvo em: {out_path}")


if __name__ == "__main__":
    main()
