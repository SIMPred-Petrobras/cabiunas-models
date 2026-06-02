#!/usr/bin/env python3
"""
eval_predictive_layer.py
Avalia qualquer task ClearML per_sensor usando a camada preditiva OR+EWMA,
sem re-treinar: baixa os sequence_scores_all.csv de cada sensor e aplica
EWMA → quantile → OR → sweep de threshold → curva recall × FA/dia.

Gera comparação direta com o baseline TS=60 (per_sensor_out/).

Uso:
    PYTHONPATH=. python scripts/eval_predictive_layer.py \
        --task_id 7c641d9d46d54c3a88a6db84fd1351ee \
        --label v6_ts096
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clearml import Task

SENSORS = [
    "T5_AVG_A",
    "TC382_01_A", "TC382_02_A", "TC382_03_A",
    "TC382_04_A", "TC382_05_A", "TC382_06_A",
    "TV_351X_A",  "TV_351Y_A",
    "TV_352X_A",  "TV_352Y_A",
    "TV_353X_A",  "TV_353Y_A",
    "TV_354X_A",  "TV_354Y_A",
    "TV_355X_A",  "TV_355Y_A",
]

ALARM_CSV_DEFAULT   = "../dados/alarmes_selecionados_turbina_a.csv"
BASELINE_DIR        = "per_sensor_out"
SAMPLING_INTERVAL   = "5min"   # STRIDE=10 × 30 s = 5 min
DEBOUNCE_HOURS      = 8.0      # mesmo que PREDICTIVE_ALERT_DEBOUNCE_HOURS


# ---------------------------------------------------------------------------
# 1. Carrega sequence_scores de cada sensor
# ---------------------------------------------------------------------------

def load_mae_series(task: Task, sensors: List[str]) -> Dict[str, pd.Series]:
    arts = task.artifacts
    series: Dict[str, pd.Series] = {}
    for sensor in sensors:
        key = next(
            (k for k in arts if "sequence_scores_all" in k and sensor in k), None
        )
        if key is None:
            print(f"  [WARN] {sensor}: sequence_scores_all.csv não encontrado — ignorado")
            continue
        path = arts[key].get_local_copy()
        df = pd.read_csv(path)
        df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
        s = df.set_index("seq_start_time")["mae_seq"]
        series[sensor] = s
    print(f"  {len(series)}/{len(sensors)} sensores carregados")
    return series


# ---------------------------------------------------------------------------
# 2. EWMA + normalização por quantile
# ---------------------------------------------------------------------------

def ewma_quantile(mae: pd.Series, half_life_hours: float) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(SAMPLING_INTERVAL)))
    health = mae.ewm(halflife=max(1, hl_pts)).mean()
    return health.rank(pct=True)


# ---------------------------------------------------------------------------
# 3. OR: max do quantile normalizado em grid temporal comum
# ---------------------------------------------------------------------------

def combine_or(health_dict: Dict[str, pd.Series]) -> pd.Series:
    # Grid comum: união de todos os timestamps
    all_idx = pd.DatetimeIndex([])
    for s in health_dict.values():
        all_idx = all_idx.union(s.index)
    all_idx = all_idx.sort_values()

    combined = np.zeros(len(all_idx))
    for s in health_dict.values():
        r = s.reindex(all_idx).ffill().fillna(0.0).values
        combined = np.maximum(combined, r)
    return pd.Series(combined, index=all_idx)


# ---------------------------------------------------------------------------
# 4. Carrega alarmes e agrupa em incidentes
# ---------------------------------------------------------------------------

def load_alarms(alarm_csv: str) -> List[pd.Timestamp]:
    """Carrega onset alarms (exclui 'OK' / resets) — mesmo filtro do baseline."""
    df = pd.read_csv(alarm_csv)
    date_col = next(
        (c for c in df.columns if "ocorr" in c.lower() or ("data" in c.lower() and "ação" not in c.lower())),
        df.columns[0],
    )
    times = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    cond_col = next((c for c in df.columns if "condi" in c.lower()), None)
    if cond_col:
        mask = df[cond_col].str.upper().fillna("").ne("OK")
        times = times[mask]
    return sorted(times.dropna().tolist())


def cluster_incidents(alarm_times: List[pd.Timestamp], gap_hours: float = 4.0) -> List[pd.Timestamp]:
    """Agrupa alarmes consecutivos dentro de gap_hours num único incidente.
    Usa pandas groupby para replicar exatamente ae_separation_experiment.load().
    """
    if not alarm_times:
        return []
    s = pd.Series(alarm_times, name="t").sort_values().reset_index(drop=True)
    g = (s.diff().dt.total_seconds() / 3600 > gap_hours).cumsum()
    return s.groupby(g).first().tolist()


# ---------------------------------------------------------------------------
# 5. Detecção de episódios com gap-debounce (igual ao baseline)
# ---------------------------------------------------------------------------

def detect_episodes_gap(alert: pd.Series) -> List[tuple]:
    """Agrupa pontos de alerta consecutivos com gap < DEBOUNCE_HOURS num episódio.
    Retorna lista de (start, end) como pd.Timestamp.
    Replica lógica de ae_per_sensor_experiment.py.
    """
    debounce = pd.Timedelta(hours=DEBOUNCE_HOURS)
    on_idx = alert.index[alert]
    if len(on_idx) == 0:
        return []
    episodes: List[tuple] = []
    cur_start = on_idx[0]
    cur_end   = on_idx[0]
    for t in on_idx[1:]:
        if (t - cur_end) <= debounce:
            cur_end = t
        else:
            episodes.append((cur_start, cur_end))
            cur_start = cur_end = t
    episodes.append((cur_start, cur_end))
    return episodes


# ---------------------------------------------------------------------------
# 6. Avaliação: recall × FA/dia para um horizonte H
# ---------------------------------------------------------------------------

def evaluate_horizon(
    combined: pd.Series,
    incidents: List[pd.Timestamp],
    horizon_hours: float,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """Metodologia idêntica ao ae_per_sensor_experiment.py:
    - Recall: fração dos incidentes com ≥1 ponto de alerta em [inc-H, inc].
    - FA: episódios (gap-debounced) que não cobrem nenhum incidente.
    """
    horizon_sec = horizon_hours * 3600.0
    total_days  = (combined.index[-1] - combined.index[0]).total_seconds() / 86400.0
    inc_s = np.array([t.timestamp() for t in incidents])

    thresholds = np.linspace(0.50, 0.999, n_thresholds)
    rows = []

    for q in thresholds:
        alert   = combined >= q
        alert_s = np.array([t.timestamp() for t in combined.index[alert]])
        episodes = detect_episodes_gap(alert)

        # Recall: incidente detectado se algum ponto de alerta cai em [inc-H, inc]
        n_hit = 0
        for ti in inc_s:
            if alert_s.size and np.any((alert_s >= ti - horizon_sec) & (alert_s <= ti)):
                n_hit += 1

        # FA: episódio inútil (não precede nenhum incidente no horizonte)
        n_fp = 0
        for (s0, s1) in episodes:
            s0_ts = s0.timestamp()
            s1_ts = s1.timestamp()
            useful = bool(np.any((inc_s - horizon_sec <= s1_ts) & (inc_s >= s0_ts))) if inc_s.size else False
            if not useful:
                n_fp += 1

        fa_per_day = n_fp / max(total_days, 1.0)
        recall     = n_hit / len(incidents) if incidents else 0.0
        rows.append({
            "threshold_quantile": float(q),
            "recall":             float(recall),
            "fa_per_day":         float(fa_per_day),
            "n_alert_episodes":   len(episodes),
            "n_hit":              n_hit,
            "n_fp":               n_fp,
        })

    return pd.DataFrame(rows)


def best_operating_point(df_curve: pd.DataFrame, fa_budget: float = 1.0) -> pd.Series:
    ok = df_curve[df_curve["fa_per_day"] <= fa_budget]
    if ok.empty:
        return df_curve.loc[df_curve["recall"].idxmax()]
    return ok.loc[ok["recall"].idxmax()]


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia camada preditiva OR+EWMA de task ClearML")
    parser.add_argument("--task_id",   required=True, help="ID completo da task ClearML")
    parser.add_argument("--label",     default="experimento",  help="Nome para logs e figuras")
    parser.add_argument("--alarm_csv", default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_life", type=float, default=4.0,  help="Half-life EWMA em horas")
    parser.add_argument("--horizons",  type=float, nargs="+", default=[8.0, 24.0, 72.0])
    parser.add_argument("--fa_budget",  type=float, default=1.0,  help="FA/dia máximo para ponto de operação")
    parser.add_argument("--eval_start", default=None, help="Filtro de data início para avaliação OOS (ex: 2026-01-01)")
    parser.add_argument("--eval_end",   default=None, help="Filtro de data fim para avaliação OOS (ex: 2026-04-30)")
    parser.add_argument("--out_dir",    default="eval_predictive_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Carrega task ---
    print(f"\n[1/5] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"      {task.name}  |  status={task.get_status()}")

    # --- Sequence scores ---
    print(f"\n[2/5] Baixando sequence_scores de {len(SENSORS)} sensores...")
    mae_dict = load_mae_series(task, SENSORS)
    if not mae_dict:
        raise RuntimeError("Nenhum sensor carregado — verifique o task_id.")

    # --- EWMA + quantile ---
    print(f"\n[3/5] EWMA (half_life={args.half_life}h) + normalização quantile...")
    health_dict = {s: ewma_quantile(mae, args.half_life) for s, mae in mae_dict.items()}

    # --- OR ---
    print("\n[4/5] Combinando via OR...")
    combined = combine_or(health_dict)

    if args.eval_start or args.eval_end:
        t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else combined.index[0]
        t1 = pd.Timestamp(args.eval_end,   tz="UTC") if args.eval_end   else combined.index[-1]
        combined = combined[(combined.index >= t0) & (combined.index <= t1)]
        print(f"      [OOS] Avaliação restrita a {t0.date()} → {t1.date()}")

    print(f"      Período: {combined.index[0]} → {combined.index[-1]}")
    print(f"      Pontos: {len(combined):,}")

    # --- Alarmes ---
    print(f"\n[5/5] Avaliando contra alarmes ({args.alarm_csv})...")
    alarm_times = load_alarms(args.alarm_csv)
    alarm_times = [a for a in alarm_times if combined.index[0] <= a <= combined.index[-1]]
    incidents = cluster_incidents(alarm_times, gap_hours=4.0)
    print(f"      {len(alarm_times)} alarmes → {len(incidents)} incidentes (gap ≥ 4h)")

    # --- Avaliação por horizonte ---
    results: Dict[str, Dict] = {}
    fig, axes = plt.subplots(1, len(args.horizons), figsize=(5 * len(args.horizons), 4), sharey=True)
    if len(args.horizons) == 1:
        axes = [axes]

    print()
    print(f"{'Horizonte':8s} | {'Baseline TS=60':25s} | {args.label}")
    print("─" * 70)

    for ax, H in zip(axes, args.horizons):
        df_curve = evaluate_horizon(combined, incidents, H)
        csv_path = os.path.join(args.out_dir, f"OR_curve_H{int(H)}h_{args.label}.csv")
        df_curve.to_csv(csv_path, index=False)

        best = best_operating_point(df_curve, args.fa_budget)
        results[f"H{int(H)}h"] = {
            "recall":             float(best["recall"]),
            "fa_per_day":         float(best["fa_per_day"]),
            "threshold_quantile": float(best["threshold_quantile"]),
        }

        # Plot
        ax.plot(df_curve["fa_per_day"], df_curve["recall"],
                label=args.label, color="steelblue", linewidth=2)
        ax.scatter([best["fa_per_day"]], [best["recall"]],
                   color="steelblue", zorder=5, s=60)

        # Baseline TS=60 para comparação
        bl_path = os.path.join(BASELINE_DIR, f"per_sensor_OR_curve_H{int(H)}h.csv")
        bl_recall = bl_fa = None
        if os.path.exists(bl_path):
            df_bl = pd.read_csv(bl_path)
            ax.plot(df_bl["fa_per_day"], df_bl["recall"],
                    label="baseline TS=60", color="orange", linestyle="--", linewidth=2)
            bl_best = best_operating_point(df_bl, args.fa_budget)
            bl_recall = float(bl_best["recall"])
            bl_fa     = float(bl_best["fa_per_day"])

        ax.set_xlabel("FA/dia")
        ax.set_ylabel("Recall")
        ax.set_title(f"H = {H:.0f}h")
        ax.set_xlim(0, max(5, args.fa_budget * 3))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Log tabela
        bl_str = f"rec={bl_recall:.3f} FA={bl_fa:.3f}" if bl_recall is not None else "N/A"
        print(f"H={H:4.0f}h    | {bl_str:25s} | rec={best['recall']:.3f} FA={best['fa_per_day']:.3f}")

    plt.suptitle(f"Camada preditiva OR+EWMA({args.half_life}h) — {args.label} vs baseline TS=60",
                 fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"fig_OR_curve_{args.label}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    results_path = os.path.join(args.out_dir, f"results_{args.label}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "task_id": args.task_id,
                   "half_life_hours": args.half_life, "operating_points": results}, f, indent=2)

    print(f"\n[OK] Figura  : {fig_path}")
    print(f"[OK] Resultado: {results_path}")


if __name__ == "__main__":
    main()
