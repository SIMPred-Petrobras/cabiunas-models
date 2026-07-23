"""Testa se exigir corroboração entre sensores (K-de-N alertando ao mesmo tempo) reduz o
falso alarme sem perder antecedência, nos 2 equipamentos originais (B-4064A, B-90001A).
Ideia: em vez de confiar no melhor sensor isolado, combinar os sensores com sinal válido
(do sweep de half-life já feito) e só considerar "alerta" quando >=K deles concordam na
mesma janela — reduz ruído específico de 1 sensor sem exigir retreino nem features novas.

Reusa `sweep_sensor` (health-index no half-life vencedor H72h de cada sensor, já validado)
e `_detect_episodes`/`fa_per_day_in_window` (mesma contagem rigorosa de episódios com
debounce, e mesmo teste de sanidade contra o trecho distante da falha) dos scripts
anteriores. Compara o resultado combinado contra o melhor sensor isolado de cada
equipamento (a referência já reportada à Transpetro).

Uso:
  PYTHONPATH=. python scripts/eval_transpetro_sensor_corroboration.py
"""
import numpy as np
import pandas as pd
from clearml import Task

from src.cnn1d_ae.predictive import _detect_episodes
from scripts.eval_transpetro_halflife_sweep import EQUIPS, load_mae, sweep_sensor, TIME_STEPS
from scripts.sanity_check_transpetro_healthy_window import fa_per_day_in_window, NEAR_WINDOW_DAYS

DEBOUNCE_HOURS = 8.0
HORIZON_HOURS = 72.0

VALID_SENSORS = {
    "B-4064A": ["Pressão Sucção", "Corrente", "Vibração Bomba LNA", "Temperatura Bomba LA",
               "Temperatura Bomba LNA", "Temperatura Motor LA", "Temperatura Motor LNA",
               "Densidade"],
    "B-90001A": ["Pressão Descarga", "Pressão Sucção", "Vibração Motor LNA Y",
                "Vibração Motor LA X", "Vibração Motor LA Y", "Vibração Bomba LA X",
                "Vibração Bomba LA Y", "Vibração Bomba LNA X", "Vibração Bomba LNA Y"],
}


def build_alert_frame(task, cfg, sensors) -> pd.DataFrame:
    """Series booleana de alerta por sensor (health>=threshold vencedor H72h), alinhadas
    no mesmo indice de tempo (t_end = seq_start_time + TIME_STEPS-1 min)."""
    cols = {}
    for sensor in sensors:
        best_by_h, best_ew_hl = sweep_sensor(task, cfg, sensor)
        if not best_by_h or 72.0 not in best_by_h or best_ew_hl is None:
            continue
        thr = best_by_h[72.0]["threshold"]
        hl_used, health = best_ew_hl
        mae = load_mae(task, sensor)
        t_end = mae.index + pd.Timedelta(minutes=TIME_STEPS - 1)
        cols[sensor] = pd.Series(health, index=t_end) >= thr
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols).dropna(how="any")
    return df


def evaluate_k(alert_df: pd.DataFrame, K: int, cfg: dict) -> dict:
    count = alert_df.sum(axis=1).to_numpy(dtype=float)
    t_end = alert_df.index
    alert = count >= K
    idx = np.where(alert)[0]
    t_s = t_end.values.astype("datetime64[s]").astype("int64").astype(float)
    episodes = _detect_episodes(idx, t_s, DEBOUNCE_HOURS * 3600.0)

    inc_s = cfg["detection_ts"].timestamp()
    H = HORIZON_HOURS * 3600.0
    alert_s = t_s[idx]
    w = alert_s[(alert_s >= inc_s - H) & (alert_s <= inc_s)]
    hit = w.size > 0
    lead_hours = (inc_s - w.min()) / 3600.0 if hit else None

    fa_total = sum(1 for (s0, s1) in episodes if not ((inc_s - H) <= s1 and inc_s >= s0))
    test_mask = t_end >= cfg["train_end"]
    span_days = max((t_end[test_mask].max() - cfg["train_end"]).total_seconds() / 86400.0, 1e-9)
    fa_per_day = fa_total / span_days
    duty = float(alert[test_mask].mean())

    near_start = cfg["detection_ts"] - pd.Timedelta(days=NEAR_WINDOW_DAYS)
    fa_far, n_far = fa_per_day_in_window(count, t_end, float(K), cfg["train_end"], near_start)
    honest_fa = max(fa_per_day, fa_far) if pd.notna(fa_far) else fa_per_day

    return dict(K=K, hit=hit, lead_hours=lead_hours, fa_per_day=fa_per_day,
               fa_distante=fa_far, fa_honesto=honest_fa, duty=duty, n_episodes=len(episodes))


def main():
    rows = []
    for equip, sensors in VALID_SENSORS.items():
        cfg = EQUIPS[equip]
        task = Task.get_task(task_id=cfg["task_id"])
        print(f"\n===== {equip} ({len(sensors)} sensores com sinal valido) =====")
        alert_df = build_alert_frame(task, cfg, sensors)
        n_sensors = alert_df.shape[1]
        print(f"  sensores alinhados: {n_sensors}/{len(sensors)}, {len(alert_df)} pontos")

        for K in range(2, min(4, n_sensors) + 1):
            r = evaluate_k(alert_df, K, cfg)
            rows.append(dict(equip=equip, **r))
            lead_s = f"{r['lead_hours']:.1f}h" if r['lead_hours'] is not None else "NAO PEGOU"
            print(f"  K={K}: pegou_falha={r['hit']} lead={lead_s} "
                 f"FA_reportado={r['fa_per_day']:.3f}/d FA_honesto={r['fa_honesto']:.3f}/d "
                 f"duty={r['duty']*100:.1f}% n_ep={r['n_episodes']}")

    df = pd.DataFrame(rows)
    out_csv = "eval_predictive_out/transpetro/sensor_corroboration_compare.csv"
    df.to_csv(out_csv, index=False)
    print(f"\ncsv: {out_csv}")


if __name__ == "__main__":
    main()
