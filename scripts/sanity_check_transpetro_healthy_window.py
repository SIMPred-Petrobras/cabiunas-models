"""Teste de sanidade: o threshold/half-life escolhido por sensor (sweep H72h) foi
calibrado testando contra o ÚNICO incidente que também é o alvo — não há validação
cruzada real com N=1. Este script checa se o resultado é overfit: aplica o MESMO ponto
de operação escolhido a um trecho "distante" (longe da falha, sem nenhum incidente
próximo) e mede a taxa de falso alarme lá. Se disparar tanto no trecho distante quanto
no reportado, o resultado é ruído, não sinal — se ficar quieto no trecho distante e só
acender perto da falha, é evidência real de comportamento de aviso antecipado genuíno.

Divide o período de teste (pós TRAIN_END_DATE) em dois:
  - "distante": logo após o fim do treino até 14 dias antes da detecção formal — nenhum
    incidente aqui, então QUALQUER episódio é, por construção, falso alarme.
  - "próximo": últimos 14 dias antes da detecção — onde o aviso antecipado deveria
    aparecer.

Uso:
  PYTHONPATH=. python scripts/sanity_check_transpetro_healthy_window.py
"""
import numpy as np
import pandas as pd
from clearml import Task

from src.cnn1d_ae.predictive import compute_health_index_ewma, _detect_episodes
from scripts.eval_transpetro_halflife_sweep import (
    EQUIPS, load_mae, running_frac_forward, TIME_STEPS,
)

DEBOUNCE_HOURS = 8.0
NEAR_WINDOW_DAYS = 14


def fa_per_day_in_window(health: np.ndarray, t_end: pd.DatetimeIndex, thr: float,
                         t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[float, int]:
    mask = (t_end >= t0) & (t_end < t1)
    if mask.sum() < 2:
        return float("nan"), 0
    idx_local = np.where(mask)[0]
    alert_local = health[idx_local] >= thr
    alert_idx = idx_local[alert_local]
    t_s = t_end.values.astype("datetime64[s]").astype("int64").astype(float)
    episodes = _detect_episodes(alert_idx, t_s, DEBOUNCE_HOURS * 3600.0)
    span_days = (t1 - t0).total_seconds() / 86400.0
    return len(episodes) / max(span_days, 1e-9), len(episodes)


def main():
    from scripts.eval_transpetro_halflife_sweep import sweep_sensor

    rows = []
    for equip, cfg in EQUIPS.items():
        task = Task.get_task(task_id=cfg["task_id"])
        print(f"\n===== {equip} =====")
        near_start = cfg["detection_ts"] - pd.Timedelta(days=NEAR_WINDOW_DAYS)
        for sensor in cfg["sensors"]:
            best_by_h, best_ew_hl = sweep_sensor(task, cfg, sensor)
            if not best_by_h or 72.0 not in best_by_h:
                print(f"  {sensor:22s} sem ponto valido (H72h) — pulei")
                continue
            b72 = best_by_h[72.0]
            hl_used, health = best_ew_hl
            mae = load_mae(task, sensor)
            t_end = mae.index + pd.Timedelta(minutes=TIME_STEPS - 1)

            fa_far, n_far = fa_per_day_in_window(health, t_end, b72["threshold"],
                                                 cfg["train_end"], near_start)
            fa_near, n_near = fa_per_day_in_window(health, t_end, b72["threshold"],
                                                    near_start, mae.index.max() + pd.Timedelta(days=1))

            veredito = "OK (quieto longe da falha)" if fa_far <= b72["fa_per_day"] * 1.5 else "SUSPEITO (ruidoso tambem longe)"
            rows.append(dict(equip=equip, sensor=sensor, hl=hl_used,
                             fa_reportado=b72["fa_per_day"], fa_distante=fa_far, n_ep_distante=n_far,
                             fa_proximo=fa_near, n_ep_proximo=n_near, veredito=veredito))
            print(f"  {sensor:22s} FA reportado={b72['fa_per_day']:.2f}/d | "
                  f"distante={fa_far:.2f}/d ({n_far} ep, {(near_start-cfg['train_end']).days}d) | "
                  f"proximo={fa_near:.2f}/d ({n_near} ep, {NEAR_WINDOW_DAYS}d) -> {veredito}")

    df = pd.DataFrame(rows)
    out_csv = "eval_predictive_out/transpetro/sanity_check_healthy_window.csv"
    df.to_csv(out_csv, index=False)
    print(f"\ncsv: {out_csv}")
    n_susp = (df["veredito"].str.startswith("SUSPEITO")).sum()
    print(f"\n{len(df)-n_susp}/{len(df)} sensores OK (quietos no trecho distante); {n_susp} suspeitos de overfit")


if __name__ == "__main__":
    main()
