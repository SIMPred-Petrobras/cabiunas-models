#!/usr/bin/env python3
"""
sweep_threshold_mean_std.py
Responde à sugestão "threshold = média + y * desvio-padrão; testar valores de y".

Compara, no MESMO sinal e na MESMA janela, duas regras de threshold:
  A) quantil da EWMA do erro (regra atual do eval de produção, varrida em q)
  B) mu + y*sigma sobre a EWMA do erro, com mu/sigma calibrados no TREINO
     (jan→jul/2025, só equipamento ligado — sem vazamento)

Métricas idênticas nas duas (recall por episódio com horizonte de 8h, FA/dia por
episódio com debounce, duty-cycle), então a comparação é direta.

Roda offline, lendo os artefatos já baixados em ~/.clearml/cache (o servidor
ClearML pode estar fora).

Uso:
    PYTHONPATH=. python scripts/sweep_threshold_mean_std.py
"""
from __future__ import annotations

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
off = _load("plot_spike_experiment_offline")

TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
HALF_LIFE, HORIZON = off.HALF_LIFE, off.HORIZON
Y_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
OUT_CSV = os.path.join(off.OUT_DIR, "threshold_mean_std_sweep.csv")


def metrics(alert: pd.Series, incidents: list, total_days: float) -> dict:
    """Mesmas definições do eval: recall por episódio (janela H antes do incidente),
    FA/dia = episódios de alerta que não casam com nenhum incidente."""
    inc_s = np.array([t.timestamp() for t in incidents])
    horizon_sec = HORIZON * 3600.0
    episodes = ev.detect_episodes_gap(alert)
    alert_s = np.array([t.timestamp() for t in alert.index[alert]])

    n_hit, leads = 0, []
    for ti in inc_s:
        w = alert_s[(alert_s >= ti - horizon_sec) & (alert_s <= ti)] if alert_s.size else np.array([])
        if w.size:
            n_hit += 1
            leads.append((ti - w.min()) / 3600.0)
    n_fp = sum(
        1 for (s0, s1) in episodes
        if not (np.any((inc_s - horizon_sec <= s1.timestamp()) & (inc_s >= s0.timestamp()))
                if inc_s.size else False)
    )
    return {
        "recall": n_hit / len(incidents) if incidents else 0.0,
        "n_hit": n_hit,
        "n_fp": n_fp,
        "fa_per_day": n_fp / max(total_days, 1.0),
        "duty": float(alert.mean()),
        "median_lead_h": float(np.median(leads)) if leads else 0.0,
    }


def main() -> None:
    print("[1/3] Identificando braços no cache local...")
    found, on_mask, incidents = off.identify_arms()
    if "base" not in found:
        raise SystemExit("Braço 'base' não encontrado no cache.")

    mae = off.read_mae(found["base"])

    # Sinal suavizado (mesma EWMA do eval, mas SEM o rank — mu+y*sigma precisa da
    # escala física do erro; o rank é uniforme em [0,1] e tornaria sigma sem sentido).
    hl_pts = int(round(pd.Timedelta(hours=HALF_LIFE) / pd.Timedelta(ev.SAMPLING_INTERVAL)))
    smooth = mae.ewm(halflife=max(1, hl_pts)).mean()
    smooth = smooth[(smooth.index >= off.EVAL_START) & (smooth.index <= off.EVAL_END)]

    om = on_mask.reindex(smooth.index, method="nearest",
                         tolerance=pd.Timedelta("6min")).fillna(False).astype(bool)

    total_days = (smooth.index[-1] - smooth.index[0]).total_seconds() / 86400.0

    # mu/sigma calibrados só no treino e só com equipamento ligado (sem vazamento)
    tr = smooth[(smooth.index >= TRAIN_START) & (smooth.index < TRAIN_END) & om]
    mu, sigma = float(tr.mean()), float(tr.std(ddof=0))
    print(f"\n[2/3] Calibração no treino ({TRAIN_START.date()}→{TRAIN_END.date()}, "
          f"ON only, n={len(tr)}): mu={mu:.5f}  sigma={sigma:.5f}")

    # ---- Regra B: mu + y*sigma ----
    rows = []
    for y in Y_GRID:
        thr = mu + y * sigma
        alert = om & (smooth >= thr)
        m = metrics(alert, incidents, total_days)
        m.update({"regra": "mu+y*sigma", "y": y, "threshold_abs": thr})
        rows.append(m)
        print(f"  y={y:<5} thr={thr:.5f}  recall {m['recall']*100:5.1f}% "
              f"({m['n_hit']}/{len(incidents)})  FA {m['fa_per_day']:.3f}/dia  "
              f"duty {m['duty']:.3f}  lead {m['median_lead_h']:.1f}h")

    # ---- Regra A: quantil da EWMA-rank (regra atual), mesmo sinal/janela ----
    print("\n[3/3] Regra atual (quantil da EWMA-rank), mesmo sinal e janela:")
    health = ev.ewma_quantile(mae, HALF_LIFE)
    health = health[(health.index >= off.EVAL_START) & (health.index <= off.EVAL_END)]
    health = health.where(om, other=0.0)
    res = ev.best_point_for_sensor(health, incidents, HORIZON, max_duty_cycle=off.MAX_DUTY)
    alert_q = health >= res["threshold_q"]
    mq = metrics(alert_q, incidents, total_days)
    mq.update({"regra": "quantil (atual)", "y": np.nan,
               "threshold_abs": float(smooth[alert_q].min()) if alert_q.any() else np.nan})
    rows.append(mq)
    print(f"  q={res['threshold_q']:.4f}  recall {mq['recall']*100:5.1f}% "
          f"({mq['n_hit']}/{len(incidents)})  FA {mq['fa_per_day']:.3f}/dia  "
          f"duty {mq['duty']:.3f}  lead {mq['median_lead_h']:.1f}h")

    # y equivalente ao ponto que o sweep atual escolheu (ponte entre as duas regras)
    if alert_q.any():
        y_eq = (float(smooth[alert_q].min()) - mu) / sigma
        print(f"  → o ponto escolhido pelo sweep atual equivale a y ≈ {y_eq:.2f}")

    os.makedirs(off.OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nsalvo: {OUT_CSV}")


if __name__ == "__main__":
    main()
