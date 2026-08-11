#!/usr/bin/env python3
"""
baseline_trivial_vs_ae.py
A pergunta que nunca foi feita neste projeto: **o autoencoder ganha de um limiar
na própria temperatura?**

Medimos que o alarme do DCS é `TC382_03_A > 760 °C` (cruzamentos/dia batem 1:1 com
alarmes/dia em 2024H2 e em 2026). Logo o produto é prever um cruzamento de linha
fixa em 8 h — e um limiar na temperatura é o baseline trivial obrigatório. Se ele
empatar com a rede, o projeto precisa ser reformulado, não ajustado.

COMPARAÇÃO PAREADA — o mesmo maquinário para todos os braços, para isolar a única
pergunta que importa: o erro de reconstrução carrega informação ALÉM da temperatura?

    ae        EWMA(MAE do autoencoder) → rank        (sistema atual)
    temp      EWMA(TC382_03_A)         → rank        (baseline trivial)
    slope     EWMA(dT/dt)              → rank        (só a tendência)

Todos passam por `best_point_for_sensor` com parâmetros idênticos (horizonte 8 h,
sticky 12 h, FA ≤ 1/dia, duty pós-sticky ≤ 0,25) e varrem o MESMO grid de half-life,
sobre a MESMA grade temporal (índice do MAE) — então duty, FA e denominador de
incidente são exatamente os mesmos. Nenhum braço leva vantagem de amostragem.

Uso:
    PYTHONPATH=. python scripts/baseline_trivial_vs_ae.py
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

SENSOR = "TC382_03_A"
RAW = "../dados/sensores_full_2024_2026_30s.csv"
TASK = "3b34a312aa234aae9ac1f5c1f922791f"        # controle = melhor braço atual
OUT = "eval_predictive_out/baseline_trivial_vs_ae.csv"

HL_GRID = [0.5, 1.0, 2.0, 4.0]
HORIZON, STICKY, FA_BUDGET = 8.0, 12.0, 1.0
MAX_DUTY, MAX_STICKY = 0.35, 0.25

JANELAS = [("2024 jun–dez", "2024-06-01", "2025-01-01"),
           ("2026 jan–abr", "2026-01-01", "2026-05-01"),
           ("OOS jul/25→abr/26", "2025-07-01", "2026-05-01"),
           ("FULL jan/24→abr/26", "2024-01-01", "2026-05-01")]


def best_over_hl(score: pd.Series, inc: list, running: pd.Series) -> dict:
    """Varre half-life e devolve o melhor ponto — mesma regra do protocolo honesto."""
    best = None
    for hl in HL_GRID:
        h = sw.ewma_on(score, hl, running).rank(pct=True)
        if h.empty:
            continue
        r = ev.best_point_for_sensor(h, inc, horizon_hours=HORIZON, sticky_hours=STICKY,
                                     fa_budget=FA_BUDGET, n_thresholds=120,
                                     max_duty_cycle=MAX_DUTY, max_sticky_duty=MAX_STICKY)
        r["hl"] = hl
        key = (r.get("recall_raw") or 0.0, -(r.get("fa_per_day") or 9e9))
        if best is None or key > (best.get("recall_raw") or 0.0, -(best.get("fa_per_day") or 9e9)):
            best = r
    return best or {}


def main() -> None:
    raw = pd.read_csv(RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    mae = ev.load_mae_series(Task.get_task(task_id=TASK), [SENSOR])[SENSOR]

    # baselines na MESMA grade do MAE — duty/FA/denominador ficam idênticos
    t_on_grid = tc03.reindex(mae.index, method="nearest")
    dt_h = 30.0 / 3600.0
    slope = (tc03.diff() / dt_h).reindex(mae.index, method="nearest")

    BRACOS = {"ae  (autoencoder)": mae,
              "temp (limiar trivial)": t_on_grid,
              "slope (só tendência)": slope}

    rows = []
    for wlab, a, b in JANELAS:
        t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        inc = sw.incidents_on(running, tc03, t0, t1)
        print(f"\n=== {wlab} — {len(inc)} incidentes ON ===")
        print(f"  {'braço':<24}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'lead h':>9}{'hl':>6}")
        for name, s in BRACOS.items():
            sc = s[(s.index >= t0) & (s.index < t1)].dropna()
            r = best_over_hl(sc, inc, running)
            rr = r.get("recall_raw")
            rows.append(dict(janela=wlab, braco=name, n_inc=len(inc), recall_raw=rr,
                             fa_per_day=r.get("fa_per_day"), duty=r.get("duty_sticky"),
                             lead_h=r.get("median_lead_hours"), hl=r.get("hl")))
            print(f"  {name:<24}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                  f"{r.get('fa_per_day', float('nan')):>10.3f}"
                  f"{r.get('duty_sticky', float('nan')):>8.2f}"
                  f"{r.get('median_lead_hours', float('nan')):>9.1f}"
                  f"{str(r.get('hl')):>6}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGravado: {OUT}")

    print("\n=== VEREDITO: o AE ganha do limiar trivial? ===")
    for wlab, _, _ in JANELAS:
        d = df[df.janela == wlab].set_index("braco")
        ae, tp = d.loc["ae  (autoencoder)", "recall_raw"], d.loc["temp (limiar trivial)", "recall_raw"]
        if pd.isna(ae) or pd.isna(tp):
            continue
        dpp = (ae - tp) * 100
        v = "AE GANHA" if dpp > 10 else ("empate" if abs(dpp) <= 10 else "LIMIAR GANHA")
        print(f"  {wlab:<22} AE {ae*100:5.1f}%  ×  limiar {tp*100:5.1f}%   "
              f"Δ={dpp:+6.1f}pp   → {v}")
    print("\n  (margem de 10pp = ruído de semente medido no projeto, ±27pp no pior caso)")


if __name__ == "__main__":
    main()
