#!/usr/bin/env python3
"""
baseline_vs_melhor_ae.py
Refaz a comparação AE × limiar trivial contra o MELHOR autoencoder, não contra o controle.

Por que existe. O `baseline_trivial_vs_ae.py` usou a task `3b34a312`, que é o CONTROLE do
experimento v14 — não o melhor modelo. O melhor braço do projeto é o `1a15c26d` (b2024),
que na auditoria marca recall_raw 86,2% @ FA 0,103 contra os 62,0% do controle. Se o AE
bom empata ou ganha do limiar, a conclusão do projeto inverte, então isso precisa ser
medido, não argumentado.

⚠️ O QUE INVALIDARIA A COMPARAÇÃO. Os dois modelos cobrem janelas diferentes:

    b2024        2024-06-01 → 2026-04-30   (201.267 pontos)
    v14 control  2024-01-01 → 2026-04-30   (245.043 pontos)

Os 86,2% do b2024 foram medidos sobre 58 incidentes; os 62,0% do controle, sobre 79.
Comparar os dois números direto é o confound de janela que já derrubou três conclusões
neste projeto. Aqui TUDO roda na INTERSEÇÃO dos dois índices de MAE — mesma grade, mesmo
período, mesmo denominador de incidentes, para os três braços.

Uso:
    PYTHONPATH=. python scripts/baseline_vs_melhor_ae.py
"""
from __future__ import annotations

import importlib.util
import os

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
bl = _load("baseline_trivial_vs_ae")

SENSOR = "TC382_03_A"
OUT = "eval_predictive_out/baseline_vs_melhor_ae.csv"

TASKS = {"AE b2024 (melhor braço)": "1a15c26d994e44febb77f0bec8c2b378",
         "AE v14-control (o usado antes)": "3b34a312aa234aae9ac1f5c1f922791f"}

JANELAS = [("INTERSEÇÃO jun/24→abr/26", "2024-06-01", "2026-05-01"),
           ("2024 jun–dez", "2024-06-01", "2025-01-01"),
           ("OOS jul/25→abr/26", "2025-07-01", "2026-05-01"),
           ("2026 jan–abr", "2026-01-01", "2026-05-01")]


def main() -> None:
    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    maes = {}
    for lab, tid in TASKS.items():
        m = ev.load_mae_series(Task.get_task(task_id=tid), [SENSOR])[SENSOR]
        maes[lab] = m
        print(f"{lab:<32} {len(m):>8,} pts  {m.index[0].date()} → {m.index[-1].date()}")

    # INTERSEÇÃO: é isto que torna a comparação legítima
    idx = None
    for m in maes.values():
        idx = m.index if idx is None else idx.intersection(m.index)
    print(f"\nInterseção: {len(idx):,} pontos  {idx[0].date()} → {idx[-1].date()}")

    bracos = {lab: m.reindex(idx) for lab, m in maes.items()}
    bracos["limiar trivial (temperatura)"] = tc03.reindex(idx, method="nearest")

    rows = []
    for wlab, a, b in JANELAS:
        t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        inc = sw.incidents_on(running, tc03, t0, t1)
        print(f"\n=== {wlab} — {len(inc)} incidentes ON ===")
        print(f"  {'braço':<32}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'lead h':>9}{'hl':>6}")
        for name, sc in bracos.items():
            s = sc[(sc.index >= t0) & (sc.index < t1)].dropna()
            r = bl.best_over_hl(s, inc, running) if not s.empty else {}
            rr = r.get("recall_raw")
            rows.append(dict(janela=wlab, braco=name, n_inc=len(inc), recall_raw=rr,
                             fa_per_day=r.get("fa_per_day"), duty=r.get("duty_sticky"),
                             lead_h=r.get("median_lead_hours"), hl=r.get("hl")))
            print(f"  {name:<32}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                  f"{r.get('fa_per_day', float('nan')):>10.3f}"
                  f"{r.get('duty_sticky', float('nan')):>8.2f}"
                  f"{r.get('median_lead_hours', float('nan')):>9.1f}"
                  f"{str(r.get('hl')):>6}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGravado: {OUT}")

    print("\n=== VEREDITO: o MELHOR AE ganha do limiar trivial? ===")
    for wlab, _, _ in JANELAS:
        d = df[df.janela == wlab].set_index("braco")
        ae = d.loc["AE b2024 (melhor braço)", "recall_raw"]
        tv = d.loc["limiar trivial (temperatura)", "recall_raw"]
        if pd.isna(ae) or pd.isna(tv):
            continue
        dpp = (ae - tv) * 100
        v = "AE GANHA" if dpp > 10 else ("empate" if abs(dpp) <= 10 else "LIMIAR GANHA")
        fa_ae = d.loc["AE b2024 (melhor braço)", "fa_per_day"]
        fa_tv = d.loc["limiar trivial (temperatura)", "fa_per_day"]
        print(f"  {wlab:<26} AE {ae*100:5.1f}% (FA {fa_ae:.3f})  ×  limiar {tv*100:5.1f}% "
              f"(FA {fa_tv:.3f})   Δ={dpp:+6.1f}pp → {v}")
    print("\n  (margem de 10pp = ruído de semente medido no projeto)")


if __name__ == "__main__":
    main()
