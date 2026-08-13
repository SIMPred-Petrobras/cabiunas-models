#!/usr/bin/env python3
"""
verifica_semeadura.py
Verifica se a correção de semeadura tornou o pipeline reprodutível.

Contexto: `RANDOM_SEED` era config morta (split temporal ignora o seed; nada em `src/`
semeava TF/Keras; o KerasTuner sorteava 20 arquiteturas sem semente). Medimos 27,6pp de
amplitude entre 5 execuções IDÊNTICAS do b2024. A correção semeia globalmente
(`keras.utils.set_random_seed`) e passa `seed` ao `kt.RandomSearch`.

O teste tem DUAS direções, e as duas importam:

    A42 vs B42  (mesma semente)      → tem de bater. Mede reprodutibilidade.
    A42 vs S07  (sementes distintas) → tem de diferir. Sem este braço, um bug que
                                        congelasse o pipeline num caminho único faria o
                                        teste passar por acidente.

⚠️ `set_random_seed` NÃO garante determinismo bit a bit em GPU: cuDNN faz autotuning e
algumas ops usam atomics. Por isso o veredito é em dois níveis — idêntico bit a bit
(ideal) ou equivalente na métrica (aceitável na prática). Se nem a métrica bater, falta
`tf.config.experimental.enable_op_determinism()`.

Uso:
    PYTHONPATH=. python scripts/verifica_semeadura.py
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
bl = _load("baseline_trivial_vs_ae")

SENSOR = "TC382_03_A"
OUT = "eval_predictive_out/verifica_semeadura.csv"
TASKS = {"A (seed 42)": "6cfbec162c3c48f798549fd97ca557b9",
         "B (seed 42)": "d959460d8fc94a209fb6ed0beff33e0c",
         "C (seed 07)": "8bbe1a0ad9b946cca87787ac6ff836de"}
T0, T1 = pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")


def main() -> None:
    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    maes = {}
    for lab, tid in TASKS.items():
        t = Task.get_task(task_id=tid)
        if t.get_status() != "completed":
            raise SystemExit(f"{lab} ainda em '{t.get_status()}' — rode quando terminar.")
        maes[lab] = ev.load_mae_series(t, [SENSOR])[SENSOR]

    idx = None
    for s in maes.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    maes = {k: v.reindex(idx) for k, v in maes.items()}
    print(f"interseção: {len(idx):,} pontos")

    inc = sw.incidents_on(running, tc03, T0, T1)
    print(f"incidentes na FULL: {len(inc)}\n")

    met = {}
    print(f"  {'execução':<14}{'recall_raw':>12}{'FA/dia':>10}{'hl':>6}{'MAE p50':>12}")
    for lab, s in maes.items():
        r = bl.best_over_hl(s.dropna(), inc, running)
        met[lab] = r
        print(f"  {lab:<14}{r.get('recall_raw', float('nan'))*100:>11.1f}%"
              f"{r.get('fa_per_day', float('nan')):>10.3f}{str(r.get('hl')):>6}"
              f"{s.median():>12.6f}", flush=True)

    def compara(x: str, y: str) -> dict:
        a, b = maes[x].dropna(), maes[y].dropna()
        j = a.index.intersection(b.index)
        d = (a.reindex(j) - b.reindex(j)).abs()
        return dict(par=f"{x} × {y}", n=len(j), max_abs=float(d.max()),
                    media_abs=float(d.mean()),
                    corr=float(np.corrcoef(a.reindex(j), b.reindex(j))[0, 1]),
                    identico=bool(d.max() == 0.0),
                    d_recall_pp=(met[x]["recall_raw"] - met[y]["recall_raw"]) * 100)

    pares = [compara("A (seed 42)", "B (seed 42)"), compara("A (seed 42)", "C (seed 07)")]
    df = pd.DataFrame(pares)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== COMPARAÇÃO DAS SÉRIES DE ERRO ===")
    for p in pares:
        print(f"\n  {p['par']}")
        print(f"    idêntico bit a bit: {'SIM' if p['identico'] else 'não'}")
        print(f"    max|Δ| = {p['max_abs']:.3e}   média|Δ| = {p['media_abs']:.3e}   "
              f"corr = {p['corr']:.6f}")
        print(f"    Δ recall_raw = {p['d_recall_pp']:+.1f} pp")

    ab, ac = pares
    print("\n=== VEREDITO ===")
    if ab["identico"]:
        v1 = "REPRODUTÍVEL bit a bit"
    elif abs(ab["d_recall_pp"]) < 1e-9:
        v1 = ("reprodutível NA MÉTRICA (série difere um pouco — não-determinismo de GPU; "
              "para bit a bit falta enable_op_determinism)")
    else:
        v1 = (f"AINDA NÃO REPRODUTÍVEL — mesma semente muda o recall em "
              f"{ab['d_recall_pp']:+.1f} pp; falta enable_op_determinism()")
    print(f"  mesma semente (A×B):      {v1}")

    v2 = ("OK — a semente tem efeito" if not ac["identico"]
          else "SUSPEITO: sementes diferentes deram resultado idêntico; o seed pode não "
               "estar chegando ao tuner")
    print(f"  sementes distintas (A×C): {v2}")
    print(f"\n  Referência: 5 execuções SEM semeadura variaram 27,6 pp na mesma janela.")
    print(f"  Gravado: {OUT}")


if __name__ == "__main__":
    main()
