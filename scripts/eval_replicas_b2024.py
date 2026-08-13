#!/usr/bin/env python3
"""
eval_replicas_b2024.py
Mede o RUÍDO DE EXECUÇÃO da configuração b2024: 4 réplicas idênticas, avaliadas no mesmo
protocolo, contra o b2024 original e o limiar trivial.

A pergunta. O b2024 marca recall_raw 86,2% e é o nosso melhor braço — mas é UMA execução.
O "rerun" que existia no repo não é réplica (difere em `TRAIN_START_DATE`), então até agora
não havia nenhuma medida de quanto essa configuração varia sozinha. Sem isso não dá para
dizer se os 86,2% são o modelo ou a sorte, nem se o +1,7pp sobre o limiar significa algo.

⚠️ POR QUE AS RÉPLICAS SÃO IDÊNTICAS NO ARQUIVO. Testado: `RANDOM_SEED` não afeta este
pipeline. Com `SPLIT_MODE="temporal"` o `train_val_split` devolve `x[:n], x[n:]` e ignora o
seed, e não existe `tf.random.set_seed` nem `keras.utils.set_random_seed` em `src/`. A
variação entre execuções vem da inicialização de pesos não semeada e — o que pesa mais — do
KerasTuner com `MAX_TRIALS=20` também não semeado, que a cada execução sorteia um conjunto
diferente de hiperparâmetros e escolhe o melhor por `val_loss`. Não é ruído de semente: é
loteria de arquitetura. Logo, reexecutar o mesmo arquivo JÁ é o experimento.

Tudo na interseção das grades de MAE, mesmo denominador de incidentes, ponto de operação
buscado igual para todos.

Uso:
    PYTHONPATH=. python scripts/eval_replicas_b2024.py
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
OUT = "eval_predictive_out/replicas_b2024.csv"

TASKS = {
    "b2024 ORIGINAL": "1a15c26d994e44febb77f0bec8c2b378",
    "réplica 1": "6b76813bed574baea57ef14da233964a",
    "réplica 2": "067ade9c0c5c455886e68c53ad3a5307",
    "réplica 3": "954f2d356c09423dbf4044c449453793",
    "réplica 4": "91c37f801a3945b885cc953cbd8d37ec",
}
JANELAS = [("FULL jun/24→abr/26", "2024-06-01", "2026-05-01"),
           ("2024 jun–dez", "2024-06-01", "2025-01-01"),
           ("OOS jul/25→abr/26", "2025-07-01", "2026-05-01")]


def main() -> None:
    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    maes, faltando = {}, []
    for lab, tid in TASKS.items():
        t = Task.get_task(task_id=tid)
        if t.get_status() != "completed":
            faltando.append(f"{lab} ({t.get_status()})")
            continue
        s = ev.load_mae_series(t, [SENSOR]).get(SENSOR)
        if s is None:
            faltando.append(f"{lab} (sem artefato)")
            continue
        maes[lab] = s
    if faltando:
        print(f"[aviso] fora da análise: {', '.join(faltando)}")
    if len(maes) < 2:
        raise SystemExit("réplicas insuficientes — rode quando os treinos terminarem.")

    idx = None
    for s in maes.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    print(f"interseção das grades: {len(idx):,} pontos  {idx[0].date()} → {idx[-1].date()}")

    bracos = {k: v.reindex(idx) for k, v in maes.items()}
    bracos["limiar trivial"] = tc03.reindex(idx, method="nearest")

    rows = []
    for wlab, a, b in JANELAS:
        t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        inc = sw.incidents_on(running, tc03, t0, t1)
        print(f"\n=== {wlab} — {len(inc)} incidentes ON ===")
        print(f"  {'braço':<20}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'hl':>6}")
        for name, sc in bracos.items():
            s = sc[(sc.index >= t0) & (sc.index < t1)].dropna()
            r = bl.best_over_hl(s, inc, running) if not s.empty else {}
            rr = r.get("recall_raw")
            rows.append(dict(janela=wlab, braco=name, n_inc=len(inc), recall_raw=rr,
                             fa_per_day=r.get("fa_per_day"), duty=r.get("duty_sticky"),
                             hl=r.get("hl")))
            print(f"  {name:<20}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                  f"{r.get('fa_per_day', float('nan')):>10.3f}"
                  f"{r.get('duty_sticky', float('nan')):>8.2f}"
                  f"{str(r.get('hl')):>6}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGravado: {OUT}")

    print("\n=== DISPERSÃO DA MESMA CONFIGURAÇÃO ===")
    for wlab, _, _ in JANELAS:
        d = df[df.janela == wlab].set_index("braco")
        reps = [k for k in d.index if k.startswith("réplica") or k.startswith("b2024")]
        v = d.loc[reps, "recall_raw"].dropna() * 100
        tv = d.loc["limiar trivial", "recall_raw"] * 100
        if v.empty:
            continue
        print(f"\n  {wlab}  (n={int(d.n_inc.iloc[0])} incidentes)")
        print(f"    execuções: {', '.join(f'{x:.1f}%' for x in v)}")
        print(f"    min {v.min():.1f}%   mediana {v.median():.1f}%   max {v.max():.1f}%   "
              f"amplitude {v.max()-v.min():.1f}pp   dp {v.std():.1f}pp")
        print(f"    limiar trivial: {tv:.1f}%  →  execuções acima dele: "
              f"{int((v > tv).sum())}/{len(v)}")
    print("\n  Leitura: se a amplitude entre execuções idênticas for da ordem da diferença")
    print("  para o limiar, então aquela diferença não é o modelo — é a rodada.")


if __name__ == "__main__":
    main()
