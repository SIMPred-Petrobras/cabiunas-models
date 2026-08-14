#!/usr/bin/env python3
"""
eval_multivar_antigo.py
Re-pontua as tasks multivariadas ANTIGAS sob o protocolo honesto atual, sem retreinar.

Motivação: a pergunta "um AE multivariado reduz o falso positivo do TC382_03_A?" já foi
respondida uma vez (91,2% @ FA 0,092, H=72h), mas naquela metodologia — threshold buscado
na janela de avaliação, denominador diferente, e H=72h, onde o nulo embaralhado sozinho
marca 88,2%. Aqui as MESMAS séries de erro passam pelo maquinário que produziu os
86,2%/84,5% da mesa atual.

⚠️ NÃO É VEREDITO. Estas tasks divergem do braço b2024 em várias frentes além do
multivariado (treino até 2025-01-14 contra 2025-07-01, SENTINEL_MODE=none, RUNNING_COL=
NGP_A, sem filtro HI/HIHI). A evidência é ASSIMÉTRICA:

    vencerem apesar do handicap de janela de treino  → forte, motiva o experimento
    perderem                                          → inconclusivo, não refuta nada

O experimento que decide é a ablação pareada (`eval_multivariado_vs_control.py`), onde só
3 chaves mudam entre os braços.

⚠️ Os 8 runs `*_oos_2026_runa` NÃO são multivariados apesar do nome
(`MULTIVARIATE_JOINT: false`, `MODEL_MODE: per_sensor`) — não servem como réplicas.

Uso:
    PYTHONPATH=. python scripts/eval_multivar_antigo.py
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
OUT = "eval_predictive_out/multivar_antigo.csv"

# Só tasks com MULTIVARIATE_JOINT=true e MODEL_MODE=multivariate.
TASKS = {
    "multi 2025 (in-sample)": "085dd109e8874f06ae6afb2a8fd2ec1d",
    "multi OOS (jan25)":      "638a695d957a476ba280f5846ec455cf",
    "multi OOS 2026":         "8de61587255943d88b42a868d7acdc25",
}


def load_multi_mae(task: Task, sensor: str) -> pd.Series:
    """Lê o canal do sensor no artifact do GRUPO.

    Tasks multivariadas nomeiam o artifact pelo grupo (`MULTI_*_csv_sequence_scores_all`),
    não pelo sensor — `ev.load_mae_series` casa por substring do sensor e devolve vazio
    em silêncio nessas tasks. Daí o loader dedicado.
    """
    key = next((k for k in task.artifacts if "sequence_scores_all" in k), None)
    if key is None:
        raise RuntimeError(f"sem artifact sequence_scores_all em {task.id[:8]}")
    df = pd.read_csv(task.artifacts[key].get_local_copy())
    col = f"mae_{sensor}"
    if col not in df.columns:
        raise RuntimeError(f"coluna {col} ausente em {key} (tem: {list(df.columns)[:8]})")
    df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
    return df.set_index("seq_start_time")[col].dropna()


def main() -> None:
    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    series = {}
    for lab, tid in TASKS.items():
        t = Task.get_task(task_id=tid)
        if t.get_status() != "completed":
            print(f"  [skip] {lab}: status={t.get_status()}")
            continue
        try:
            s = load_multi_mae(t, SENSOR)
        except Exception as exc:
            print(f"  [skip] {lab}: {exc}")
            continue
        series[lab] = s
        print(f"  {lab:<24} {len(s):>7,} pts  {s.index.min().date()} → {s.index.max().date()}")

    if not series:
        raise SystemExit("nenhuma task multivariada carregada")

    # Cada task cobre um período diferente (duas param em nov/2025). Comparar as três
    # entre si num índice comum jogaria fora meses de dado; comparar cada uma contra o
    # limiar trivial NA SUA PRÓPRIA GRADE mantém cada par honesto — duty, FA e
    # denominador idênticos dentro do par, que é o que a comparação exige.
    #
    # ⚠️ A série É recortada na janela, como em `eval_replicas_b2024.py` — e ao contrário
    # de `baseline_trivial_vs_ae.py`, que passa a série inteira. Sem recorte, FA/dia e
    # duty saem medidos sobre a série toda enquanto o recall sai da janela. As duas
    # convenções coincidem na FULL, o que torna o erro fácil de não ver.
    rows = []
    for lab, s in series.items():
        s0, s1 = s.index.min(), s.index.max()
        span_d = (s1 - s0).total_seconds() / 86400.0
        print(f"\n\n########  {lab}  —  grade {s0.date()} → {s1.date()} ({span_d:.0f} d)")

        # limiar trivial reamostrado NA MESMA grade: nenhum braço leva vantagem de amostragem
        arms = {"temp (limiar trivial)": tc03.reindex(s.index, method="nearest"), lab: s}

        for wlab, a, b in bl.JANELAS:
            t0 = max(pd.Timestamp(a, tz="UTC"), s0)
            t1 = min(pd.Timestamp(b, tz="UTC"), s1)
            if t1 <= t0:
                continue
            cov = (t1 - t0) / (pd.Timestamp(b, tz="UTC") - pd.Timestamp(a, tz="UTC"))
            inc = sw.incidents_on(running, tc03, t0, t1)
            if len(inc) < 5:
                print(f"\n=== {wlab} ∩ grade: {len(inc)} incidentes — amostra insuficiente, pulado")
                continue
            print(f"\n=== {wlab} ∩ grade ({cov:.0%} da janela) — {len(inc)} incidentes ON ===")
            print(f"  {'braço':<24}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'hl':>6}")
            for alab, ser in arms.items():
                w = ser[(ser.index >= t0) & (ser.index < t1)].dropna()
                if w.empty:
                    continue
                r = bl.best_over_hl(w, inc, running)
                print(f"  {alab:<24}{(r.get('recall_raw') or 0)*100:>11.1f}%"
                      f"{r.get('fa_per_day', float('nan')):>10.3f}"
                      f"{r.get('duty_sticky', r.get('duty', float('nan'))):>8.3f}"
                      f"{str(r.get('hl')):>6}", flush=True)
                rows.append(dict(task=lab, janela=wlab, braco=alab, n_inc=len(inc),
                                 cobertura=float(cov),
                                 recall_raw=r.get("recall_raw"), fa_per_day=r.get("fa_per_day"),
                                 duty=r.get("duty_sticky", r.get("duty")), hl=r.get("hl")))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nGravado: {OUT}")
    print("\n⚠️ Diagnóstico, não veredito: treino/pré-processamento diferem do b2024 em "
          "mais coisas que o multivariado. Vitória aqui é forte; derrota é inconclusiva.")


if __name__ == "__main__":
    main()
