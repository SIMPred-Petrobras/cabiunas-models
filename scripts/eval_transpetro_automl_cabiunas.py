#!/usr/bin/env python3
"""
eval_transpetro_automl_cabiunas.py
Passa o score do AutoML da Transpetro (task `efe4884d`, Dense multivariado sobre 36
sensores) pelo NOSSO protocolo, para produzir um número comparável com os nossos braços.

Por que refazer. O notebook deles reporta "FP 0,00% com 28,1% de pré-falha", mas:
  · o rótulo do evento é `2025-01-17 22:55:21`, que é o alarme NORMALIZANDO (OK) — o HI
    foi 18,0 h antes, então a janela de "pré-falha" contém o incidente inteiro (43,3% dela
    já está acima de 760 °C);
  · a janela "normal" é 01–08/01/2025, com a turbina ligada em só 7,0% do tempo;
  · a avaliação do notebook usa 1 evento, com 249 trials + 56 combinações de limiar;
  · no treino do AutoML foram 368 eventos com intervalo mediano de 0,81 dia, o que faz as
    janelas de pré-falha cobrirem 83,3% da série — "prever evento em 3 dias" é quase
    sempre verdade.

Nada disso diz que o modelo é ruim. Diz que aquele número não mede. Aqui o mesmo score é
avaliado como avaliamos todo o resto: onset do HI/HIHI (não do OK), máquina ligada,
horizonte 8 h, sticky 12 h, FA ≤ 1/dia, duty ≤ 0,25, com o limiar trivial e o nosso melhor
AE na MESMA grade e janela, e com o piso do acaso ao lado.

Como o modelo deles é multivariado (um score para o equipamento), avaliamos contra dois
conjuntos de incidentes: os do TC382_03_A (comparável com tudo que já medimos) e os de
QUALQUER canal térmico (mais justo para um modelo de equipamento).

Uso:
    PYTHONPATH=. python scripts/eval_transpetro_automl_cabiunas.py
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

TASK_TP = "efe4884dc12c4a179765e3ffe2579a03"      # AutoML Transpetro
TASK_AE = "1a15c26d994e44febb77f0bec8c2b378"      # nosso melhor braço (b2024)
SENSOR = "TC382_03_A"
TERMICOS = ["T5_AVG_A"] + [f"TC382_0{i}_A" for i in range(1, 7)]
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
OUT = "eval_predictive_out/eval_transpetro_automl.csv"
HOT = 500.0
EXCL_HI = ["UNDER", "CFN", "LOLO", "OVER"]        # só superaquecimento


def incidentes(sensores: list, running: pd.Series, maq_on: pd.Series,
               t0, t1) -> list:
    """Onset de HI/HIHI, com MÁQUINA ligada (árbitro robusto, não o valor do sensor)."""
    todos = []
    al = ev.load_alarms_gap(ALARM, exclude_conditions=EXCL_HI)
    for s in sensores:
        todos += [a for a in al.get(s, []) if t0 <= a <= t1]
    inc = ev.cluster_incidents(sorted(todos), gap_hours=ev.GAP_HOURS)
    if not inc:
        return []
    on = maq_on.reindex(pd.DatetimeIndex(inc), method="nearest")
    return [t for t, o in zip(inc, on.values) if bool(o)]


def main() -> None:
    print("[1/4] dados brutos...", flush=True)
    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A"] + TERMICOS,
                      low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    vals = {s: pd.to_numeric(raw[s], errors="coerce") for s in TERMICOS}
    maq_on = (pd.concat(vals.values(), axis=1).max(axis=1) > HOT) & (running > 0.5)

    print("[2/4] score do AutoML Transpetro...", flush=True)
    tp = Task.get_task(task_id=TASK_TP).artifacts["best_scores"].get()
    tp.index = pd.to_datetime(tp.index)
    if tp.index.tz is None:
        # o evento do notebook (22:55:21) bate ao segundo com o alarme OK em UTC,
        # então o carimbo é UTC sem tz declarada
        tp.index = tp.index.tz_localize("UTC")
    tp = tp.sort_index()["reconstruction_error"]
    print(f"      {len(tp):,} pontos  {tp.index[0]} → {tp.index[-1]}")

    print("[3/4] nosso AE b2024...", flush=True)
    mae = ev.load_mae_series(Task.get_task(task_id=TASK_AE), [SENSOR])[SENSOR]

    # INTERSEÇÃO: grade do nosso MAE (5 min) dentro do período coberto pelos dois
    t0 = max(tp.index[0], mae.index[0])
    t1 = min(tp.index[-1], mae.index[-1])
    idx = mae.index[(mae.index >= t0) & (mae.index <= t1)]
    print(f"      interseção: {len(idx):,} pontos  {t0.date()} → {t1.date()}")

    bracos = {
        "AutoML Transpetro (Dense, 36 sensores)": tp.reindex(idx, method="nearest"),
        "AE b2024 (nosso melhor)": mae.reindex(idx),
        "limiar trivial (TC382_03_A)": vals[SENSOR].reindex(idx, method="nearest"),
    }

    rows = []
    conjuntos = [("incidentes do TC382_03_A", [SENSOR]),
                 ("incidentes de QUALQUER térmico", TERMICOS)]
    janelas = [("interseção completa", t0, t1),
               ("OOS jul/25→abr/26", pd.Timestamp("2025-07-01", tz="UTC"), t1)]

    print("[4/4] avaliação...", flush=True)
    for clab, sens in conjuntos:
        for wlab, a, b in janelas:
            inc = incidentes(sens, running, maq_on, a, b)
            print(f"\n=== {clab} · {wlab} — {len(inc)} incidentes HI/HIHI com máquina ON ===")
            print(f"  {'braço':<40}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'hl':>6}")
            for name, sc in bracos.items():
                s = sc[(sc.index >= a) & (sc.index <= b)].dropna()
                r = bl.best_over_hl(s, inc, running) if not s.empty else {}
                rr = r.get("recall_raw")
                rows.append(dict(conjunto=clab, janela=wlab, braco=name, n_inc=len(inc),
                                 recall_raw=rr, fa_per_day=r.get("fa_per_day"),
                                 duty=r.get("duty_sticky"), hl=r.get("hl")))
                print(f"  {name:<40}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                      f"{r.get('fa_per_day', float('nan')):>10.3f}"
                      f"{r.get('duty_sticky', float('nan')):>8.2f}"
                      f"{str(r.get('hl')):>6}", flush=True)
            # piso do acaso na MESMA janela e grade
            recs, fas = [], []
            for seed in range(5):
                rnd = pd.Series(np.random.default_rng(500 + seed).random(len(idx)), index=idx)
                rr = bl.best_over_hl(rnd[(rnd.index >= a) & (rnd.index <= b)], inc, running)
                recs.append(rr.get("recall_raw") or 0.0)
                fas.append(rr.get("fa_per_day") or np.nan)
            rows.append(dict(conjunto=clab, janela=wlab, braco="PISO DO ACASO (5 sementes)",
                             n_inc=len(inc), recall_raw=float(np.mean(recs)),
                             fa_per_day=float(np.nanmean(fas)), duty=None, hl=None))
            print(f"  {'PISO DO ACASO (5 sementes)':<40}{np.mean(recs)*100:>11.1f}%"
                  f"{np.nanmean(fas):>10.3f}{'':>8}{'':>6}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGravado: {OUT}")

    print("\n=== VEREDITO ===")
    for clab, _ in conjuntos:
        for wlab, _, _ in janelas:
            d = df[(df.conjunto == clab) & (df.janela == wlab)].set_index("braco")
            tpv = d.loc["AutoML Transpetro (Dense, 36 sensores)", "recall_raw"]
            aev = d.loc["AE b2024 (nosso melhor)", "recall_raw"]
            tvv = d.loc["limiar trivial (TC382_03_A)", "recall_raw"]
            piso = d.loc["PISO DO ACASO (5 sementes)", "recall_raw"]
            n = int(d.n_inc.iloc[0])
            print(f"\n  {clab} · {wlab}  (n={n})")
            print(f"    AutoML Transpetro {tpv*100:5.1f}%   ·  acima do acaso: {(tpv-piso)*100:+5.1f}pp")
            print(f"    AE b2024          {aev*100:5.1f}%   ·  acima do acaso: {(aev-piso)*100:+5.1f}pp")
            print(f"    limiar trivial    {tvv*100:5.1f}%   ·  acima do acaso: {(tvv-piso)*100:+5.1f}pp")
            print(f"    piso do acaso     {piso*100:5.1f}%")


if __name__ == "__main__":
    main()
