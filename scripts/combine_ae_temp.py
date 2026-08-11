#!/usr/bin/env python3
"""
combine_ae_temp.py
Testa combinar o autoencoder com o limiar na própria temperatura.

Contexto: o limiar sozinho supera o AE na janela FULL (81,0% × 62,0%, FA 2,4× menor),
mas o AE ganha em 2026 (93,8% × 68,8%). A hipótese é que eles errem em regimes
DIFERENTES — o limiar falha quando há folga térmica (2025, máquina a 650 °C, um alarme
exige excursão brusca) e o AE falha quando não há (2024/2026, máquina a ~708 °C).

⚠️ CONTROLE DE GRAU DE LIBERDADE. Uma combinação com dois limiares independentes
ganharia por busca, não por informação. Por isso as fusões aqui são ALGÉBRICAS sobre os
dois ranks e gastam exatamente os mesmos parâmetros que um braço isolado — um half-life
(compartilhado) e um threshold:

    max    = OR suave   (dispara se QUALQUER canal está alto)
    mean   = consenso
    min    = AND suave  (exige os dois altos)
    w·temp + (1−w)·ae   — tem 1 parâmetro a mais (w); marcado como tal

Critério de promoção fixado ANTES de rodar: a fusão só vale se ganhar do MELHOR braço
isolado **na janela FULL e também nos sub-regimes** (2024 jun–dez quente, 2026 quente,
OOS). Ganhar só na FULL, onde a busca roda, é sinal de sobreajuste do ponto de operação.

Uso:
    PYTHONPATH=. python scripts/combine_ae_temp.py
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

SENSOR = bl.SENSOR
OUT = "eval_predictive_out/combine_ae_temp.csv"
HL_GRID = bl.HL_GRID
JANELAS = bl.JANELAS


def fusions(rank_ae: pd.Series, rank_t: pd.Series) -> dict[str, pd.Series]:
    a, t = rank_ae.align(rank_t, join="inner")
    return {
        "ae  (isolado)": a,
        "temp (isolado)": t,
        "max  (OR suave)": np.maximum(a, t),
        "mean (consenso)": (a + t) / 2.0,
        "min  (AND suave)": np.minimum(a, t),
        "0.75·temp+0.25·ae  [+1 gl]": 0.75 * t + 0.25 * a,
        "0.25·temp+0.75·ae  [+1 gl]": 0.25 * t + 0.75 * a,
    }


def best_point(score: pd.Series, inc: list) -> dict:
    return ev.best_point_for_sensor(score.rank(pct=True), inc,
                                    horizon_hours=bl.HORIZON, sticky_hours=bl.STICKY,
                                    fa_budget=bl.FA_BUDGET, n_thresholds=120,
                                    max_duty_cycle=bl.MAX_DUTY,
                                    max_sticky_duty=bl.MAX_STICKY)


def main() -> None:
    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    mae = ev.load_mae_series(Task.get_task(task_id=bl.TASK), [SENSOR])[SENSOR]
    t_grid = tc03.reindex(mae.index, method="nearest")

    rows = []
    for wlab, a, b in JANELAS:
        t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        inc = sw.incidents_on(running, tc03, t0, t1)
        m = mae[(mae.index >= t0) & (mae.index < t1)].dropna()
        tt = t_grid[(t_grid.index >= t0) & (t_grid.index < t1)].dropna()
        print(f"\n=== {wlab} — {len(inc)} incidentes ON ===")
        print(f"  {'fusão':<30}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'hl':>6}")

        best_por_fusao: dict[str, dict] = {}
        for hl in HL_GRID:      # half-life COMPARTILHADO — mesmo gl dos braços isolados
            ra = sw.ewma_on(m, hl, running).rank(pct=True)
            rt = sw.ewma_on(tt, hl, running).rank(pct=True)
            for name, sc in fusions(ra, rt).items():
                if sc.empty:
                    continue
                r = best_point(sc, inc)
                r["hl"] = hl
                cur = best_por_fusao.get(name)
                key = (r.get("recall_raw") or 0.0, -(r.get("fa_per_day") or 9e9))
                if cur is None or key > (cur.get("recall_raw") or 0.0,
                                         -(cur.get("fa_per_day") or 9e9)):
                    best_por_fusao[name] = r

        for name, r in best_por_fusao.items():
            rr = r.get("recall_raw")
            rows.append(dict(janela=wlab, fusao=name, n_inc=len(inc), recall_raw=rr,
                             fa_per_day=r.get("fa_per_day"), duty=r.get("duty_sticky"),
                             hl=r.get("hl")))
            print(f"  {name:<30}{(f'{rr*100:.1f}%' if rr is not None else '—'):>12}"
                  f"{r.get('fa_per_day', float('nan')):>10.3f}"
                  f"{r.get('duty_sticky', float('nan')):>8.2f}{str(r.get('hl')):>6}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGravado: {OUT}")

    # ---- veredito: precisa ganhar do melhor isolado em TODAS as janelas
    print("\n=== VEREDITO — ganho sobre o melhor braço isolado, por janela ===")
    iso = {"ae  (isolado)", "temp (isolado)"}
    print(f"  {'fusão':<30}" + "".join(f"{w[:16]:>18}" for w, _, _ in JANELAS))
    veredito = {}
    for name in df.fusao.unique():
        if name in iso:
            continue
        cells, ganhos = [], []
        for wlab, _, _ in JANELAS:
            d = df[df.janela == wlab]
            melhor_iso = d[d.fusao.isin(iso)].recall_raw.max()
            v = d[d.fusao == name].recall_raw.iloc[0]
            dpp = (v - melhor_iso) * 100
            ganhos.append(dpp)
            cells.append(f"{v*100:5.1f}% ({dpp:+5.1f})")
        veredito[name] = min(ganhos)
        print(f"  {name:<30}" + "".join(f"{c:>18}" for c in cells))
    print("\n  (valor = recall da fusão; entre parênteses = Δpp contra o MELHOR isolado daquela janela)")
    campeao = max(veredito.items(), key=lambda kv: kv[1])
    print(f"\n  Melhor pior-caso: {campeao[0]}  →  {campeao[1]:+.1f}pp na janela mais adversa")
    print("  Promover só se esse pior caso for > +10pp (ruído de semente do projeto).")


if __name__ == "__main__":
    main()
