#!/usr/bin/env python3
"""
eval_multivariado_vs_control.py
Ablação PAREADA: AE conjunto (7 canais térmicos) × AE univariado, no TC382_03_A.

A pergunta é de FALSO POSITIVO. O mecanismo do FP está medido: em [-6h,+3h] do onset,
`|dT5/dt|` mediano é 172 °C/h no FP contra 21 °C/h no TP — o falso positivo é a máquina
MANOBRANDO. Um AE univariado em TC03 não vê a manobra; um AE conjunto que reconstrói TC03
condicionado ao T5 e aos irmãos deveria representá-la como normal e não disparar.

Desenho:
  • entre os braços mudam 3 chaves (MULTIVARIATE_JOINT, MODEL_MODE, TARGET_SENSOR) e mais
    nada — mesmo CSV, mesma janela de treino, mesmo TIME_STEPS/STRIDE, mesmo pré-proc;
  • `MULTI_EXCLUSION_SCOPE="target"` iguala as exclusões de treino dos dois caminhos
    (alarmes, gaps longos, forward-fill), senão "multivariado" viria confundido com
    "treinado em menos dado";
  • MESMAS sementes nos dois braços, comparação Δ PAREADA por semente — com n=3 isso é
    muito mais potente que comparar medianas independentes, e o ruído entre sementes
    já foi medido em 10,3pp (27,6pp sem semeadura).

CRITÉRIO PRÉ-REGISTRADO (fixado antes de ver resultado, sem folga depois):
  vence se, na MEDIANA das 3 sementes, na FULL (58 inc), atingir
      FA/dia <= 0,026  mantendo  recall_raw >= 84,5%     (= o limiar trivial)
  e isso se sustentar no OOS jul/25→abr/26 sem piorar contra o trivial lá.
  Reportado à parte, sem virar o critério: o Δ pareado M−C.

Uso:
    PYTHONPATH=. python scripts/eval_multivariado_vs_control.py
    PYTHONPATH=. python scripts/eval_multivariado_vs_control.py --task ctrl:13=<id> ...
"""
from __future__ import annotations

import argparse
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
ma = _load("eval_multivar_antigo")

SENSOR = "TC382_03_A"
SEEDS = [42, 7, 13]
OUT = "eval_predictive_out/multivariado_vs_control.csv"
PROJECT = "TesteMLCab"

# Tasks já rodadas (controle semeado). As demais são resolvidas por nome do config.
KNOWN = {
    ("ctrl", 42): "6cfbec162c3c48f798549fd97ca557b9",
    ("ctrl", 7): "8bbe1a0ad9b946cca87787ac6ff836de",
}
CONFIG_NAME = {"ctrl": "v15_ctrl_seed{s:02d}", "multi": "v15_multi_seed{s:02d}"}

# Âncora: o controle semeado 42 tem de reproduzir o que já está auditado.
# (verifica_semeadura.csv / a sessão de semeadura: 62,1% @ 0,118 na FULL, hl=4,0)
ANCHOR = {"recall_raw": 0.6207, "fa_per_day": 0.118}


def resolve_task(arm: str, seed: int, overrides: dict) -> str | None:
    if (arm, seed) in overrides:
        return overrides[(arm, seed)]
    if (arm, seed) in KNOWN:
        return KNOWN[(arm, seed)]
    name = "cnn1d-ae-" + CONFIG_NAME[arm].format(s=seed)
    cands = [t for t in Task.get_tasks(project_name=PROJECT, task_name=name)
             if t.name == name and t.get_status() == "completed"]
    if not cands:
        return None
    return cands[0].id  # get_tasks devolve mais recente primeiro


def load_score(arm: str, task_id: str) -> pd.Series:
    """MAE do canal TC382_03_A. O caminho multivariado nomeia o artifact pelo GRUPO
    (`MULTI_*`), e `ev.load_mae_series` casa por substring do sensor — devolve vazio em
    silêncio nessas tasks. Daí o loader dedicado por braço."""
    t = Task.get_task(task_id=task_id)
    if arm == "multi":
        return ma.load_multi_mae(t, SENSOR)
    return ev.load_mae_series(t, [SENSOR])[SENSOR]


def check_anchor(row: dict) -> None:
    r, f = row.get("recall_raw"), row.get("fa_per_day")
    ok = (r is not None and abs(r - ANCHOR["recall_raw"]) < 0.02
          and abs(f - ANCHOR["fa_per_day"]) < 0.01)
    print(f"\n[ÂNCORA] controle semente 42 na FULL: "
          f"{(r or 0)*100:.1f}% @ {f:.3f}  (esperado {ANCHOR['recall_raw']*100:.1f}% @ "
          f"{ANCHOR['fa_per_day']:.3f}) → {'OK' if ok else 'DIVERGIU'}")
    if not ok:
        raise SystemExit(
            "Âncora não bate: o controle não reproduz a linha auditada. Algo mudou no "
            "protocolo ou na task — resolver isso ANTES de ler qualquer comparação."
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", nargs="*", default=[],
                   help="override braço:semente=task_id (ex: multi:13=abc123...)")
    args = p.parse_args()
    overrides = {}
    for item in args.task:
        k, v = item.split("=", 1)
        arm, seed = k.split(":")
        overrides[(arm, int(seed))] = v.strip()

    raw = pd.read_csv(bl.RAW, usecols=["data_datetime", "RUNNING_A", SENSOR], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    running = pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)
    tc03 = pd.to_numeric(raw[SENSOR], errors="coerce")

    print("[1/3] Resolvendo tasks...")
    series: dict[tuple[str, int], pd.Series] = {}
    for arm in ("ctrl", "multi"):
        for s in SEEDS:
            tid = resolve_task(arm, s, overrides)
            if tid is None:
                print(f"  {arm}/{s:02d}: NÃO ENCONTRADA — pulada")
                continue
            try:
                series[(arm, s)] = load_score(arm, tid).dropna()
            except Exception as exc:
                print(f"  {arm}/{s:02d}: erro ao carregar ({exc})")
                continue
            print(f"  {arm}/{s:02d}: {tid[:8]}  {len(series[(arm, s)]):,} pts")

    if not series:
        raise SystemExit("nenhuma task carregada")

    # Grade comum a TODOS os braços: duty, FA e denominador ficam idênticos, e nenhum
    # braço leva vantagem de amostragem. (Mesmo padrão de verifica_semeadura.py.)
    idx = None
    for s in series.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    series = {k: v.reindex(idx).dropna() for k, v in series.items()}
    print(f"[2/3] Grade comum: {len(idx):,} pontos "
          f"({idx.min().date()} → {idx.max().date()})")

    arms = {"temp (limiar trivial)": tc03.reindex(idx, method="nearest")}
    for (arm, s), v in series.items():
        arms[f"{arm} semente {s:02d}"] = v

    print("[3/3] Avaliando...")
    rows = []
    for wlab, a, b in bl.JANELAS:
        t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        inc = sw.incidents_on(running, tc03, t0, t1)
        if len(inc) < 5:
            continue
        print(f"\n=== {wlab} — {len(inc)} incidentes ON ===")
        print(f"  {'braço':<24}{'recall_raw':>12}{'FA/dia':>10}{'duty':>8}{'hl':>6}")
        for alab, ser in arms.items():
            r = bl.best_over_hl(ser, inc, running)
            print(f"  {alab:<24}{(r.get('recall_raw') or 0)*100:>11.1f}%"
                  f"{r.get('fa_per_day', float('nan')):>10.3f}"
                  f"{r.get('duty_sticky', r.get('duty', float('nan'))):>8.3f}"
                  f"{str(r.get('hl')):>6}", flush=True)
            rows.append(dict(janela=wlab, braco=alab, n_inc=len(inc),
                             recall_raw=r.get("recall_raw"), fa_per_day=r.get("fa_per_day"),
                             duty=r.get("duty_sticky", r.get("duty")), hl=r.get("hl")))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    if not df.empty:
        anchor_row = df[(df.janela.str.startswith("FULL")) & (df.braco == "ctrl semente 42")]
        if len(anchor_row):
            check_anchor(anchor_row.iloc[0].to_dict())

    veredito(df)
    print(f"\nGravado: {OUT}")


def veredito(df: pd.DataFrame) -> None:
    for wlab in df.janela.unique():
        w = df[df.janela == wlab].set_index("braco")
        n = int(w.n_inc.iloc[0])
        tri = w.loc["temp (limiar trivial)"]

        print(f"\n\n########  {wlab} — {n} incidentes  ########")

        # Δ pareado por semente: cada semente é o seu próprio controle.
        deltas_r, deltas_f = [], []
        print(f"  {'semente':<10}{'recall C→M':>22}{'FA C→M':>22}")
        for s in SEEDS:
            c, m = f"ctrl semente {s:02d}", f"multi semente {s:02d}"
            if c not in w.index or m not in w.index:
                continue
            dr = (w.loc[m, "recall_raw"] - w.loc[c, "recall_raw"]) * 100
            dfa = w.loc[m, "fa_per_day"] - w.loc[c, "fa_per_day"]
            deltas_r.append(dr)
            deltas_f.append(dfa)
            print(f"  {s:<10}{w.loc[c,'recall_raw']*100:>8.1f}% →{w.loc[m,'recall_raw']*100:>7.1f}%"
                  f" ({dr:+.1f}pp){w.loc[c,'fa_per_day']:>10.3f} →{w.loc[m,'fa_per_day']:>7.3f}"
                  f" ({dfa:+.3f})")
        if deltas_r:
            print(f"  {'Δ mediano':<10}{np.median(deltas_r):>21.1f}pp"
                  f"{np.median(deltas_f):>22.3f}")

        med = {}
        for arm in ("ctrl", "multi"):
            sel = w[w.index.str.startswith(arm + " semente")]
            if len(sel):
                med[arm] = (sel.recall_raw.median(), sel.fa_per_day.median())
                print(f"  mediana {arm:<6}: {med[arm][0]*100:>5.1f}% @ FA {med[arm][1]:.3f} "
                      f"(n={len(sel)} sementes)")
        print(f"  limiar trivial : {tri.recall_raw*100:>5.1f}% @ FA {tri.fa_per_day:.3f}")

        if wlab.startswith("FULL") and "multi" in med:
            r, f = med["multi"]
            passou = (f <= tri.fa_per_day) and (r >= tri.recall_raw)
            print(f"\n  CRITÉRIO PRÉ-REGISTRADO (FA <= {tri.fa_per_day:.3f} e "
                  f"recall >= {tri.recall_raw*100:.1f}%): "
                  f"{'ATINGIDO' if passou else 'NÃO ATINGIDO'}")


if __name__ == "__main__":
    main()
