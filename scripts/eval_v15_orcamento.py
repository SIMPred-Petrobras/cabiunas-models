#!/usr/bin/env python3
"""
eval_v15_orcamento.py
Avalia os braços do experimento v15 (orçamento de busca: MAX_TRIALS × EPOCHS) no
MESMO protocolo da auditoria, para que os números saiam comparáveis com a tabela
da Frente B (v13): horizonte 8h, sticky 12h, FA≤1/dia, duty bruto ≤0.35, duty
pós-sticky ≤0.25, incidentes HI/HIHI com máquina ON e filtro de fantasma (<500°C).
Cenários FULL / BACKCAST_2024 / OOS, melhor ponto por cenário sobre HL {0.5,1,2,4}.

Desenho do v15 (2×2, tudo o mais idêntico ao v13):

    braço b2024  (TRAIN_START_DATE=2024-06-01)  × orçamento t20e20 | t50e80
    braço rerun  (TRAIN_START_DATE=2025-01-01)  × orçamento t20e20 | t50e80

Os braços t20e20 são réplicas do orçamento do v13 — sem eles o delta de orçamento
fica confundido com ruído de retreino (±27pp neste problema), e comparar contra o
número histórico do v13 não prova nada.

Além da métrica de detecção, reporta o MELHOR val_loss do tuner e quantos trials
rodaram: mais orçamento pode melhorar a otimização (val_loss menor) sem mover a
detecção — são perguntas diferentes e o script separa as duas.

O script é INCREMENTAL: cada execução funde os braços pedidos com o CSV já gravado,
então dá para rodar os treinos um a um e ir acumulando a tabela.

Uso:
    # avalia os braços já concluídos (pula os que não completaram)
    PYTHONPATH=. CLEARML_CONFIG_FILE=$(pwd)/clearml.conf \
        python scripts/eval_v15_orcamento.py

    # adiciona um braço novo (ou re-mapeia um id que voce re-rodou)
    PYTHONPATH=. CLEARML_CONFIG_FILE=$(pwd)/clearml.conf \
        python scripts/eval_v15_orcamento.py --arm rerun_t50e80=<task_id>

    PYTHONPATH=. python scripts/eval_v15_orcamento.py --sensor T5_AVG_A
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")


def _resolve_dados() -> str:
    """`dados/` mudou de nível quando o repo passou a viver sob `cabv2/`; resolve
    pelo primeiro caminho que existir em vez de fixar a profundidade."""
    for up in ("..", "../..", "../../.."):
        cand = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("diretório 'dados/' não encontrado a partir do repo.")


DADOS = _resolve_dados()
sw.RAW_CSV = os.path.join(DADOS, "sensores_2024h2_2025_2026_30s.csv")
sw.ALARM_CSV = os.path.join(DADOS, "alarmes_selecionados_turbina_a.csv")
ev.ALARM_CSV_DEFAULT = sw.ALARM_CSV

# Tasks do v15 (commit 9370f21). Reenfileirar um braço gera id novo: passe --arm.
ARMS_DEFAULT = {
    "b2024_t50e80": "158cca6631a54b4fae8db86c79d5010c",
    "rerun_t50e80": "05824cc74b6d4daa8706afcb8eb4a741",
    "b2024_t20e20": "b4438959e7834456bc35980c21a3fc03",
    "rerun_t20e20": "9d1328094ff6430a8fa50ef1ca269861",
}

SCENARIOS = [("FULL", None, None),
             ("BACKCAST_2024", *sw.BACKCAST),
             ("OOS", *sw.OOS)]

OUT_TMPL = "eval_predictive_out/v15_orcamento_{sensor}.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--arm", action="append", default=[],
                   help="label=task_id (repetível); sobrepõe/estende o registro padrão")
    p.add_argument("--out", default=None)
    return p.parse_args()


def task_meta(task) -> dict:
    """MAX_TRIALS/EPOCHS efetivos e o melhor val_loss do tuner, do próprio artefato."""
    meta = {"trials_cfg": None, "epochs_cfg": None, "n_trials": None, "val_loss": None}
    try:
        cfg = task.get_parameters_as_dict().get("pipeline_config", {})
        meta["trials_cfg"] = int(cfg.get("MAX_TRIALS"))
        meta["epochs_cfg"] = int(cfg.get("EPOCHS"))
    except Exception:
        pass
    return meta


def tuner_stats(task, sensor: str) -> dict:
    name = f"{sensor}_csv_trials_ranking.csv"
    if name not in task.artifacts:
        return {"n_trials": None, "val_loss": None}
    try:
        df = pd.read_csv(task.artifacts[name].get_local_copy())
        return {"n_trials": int(len(df)),
                "val_loss": float(df["score_val_loss"].min())}
    except Exception:
        return {"n_trials": None, "val_loss": None}


def main() -> None:
    args = parse_args()
    sensor = args.sensor
    sw.SENSOR = sensor                      # incidents_on lê o global do módulo
    out_csv = args.out or OUT_TMPL.format(sensor=sensor)

    arms = dict(ARMS_DEFAULT)
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm espera label=task_id, recebi {spec!r}")
        k, v = spec.split("=", 1)
        arms[k.strip()] = v.strip()

    from clearml import Task

    running, tc03_unused, t5 = sw.load_raw()
    raw = pd.read_csv(sw.RAW_CSV, usecols=["data_datetime", sensor], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    serie = pd.to_numeric(
        raw.dropna(subset=["data_datetime"]).set_index("data_datetime")[sensor],
        errors="coerce").sort_index()
    t5s = t5.ewm(halflife=int(pd.Timedelta(hours=24) / pd.Timedelta("30s"))).mean()

    rows = []
    for label, tid in arms.items():
        task = Task.get_task(task_id=tid)
        art = f"{sensor}_csv_sequence_scores_all.csv"
        if task.status != "completed" or art not in task.artifacts:
            print(f"[skip] {label:<14} status={task.status} "
                  f"({'sem artefato ' + art if task.status == 'completed' else 'não completou'})")
            continue
        mae = sw.read_mae(task.artifacts[art].get_local_copy())
        meta = task_meta(task) | tuner_stats(task, sensor)
        print(f"[eval] {label:<14} {tid[:8]}  MAX_TRIALS={meta['trials_cfg']} "
              f"EPOCHS={meta['epochs_cfg']}  trials={meta['n_trials']} "
              f"val_loss={meta['val_loss']}")

        for cen, t0, t1 in SCENARIOS:
            m = mae
            if t0 is not None:
                m = m[m.index >= t0]
            if t1 is not None:
                m = m[m.index < t1]
            inc = sw.incidents_on(running, serie, m.index.min(), m.index.max())
            r = sw.best_over_hl(m, inc, running, t5s, sw.health_global)
            rows.append({
                "braco": label, "task_id": tid, "sensor": sensor, "cenario": cen,
                "max_trials": meta["trials_cfg"], "epochs": meta["epochs_cfg"],
                "n_trials": meta["n_trials"], "best_val_loss": meta["val_loss"],
                "inc_on": len(inc),
                "recall": r.get("recall"), "recall_raw": r.get("recall_raw"),
                "fa_per_day": r.get("fa_per_day"), "duty_sticky": r.get("duty_sticky"),
                "hl": r.get("hl"), "threshold_q": r.get("threshold_q"),
                "lead_med_h": r.get("median_lead_hours"),
            })
            rr = r.get("recall_raw")
            print(f"    {cen:<14} inc={len(inc):>2}  "
                  f"recall_raw={'—' if rr is None else f'{rr:.1%}'}  "
                  f"fa={r.get('fa_per_day', float('nan')):.3f}  "
                  f"duty={r.get('duty_sticky', float('nan')):.3f}  hl={r.get('hl')}")

    if not rows:
        raise SystemExit("nenhum braço avaliável — nada gravado.")

    df = pd.DataFrame(rows)
    if os.path.isfile(out_csv):     # incremental: braço re-avaliado sobrescreve o antigo
        old = pd.read_csv(out_csv)
        keep = ~old.set_index(["braco", "cenario"]).index.isin(
            df.set_index(["braco", "cenario"]).index)
        df = pd.concat([old[keep.tolist()], df], ignore_index=True)
    df = df.sort_values(["braco", "cenario"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nGravado em {out_csv}  ({df['braco'].nunique()} braço(s))")
    faltam = [k for k in arms if k not in set(df["braco"])]
    if faltam:
        print(f"Ainda faltam (não completaram): {', '.join(faltam)}")


if __name__ == "__main__":
    main()
