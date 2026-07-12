#!/usr/bin/env python3
"""Baixa os produtos das tasks MULTIVARIADAS v3 concluídas no ClearML e monta
resultados/Mult_sensor/<eq>/: figs/, csv/, MODEL_CARD.md por equipamento e um
MODELS_INDEX.md consolidado.

Baixa SOMENTE os artefatos do prefixo do GRUPO (== EQUIPMENT_ID) — a task
também sobe point_anomalies/figs de modelos univariados extras (subproduto),
que não fazem parte do resultado do experimento multivariado (ver bug
corrigido em analyze_failure_detection.py).

A partir da pipeline v3 (ver pipeline.py::run_one_group), o próprio treino já
separa as figuras do sensor-alvo (prefixo TARGET_, raiz de figs/) das de
contexto (figs/contexto/) — este script preserva essa estrutura ao baixar.
Para tasks rodadas ANTES dessa mudança, rode uma vez
scripts/reorganize_mult_figs.py depois deste script para reorganizar.

Uso:
    PYTHONPATH=. python scripts/collect_multivar_results.py \
        --out resultados/Mult_sensor
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict
from src.cnn1d_ae.model_card import write_model_card
from src.cnn1d_ae.pipeline import parse_failure_dates

TASK_IDS_JSON = Path("analysis/multivar_v3_task_ids.json")


def _ensure_clearml_config() -> None:
    import os
    if os.getenv("CLEARML_CONFIG_FILE"):
        return
    local = Path.cwd() / "clearml.conf"
    if local.is_file():
        os.environ["CLEARML_CONFIG_FILE"] = str(local)


def _get_local_copy_retry(art, attempts: int = 3):
    for _ in range(attempts):
        try:
            p = art.get_local_copy()
        except Exception:
            p = None
        if p:
            return p
    return None


def download_group(task, eq: str, eq_dir: Path) -> list:
    """Baixa todos os artefatos cujo prefixo == eq (nome do grupo). Retorna
    a lista de nomes de artefatos que falharam no download."""
    missed = []
    prefix = f"{eq}/"
    for name, art in task.artifacts.items():
        if not name.startswith(prefix):
            continue
        local = _get_local_copy_retry(art)
        if not local:
            missed.append(name)
            print(f"  [WARN] download falhou: {name}")
            continue
        rest = name[len(prefix):]  # "csv/foo.csv" | "figs/bar.png" | "best_hyperparameters_json"
        if rest.startswith("csv/"):
            target = eq_dir / "csv" / rest[len("csv/"):]
        elif rest.startswith("figs/"):
            target = eq_dir / "figs" / rest[len("figs/"):]
        else:
            target = eq_dir / f"{rest}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
    return missed


def rebuild_card(eq: str, eq_dir: Path) -> str | None:
    csv_dir = eq_dir / "csv"
    run_cfg_p = csv_dir / "run_config.json"
    calib_p = csv_dir / "calibration_report.json"
    group_def_p = csv_dir / "group_definition.json"
    hp_p = eq_dir / "best_hyperparameters_json.json"
    if not (run_cfg_p.exists() and calib_p.exists()):
        return None

    cfg = update_cfg_from_dict(PipelineConfig(), json.loads(run_cfg_p.read_text(encoding="utf-8")))
    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    best_hp = json.loads(hp_p.read_text(encoding="utf-8")) if hp_p.exists() else None
    group_def = json.loads(group_def_p.read_text(encoding="utf-8")) if group_def_p.exists() else {}
    sensors = group_def.get("sensors") or calib.get("sensors") or [eq]

    data_period = None
    seq_p = csv_dir / "sequence_scores_all.csv"
    if seq_p.exists():
        try:
            ts = pd.read_csv(seq_p, usecols=[0]).iloc[:, 0]
            ts = pd.to_datetime(ts, errors="coerce").dropna()
            if len(ts):
                data_period = f"{ts.min()} → {ts.max()}"
        except Exception:
            pass

    out_dirs = {"root": str(eq_dir), "figs": str(eq_dir / "figs"), "csv": str(csv_dir)}
    return write_model_card(
        cfg, out_dirs,
        kind="group", name=calib.get("group", eq), sensors=sensors,
        best_hp=best_hp, threshold=float(calib.get("threshold", float("nan"))),
        calibration_report=calib, failure_times=parse_failure_dates(cfg.FAILURE_DATE),
        n_features=len(sensors), data_period=data_period,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resultados/Mult_sensor")
    ap.add_argument("--only", nargs="*", help="Subconjunto de equipamentos (default: todos).")
    args = ap.parse_args()

    _ensure_clearml_config()
    from clearml import Task

    ids = json.loads(TASK_IDS_JSON.read_text(encoding="utf-8"))
    wanted = args.only or list(ids.keys())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows, pending, failed = [], [], []
    for eq in wanted:
        tid = ids[eq]
        task = Task.get_task(task_id=tid)
        status = task.get_status()
        if status != "completed":
            (failed if status in ("failed", "closed") else pending).append((eq, status))
            print(f"[SKIP] {eq}: status={status}")
            continue

        eq_dir = out_root / eq
        print(f"[GET ] {eq} (task {tid[:8]}...) -> {eq_dir}")
        try:
            missed = download_group(task, eq, eq_dir)
            card = rebuild_card(eq, eq_dir)
            if card:
                print(f"[CARD] {card}")
            if missed:
                print(f"[WARN] {eq}: {len(missed)} artefato(s) não baixados")
        except Exception as exc:
            print(f"[ERRO] {eq}: {exc}")
            failed.append((eq, f"coleta: {exc}"))
            continue

        calib_p = eq_dir / "csv" / "calibration_report.json"
        calib = json.loads(calib_p.read_text(encoding="utf-8")) if calib_p.exists() else {}
        rows.append({
            "equip": eq, "task_id": tid,
            "sensors": ", ".join(calib.get("sensors", [])),
            "target_sensor": calib.get("target_sensor", ""),
            "hit_rate": calib.get("hit_rate"),
            "threshold": calib.get("threshold"),
            "n_alarms": calib.get("n_alarms"),
            "hits": calib.get("alarms_with_detected_anomaly_in_window"),
            "rate_per_day": calib.get("anomaly_rate_points_per_day"),
        })

    lines = [
        "# Resultados multivariados v3 — Transpetro (12 equipamentos)",
        "",
        f"- **Concluídos:** {len(rows)} | **Em progresso:** {len(pending)} | **Falhas:** {len(failed)}",
        "",
        "| Equipamento | Sensor-alvo | Hit rate | Alarmes (hit/total) | Limiar (MAE) | Anom./dia | Card |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["equip"]):
        thr = r["threshold"]
        thr_s = f"{thr:.6g}" if isinstance(thr, (int, float)) else "—"
        card_rel = f"{r['equip']}/MODEL_CARD.md"
        lines.append(
            f"| `{r['equip']}` | {r['target_sensor']} | {r['hit_rate'] if r['hit_rate'] is not None else '—'} | "
            f"{r['hits']}/{r['n_alarms']} | {thr_s} | {r['rate_per_day']} | [card](<{card_rel}>) |")
    if pending:
        lines += ["", "## Em progresso", ""] + [f"- `{eq}` — {st}" for eq, st in pending]
    if failed:
        lines += ["", "## Com erro", ""] + [f"- `{eq}` — {st}" for eq, st in failed]
    lines.append("")

    index_p = out_root / "MODELS_INDEX.md"
    index_p.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[INDEX] {index_p}")
    print(f"[DONE] concluídos={len(rows)} pendentes={len(pending)} falhas={len(failed)}")


if __name__ == "__main__":
    main()
