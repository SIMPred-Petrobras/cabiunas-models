#!/usr/bin/env python3
"""Baixa os produtos das tasks por-sensor concluídas no ClearML e monta a
pasta de resultados: figs/, csv/, MODEL_CARD.md por equipamento e um
MODELS_INDEX.md consolidado.

O MODEL_CARD.md é reconstruído localmente porque a pipeline o grava no
filesystem do worker mas não o envia como artefato — todos os dados
necessários (run_config, calibration_report, best_hyperparameters) estão
disponíveis nos artefatos csv/.

Uso:
    PYTHONPATH=. python scripts/collect_persensor_results.py \
        --out analysis/persensor_results
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict
from src.cnn1d_ae.model_card import write_model_card
from src.cnn1d_ae.pipeline import parse_failure_dates

# EQUIP -> task_id (execução por-sensor de 10/07/2026)
TASKS = {
    "B-0302C": "135e8db1765a472186c97ed992aa1263",
    "B-24001B": "2f8dfd667cad4da993aee2d5c82a0182",
    "B-3403C": "52199deb9d3a4fdabde4e6fbad354044",
    "B-402E": "e5e3be3177b74cedb2e95666c4ea8d13",
    "B-4064A": "6d9dc64a83aa4f9aab113c17085da475",
    "B-4703.24001B": "4ff252ebd40a4ec88a7fcc27cf8b1ab3",
    "B-5401A": "54d332c5a57349c9a270aa659ee25078",
    "B-5501B": "03d819504ba84908b60a68c7114326c0",
    "B-6511502A": "5eab2bd8a09b4eb994346db82330dd45",
    "B-8801C": "a4dc5f94d1734b498466a709224eb9ad",
    "B-8802B": "4479791aba0742219048b7cfadb894ed",
    "B-90001A": "95aa47778c8a4a429d0e4dc8272c2dd8",
}


def _ensure_clearml_config() -> None:
    if os.getenv("CLEARML_CONFIG_FILE"):
        return
    local = Path.cwd() / "clearml.conf"
    if local.is_file():
        os.environ["CLEARML_CONFIG_FILE"] = str(local)


def _get_local_copy_retry(art, attempts: int = 3):
    """get_local_copy() pode retornar None em timeout de rede (servidor via
    túnel). Tenta algumas vezes antes de desistir."""
    for i in range(attempts):
        try:
            local = art.get_local_copy()
        except Exception as exc:
            local = None
            last = exc
        if local:
            return local
    return None


def _download_artifacts(task, eq_dir: Path) -> dict:
    """Baixa todos os artefatos preservando o prefixo (csv/, figs/). Retorna
    um índice {nome_curto: caminho_local}. Tolera artefatos que falham no
    download (registra e segue)."""
    saved = {}
    missed = []
    for name, art in task.artifacts.items():
        # name = "<sensor>/csv/foo.csv" | "<sensor>/figs/bar.png" | "<sensor>/best_hyperparameters_json"
        local_str = _get_local_copy_retry(art)
        if not local_str:
            missed.append(name)
            print(f"[WARN] download falhou (pulado): {name}")
            continue
        local = Path(local_str)
        parts = name.split("/")
        if "csv" in parts:
            sub = eq_dir / "csv"
            fname = parts[-1]
        elif "figs" in parts:
            sub = eq_dir / "figs"
            fname = parts[-1]
        else:
            sub = eq_dir
            fname = Path(urlparse(art.url).path).name or (parts[-1] + ".json")
        sub.mkdir(parents=True, exist_ok=True)
        target = sub / fname
        shutil.copy2(local, target)
        saved[name] = target
    if missed:
        print(f"[WARN] {len(missed)} artefato(s) não baixados para {eq_dir.name}: {missed}")
    return saved


def _rebuild_card(eq: str, eq_dir: Path) -> str | None:
    csv_dir = eq_dir / "csv"
    run_cfg_p = csv_dir / "run_config.json"
    calib_p = csv_dir / "calibration_report.json"
    hp_p = eq_dir / "best_hyperparameters.json"
    if not (run_cfg_p.exists() and calib_p.exists()):
        return None

    cfg = update_cfg_from_dict(PipelineConfig(), json.loads(run_cfg_p.read_text(encoding="utf-8")))
    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    best_hp = json.loads(hp_p.read_text(encoding="utf-8")) if hp_p.exists() else None
    sensor = calib.get("sensor") or (cfg.SENSOR_LIST or [eq])[0]

    # Período coberto a partir do sequence_scores (timestamps).
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
        kind="sensor", name=sensor, sensors=[sensor],
        best_hp=best_hp, threshold=float(calib.get("threshold", float("nan"))),
        calibration_report=calib, failure_times=parse_failure_dates(cfg.FAILURE_DATE),
        n_features=1, data_period=data_period,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/persensor_results")
    ap.add_argument("--only", nargs="*", help="Subconjunto de equipamentos (default: todos concluídos).")
    args = ap.parse_args()

    _ensure_clearml_config()
    from clearml import Task

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    pending = []
    failed = []
    wanted = args.only or list(TASKS.keys())

    for eq in wanted:
        tid = TASKS[eq]
        task = Task.get_task(task_id=tid)
        status = task.get_status()
        if status != "completed":
            (failed if status in ("failed", "closed") else pending).append((eq, status))
            print(f"[SKIP] {eq}: status={status}")
            continue

        eq_dir = out_root / eq
        print(f"[GET ] {eq} ({len(task.artifacts)} artefatos) -> {eq_dir}")
        try:
            _download_artifacts(task, eq_dir)
            card = _rebuild_card(eq, eq_dir)
            if card:
                print(f"[CARD] {card}")
        except Exception as exc:
            print(f"[ERRO] {eq}: {exc}")
            failed.append((eq, f"coleta: {exc}"))
            continue

        calib_p = eq_dir / "csv" / "calibration_report.json"
        calib = json.loads(calib_p.read_text(encoding="utf-8")) if calib_p.exists() else {}
        rows.append({
            "equip": eq, "task_id": tid,
            "sensor": calib.get("sensor", ""),
            "hit_rate": calib.get("hit_rate"),
            "threshold": calib.get("threshold"),
            "n_alarms": calib.get("n_alarms"),
            "hits": calib.get("alarms_with_detected_anomaly_in_window"),
            "rate_per_day": calib.get("anomaly_rate_points_per_day"),
        })

    # Índice consolidado
    lines = [
        "# Resultados por-sensor — Transpetro (12 equipamentos)",
        "",
        f"- **Concluídos:** {len(rows)} | **Em progresso:** {len(pending)} | **Falhas:** {len(failed)}",
        "",
        "| Equipamento | Sensor-alvo | Hit rate | Alarmes (hit/total) | Limiar (MAE) | Anom./dia | Card |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["equip"]):
        hr = r["hit_rate"]
        thr = r["threshold"]
        thr_s = f"{thr:.6g}" if isinstance(thr, (int, float)) else "—"
        card_rel = f"{r['equip']}/MODEL_CARD.md"
        lines.append(
            f"| `{r['equip']}` | {r['sensor']} | {hr if hr is not None else '—'} | "
            f"{r['hits']}/{r['n_alarms']} | {thr_s} | {r['rate_per_day']} | [card](<{card_rel}>) |"
        )
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
