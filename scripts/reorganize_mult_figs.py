#!/usr/bin/env python3
"""Reorganiza figs/ já baixadas em resultados/experimento_1_mascara_v3/Mult_sensor/<eq>/ para o novo
padrão: sensor-alvo em destaque (prefixo TARGET_, raiz de figs/) e demais
sensores de entrada como contexto leve (figs/contexto/, um PNG duplo-eixo
cada). Não re-treina nem rebaixa nada — só renomeia/move arquivos locais já
existentes, para equivaler ao que a pipeline v3+ passa a gerar direto.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("resultados/experimento_1_mascara_v3/Mult_sensor")


def reorganize_equip(eq_dir: Path) -> None:
    calib_p = eq_dir / "csv" / "calibration_report.json"
    if not calib_p.exists():
        print(f"[SKIP] {eq_dir.name}: sem calibration_report.json")
        return
    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    target = calib.get("target_sensor")
    sensors = calib.get("sensors", [])
    figs_dir = eq_dir / "figs"
    contexto_dir = figs_dir / "contexto"
    contexto_dir.mkdir(exist_ok=True)

    if not target or target not in sensors:
        print(f"[WARN] {eq_dir.name}: sem target_sensor válido (target={target!r}); "
              f"movendo tudo para contexto/ (nenhum plot em destaque)")
        target = None

    safe = lambda s: s.replace("/", "_").replace("\\", "_")
    moved_target, moved_ctx, kept = 0, 0, 0

    for f in list(figs_dir.iterdir()):
        if f.is_dir() or not f.suffix == ".png":
            continue
        name = f.name
        if name in ("loss_curve.png", "train_mae_hist.png"):
            kept += 1
            continue

        matched_sensor = next((s for s in sensors if name.endswith(f"_{safe(s)}.png")), None)
        if matched_sensor is None:
            kept += 1
            continue

        kind = name[: name.rindex(f"_{safe(matched_sensor)}.png")]
        if target is not None and matched_sensor == target:
            new_name = f"TARGET_{safe(matched_sensor)}_{kind}.png"
            f.rename(figs_dir / new_name)
            moved_target += 1
        else:
            # Contexto: mantém só o painel duplo-eixo (mais informativo);
            # os demais (series_with_anomalies, subplots, zoom) são redundantes
            # com ele e só ocupariam espaço sem agregar leitura nova.
            if kind == "signal_mae_anomaly":
                f.rename(contexto_dir / f"{safe(matched_sensor)}.png")
                moved_ctx += 1
            else:
                f.unlink()

    print(f"[OK] {eq_dir.name}: alvo='{target}' | {moved_target} figs em destaque, "
          f"{moved_ctx} em contexto/, {kept} mantidas (loss/hist)")


def main() -> None:
    for eq_dir in sorted(ROOT.iterdir()):
        if eq_dir.is_dir() and eq_dir.name.startswith("B-"):
            reorganize_equip(eq_dir)


if __name__ == "__main__":
    main()
