#!/usr/bin/env python3
"""
build_mapped_alarm_csv.py
Gera `alarmes_mapeados_colunas.csv`: cópia da base de alarmes com a `Tag Alarme`
renomeada para o NOME DA COLUNA de curva correspondente, via
`configs/calibracao_v12_pressao/tag_column_map.csv`.

Motivo: `pipeline.py` casa alarme↔sensor por igualdade exata
(`df_alarm["Tag"] == sensor`). Os sensores novos têm coluna com nome de historiador
(`954005_624_TI_0305`) mas alarmes com nome de tag (`TAH_6240305`) — sem o remap, o
treino não gera janela de exclusão nem métrica preditiva para eles.

Tags já modeladas (TC382_*, T5, TV_*) ficam intactas — a coluna delas É a tag.
Várias tags podem apontar para a mesma coluna (TAH+TAHH_6240305 → TI_0305): ambas
viram a mesma Tag, o que é o comportamento desejado (o clustering por gap junta).

Uso:
    PYTHONPATH=. python scripts/build_mapped_alarm_csv.py
"""
from __future__ import annotations

import argparse

import pandas as pd

ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
MAP_CSV = "configs/calibracao_v12_pressao/tag_column_map.csv"
OUT_CSV = "../dados/alarmes_mapeados_colunas.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alarm_csv", default=ALARM_CSV)
    p.add_argument("--map_csv", default=MAP_CSV)
    p.add_argument("--out", default=OUT_CSV)
    args = p.parse_args()

    alarms = pd.read_csv(args.alarm_csv)
    mp = pd.read_csv(args.map_csv)
    ok = mp[(mp["status"] == "ok") & mp["coluna"].astype(str).str.len().gt(0)]
    remap = dict(zip(ok["tag"], ok["coluna"]))

    before = alarms["Tag Alarme"].copy()
    alarms["Tag Alarme"] = alarms["Tag Alarme"].map(remap).fillna(alarms["Tag Alarme"])
    n_changed = int((alarms["Tag Alarme"] != before).sum())

    alarms.to_csv(args.out, index=False)
    changed = sorted(set(before[alarms["Tag Alarme"] != before]))
    print(f"{n_changed} linhas remapeadas ({len(changed)} tags): {changed}")
    print(f"Gravado em {args.out}")


if __name__ == "__main__":
    main()
