"""Verifica se os alarmes UNDER dos 6 termopares TC382 são eventos COORDENADOS
(plant-wide) ou independentes. Se ~16/termopar forem os mesmos timestamps, o
denominador real da frota é ~16 eventos distintos, não a soma (96).

Junta os onsets UNDER (ON, na janela 2025) dos 6 canais, clusteriza por proximidade
(±tol) e conta: nº de eventos distintos × em quantos canais cada um aparece.

Uso:
  PYTHONPATH=. python scripts/check_under_coordination.py \
    --alarm_csv ../dados/alarmes_selecionados_turbina_a.csv --tol_min 30
"""
import argparse
import pandas as pd

import scripts.eval_per_sensor_level as E

DS = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9"
RAWCSV = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
RUN_THR = 50
TC = ["TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A", "TC382_06_A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alarm_csv", default="../dados/alarmes_selecionados_turbina_a.csv")
    ap.add_argument("--tol_min", type=float, default=30.0)
    args = ap.parse_args()

    ngp = pd.read_csv(RAWCSV, usecols=["data_datetime", "NGP_A"])
    ngp["data_datetime"] = pd.to_datetime(ngp["data_datetime"], utc=True, errors="coerce")
    ngp = ngp.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()["NGP_A"]
    t_lo, t_hi = ngp.index.min(), ngp.index.max()  # janela 2025 dos sinais

    df, _, cond_col, tag_col = E._parse_alarm_df(args.alarm_csv)
    df = df[df[cond_col].astype(str).str.upper() == "UNDER"]

    # coleta (tempo, sensor) de UNDER ON na janela 2025
    events = []
    per_ch = {}
    for s in TC:
        ts = sorted(pd.to_datetime(df[df[tag_col] == s]["_time"], utc=True).tolist())
        ts = [t for t in ts if t_lo <= t <= t_hi]
        ts = E.cluster_incidents(ts, gap_hours=E.GAP_HOURS)
        on = ngp.reindex(pd.DatetimeIndex(ts), method="nearest") > RUN_THR if ts else pd.Series([], dtype=bool)
        ts = [t for t, o in zip(ts, on.values) if o]
        per_ch[s] = len(ts)
        events += [(t, s) for t in ts]

    events.sort()
    tol = pd.Timedelta(minutes=args.tol_min)
    clusters = []  # cada um: dict(t0, sensors set)
    for t, s in events:
        if clusters and (t - clusters[-1]["t_last"]) <= tol:
            clusters[-1]["sensors"].add(s)
            clusters[-1]["t_last"] = t
        else:
            clusters.append({"t0": t, "t_last": t, "sensors": {s}})

    soma = sum(per_ch.values())
    print(f"Alarme: {args.alarm_csv}  tol=±{args.tol_min:.0f}min  janela {t_lo.date()}..{t_hi.date()}\n")
    print("UNDER ON por canal:", per_ch)
    print(f"\nsoma por canal = {soma}")
    print(f"eventos UNDER DISTINTOS (plant-wide, clusterizados) = {len(clusters)}")
    multi = sum(1 for c in clusters if len(c["sensors"]) >= 3)
    print(f"  dos quais aparecem em >=3 canais (coordenados) = {multi}")
    print(f"  fator de redundância = {soma/max(len(clusters),1):.1f}x")
    print("\ndistribuição de quantos canais por evento:")
    print(pd.Series([len(c["sensors"]) for c in clusters]).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
