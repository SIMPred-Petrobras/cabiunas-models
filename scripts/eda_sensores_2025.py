"""
EDA dos sensores da Turbina A — 2025 (30s).

Gera relatório em console e figuras em eda_output/.

Uso:
    PYTHONPATH=. python scripts/eda_sensores_2025.py
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_SENSORS = Path("../dados/sensores_brutos_2025_30s.csv")
CSV_ALARMS  = Path("../dados/alarmes_selecionados_turbina_a.csv")
OUT_DIR     = Path("eda_output")
OUT_DIR.mkdir(exist_ok=True)

# Sensores que possuem alarmes associados (foco do modelo)
SENSORS_WITH_ALARMS = [
    "T5_AVG_A",
    "TC382_01_A", "TC382_02_A", "TC382_03_A",
    "TC382_04_A", "TC382_05_A", "TC382_06_A",
    "TV_351X_A", "TV_351Y_A",
    "TV_352X_A", "TV_352Y_A",
    "TV_353X_A", "TV_353Y_A",
    "TV_354X_A", "TV_354Y_A",
    "TV_355X_A", "TV_355Y_A",
]


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Carregando CSV de sensores ...")
    df = pd.read_csv(CSV_SENSORS, parse_dates=["data_datetime"], low_memory=False)
    df = df.sort_values("data_datetime").reset_index(drop=True)

    print("Carregando alarmes ...")
    alarms = pd.read_csv(CSV_ALARMS, parse_dates=["Data da Ocorrência"])
    return df, alarms


def _detect_constant_runs(series: pd.Series, min_length: int = 3) -> pd.Series:
    is_eq = (series == series.shift(1)) & series.notna()
    grp = (~is_eq).cumsum()
    run_len = is_eq.groupby(grp).transform("sum")
    suspect = run_len >= (min_length - 1)
    return (suspect | suspect.shift(-1).fillna(False))


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def report_overview(df: pd.DataFrame) -> None:
    section("VISÃO GERAL DO DATASET")
    print(f"Período  : {df['data_datetime'].min()} → {df['data_datetime'].max()}")
    print(f"Total pts: {len(df):,}")
    print(f"Colunas  : {len(df.columns)}")

    if "RUNNING_A" in df.columns:
        on = df["RUNNING_A"].sum()
        off = len(df) - on
        print(f"\nRUNNING_A:")
        print(f"  ON  (=1): {on:,} pts  ({100*on/len(df):.1f}%)")
        print(f"  OFF (=0): {off:,} pts  ({100*off/len(df):.1f}%)")

        # Por mês
        df["month"] = df["data_datetime"].dt.to_period("M")
        monthly = df.groupby("month")["RUNNING_A"].agg(["sum", "count"])
        monthly["pct_on"] = 100 * monthly["sum"] / monthly["count"]
        print("\nDisponibilidade ON por mês:")
        for m, row in monthly.iterrows():
            print(f"  {m}: {row['pct_on']:.1f}% ON  ({int(row['sum']):,} / {int(row['count']):,})")


def report_constant_runs(df: pd.DataFrame) -> None:
    section("ANÁLISE DE RUNS CONSTANTES (FORWARD-FILL UPSTREAM)")
    print("Runs de >= 3 valores idênticos consecutivos = artefato de interpolação upstream")
    print()

    running = df.get("RUNNING_A", pd.Series(1, index=df.index))
    on_mask = running == 1

    rows = []
    for sensor in SENSORS_WITH_ALARMS:
        if sensor not in df.columns:
            continue
        s = df[sensor]
        runs_mask = _detect_constant_runs(s, min_length=3)
        total_const = runs_mask.sum()
        const_during_on = (runs_mask & on_mask).sum()

        # Contar episódios e tamanhos
        is_eq = (s == s.shift(1)) & s.notna() & on_mask
        grp = (~is_eq).cumsum()
        run_len_series = is_eq.groupby(grp).transform("sum")
        episodes_3plus = (run_len_series[on_mask] >= 2).sum()  # 2 = run de 3 (pois primeiro ponto não conta)

        rows.append({
            "sensor": sensor,
            "total_const_pts": int(total_const),
            "const_during_ON": int(const_during_on),
            "pct_of_ON_pts": f"{100*const_during_on/max(1, on_mask.sum()):.2f}%",
        })
        print(f"  {sensor:<20} | total={total_const:>7,} | durante_ON={const_during_on:>6,} ({100*const_during_on/max(1,on_mask.sum()):.2f}%)")

    return pd.DataFrame(rows)


def report_per_sensor(df: pd.DataFrame, alarms: pd.DataFrame) -> None:
    section("ESTATÍSTICAS POR SENSOR (período ON)")
    on_mask = df.get("RUNNING_A", pd.Series(1, index=df.index)) == 1
    df_on = df[on_mask]

    # Tags com alarmes
    alarm_tags = set(alarms["Tag Alarme"].dropna())

    for sensor in SENSORS_WITH_ALARMS:
        if sensor not in df.columns:
            continue
        s = df_on[sensor].dropna()
        n_alarms = len(alarms[alarms["Tag Alarme"] == sensor])

        # Runs constantes durante ON
        runs_on = _detect_constant_runs(df[sensor], min_length=3) & on_mask
        n_runs_on = runs_on.sum()

        print(f"\n{sensor}  (alarmes={n_alarms}, forward-fill_ON={n_runs_on:,})")
        if len(s) == 0:
            print("  (sem dados ON)")
            continue
        print(f"  n={len(s):,}  min={s.min():.3f}  mean={s.mean():.3f}  "
              f"std={s.std():.3f}  p99={s.quantile(0.99):.3f}  max={s.max():.3f}")

        # Detectar sentinelas (valores extremamente baixos / altos)
        if s.min() < -5:
            pct_sentinel = 100 * (s < -5).mean()
            print(f"  *** SENTINELA detectada: {pct_sentinel:.1f}% dos pts ON < -5 ***")


def plot_sensor_overview(df: pd.DataFrame, alarms: pd.DataFrame) -> None:
    section("GERANDO PLOTS")

    alarm_dt_col = "Data da Ocorrência"
    for sensor in SENSORS_WITH_ALARMS:
        if sensor not in df.columns:
            continue

        s = df.set_index("data_datetime")[sensor].dropna()
        running = df.set_index("data_datetime").get("RUNNING_A")
        alarm_times = (
            alarms.loc[alarms["Tag Alarme"] == sensor, alarm_dt_col].dropna()
            if alarm_dt_col in alarms.columns else pd.Series(dtype="datetime64[ns]")
        )

        fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1, 1]})

        # Série temporal
        ax = axes[0]
        ax.plot(s.index, s.values, lw=0.5, color="#1f77b4", alpha=0.8)
        for t in alarm_times:
            ax.axvline(t, color="red", lw=0.8, alpha=0.5)
        if len(alarm_times):
            ax.scatter(alarm_times, [s.median()] * len(alarm_times),
                       color="red", s=25, zorder=5, label=f"Alarmes ({len(alarm_times)})")
        ax.set_title(f"{sensor} — série completa 2025")
        ax.set_ylabel("valor")
        ax.legend(loc="upper right", fontsize=8)

        # RUNNING_A
        if running is not None:
            axes[1].fill_between(running.index, running.values, alpha=0.6, color="green")
            axes[1].set_ylabel("RUNNING_A")
            axes[1].set_ylim(-0.1, 1.3)

        # Forward-fill mask
        ff = _detect_constant_runs(df.set_index("data_datetime")[sensor], min_length=3)
        axes[2].fill_between(ff.index, ff.astype(int).values, alpha=0.7, color="orange")
        axes[2].set_ylabel("forward-fill")
        axes[2].set_ylim(-0.1, 1.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        out_path = OUT_DIR / f"eda_{sensor}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: {out_path}")


def report_alarm_coverage(df: pd.DataFrame, alarms: pd.DataFrame) -> None:
    section("ALARMES POR SENSOR (2025)")
    dt_col = "Data da Ocorrência"
    if dt_col not in alarms.columns:
        return

    df_2025 = alarms[alarms[dt_col].dt.year == 2025] if pd.api.types.is_datetime64_any_dtype(alarms[dt_col]) else alarms
    for sensor in SENSORS_WITH_ALARMS:
        sub = df_2025[df_2025["Tag Alarme"] == sensor]
        if len(sub) == 0:
            continue
        print(f"\n  {sensor}: {len(sub)} eventos")
        for _, row in sub.head(5).iterrows():
            print(f"    {row[dt_col]}  {row.get('Descrição Alarme', '')}  [{row.get('Status', '')}]")
        if len(sub) > 5:
            print(f"    ... (+{len(sub)-5} outros)")


def main() -> None:
    df, alarms = _load()
    report_overview(df)
    report_constant_runs(df)
    report_per_sensor(df, alarms)
    report_alarm_coverage(df, alarms)
    plot_sensor_overview(df, alarms)
    print("\nEDA concluído.")


if __name__ == "__main__":
    main()
