"""
Task 1: Análise de descontinuidades e gradientes em todas as séries.

Pergunta central: gradientes altos (delta_y/delta_x) ocorrem consistentemente
nas faixas verdes (operação normal), ou são exclusivos dos períodos de alarme?

Saídas (OUTPUT_DIR/):
  - figs/grad_<sensor>.png  : série + gradiente + zonas por sensor
  - figs/grad_summary.png   : boxplot de gradientes por zona (todos sensores)
  - csv/gradient_stats.csv  : tabela com percentis de gradiente por zona/sensor

Uso:
    PYTHONPATH=. python scripts/analise_gradientes.py --config configs/calibracao_v5_brutos/v5brutos_TC382_03_A.json
    PYTHONPATH=. python scripts/analise_gradientes.py --config configs/local.json --output_dir analise_gradientes_out
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")

from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict
from src.cnn1d_ae.io import load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_alarm_mask(index: pd.DatetimeIndex, alarm_times: pd.Series, minutes: int) -> pd.Series:
    mask = pd.Series(False, index=index)
    win = pd.Timedelta(minutes=minutes)
    for t in alarm_times.dropna():
        mask.loc[(mask.index >= t - win) & (mask.index <= t + win)] = True
    return mask


def _build_startup_mask(index: pd.DatetimeIndex, running: pd.Series, minutes: int) -> pd.Series:
    """Marca os primeiros N minutos após cada transição RUNNING_A 0→1."""
    mask = pd.Series(False, index=index)
    if minutes <= 0:
        return mask
    run_b = (running.reindex(index).fillna(0) == 1.0).astype(int)
    starts = run_b.index[run_b.diff().fillna(0) > 0].tolist()
    win = pd.Timedelta(minutes=minutes)
    for t in starts:
        mask.loc[(mask.index >= t) & (mask.index <= t + win)] = True
    return mask


def _classify_zone(
    index: pd.DatetimeIndex,
    running: pd.Series,
    alarm_mask: pd.Series,
    startup_mask: pd.Series,
) -> pd.Series:
    """
    Classifica cada ponto em:
      'normal'   → RUNNING=1, sem alarme, sem startup   (faixa verde)
      'startup'  → RUNNING=1, primeiros N min após partida
      'alarm'    → dentro da janela de alarme
      'off'      → RUNNING=0 ou desconhecido
    """
    run_aligned = (running.reindex(index).fillna(0) == 1.0)
    zones = pd.Series("off", index=index)
    zones[run_aligned] = "normal"
    zones[run_aligned & startup_mask.reindex(index).fillna(False)] = "startup"
    zones[alarm_mask.reindex(index).fillna(False)] = "alarm"
    return zones


def _compute_gradient(series: pd.Series) -> pd.Series:
    """Gradiente em unidades/minuto (valor absoluto)."""
    dt_min = series.index.to_series().diff().dt.total_seconds().fillna(np.nan) / 60.0
    dy = series.diff().fillna(np.nan)
    grad = (dy / dt_min).abs()
    # Clips extremos devidos a gaps temporais grandes
    p999 = float(grad.quantile(0.999)) if grad.notna().sum() > 10 else np.inf
    return grad.clip(upper=p999)


# ---------------------------------------------------------------------------
# Plot por sensor
# ---------------------------------------------------------------------------

ZONE_COLORS = {
    "normal":  "#2ECC71",   # verde
    "startup": "#F39C12",   # laranja
    "alarm":   "#E74C3C",   # vermelho
    "off":     "#BDC3C7",   # cinza
}


def _plot_sensor(
    sensor: str,
    series: pd.Series,
    gradient: pd.Series,
    zones: pd.Series,
    alarm_times: pd.Series,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"Gradiente — {sensor}", fontsize=12, fontweight="bold")

    ax_s, ax_g = axes

    # --- fundo colorido por zona (blocos contínuos) ---
    prev_zone, span_start = zones.iloc[0], zones.index[0]
    def _flush_span(z, t0, t1, ax):
        ax.axvspan(t0, t1, alpha=0.18, color=ZONE_COLORS.get(z, "#cccccc"), linewidth=0)

    for t, z in zip(zones.index[1:], zones.values[1:]):
        if z != prev_zone:
            _flush_span(prev_zone, span_start, t, ax_s)
            _flush_span(prev_zone, span_start, t, ax_g)
            span_start = t
            prev_zone = z
    _flush_span(prev_zone, span_start, zones.index[-1], ax_s)
    _flush_span(prev_zone, span_start, zones.index[-1], ax_g)

    # --- série bruta ---
    ax_s.plot(series.index, series.values, lw=0.6, color="#2C3E50", label=sensor)
    ax_s.set_ylabel("valor (unidade bruta)", fontsize=9)

    # --- alarmes ---
    for t in alarm_times.dropna():
        ax_s.axvline(t, color="#C0392B", lw=0.8, alpha=0.7, linestyle="--")

    # --- gradiente ---
    ax_g.plot(gradient.index, gradient.values, lw=0.5, color="#8E44AD", alpha=0.8)

    # linha de referência: p95 do gradiente em zona normal
    grad_normal = gradient[zones == "normal"].dropna()
    if len(grad_normal) > 10:
        p95_n = float(grad_normal.quantile(0.95))
        ax_g.axhline(p95_n, color="#27AE60", lw=1.0, linestyle="--",
                     label=f"p95 normal = {p95_n:.3f}")
        p99_n = float(grad_normal.quantile(0.99))
        ax_g.axhline(p99_n, color="#E67E22", lw=1.0, linestyle="--",
                     label=f"p99 normal = {p99_n:.3f}")

    ax_g.set_ylabel("|Δy/Δt|\n(unid/min)", fontsize=9)
    ax_g.set_xlabel("tempo", fontsize=9)
    ax_g.legend(fontsize=7, loc="upper right")

    # legenda de zonas
    patches = [mpatches.Patch(color=c, alpha=0.5, label=z) for z, c in ZONE_COLORS.items()]
    ax_s.legend(handles=patches + [ax_s.lines[0]], fontsize=7, loc="upper left", ncol=5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary: boxplot e tabela de estatísticas
# ---------------------------------------------------------------------------

def _plot_summary(stats_rows: list[dict], out_path: str) -> None:
    sensors = sorted({r["sensor"] for r in stats_rows})
    zones   = ["normal", "startup", "alarm"]
    zone_labels = {"normal": "Normal (verde)", "startup": "Startup", "alarm": "Janela alarme"}

    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(max(12, 3 * n_sensors), 5), sharey=False)
    if n_sensors == 1:
        axes = [axes]

    for ax, sensor in zip(axes, sensors):
        data  = []
        labels = []
        colors = []
        for z in zones:
            row = next((r for r in stats_rows if r["sensor"] == sensor and r["zone"] == z), None)
            if row and row["n"] > 0:
                # aproximação da distribuição pelo boxplot usando percentis
                data.append([row["p25"], row["p50"], row["p75"]])
                labels.append(zone_labels[z])
                colors.append(ZONE_COLORS[z])

        if data:
            bp = ax.boxplot(
                [[d[0], d[1], d[2]] for d in data],
                labels=labels,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(sensor, fontsize=8)
        ax.set_ylabel("|Δy/Δt| (unid/min)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Distribuição de Gradientes por Zona — Todos os Sensores", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Análise de gradientes por zona operacional")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="analise_gradientes_out")
    parser.add_argument("--exclude_minutes", type=int, default=None,
                        help="Janela de alarme em minutos (usa config se omitido)")
    parser.add_argument("--startup_minutes", type=int, default=None,
                        help="Minutos de startup a excluir (usa config se omitido)")
    parser.add_argument("--max_sensors", type=int, default=None,
                        help="Limita número de sensores (para debug)")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Data inicial para filtrar (ex: 2025-01-01)")
    parser.add_argument("--end_date", type=str, default=None,
                        help="Data final para filtrar (ex: 2025-12-31)")
    parser.add_argument("--use_local", action="store_true",
                        help="Lê CSVs locais diretamente (ignora ClearML dataset)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    with open(args.config) as f:
        cfg = update_cfg_from_dict(cfg, json.load(f))

    if args.use_local:
        cfg.USE_CLEARML_DATASET = False
        print("[LOCAL] Usando arquivos CSV locais (ClearML dataset ignorado)")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "csv"), exist_ok=True)

    alarm_win = args.exclude_minutes if args.exclude_minutes is not None else cfg.EXCLUDE_MINUTES_AROUND_ALARM
    startup_win = args.startup_minutes if args.startup_minutes is not None else cfg.EXCLUDE_STARTUP_MINUTES

    print(f"[CONFIG] alarm_window={alarm_win}min | startup_window={startup_win}min")
    print("[LOAD] Carregando dados...")

    df_alarm, df_feat, df_raw, _ = load_data(cfg)

    # Decide fonte de dados
    source_df = df_raw if cfg.TRAIN_SOURCE.lower() == "raw" else df_feat

    # Filtro de período
    if args.start_date or args.end_date:
        time_col = cfg.TIME_COL
        source_df = source_df.copy()
        source_df[time_col] = pd.to_datetime(source_df[time_col], errors="coerce")
        if args.start_date:
            source_df = source_df[source_df[time_col] >= pd.Timestamp(args.start_date)]
        if args.end_date:
            source_df = source_df[source_df[time_col] <= pd.Timestamp(args.end_date)]
        print(f"[FILTER] Período: {args.start_date or 'início'} → {args.end_date or 'fim'} "
              f"({len(source_df):,} pontos)")
    sensor_cols = [c for c in source_df.columns if c not in {cfg.TIME_COL, cfg.RUNNING_COL or ""}
                   and not c.startswith("_")]

    if args.max_sensors:
        sensor_cols = sensor_cols[:args.max_sensors]

    print(f"[INFO] {len(sensor_cols)} sensores encontrados: {sensor_cols}")

    # Alarmes de todos os sensores (para a máscara combinada)
    all_alarm_times = pd.to_datetime(
        df_alarm["Data da Ocorrencia"], errors="coerce"
    ).dropna().drop_duplicates().sort_values()

    # RUNNING_COL
    running_col = cfg.RUNNING_COL
    if running_col and running_col in source_df.columns:
        running = source_df.set_index(cfg.TIME_COL)[running_col]
    else:
        running = None
        print("[WARN] RUNNING_COL não encontrado — todos os pontos tratados como ON.")

    stats_rows: list[dict] = []

    for sensor in sensor_cols:
        print(f"[SENSOR] {sensor} ...", end=" ", flush=True)

        series_raw = (
            source_df[[cfg.TIME_COL, sensor]]
            .copy()
            .dropna(subset=[cfg.TIME_COL])
            .sort_values(cfg.TIME_COL)
            .drop_duplicates(subset=[cfg.TIME_COL])
        )
        series_raw[sensor] = pd.to_numeric(series_raw[sensor], errors="coerce")
        series_raw = series_raw.set_index(cfg.TIME_COL)[sensor]

        if series_raw.isna().all() or len(series_raw) < 10:
            print("skipped (sem dados)")
            continue

        # Alarmes deste sensor especificamente (para linhas verticais no plot)
        if "Tag" in df_alarm.columns:
            sensor_alarms = df_alarm[df_alarm["Tag"] == sensor]["Data da Ocorrencia"]
        elif "Tag Alarme" in df_alarm.columns:
            sensor_alarms = df_alarm[df_alarm["Tag Alarme"] == sensor]["Data da Ocorrencia"]
        else:
            sensor_alarms = all_alarm_times

        sensor_alarms = pd.to_datetime(sensor_alarms, errors="coerce").dropna()

        index = series_raw.index

        # Máscaras
        alarm_mask = _build_alarm_mask(index, all_alarm_times, alarm_win)
        run_for_mask = running.reindex(index).fillna(1.0) if running is not None else pd.Series(1.0, index=index)
        startup_mask = _build_startup_mask(index, run_for_mask, startup_win)
        zones = _classify_zone(index, run_for_mask, alarm_mask, startup_mask)

        # Gradiente
        gradient = _compute_gradient(series_raw)

        # Estatísticas por zona
        for zone in ["normal", "startup", "alarm", "off"]:
            g_zone = gradient[zones == zone].dropna()
            n = len(g_zone)
            row = {
                "sensor": sensor,
                "zone": zone,
                "n": n,
                "pct_total": round(100 * n / max(len(gradient), 1), 2),
                "mean": round(float(g_zone.mean()), 6) if n > 0 else None,
                "p25": round(float(g_zone.quantile(0.25)), 6) if n > 0 else None,
                "p50": round(float(g_zone.quantile(0.50)), 6) if n > 0 else None,
                "p75": round(float(g_zone.quantile(0.75)), 6) if n > 0 else None,
                "p95": round(float(g_zone.quantile(0.95)), 6) if n > 0 else None,
                "p99": round(float(g_zone.quantile(0.99)), 6) if n > 0 else None,
                "max": round(float(g_zone.max()), 6) if n > 0 else None,
            }
            stats_rows.append(row)

        # Compara p95 normal vs p95 alarme para flaggar sensores "problemáticos"
        p95_n = next((r["p95"] for r in stats_rows if r["sensor"] == sensor and r["zone"] == "normal"), None)
        p95_a = next((r["p95"] for r in stats_rows if r["sensor"] == sensor and r["zone"] == "alarm"), None)
        ratio_str = f"ratio_alarm/normal={p95_a/p95_n:.1f}" if (p95_n and p95_a and p95_n > 0) else "N/A"
        print(f"p95_normal={p95_n} | p95_alarm={p95_a} | {ratio_str}")

        # Plot individual
        out_fig = os.path.join(args.output_dir, "figs", f"grad_{sensor}.png")
        _plot_sensor(sensor, series_raw, gradient, zones, sensor_alarms, out_fig)

    # Salva tabela CSV
    df_stats = pd.DataFrame(stats_rows)
    csv_path = os.path.join(args.output_dir, "csv", "gradient_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"\n[SAVE] Estatísticas: {csv_path}")

    # Plot sumário
    summary_path = os.path.join(args.output_dir, "figs", "grad_summary.png")
    _plot_summary(stats_rows, summary_path)
    print(f"[SAVE] Sumário: {summary_path}")

    # Imprime tabela resumo no terminal
    print("\n=== RESUMO: p95 do gradiente por sensor e zona ===")
    pivot = df_stats[df_stats["zone"].isin(["normal", "alarm"])].pivot_table(
        index="sensor", columns="zone", values="p95", aggfunc="first"
    )
    if "normal" in pivot.columns and "alarm" in pivot.columns:
        pivot["ratio_alarm/normal"] = (pivot["alarm"] / pivot["normal"].replace(0, np.nan)).round(2)
    print(pivot.to_string())
    print(f"\n[DONE] Resultados em: {args.output_dir}/")


if __name__ == "__main__":
    main()
