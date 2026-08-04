#!/usr/bin/env python3
"""
plot_pressure_series.py
Figuras exploratórias das séries de pressão que estavam no dataset e nunca foram
modeladas (`954005_624_{PI,PDI,PDIT}_0xxx` em sensores_brutos_2025_2026_30s.csv).

Duas saídas:
  panorama  — small multiples, todos os instrumentos de pressão no período inteiro,
              com o período OFF (RUNNING_A=0) sombreado.
  zoom      — recorte de alguns dias em torno de um onset de alarme, para ver se a
              curva se mexe no evento.

Escalas diferem em uma ordem de grandeza entre instrumentos (PI_0315 ≈ 17 bar,
PDI_0302 ≈ 1,4), então cada série tem seu próprio painel e seu próprio eixo —
NUNCA dois eixos y no mesmo painel.

Uso:
    PYTHONPATH=. python scripts/plot_pressure_series.py
    PYTHONPATH=. python scripts/plot_pressure_series.py --zoom_tag PDAL_6240302 --zoom_days 3
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

RAW_CSV = "../dados/sensores_brutos_2025_2026_30s.csv"
ALARM_CSV = "../dados/alarmes_selecionados_turbina_a.csv"
MAP_CSV = "configs/calibracao_v12_pressao/tag_column_map.csv"
OUT_DIR = "eval_pressure_out"

# Tokens do design system (referência validada) — série em um único tom, porque
# cada painel tem UMA série; a identidade vem do título, não da cor.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
OFF_BAND = "#d8d7d2"     # neutro: estado do equipamento, não é uma série
CRITICAL = "#d03b3b"     # status: onset de alarme


def load(raw_csv: str) -> pd.DataFrame:
    d = pd.read_csv(raw_csv, low_memory=False)
    d["data_datetime"] = pd.to_datetime(d["data_datetime"], format="ISO8601", utc=True)
    return d.set_index("data_datetime").apply(pd.to_numeric, errors="coerce").sort_index()


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=7, length=3)


def shade_off(ax, on: pd.Series) -> None:
    """Sombreia os blocos com RUNNING_A=0. Faixa neutra, atrás da série."""
    blk = (on != on.shift()).cumsum()
    for _, g in on.groupby(blk):
        if not bool(g.iloc[0]):
            ax.axvspan(g.index[0], g.index[-1], color=OFF_BAND, alpha=0.55, lw=0, zorder=0)


def panorama(d: pd.DataFrame, cols: list[str], labels: dict, out: str) -> None:
    on = d["RUNNING_A"] > 0.5
    # decima para plotagem: 30s × 1.4M pontos não acrescenta nada em 16 meses
    step = max(1, len(d) // 6000)
    dd, oo = d.iloc[::step], on.iloc[::step]

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.35 * n), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, c in zip(axes, cols):
        shade_off(ax, oo)
        ax.plot(dd.index, dd[c].values, lw=0.7, color=SERIES, solid_capstyle="round")
        style(ax)
        ax.set_ylabel(labels.get(c, c), fontsize=7.5, color=INK, rotation=0,
                      ha="right", va="center", labelpad=10)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    fig.suptitle("Instrumentos de pressão — 2025-01 a 2026-04  "
                 "(faixa cinza = RUNNING_A desligado)",
                 fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  {out}")


def zoom(d: pd.DataFrame, col: str, tag: str, onsets, days: float, out: str) -> None:
    on = d["RUNNING_A"] > 0.5
    n = min(4, len(onsets))
    if n == 0:
        print(f"  [skip] {tag}: sem onset no período")
        return
    picks = onsets[:: max(1, len(onsets) // n)][:n]
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.0), squeeze=False)
    fig.patch.set_facecolor("white")
    half = pd.Timedelta(days=days / 2)
    for ax, t in zip(axes[0], picks):
        w = d[col][(d.index >= t - half) & (d.index <= t + half)].dropna()
        if w.empty:
            continue
        shade_off(ax, on[(on.index >= t - half) & (on.index <= t + half)])
        ax.plot(w.index, w.values, lw=0.9, color=SERIES, solid_capstyle="round")
        ax.axvline(t, color=CRITICAL, lw=1.4, zorder=3)
        ax.annotate("alarme", xy=(t, ax.get_ylim()[1]), xytext=(3, -8),
                    textcoords="offset points", fontsize=7, color=CRITICAL,
                    va="top", ha="left")
        style(ax)
        ax.set_title(str(t)[:16], fontsize=8, color=INK_MUTED)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
        for lb in ax.get_xticklabels():
            lb.set_rotation(30)
            lb.set_ha("right")
    fig.suptitle(f"{tag} → {col}", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_csv", default=RAW_CSV)
    p.add_argument("--alarm_csv", default=ALARM_CSV)
    p.add_argument("--map_csv", default=MAP_CSV)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--zoom_tag", default="PDAL_6240302")
    p.add_argument("--zoom_days", type=float, default=2.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load(args.raw_csv)
    press = [c for c in d.columns if "_PI_" in c or "_PDI_" in c or "_PDIT_" in c]
    labels = {c: c.replace("954005_624_", "") for c in press}
    print(f"{len(press)} instrumentos de pressão | {d.index.min()} → {d.index.max()}")

    panorama(d, press, labels, os.path.join(args.out_dir, "pressao_panorama.png"))

    mp = pd.read_csv(args.map_csv)
    row = mp[mp["tag"] == args.zoom_tag]
    if row.empty or not isinstance(row.iloc[0]["coluna"], str):
        print(f"[skip] zoom: {args.zoom_tag} sem coluna casada")
        return
    col = row.iloc[0]["coluna"]
    a = pd.read_csv(args.alarm_csv)
    a["_t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce", utc=True)
    ons = a[(a["Tag Alarme"] == args.zoom_tag) &
            (a["Condição do Alarme"].astype(str).str.upper() != "OK")]["_t"]
    ons = sorted(ons[(ons >= d.index.min()) & (ons <= d.index.max())])
    zoom(d, col, args.zoom_tag, ons, args.zoom_days,
         os.path.join(args.out_dir, f"pressao_zoom_{args.zoom_tag}.png"))


if __name__ == "__main__":
    main()
