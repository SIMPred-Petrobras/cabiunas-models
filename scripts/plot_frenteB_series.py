#!/usr/bin/env python3
"""
plot_frenteB_series.py
Série temporal do resultado da Frente B no TC382_03_A, janela FULL (jun/24→abr/26):

  painel 1 — temperatura + incidentes HI/HIHI + OFF sombreado;
  painel 2 — health do B2024 (treino jun/24→jul/25) no ponto FULL da auditoria;
  painel 3 — health do RERUN-controle (treino jan→jul/25) no ponto FULL dele.

Cada painel de health marca hit (verde) / miss (vermelho) por cruzamento BRUTO na
janela de 8h — é o recall_raw da auditoria: 86,2% (b2024) vs 51,7% (rerun). A figura
mostra o mecanismo: sob UM threshold global, o b2024 "enxerga" os incidentes de
jun–dez/2024 que o rerun perde, porque aprendeu o normal daquele regime.

Sanidade: recomputa recall_raw/FA/duty de cada braço no ponto gravado e ABORTA se
não bater com fleet_v13_{b2024,rerun}_FULL_hihihi.csv.

Uso:
    PYTHONPATH=. CLEARML_CONFIG_FILE=$(pwd)/clearml.conf python scripts/plot_frenteB_series.py
"""
from __future__ import annotations

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

SENSOR = "TC382_03_A"
ARMS = {
    "b2024": ("1a15c26d994e44febb77f0bec8c2b378", "eval_predictive_out/fleet_v13_b2024_FULL_hihihi.csv",
              "B2024 — treino jun/24→jul/25"),
    "rerun": ("8700f3a2c7ec4f57887d6d0fdc38017e", "eval_predictive_out/fleet_v13_rerun_FULL_hihihi.csv",
              "Controle — treino jan→jul/25 (mesmo config do v10)"),
}
OUT = "eval_predictive_out/fig_frenteB_TC382_03_A_serie.png"

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2df"
SERIES, OFF_BAND = "#2a78d6", "#d8d7d2"
ALERT, THR = "#f6e3b4", "#b3541e"
HIT, MISS = "#0ca30c", "#d03b3b"


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def shade_off(ax, on: pd.Series):
    blk = (on != on.shift()).cumsum()
    for _, g in on.groupby(blk):
        if not bool(g.iloc[0]):
            ax.axvspan(g.index[0], g.index[-1], color=OFF_BAND, alpha=0.5, lw=0, zorder=0)


def alarm_active_spans(t_lo, t_hi) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Janelas onset→OK do alarme HI/HIHI real do DCS — 'quando o alarme existiu'.

    Um onset HI/HIHI abre a janela; o próximo OK fecha. Sem OK registrado, fecha no
    fim do recorte (alarme ficou pendente)."""
    df = pd.read_csv(sw.ALARM_CSV)
    df = df[df["Tag Alarme"] == SENSOR].copy()
    df["_t"] = pd.to_datetime(df["Data da Ocorrência"], errors="coerce", utc=True)
    df = df.dropna(subset=["_t"]).sort_values("_t")
    spans, onset = [], None
    for _, r in df.iterrows():
        cond = str(r["Condição do Alarme"]).upper()
        if cond in ("HI", "HIHI") and onset is None:
            onset = r["_t"]
        elif cond == "OK" and onset is not None:
            spans.append((onset, r["_t"]))
            onset = None
    if onset is not None:
        spans.append((onset, t_hi))
    return [(a, b) for a, b in spans if b >= t_lo and a <= t_hi]


def raw_hits(h: pd.Series, q: float, incidents: list) -> tuple[list, list]:
    raw_s = np.array([t.timestamp() for t in h.index[h >= q]])
    hs = sw.HORIZON * 3600.0
    hits = [t for t in incidents if raw_s.size and
            np.any((raw_s >= t.timestamp() - hs) & (raw_s <= t.timestamp()))]
    return hits, [t for t in incidents if t not in hits]


def main() -> None:
    running, tc03, _ = sw.load_raw()
    on_full = running > 0.5

    arms = {}
    for name, (tid, csv, label) in ARMS.items():
        row = pd.read_csv(csv).set_index("sensor").loc[SENSOR]
        task = Task.get_task(task_id=tid)
        mae = ev.load_mae_series(task, [SENSOR])[SENSOR]
        h = sw.ewma_on(mae, float(row["hl"]), running).rank(pct=True)
        inc = sw.incidents_on(running, tc03, mae.index.min(), mae.index.max())
        q = float(row["threshold_q"])
        rr, fa, duty = sw.metrics_at(h, q, inc)
        exp = (float(row["recall_raw"]), float(row["fa_per_day"]), float(row["duty_sticky"]))
        print(f"{name}: recall_raw={rr:.3f} (esp {exp[0]:.3f})  fa={fa:.3f} (esp {exp[1]:.3f})  "
              f"duty={duty:.3f} (esp {exp[2]:.3f})  hl={row['hl']} q={q:.3f}")
        if not (abs(rr - exp[0]) < 0.01 and abs(fa - exp[1]) < 0.01 and abs(duty - exp[2]) < 0.01):
            raise SystemExit(f"{name}: ponto de operação não reproduz a auditoria — abortando.")
        hits, misses = raw_hits(h, q, inc)
        alert = ev.apply_sticky(h, q, sw.STICKY)
        arms[name] = dict(h=h, q=q, inc=inc, hits=hits, misses=misses, alert=alert,
                          label=label, rr=rr, fa=fa)

    step = max(1, len(tc03) // 20000)
    temp, on_p = tc03.iloc[::step], on_full.iloc[::step]
    inc_all = arms["b2024"]["inc"]

    fig, axes = plt.subplots(4, 1, figsize=(13.5, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [0.85, 0.12, 1, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    shade_off(ax, on_p)
    ax.plot(temp.index, temp.values, lw=0.55, color=INK, alpha=0.85, zorder=2)
    for t in inc_all:
        ax.axvline(t, color=MISS, lw=0.7, alpha=0.45, zorder=2)
    style(ax)
    ax.set_ylabel("TC382_03_A (°C)", fontsize=9, color=INK)
    ax.set_title("Frente B — janela FULL jun/2024 → abr/2026, um único ponto de operação por braço "
                 f"({len(inc_all)} incidentes HI/HIHI)", fontsize=11, color=INK, loc="left")

    # pista dedicada: blocos sólidos = alarme HI/HIHI ativo no DCS (onset→OK)
    lane = axes[1]
    spans = alarm_active_spans(temp.index.min(), temp.index.max())
    for a, b in spans:
        b_draw = max(b, a + pd.Timedelta(hours=24))   # largura mínima p/ 23 meses de eixo
        lane.axvspan(a, b_draw, ymin=0.15, ymax=0.85, color=MISS, alpha=0.9, lw=0)
    lane.set_ylim(0, 1)
    lane.set_yticks([])
    lane.set_ylabel("alarme\nDCS", fontsize=8, color=MISS, rotation=0,
                    ha="right", va="center", labelpad=12)
    for s in ("top", "right", "left"):
        lane.spines[s].set_visible(False)
    lane.spines["bottom"].set_color(GRID)
    lane.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    lane.text(0.999, 0.5, f"{len(spans)} eventos HI/HIHI (onset→OK)", transform=lane.transAxes,
              fontsize=7.5, color=INK_MUTED, ha="right", va="center")

    for ax, name in zip(axes[2:], ["b2024", "rerun"]):
        a = arms[name]
        shade_off(ax, on_p)
        blk = (a["alert"] != a["alert"].shift()).cumsum()
        for _, g in a["alert"].groupby(blk):
            if bool(g.iloc[0]):
                ax.axvspan(g.index[0], g.index[-1], color=ALERT, alpha=0.8, lw=0, zorder=1)
        ax.plot(a["h"].index, a["h"].values, lw=0.6, color=SERIES)
        ax.axhline(a["q"], color=THR, lw=1.1, ls="--")
        for t in a["hits"]:
            ax.axvline(t, color=HIT, lw=1.2, zorder=3)
        for t in a["misses"]:
            ax.axvline(t, color=MISS, lw=1.2, zorder=3)
        style(ax)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("health index", fontsize=9, color=INK)
        ax.text(0.005, 0.96, f"{a['label']}  —  raw {a['rr']:.1%} "
                f"({len(a['hits'])}/{len(a['inc'])})  ·  FA {a['fa']:.3f}/dia",
                transform=ax.transAxes, fontsize=9, color=INK, va="top",
                bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.25", alpha=0.9))

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[
        Patch(facecolor=MISS, alpha=0.9, label="alarme HI/HIHI ativo no DCS (pista própria)"),
        Line2D([], [], color=HIT, lw=1.4, label="incidente detectado (cruzamento bruto, 8h)"),
        Line2D([], [], color=MISS, lw=1.4, label="incidente perdido"),
        Line2D([], [], color=THR, lw=1.1, ls="--", label="threshold (1 por braço)"),
        Patch(facecolor=ALERT, label="alerta ativo (sticky 12h)"),
        Patch(facecolor=OFF_BAND, alpha=0.6, label="equipamento OFF"),
    ], loc="lower right", fontsize=7.5, framealpha=0.95, ncol=3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"Figura: {OUT}")


if __name__ == "__main__":
    main()
