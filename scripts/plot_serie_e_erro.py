#!/usr/bin/env python3
"""
plot_serie_e_erro.py
Série temporal do sensor e a série de erro de reconstrução do autoencoder, no
mesmo eixo de tempo, mais o índice de saúde que transforma o erro em alerta.

Três painéis compartilhando o eixo x:

  1. temperatura do sensor (envelope min/max por bin, para 23 meses de dado a 30s
     não virarem uma mancha), fundo cinza onde a máquina está desligada;
  2. erro de reconstrução — MAE bruto por janela em cinza fino, e a EWMA por cima;
     é a EWMA que o detector usa, o bruto está ali só para mostrar o quanto ela
     suaviza;
  3. índice de saúde = rank percentual da EWMA sobre o período pontuado, com a
     linha de corte e os episódios de alerta pintados. Verde = o episódio pega um
     incidente dentro do horizonte; vermelho = falso positivo.

Marcas verticais = incidentes HI/HIHI com máquina ON (mesmo filtro do protocolo
de auditoria: fantasma <500°C descartado, eventos a menos de GAP_HOURS agrupados).
A faixa hachurada no topo é a janela de treino — tudo à esquerda do fim dela é
retrodição, não previsão.

Uso:
    PYTHONPATH=. python scripts/plot_serie_e_erro.py                  # excl_24h
    PYTHONPATH=. python scripts/plot_serie_e_erro.py --task <id> --hl 1.0
    PYTHONPATH=. python scripts/plot_serie_e_erro.py --zoom 2025-02-01 2025-03-01
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))

# paleta fixa, mesma dos outros plots desta análise
COR_TEMP = "#1f2933"
COR_OFF = "#e4e7eb"
COR_MAE_BRUTO = "#b8c2cc"
COR_EWMA = "#2a78d6"
COR_TP = "#1baf7a"
COR_FP = "#eb6834"
COR_INC = "#9b1d20"
COR_TREINO = "#4a3aa7"

DEFAULT_TASK = "0fdeb5318361420e904b7994a65e3593"   # v16_excl_24h
DEFAULT_HL = 0.5
DEFAULT_Q = 0.8564                                  # melhor ponto FULL do braço
TREINO = ("2024-06-01", "2025-07-01")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")


def _resolve_dados() -> str:
    for up in ("..", "../..", "../../.."):
        cand = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("diretório 'dados/' não encontrado.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--hl", type=float, default=DEFAULT_HL)
    p.add_argument("--q", type=float, default=DEFAULT_Q)
    p.add_argument("--label", default="v16_excl_24h")
    p.add_argument("--zoom", nargs=2, default=None, metavar=("INICIO", "FIM"))
    p.add_argument("--treino", nargs=2, default=TREINO, metavar=("INICIO", "FIM"),
                   help="janela de treino do braço, para a faixa no topo; o v17 usa "
                        "2025-07-01 2026-05-01, não o padrão do v13/v16")
    p.add_argument("--out", default=None)
    return p.parse_args()


def envelope(s: pd.Series, n_bins: int = 2200):
    """min/max por bin — preserva picos que uma reamostragem por média apagaria."""
    if s.empty:
        return s.index, s.values, s.values
    edges = pd.date_range(s.index[0], s.index[-1], periods=n_bins + 1)
    g = s.groupby(pd.cut(s.index, edges, labels=edges[:-1]), observed=False)
    lo, hi = g.min(), g.max()
    idx = pd.DatetimeIndex(edges[:-1])
    return idx, lo.values.astype(float), hi.values.astype(float)


def spans_off(running: pd.Series, t0, t1, min_minutes: float = 30.0):
    """(inicio, fim) dos trechos com máquina desligada, por run-length.

    Emparelhar `diff()==1` com `diff()==-1` erra quando a série começa desligada:
    o primeiro evento é OFF→ON e os pares saem trocados, com duração negativa.
    Aqui os blocos são construídos direto dos limites de mudança.
    """
    r = running.loc[(running.index >= t0) & (running.index <= t1)]
    if r.empty:
        return []
    off = (r <= 0.5).to_numpy()
    idx = r.index
    corte = np.flatnonzero(off[1:] != off[:-1]) + 1
    ini_i = np.concatenate(([0], corte))
    fim_i = np.concatenate((corte, [len(off)]))
    out = []
    for a, b in zip(ini_i, fim_i):
        if not off[a]:
            continue
        ta, tb = idx[a], idx[min(b, len(idx) - 1)]
        if (tb - ta) >= pd.Timedelta(minutes=min_minutes):
            out.append((ta, tb))
    return out


def quebra_vazios(s: pd.Series, max_gap_h: float = 2.0) -> pd.Series:
    """Insere NaN nos buracos para o plot não ligar pontos separados por dias."""
    if s.empty:
        return s
    dt = s.index.to_series().diff()
    buracos = s.index[dt > pd.Timedelta(hours=max_gap_h)]
    if len(buracos) == 0:
        return s
    extra = pd.Series(np.nan, index=buracos - pd.Timedelta(seconds=1))
    return pd.concat([s, extra]).sort_index()


def main() -> None:
    args = parse_args()
    S = args.sensor
    sw.SENSOR = S
    dados = _resolve_dados()
    sw.RAW_CSV = os.path.join(dados, "sensores_2024h2_2025_2026_30s.csv")
    sw.ALARM_CSV = os.path.join(dados, "alarmes_selecionados_turbina_a.csv")
    ev.ALARM_CSV_DEFAULT = sw.ALARM_CSV

    from clearml import Task

    running, _, t5 = sw.load_raw()
    raw = pd.read_csv(sw.RAW_CSV, usecols=["data_datetime", S], low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    serie = pd.to_numeric(
        raw.dropna(subset=["data_datetime"]).set_index("data_datetime")[S],
        errors="coerce").sort_index()
    t5s = t5.ewm(halflife=int(pd.Timedelta(hours=24) / pd.Timedelta("30s"))).mean()

    task = Task.get_task(task_id=args.task)
    mae = sw.read_mae(task.artifacts[f"{S}_csv_sequence_scores_all.csv"].get_local_copy())
    health = sw.health_global(mae, args.hl, running, t5s)
    ewma = sw.ewma_on(mae, args.hl, running)

    t0, t1 = mae.index.min(), mae.index.max()
    if args.zoom:
        t0 = pd.Timestamp(args.zoom[0], tz="UTC")
        t1 = pd.Timestamp(args.zoom[1], tz="UTC")

    inc = sw.incidents_on(running, serie, mae.index.min(), mae.index.max())
    inc_vis = [t for t in inc if t0 <= t <= t1]
    inc_s = np.array([t.timestamp() for t in inc])

    alert = ev.apply_sticky(health, args.q, sw.STICKY)
    eps = ev.detect_episodes_gap(alert)
    hs = sw.HORIZON * 3600.0
    eps_cls = [(a, b, bool(inc_s.size) and bool(
        np.any((inc_s - hs <= b.timestamp()) & (inc_s >= a.timestamp()))))
        for a, b in eps if b >= t0 and a <= t1]

    ser_v = serie.loc[(serie.index >= t0) & (serie.index <= t1)]
    mae_v = mae.loc[(mae.index >= t0) & (mae.index <= t1)]
    ewm_v = ewma.loc[(ewma.index >= t0) & (ewma.index <= t1)]
    hea_v = health.loc[(health.index >= t0) & (health.index <= t1)]

    nb = 2200 if not args.zoom else 900
    fig, axes = plt.subplots(3, 1, figsize=(16.5, 10.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0, 0.95],
                                          "hspace": 0.13})
    ax1, ax2, ax3 = axes

    for a, b in spans_off(running, t0, t1):
        for ax in axes:
            ax.axvspan(a, b, color=COR_OFF, lw=0, zorder=0)

    # 1 -------------------------------------------------- temperatura
    xi, lo, hi = envelope(ser_v, nb)
    ax1.fill_between(xi, lo, hi, color=COR_TEMP, alpha=0.55, lw=0, zorder=3)
    v_on = ser_v.where(running.reindex(ser_v.index, method="nearest") > 0.5).dropna()
    if len(v_on):
        ax1.set_ylim(495, np.nanpercentile(v_on, 99.97) + 8)
    ax1.set_ylabel("temperatura (°C)")
    ax1.set_title(f"{S} — série do sensor e erro de reconstrução  ·  {args.label}  ·  "
                  f"EWMA half-life {args.hl}h, corte no quantil {args.q}",
                  loc="left", fontsize=12.5, pad=11)

    # faixa da janela de treino
    tr0, tr1 = pd.Timestamp(args.treino[0], tz="UTC"), pd.Timestamp(args.treino[1], tz="UTC")
    if tr1 >= t0 and tr0 <= t1:
        ax1.axvspan(max(tr0, t0), min(tr1, t1), ymin=0.955, ymax=1.0,
                    color=COR_TREINO, alpha=0.75, lw=0, zorder=6)

    # 2 -------------------------------------------------- erro de reconstrução
    xi, lo, hi = envelope(mae_v, nb)
    ax2.fill_between(xi, lo, hi, color=COR_MAE_BRUTO, lw=0, zorder=2,
                     label="MAE bruto por janela (min–max)")
    ewm_q = quebra_vazios(ewm_v)
    ax2.plot(ewm_q.index, ewm_q.values, color=COR_EWMA, lw=0.9, zorder=4,
             label=f"EWMA hl={args.hl}h (o que o detector usa)")
    ax2.set_ylabel("erro de reconstrução (MAE)")
    ax2.set_yscale("log")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # 3 -------------------------------------------------- índice de saúde
    for a, b, tp in eps_cls:
        ax3.axvspan(max(a, t0), min(b, t1), color=COR_TP if tp else COR_FP,
                    alpha=0.30, lw=0, zorder=1)
    hea_q = quebra_vazios(hea_v)
    ax3.plot(hea_q.index, hea_q.values, color=COR_TEMP, lw=0.5, zorder=4)
    ax3.axhline(args.q, color=COR_INC, lw=1.1, ls="--", zorder=5)
    ax3.set_ylabel("índice de saúde\n(rank percentual)")
    ax3.set_ylim(0, 1.02)
    ax3.set_xlabel("tempo (UTC)")

    for t in inc_vis:
        for ax in axes:
            ax.axvline(t, color=COR_INC, lw=0.7, alpha=0.55, zorder=7)

    n_fp = sum(1 for *_, tp in eps_cls if not tp)
    n_tp = sum(1 for *_, tp in eps_cls if tp)
    handles = [Patch(fc=COR_TREINO, alpha=0.75, label="janela de treino"),
               Patch(fc=COR_OFF, label="máquina desligada"),
               plt.Line2D([], [], color=COR_INC, lw=1.0,
                          label=f"incidente HI/HIHI com máquina ON (n={len(inc_vis)})"),
               Patch(fc=COR_TP, alpha=0.30, label=f"episódio que pega incidente (n={n_tp})"),
               Patch(fc=COR_FP, alpha=0.30, label=f"falso positivo (n={n_fp})")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.008))

    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))
    for ax in axes:
        ax.grid(axis="y", color="#ffffff", lw=0.8, alpha=0.9, zorder=1)
        ax.set_axisbelow(False)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    fig.subplots_adjust(left=0.062, right=0.985, top=0.945, bottom=0.085)
    out = args.out or os.path.join(
        "eval_predictive_out",
        f"fig_serie_e_erro_{S}_{args.label}"
        + (f"_{args.zoom[0]}_{args.zoom[1]}" if args.zoom else "") + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"gravado: {out}")
    print(f"  incidentes visíveis={len(inc_vis)}  episódios TP={n_tp}  FP={n_fp}")


if __name__ == "__main__":
    main()
