#!/usr/bin/env python
"""Diagnóstico visual dos incidentes de teste perdidos.

Carrega os MAE raw (calibrate_halflife_out/mae_raw.npz) — sem retreinar —
e responde: o sinal sobe mas fica abaixo do threshold, ou não sobe?

Saídas em calibrate_halflife_out/:
  diag_1_gap_signal.png  — distribuição de max-saúde pré-incidente (pegos vs perdidos)
  diag_2_missed_grid.png — grade de janelas ±72h dos incidentes perdidos
  diag_3_per_sensor.png  — contribuição por sensor nos N piores casos

Execução (~10s, sem treino):
    PYTHONPATH=. python scripts/diag_missed_incidents.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma

TS          = 60
STRIDE      = 10
HALF_LIFE_H = 4.0
DT_SECONDS  = STRIDE * 30.0
TRAIN_FRAC  = 0.66
THRESH_Q    = 0.50      # ponto de operação do sweep (recall máximo dentro do budget)
HORIZON_H   = 8.0
WINDOW_H    = 72.0
OUT         = "calibrate_halflife_out"
NPZ_PATH    = f"{OUT}/mae_raw.npz"


def log(m): print(f"[DIAG] {m}", flush=True)


def health_or_normalized(per_sensor_health, thrs):
    """Max saúde normalizada pelo threshold de cada sensor. >= 1.0 = alerta."""
    stacked = np.stack(
        [per_sensor_health[s] / max(thrs[s], 1e-9) for s in SENSORS], axis=1
    )
    return stacked.max(axis=1), stacked  # or_norm, per_sensor_norm


def classify_incidents(or_norm, seq_run_full, t_end_sec, inc_s, horizon_h):
    """Pego / perdido_sem_sinal / perdido_sem_running — aplica seq_run_full."""
    H = horizon_h * 3600.0
    results = []
    for ti in inc_s:
        win     = (t_end_sec >= ti - H) & (t_end_sec <= ti)
        win_run = win & seq_run_full          # só sequências com equipamento on
        has_run = bool(win_run.any())
        peak    = float(or_norm[win_run].max()) if has_run else 0.0
        caught  = has_run and (peak >= 1.0)

        if caught:
            status = "pego"
        elif not has_run:
            status = "perdido_sem_running"    # equipamento off na janela
        else:
            status = "perdido_sem_sinal"      # rodando mas saúde abaixo do threshold

        results.append(dict(t=ti, peak=peak, caught=caught,
                            has_running=has_run, status=status))
    return results


# ---------------------------------------------------------------------------
# plot 1: distribuição de pico de sinal pré-incidente
# ---------------------------------------------------------------------------
def plot_gap_signal(results, path):
    pegos     = [r["peak"] for r in results if r["status"] == "pego"]
    sem_sinal = [r["peak"] for r in results if r["status"] == "perdido_sem_sinal"]
    sem_run   = [r["peak"] for r in results if r["status"] == "perdido_sem_running"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Incidentes de teste — threshold q={THRESH_Q}  H={int(HORIZON_H)}h",
                 fontsize=10)

    ax = axes[0]
    upper = max([max(pegos, default=0), max(sem_sinal, default=0), 2.0]) * 1.05
    bins  = np.linspace(0, upper, 35)
    ax.hist(pegos,     bins=bins, alpha=0.7, color="steelblue",
            label=f"Pegos (n={len(pegos)})")
    ax.hist(sem_sinal, bins=bins, alpha=0.7, color="tomato",
            label=f"Perdido — sem sinal (n={len(sem_sinal)})")
    ax.hist(sem_run,   bins=bins, alpha=0.7, color="gray",
            label=f"Perdido — off na janela (n={len(sem_run)})")
    ax.axvline(1.0, color="black", ls="--", lw=1.2, label="Threshold (=1.0 norm.)")
    ax.set_xlabel("Pico OR-normalizado em [t_inc − 8h, t_inc]  (só running)")
    ax.set_ylabel("Contagem")
    ax.set_title("Separação: pegos vs perdidos")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    cats = {"Pegos": (pegos, "steelblue"),
            "Perdido\nsem sinal": (sem_sinal, "tomato"),
            "Perdido\nsem running": (sem_run, "gray")}
    for xi, (label, (vals, col)) in enumerate(cats.items()):
        ax2.scatter(np.full(len(vals), xi), vals,
                    alpha=0.45, color=col, s=25)
    ax2.axhline(1.0, color="black", ls="--", lw=1.0)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Pegos", "Perdido\nsem sinal", "Perdido\nsem running"])
    ax2.set_ylabel("Pico OR-normalizado")
    ax2.set_title("Scatter por categoria")

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"salvo {path}")


# ---------------------------------------------------------------------------
# plot 2: grade ±72h dos incidentes perdidos
# ---------------------------------------------------------------------------
def plot_missed_grid(results, or_norm, seq_run_full, t_end_sec, t_end_pd, path,
                     window_h=72.0):
    missed = [r for r in results if not r["caught"]]
    if not missed:
        log("nenhum incidente perdido — pulando grade.")
        return

    W     = window_h * 3600.0
    ncols = 4
    nrows = int(np.ceil(len(missed) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, max(nrows * 2.8, 4)),
                             squeeze=False)
    fig.suptitle(
        f"Incidentes PERDIDOS no teste — janela ±{int(window_h)}h  "
        f"(threshold=1.0 norm.  |  vermelho=sem sinal  cinza=off)",
        fontsize=10, y=1.01,
    )

    for i, r in enumerate(missed):
        ax  = axes[i // ncols][i % ncols]
        ti  = r["t"]
        t_dt = pd.Timestamp(ti, unit="s", tz="UTC").tz_convert("America/Sao_Paulo")

        win = (t_end_sec >= ti - W) & (t_end_sec <= ti + W)
        if not win.any():
            ax.set_visible(False)
            continue

        t_plot = (pd.DatetimeIndex(t_end_pd[win])
                  .tz_localize("UTC").tz_convert("America/Sao_Paulo"))
        h_all  = or_norm[win]
        run_m  = seq_run_full[win]

        ax.plot(t_plot, h_all, lw=0.6, color="lightgray", zorder=1)
        if run_m.any():
            color = "tomato" if r["status"] == "perdido_sem_sinal" else "slategray"
            ax.plot(t_plot[run_m], h_all[run_m], lw=0.9, color=color, zorder=2)

        ax.axhline(1.0, color="black", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(t_dt, color="black", lw=1.5)
        t_pre = t_dt - pd.Timedelta(hours=HORIZON_H)
        color_span = "tomato" if r["status"] == "perdido_sem_sinal" else "gray"
        ax.axvspan(t_pre, t_dt, alpha=0.10, color=color_span)

        tag = "OFF" if not r["has_running"] else f"peak={r['peak']:.2f}"
        ax.set_title(f"#{i+1}  {t_dt.strftime('%d/%m %H:%M')}  {tag}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    for j in range(len(missed), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"salvo {path}")


# ---------------------------------------------------------------------------
# plot 3: contribuição por sensor nos N piores "perdido_sem_sinal"
# ---------------------------------------------------------------------------
def plot_per_sensor(results, per_sensor_norm, seq_run_full, t_end_sec, t_end_pd,
                    path, n_worst=6, window_h=48.0):
    missed = sorted(
        [r for r in results if r["status"] == "perdido_sem_sinal"],
        key=lambda r: r["peak"],
    )
    cases = missed[:n_worst]
    if not cases:
        log("sem casos 'perdido_sem_sinal' — pulando per-sensor plot.")
        return

    W = window_h * 3600.0
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(14, n * 2.8), squeeze=False)
    fig.suptitle(
        f"Contribuição por sensor — {n} piores 'perdido_sem_sinal' "
        f"(equipamento rodando mas health abaixo do threshold)",
        fontsize=10,
    )
    cmap   = plt.cm.tab20
    colors = [cmap(i / len(SENSORS)) for i in range(len(SENSORS))]

    for row, r in enumerate(cases):
        ax   = axes[row][0]
        ti   = r["t"]
        t_dt = pd.Timestamp(ti, unit="s", tz="UTC").tz_convert("America/Sao_Paulo")
        win  = (t_end_sec >= ti - W) & (t_end_sec <= ti + W)
        if not win.any():
            ax.set_visible(False)
            continue

        t_plot = (pd.DatetimeIndex(t_end_pd[win])
                  .tz_localize("UTC").tz_convert("America/Sao_Paulo"))

        for si, sensor in enumerate(SENSORS):
            ax.plot(t_plot, per_sensor_norm[win, si],
                    lw=0.7, alpha=0.55, color=colors[si], label=sensor)

        run_m = seq_run_full[win]
        ax.fill_between(t_plot, 0, 0.05, where=run_m,
                        alpha=0.3, color="green",
                        transform=ax.get_xaxis_transform())

        ax.axhline(1.0, color="black", ls="--", lw=1.0, label="Threshold")
        ax.axvline(t_dt, color="tomato", lw=1.5, label="Incidente")
        t_pre = t_dt - pd.Timedelta(hours=HORIZON_H)
        ax.axvspan(t_pre, t_dt, alpha=0.08, color="tomato")
        ax.set_title(
            f"{t_dt.strftime('%Y-%m-%d %H:%M')}  peak OR={r['peak']:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        if row == 0:
            ax.legend(fontsize=5, ncol=6, loc="upper right")

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"salvo {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)

    if not os.path.exists(NPZ_PATH):
        log(f"ERRO: {NPZ_PATH} não encontrado — rode calibrate_halflife.py primeiro.")
        sys.exit(1)

    log("carregando dados...")
    df, inc_full = load(priority=None)
    n            = len(df)
    n_train_pts  = int(TRAIN_FRAC * n)
    t_split      = df[TIME_COL].iloc[n_train_pts]

    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full     = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    incidents_test = inc_full[inc_full > t_split]
    log(f"incidentes de teste: {len(incidents_test)}")

    starts       = np.arange(0, n - TS + 1, STRIDE)
    ends         = starts + TS - 1
    t_end_pd     = df[TIME_COL].to_numpy()[ends]
    t_end_sec    = (pd.DatetimeIndex(t_end_pd)
                    .values.astype("datetime64[s]").astype("int64").astype(float))
    seq_run_frac = np.array([op[s:s + TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999
    is_train     = pd.DatetimeIndex(t_end_pd) <= t_split

    log(f"carregando MAE de {NPZ_PATH}  (hl={HALF_LIFE_H}h, q={THRESH_Q})...")
    npz = np.load(NPZ_PATH)
    per_sensor_health = {
        s: compute_health_index_ewma(npz[s], seq_run_frac, HALF_LIFE_H, DT_SECONDS)
        for s in SENSORS
    }

    thrs = {}
    for s, h in per_sensor_health.items():
        valid  = h[is_train & seq_run_full]
        thrs[s] = float(np.quantile(valid, THRESH_Q)) if len(valid) else 1e9

    or_norm, per_sensor_norm = health_or_normalized(per_sensor_health, thrs)

    inc_te_s = (pd.DatetimeIndex(incidents_test)
                .values.astype("datetime64[s]").astype("int64").astype(float))
    results  = classify_incidents(or_norm, seq_run_full, t_end_sec, inc_te_s, HORIZON_H)

    n_pego    = sum(r["status"] == "pego"               for r in results)
    n_ssinal  = sum(r["status"] == "perdido_sem_sinal"  for r in results)
    n_srun    = sum(r["status"] == "perdido_sem_running" for r in results)

    print(f"\n{'='*60}")
    print(f"  q={THRESH_Q}  hl={HALF_LIFE_H}h  H={int(HORIZON_H)}h  "
          f"total={len(results)}")
    print(f"  pegos={n_pego}  sem_sinal={n_ssinal}  sem_running={n_srun}"
          f"  recall={n_pego/len(results):.3f}")
    print(f"{'='*60}\n")

    print(f"{'#':>3}  {'data (BR)':>18}  {'peak':>6}  status")
    for i, r in enumerate(results):
        t_dt = pd.Timestamp(r["t"], unit="s", tz="UTC").tz_convert("America/Sao_Paulo")
        print(f"{i+1:>3}  {t_dt.strftime('%Y-%m-%d %H:%M'):>18}  "
              f"{r['peak']:>6.3f}  {r['status']}")

    log("\ngerando gráficos...")
    plot_gap_signal(results, f"{OUT}/diag_1_gap_signal.png")
    plot_missed_grid(results, or_norm, seq_run_full, t_end_sec, t_end_pd,
                     f"{OUT}/diag_2_missed_grid.png", window_h=WINDOW_H)
    plot_per_sensor(results, per_sensor_norm, seq_run_full, t_end_sec, t_end_pd,
                    f"{OUT}/diag_3_per_sensor.png", n_worst=6, window_h=48.0)
    log("concluído.")


if __name__ == "__main__":
    main()
