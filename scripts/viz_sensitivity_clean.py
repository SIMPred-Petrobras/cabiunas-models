#!/usr/bin/env python
"""Versao limpa e ampliada do grid de distribuicoes: 3 colunas x 6 linhas."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL

TS = 60; STRIDE = 10; Q_OP = 0.715
OUT_DIR = "relatorio_anexos"
MAE_CACHE = "improve_halflife_out/mae_per_sensor_cache.npz"


def main():
    df, _ = load(priority=None)
    n = len(df)
    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    starts = np.arange(0, n - TS + 1, STRIDE)
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999

    print("[VIZ] carregando MAE cache")
    mae = np.load(MAE_CACHE)["mae"]

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 6 linhas x 3 colunas = 18 slots para 17 sensores
    fig, axes = plt.subplots(6, 3, figsize=(18, 24))
    axes_flat = axes.flatten()

    # ordena: temperatura primeiro (T5+TC), depois vibração
    sensor_order = ["T5_AVG_A"] + [s for s in SENSORS if s.startswith("TC382")] + \
                   [s for s in SENSORS if s.startswith("TV_")]
    assert len(sensor_order) == 17

    for j, sensor in enumerate(sensor_order):
        ax = axes_flat[j]
        idx = SENSORS.index(sensor)
        mae_run = mae[seq_run_full, idx]
        thr = float(np.quantile(mae_run, Q_OP))
        med = float(np.median(mae_run))
        p99 = float(np.quantile(mae_run, 0.99))
        # histograma em escala log y
        ax.hist(mae_run, bins=100, color="steelblue", alpha=0.75, edgecolor="navy", linewidth=0.3)
        ax.axvline(thr, color="red", ls="--", lw=1.8, label=f"thr q={Q_OP} = {thr:.4f}")
        ax.axvline(med, color="green", ls=":", lw=1.2, label=f"mediana = {med:.4f}")
        ax.set_title(f"{sensor}\n(media={mae_run.mean():.4f}, p99={p99:.4f})",
                     fontsize=12)
        ax.set_yscale("log")
        ax.set_xlabel("MAE/sequencia", fontsize=10)
        ax.set_ylabel("frequencia (log)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        # ajusta xlim para xpercentil 99.5 para nao espremer
        ax.set_xlim(0, np.quantile(mae_run, 0.995))
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.3)

    # ultimo subplot vazio
    axes_flat[17].set_visible(False)

    fig.suptitle(
        "Distribuição do erro de reconstrução (MAE) por sensor — período em operação\n"
        "linha VERMELHA = threshold individual (q=0.715, define alerta) | "
        "linha VERDE = mediana | escala Y em log",
        fontsize=15, y=0.998,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = f"{OUT_DIR}/fig_mae_distribution_grid_HD.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] salvo {out}")


if __name__ == "__main__":
    main()
