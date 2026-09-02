#!/usr/bin/env python3
"""Plota os falsos positivos do ponto de operacao na serie temporal.

Duas figuras:
  1. panorama -- os 4 sinais (score / limiar) nos 16 meses inteiros, com os episodios
     de FALSO POSITIVO em vermelho, as DETECCOES (8/8) em verde, e os eventos reais
     como linhas verticais tracejadas. Da para ver de longe onde os FP se concentram
     (a borda do blackout, ver [[borda-do-blackout-explica-os-fp]]).
  2. mosaico de zoom -- um painel por episodio de FP, +-24 h de contexto, com os 4
     sinais sobrepostos e o voto (n sinais simultaneos) embaixo. E o que autopsia_fp.py
     relata em texto, aqui em figura.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo, EW
from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
INK, INK2, MUTED = "#131a20", "#3d4a55", "#6b7885"
RULE, GROUND = "#dde2e6", "#f4f6f7"
ACCENT, AMP, CTX = "#0f6e78", "#b8792a", "#2b6ca3"
GOOD, CRIT = "#2e7d4f", "#b03a2e"
SANS = "DejaVu Sans"
NOMES = {"t": "temperatura", "p": "pressão", "sp": "spread mancal", "vb": "vibração"}
CORES = {"t": "#c0392b", "p": "#2b6ca3", "sp": "#8e5b3f", "vb": "#6a4c93"}


def monta():
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    R = {c: (EW[c].where(mask) / (BASE[c] * K[c])) for c in SIN}
    return al, ns, R


def classifica(al, alvo_ts, jan_h=48.0):
    eps = AV.episodios(al)
    jw = [(t - pd.Timedelta(hours=jan_h), t) for t in alvo_ts]
    fps, tps = [], []
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw):
            tps.append((a, b))
        else:
            fps.append((a, b))
    return fps, tps


def formata_eixo(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.tick_params(labelsize=7.5, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)


def panorama(al, ns, R, fps, tps, saida):
    fig, axes = plt.subplots(len(SIN) + 1, 1, sharex=True, figsize=(15, 10.5),
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.6]},
                             facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        for a, b in fps:
            ax.axvspan(a, b, color=CRIT, alpha=0.18, lw=0)
        for a, b in tps:
            ax.axvspan(a, b, color=GOOD, alpha=0.16, lw=0)
        for t in alvo:
            ax.axvline(t, color=AMP, ls="--", lw=1.0, alpha=0.85, zorder=5)

    for i, c in enumerate(SIN):
        ax = axes[i]
        s = R[c].reindex(idx)
        ax.plot(idx, s.to_numpy(), color=CORES[c], lw=0.6, alpha=0.9)
        ax.axhline(1.0, color=MUTED, ls=":", lw=0.9)
        ax.set_ylabel(f"{NOMES[c]}\nscore/limiar", fontsize=8, color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(0.05, max(20, np.nanpercentile(s.dropna(), 99.9) * 1.3))
        formata_eixo(ax)

    axv = axes[-1]
    axv.plot(idx, ns.reindex(idx).to_numpy(), color=INK, lw=0.7, drawstyle="steps-post")
    axv.axhline(2, color=CRIT, ls="--", lw=1.0)
    axv.set_ylabel("nº sinais\nsimultâneos", fontsize=8, color=INK2)
    axv.set_ylim(-0.3, 4.3)
    axv.set_yticks([0, 1, 2, 3, 4])
    formata_eixo(axv)
    axv.set_xlabel("")

    h = [plt.Line2D([0], [0], color=AMP, ls="--", lw=1.4, label="falha real"),
        plt.Rectangle((0, 0), 1, 1, color=GOOD, alpha=0.35, label=f"detecção ({len(tps)})"),
        plt.Rectangle((0, 0), 1, 1, color=CRIT, alpha=0.35, label=f"falso positivo ({len(fps)})")]
    axes[0].legend(handles=h, loc="upper left", fontsize=8.5, ncol=3, frameon=True,
                   framealpha=0.9, edgecolor=RULE)
    fig.suptitle("TC-330.03A — panorama dos 16 meses: onde nascem os falsos positivos",
                fontsize=13.5, fontweight="bold", color=INK, y=0.995)
    fig.text(0.5, 0.005,
             "score ÷ limiar por sinal (escala log) · faixa vermelha = episódio de falso "
             "positivo · faixa verde = detecção · linha laranja = falha real",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(saida, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"-> {saida}")


def mosaico(al, ns, R, fps, saida, margem_h=24.0):
    n = len(fps)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), facecolor="white")
    axes = np.atleast_2d(axes)
    mg = pd.Timedelta(hours=margem_h)

    for i, (a, b) in enumerate(fps):
        r, cidx = divmod(i, ncol)
        ax = axes[r, cidx]
        ax.set_facecolor("white")
        w0, w1 = a - mg, b + mg
        for c in SIN:
            s = R[c].loc[w0:w1]
            ax.plot(s.index, s.to_numpy(), color=CORES[c], lw=1.1, label=NOMES[c])
        ax.axhline(1.0, color=MUTED, ls=":", lw=0.9)
        ax.axvspan(a, b, color=CRIT, alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(0.1, None)
        dur = (b - a).total_seconds() / 3600
        ax.set_title(f"{a:%d/%m/%Y %H:%M}  ·  {dur:.1f} h", fontsize=8.5, color=INK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.tick_params(labelsize=6.5, colors=INK2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6.5, loc="upper left", ncol=2, frameon=False)

    for j in range(n, nrow * ncol):
        r, cidx = divmod(j, ncol)
        axes[r, cidx].axis("off")

    fig.suptitle(f"TC-330.03A — os {n} falsos positivos, um a um (±{margem_h:.0f} h de contexto)",
                fontsize=13.5, fontweight="bold", color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(saida, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"-> {saida}")


if __name__ == "__main__":
    al, ns, R = monta()
    m = AV.avalia(al, alvo, mask)
    fps, tps = classifica(al, alvo)
    print(f"controle: {m['det']}/8, {m['episodios']} episódios, {len(fps)} FP, "
          f"{len(tps)} TP  (esperado 8/8, 20, 12, 8)")

    panorama(al, ns, R, fps, tps, "fig_panorama_fp.png")
    mosaico(al, ns, R, fps, "fig_mosaico_fp.png")
