#!/usr/bin/env python3
"""Figuras do relatorio de apresentacao.

fig_apresentacao_serie.png  serie completa, antes/depois do limite de
                            permanencia, com os 8 trips marcados
fig_apresentacao_zoom.png   quatro eventos em detalhe, mostrando os sinais
                            subindo antes da queda da maquina
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from auto_reset import trunca

K_BASE, K_VIB, LIM = 1.3, 2.2, 12
AZUL, LARANJA, VERDE, VERM = "#2E5E6E", "#C4703A", "#1F7A5E", "#9E2B2B"


def main():
    df = canonico()
    ftodas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    falhas = ftodas[ftodas >= "2025-01-01"].reset_index(drop=True)
    mask = mascara_pontuacao(df); idx = mask.index
    cal = (idx[-1] - idx[0]).total_seconds() / 3600 / 730
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in ftodas]

    out = roda(BRACO, df, ftodas)
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    E = {"t": ew("t", "1h"), "p": ew("p", "1h"),
         "sp": ew("sp", "30min"), "vb": ew("vb", "30min")}
    S = {"t": DET._sustained(E["t"], DET.THR_FAM * K_BASE),
         "p": DET._sustained(E["p"], DET.THR_FAM * K_BASE),
         "sp": DET._sustained(E["sp"], DET.THR_SPREAD * K_BASE),
         "vb": DET._sustained(E["vb"], 3.0 * K_VIB)}
    n = sum(s.astype(int) for s in S.values())
    al_sem = (n >= 2) & mask
    al_com = trunca(al_sem, LIM)

    def fps(al):
        eps = A.episodios(al)
        return [(a, b) for a, b in eps
                if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]

    # ---------------------------------------------------------- figura 1
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.6), sharex=True)
    for ax, al, titulo in [
        (axes[0], al_sem, "Detector atual — sem limite de permanência"),
        (axes[1], al_com, f"Com limite de {LIM} h — alarme 72% mais curto; um trip sai da janela de 48 h "
         f"(o alerta disparou, mas 91 h antes)"),
    ]:
        fp = fps(al)
        h = sum((b - a).total_seconds()/3600 + 2/60 for a, b in fp)
        ax.fill_between(n.index, 0, n.where(al, 0).to_numpy(), step="mid",
                        color=AZUL, lw=0, alpha=.9)
        ax.fill_between(n.index, 0, n.to_numpy(), step="mid", color=AZUL, lw=0, alpha=.18)
        for a, b in fp:
            ax.axvspan(a, b, color=VERM, alpha=.22, lw=0)
        for ev in falhas:
            pego = al[(al.index >= ev - pd.Timedelta(hours=48)) & (al.index < ev)].any()
            ax.axvline(ev, color=VERDE if pego else VERM, lw=1.5, ls="--", alpha=.9)
        ax.set_ylim(0, 4.4); ax.set_yticks([0, 2, 4])
        ax.set_ylabel("sinais\nsimultâneos", fontsize=9)
        ax.set_title(f"{titulo}   —   {len(fp)/cal:.1f} alarmes falsos/mês, "
                     f"{h/cal:.0f} h/mês em alarme", fontsize=10.5, loc="left")
        ax.grid(axis="y", alpha=.2, lw=.6)

    axes[0].plot([], [], color=VERM, alpha=.22, lw=8, label="episódio de alarme falso")
    axes[0].plot([], [], color=VERDE, ls="--", lw=1.5, label="trip detectado")
    axes[0].plot([], [], color=VERM, ls="--", lw=1.5, label="trip não detectado")
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=3, framealpha=.92)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    axes[1].set_xlim(idx[0], idx[-1])
    fig.autofmt_xdate()
    fig.suptitle("TC-330.03A — detecção de trip, 2024-01 a 2026-04", fontsize=12.5, y=.995)
    fig.tight_layout()
    fig.savefig("fig_apresentacao_serie.png", dpi=145)
    plt.close(fig)
    print("fig_apresentacao_serie.png")

    # ---------------------------------------------------------- figura 2
    casos = [pd.Timestamp("2025-04-07 21:18", tz="UTC"),
             pd.Timestamp("2025-11-04 06:22", tz="UTC"),
             pd.Timestamp("2025-12-09 08:36", tz="UTC"),
             pd.Timestamp("2026-02-26 15:34", tz="UTC")]
    rot = {"t": "temperatura", "p": "pressão", "sp": "mancal", "vb": "vibração"}
    cor = {"t": "#B5651D", "p": "#2E5E6E", "sp": "#1F7A5E", "vb": "#7A3E8F"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for ax, ev in zip(axes.ravel(), casos):
        j0, j1 = ev - pd.Timedelta(hours=72), ev + pd.Timedelta(hours=4)
        w = (idx >= j0) & (idx < j1)
        for c in ["t", "p", "sp", "vb"]:
            lim = (DET.THR_FAM if c in ("t", "p") else DET.THR_SPREAD) * K_BASE \
                  if c != "vb" else 3.0 * K_VIB
            ax.plot(idx[w], (E[c][w] / lim).to_numpy(), lw=1.15, color=cor[c],
                    label=rot[c], alpha=.9)
        ax.axhline(1.0, color="#444", lw=1, ls=":")
        ax.axvline(ev, color=VERM, lw=1.6, ls="--")
        alw = al_com[w]
        ax.fill_between(idx[w], 0, 3.2, where=alw.to_numpy(), color=LARANJA,
                        alpha=.16, lw=0, step="mid")
        lead = None
        for a, b in A.episodios(al_com):
            if a < ev and (ev - b).total_seconds()/3600 <= 2:
                lead = (ev - a).total_seconds()/3600
        ax.set_ylim(0, 3.2)
        ax.set_title(f"{ev:%d/%m/%Y}" + (f"  —  alerta {lead:.1f} h antes"
                     if lead else "  —  sem alerta ativo na queda"),
                     fontsize=10, loc="left")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
        ax.tick_params(axis="x", labelsize=7.5, rotation=30)
        ax.grid(alpha=.2, lw=.6)
        ax.set_ylabel("escore ÷ limiar", fontsize=8.5)
    axes[0, 0].legend(fontsize=8, ncol=2, loc="upper left", framealpha=.92)
    fig.suptitle("Os quatro sinais nas 72 h antes de cada trip  "
                 "(linha pontilhada = limiar; faixa = alarme confirmado)",
                 fontsize=11.5, y=.995)
    fig.tight_layout()
    fig.savefig("fig_apresentacao_zoom.png", dpi=145)
    plt.close(fig)
    print("fig_apresentacao_zoom.png")


main()
