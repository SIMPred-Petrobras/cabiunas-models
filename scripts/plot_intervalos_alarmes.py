#!/usr/bin/env python3
"""
plot_intervalos_alarmes.py
Intervalo entre mudanças de alarme do sensor ao longo da série temporal, com
TODAS as condições registradas no CSV — quanto tempo passa entre um registro
de alarme e o próximo.

Motivação: a inter-chegada é o que define se dois alarmes viram UM incidente
(GAP_HOURS=4) e se a janela de avaliação de um evento contém o evento anterior
(horizonte 8h). A distribuição é de RAJADA, não uniforme.

Painel de cima   — tira de eventos: cada registro como um traço, uma faixa por condição.
Painel principal — intervalo desde o registro anterior (qualquer condição), escala log,
                   com as linhas de referência dos parâmetros do pipeline.
Painel da direita — a mesma distribuição em bins log, com as medianas.

Cor: 4 slots categóricos validados (validate_palette.js, light, --pairs all —
CVD ΔE 9.2, normal 16.3) com atribuição FIXA por condição, para a cor seguir a
entidade e não a frequência. OK entra em neutro porque não é alarme, é a
normalização; condições fora do mapa caem em "outros", também neutro.

Uso:
    PYTHONPATH=. python scripts/plot_intervalos_alarmes.py
    PYTHONPATH=. python scripts/plot_intervalos_alarmes.py --sensor T5_AVG_A
    PYTHONPATH=. python scripts/plot_intervalos_alarmes.py --sem-ok   # só alarmes
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

_HERE = os.path.dirname(os.path.abspath(__file__))

# Atribuição FIXA condição -> cor (a cor segue a entidade, nunca o ranking).
# Slots 1-4 da paleta categórica; validados all-pairs em conjunto.
COR = {"HI": "#2a78d6", "HIHI": "#eb6834", "UNDER": "#1baf7a", "OVER": "#4a3aa7"}
COR_OK, COR_OUTROS = "#a9a7a0", "#77756e"   # neutros: OK não é alarme
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#dedcd6"
SUP = "#fcfcfb"

# Ordem de desenho/leitura: mais severo em cima
ORDEM = ["HIHI", "HI", "OVER", "UNDER", "OUTROS", "OK"]

REFS = [(4.0, "GAP_HOURS = 4 h\n(agrupa em 1 incidente)"),
        (8.0, "horizonte = 8 h"),
        (12.0, "STICKY = 12 h"),
        (72.0, "exclusão pré-alarme = 72 h")]

TREINO = (pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"))


def _resolve_dados() -> str:
    for up in ("..", "../..", "../../.."):
        cand = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(cand):
            return cand
    raise SystemExit("diretório 'dados/' não encontrado a partir do repo.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--sem-ok", action="store_true",
                   help="ignora os registros OK (só transições de alarme)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def carrega(sensor: str) -> pd.DataFrame:
    al = os.path.join(_resolve_dados(), "alarmes_selecionados_turbina_a.csv")
    df = pd.read_csv(al)
    df["t"] = pd.to_datetime(df["Data da Ocorrência"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"])
    df = df[df["Tag Alarme"] == sensor].copy()
    df["c"] = df["Condição do Alarme"].str.upper().str.strip()
    df["grp"] = np.where(df["c"].isin(list(COR) + ["OK"]), df["c"], "OUTROS")
    return df.sort_values("t").reset_index(drop=True)


def cor_de(g: str) -> str:
    return COR_OK if g == "OK" else COR.get(g, COR_OUTROS)


def main() -> None:
    args = parse_args()
    sensor = args.sensor
    suf = "_sem_ok" if args.sem_ok else ""
    out_png = args.out or f"eval_predictive_out/fig_intervalos_alarmes_{sensor}{suf}.png"

    df_all = carrega(sensor)
    df = df_all[df_all.grp != "OK"].copy() if args.sem_ok else df_all
    presentes = [g for g in ORDEM if (df_all.grp == g).any()]

    t = df["t"].to_numpy()
    grp = df["grp"].to_numpy()
    gap = np.array([np.nan] + [(t[i] - t[i - 1]) / np.timedelta64(1, "h")
                               for i in range(1, len(t))])
    m = ~np.isnan(gap)
    tg, gg, lg = t[m], gap[m], grp[m]

    alarme = lg != "OK"
    med_all = float(np.median(gg))
    med_al = float(np.median(gg[alarme])) if alarme.any() else float("nan")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRADE,
                         "text.color": TINTA, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2,
                         "figure.facecolor": SUP, "axes.facecolor": SUP})
    fig = plt.figure(figsize=(14.4, 8.2))
    gs = GridSpec(2, 2, height_ratios=[1.25, 3.4], width_ratios=[5.2, 1],
                  hspace=0.10, wspace=0.03,
                  left=0.075, right=0.755, top=0.825, bottom=0.08)
    ax_r = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_h = fig.add_subplot(gs[1, 1], sharey=ax)

    # ---------- tira de eventos ----------
    for i, g in enumerate(presentes):
        y = len(presentes) - 1 - i
        sel = df_all[df_all.grp == g]["t"]
        c = cor_de(g)
        ax_r.vlines(sel, y - 0.34, y + 0.34, color=c, lw=0.9, alpha=0.9)
        # rótulo direto (regra de relevo: aqua fica em 2.74:1)
        ax_r.text(-0.011, y, g, transform=ax_r.get_yaxis_transform(),
                  ha="right", va="center", color=c, fontsize=9, fontweight="bold")
        ax_r.text(1.008, y, f"{len(sel)}", transform=ax_r.get_yaxis_transform(),
                  ha="left", va="center", color=TINTA2, fontsize=8.5)
    ax_r.set_ylim(-0.7, len(presentes) - 0.3)
    ax_r.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_r.spines[s].set_visible(False)
    plt.setp(ax_r.get_xticklabels(), visible=False)
    ax_r.tick_params(axis="x", length=0)
    ax_r.set_title("cada traço = um registro de alarme", loc="left",
                   fontsize=8.5, color=TINTA2, pad=4)

    # ---------- painel principal ----------
    ax.axvspan(TREINO[0], TREINO[1], color="#2a78d6", alpha=0.045, lw=0, zorder=0)
    ax.text(TREINO[0] + (TREINO[1] - TREINO[0]) / 2, 9500, "janela de treino",
            ha="center", va="top", fontsize=8, color=TINTA2, zorder=2)
    for h, _ in REFS:
        ax.axhline(h, color=GRADE, lw=1.0, ls=(0, (5, 4)), zorder=1)
        ax_h.axhline(h, color=GRADE, lw=1.0, ls=(0, (5, 4)), zorder=1)

    ax.vlines(tg, 0.003, gg, color=GRADE, lw=0.55, zorder=2)
    # OK recessivo: é a normalização, não o evento — fica atrás e menor
    for g in [x for x in reversed(presentes) if x in ("OK", "OUTROS")] + \
             [x for x in reversed(presentes) if x not in ("OK", "OUTROS")]:
        s = lg == g
        if not s.any():
            continue
        rec = g in ("OK", "OUTROS")
        ax.scatter(tg[s], gg[s], s=13 if rec else 27, color=cor_de(g),
                   zorder=3 if rec else 5, alpha=0.75 if rec else 1.0,
                   edgecolors=SUP, linewidths=0.5 if rec else 0.9,
                   label=f"{g} ({int(s.sum())})")

    ax.set_yscale("log")
    ax.set_ylim(0.003, 20000)
    ax.set_ylabel("intervalo desde o registro anterior  (horas, escala log)")
    ax.set_yticks([0.0167, 0.1, 1, 4, 12, 24, 72, 168, 720, 8760])
    ax.set_yticklabels(["1 min", "6 min", "1 h", "4 h", "12 h", "1 dia",
                        "3 dias", "1 sem", "1 mês", "1 ano"])
    ax.grid(axis="y", color=GRADE, lw=0.6, alpha=0.45, zorder=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2,
                    handletextpad=0.4, borderaxespad=0.5, columnspacing=1.2,
                    title="condição do registro que chegou  (n)")
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(TINTA2)

    # ---------- distribuição marginal ----------
    bins = np.logspace(np.log10(0.003), np.log10(20000), 46)
    ax_h.hist(gg, bins=bins, orientation="horizontal",
              color="#d5d3cc", edgecolor=SUP, linewidth=0.5, label="todos")
    ax_h.hist(gg[alarme], bins=bins, orientation="horizontal",
              color="#9fa6b0", edgecolor=SUP, linewidth=0.5, label="só alarmes")
    for val, cor, lbl in ((med_all, TINTA2, f"mediana geral\n{val_fmt(med_all)}"),
                          (med_al, COR["HIHI"], f"mediana alarmes\n{val_fmt(med_al)}")):
        if np.isfinite(val):
            ax_h.axhline(val, color=cor, lw=1.6, zorder=6)
            ax_h.text(0.95, val * 1.28, lbl, transform=ax_h.get_yaxis_transform(),
                      ha="right", va="bottom", fontsize=7.6, color=cor,
                      fontweight="bold", zorder=7)
    for h, txt in REFS:
        ax_h.text(1.16, h, txt, transform=ax_h.get_yaxis_transform(),
                  ha="left", va="center", fontsize=8, color=TINTA2, zorder=6)
    ax_h.set_xlabel("registros", fontsize=8, color=TINTA2)
    ax_h.set_xticks([])
    plt.setp(ax_h.get_yticklabels(), visible=False)
    ax_h.tick_params(axis="y", length=0)
    for s in ("top", "right", "bottom"):
        ax_h.spines[s].set_visible(False)
    ax_h.grid(axis="y", color=GRADE, lw=0.6, alpha=0.45, zorder=0)

    # ---------- títulos ----------
    n4 = int((gg[alarme] < 4).sum())
    n8 = int((gg[alarme] < 8).sum())
    na = int(alarme.sum())
    fig.text(0.075, 0.955, f"Intervalo entre alarmes — {sensor}",
             fontsize=15, fontweight="bold", color=TINTA)
    fig.text(0.075, 0.918,
             f"{len(df_all)} registros, {len(presentes)} condições, 2022–2026. "
             f"Mediana de {val_fmt(med_all)} entre registros quaisquer e "
             f"{val_fmt(med_al)} entre alarmes.",
             fontsize=9.5, color=TINTA2)
    fig.text(0.075, 0.889,
             f"Chegada em rajada: {n4} pares de alarme ({n4/max(na,1):.0%}) ficam abaixo do "
             f"GAP_HOURS de 4 h e {n8} ({n8/max(na,1):.0%}) abaixo do horizonte de 8 h.",
             fontsize=9.5, color=TINTA2)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=170, facecolor=SUP)
    print(f"Gravado: {out_png}")
    print(f"  registros={len(df_all)}  condicoes={presentes}")
    print(f"  mediana_geral={med_all:.2f} h  mediana_alarmes={med_al:.2f} h  "
          f"<4h={n4}  <8h={n8}")


def val_fmt(h: float) -> str:
    if not np.isfinite(h):
        return "—"
    if h < 1:
        return f"{h*60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h/24:.1f} dias"


if __name__ == "__main__":
    main()
