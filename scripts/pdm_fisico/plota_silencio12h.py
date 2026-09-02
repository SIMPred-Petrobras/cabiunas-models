#!/usr/bin/env python3
"""Plota o ponto de silencio-12h (kb=1.5, kv=2.2) no mesmo estilo do
fig_nosso_estilo_francisco.png, para a leitura de menor-FP-possivel.

Achado em _tmp_silencio24_regrac.py: alongar o blackout pos-partida de 6h para
12h, SEM mexer no limiar (kb=1.7, kv=2.2, o mesmo do ponto padrao), da 5/8 a
0,178 FP/mes e 1,00 h/mes -- pior recall que o ponto padrao (8/8 a 0,517/7,15),
mas MELHOR em ambas as reguas de custo do que os tres finalistas do
Francisco/Lara no mesmo recall (5/8). Ver conversa que fechou nisso.

NAO CONFUNDIR com o ponto revarrido (kb=1.5) do mesmo silencio, que da um
resultado DIFERENTE (6/8, 0,535 FP/mes, 1,99 h/mes) -- esse plot e do ponto
fixo, que foi o citado ao usuario.

Classificacao usa a REGRA C (Francisco), igual ao plota_estilo_francisco.py.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import avalia as AV
from publica_clearml import GRID, SIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN, T0
from blackout_curto import cusum

BLACKOUT_H = "12h"
KB, KV = 1.7, 2.2
INK, INK2, MUTED = "#131a20", "#3d4a55", "#6b7885"
RULE, GROUND = "#dde2e6", "#f4f6f7"
AMP, TP_COR, FP_COR, NEU_COR, LINHA = "#b8792a", "#c0392b", "#131a20", "#d68a1f", "#2b6ca3"
STOP_COR = "#c9cfd4"

# --- reconstroi mascara e sinais com o blackout de 12h (pos_processamento.py fixa 6h) ---
g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
sel = idx >= T0
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))

n_bl = int(pd.Timedelta(BLACKOUT_H) / pd.Timedelta(GRID))
blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
mask = (estavel & ~blk) & sel
reset = ((~mask) | part).to_numpy()

z = np.load("piso_fisico_cache.npz")
spv = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Zv = np.abs((z["Xh"] - z["MED"]) / z["S"])
vbv = np.full(len(idx), np.nan)
vbv[z["hot"]] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
vbv[~np.isfinite(vbv)] = np.nan
cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vbv}, index=idx)
from publica_clearml import HL, BASE
EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}


def pos(voto, refrat_h, dur_min):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=refrat_h)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= dur_min:
            fin.loc[a:b] = True
    return fin & sel


def alarme():
    K_ = {"t": KB, "p": KB, "sp": KB, "vb": KV}
    ON = {}
    for c in SIN:
        thr = BASE[c] * K_[c]
        E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    return pos(v, REFRAT_H, DUR_MIN)


def paradas_reais_2h():
    g_id = (op != op.shift()).cumsum()
    runs = pd.DataFrame({"tempo": idx, "op": op.values, "grupo": g_id.values})
    resumo = runs.groupby("grupo").agg(ini=("tempo", "first"), fim=("tempo", "last"), op=("op", "first"))
    resumo["dur_h"] = (resumo.fim - resumo.ini).dt.total_seconds() / 3600 + (2 / 60)
    return resumo[(resumo.op == False) & (resumo.dur_h >= 2.0)]


def classifica_regra_c(eps, paradas):
    JAN = pd.Timedelta(hours=48)
    jw = [(t - JAN, t) for t in alvo]
    out = []
    for a, b in eps:
        tp_alvo = [t for t, (t0_, t1_) in zip(alvo, jw) if a <= t1_ and b >= t0_]
        if tp_alvo:
            lead_h = (tp_alvo[0] - a).total_seconds() / 3600
            out.append((a, b, "TP", lead_h))
            continue
        cand = paradas[(paradas.ini >= a) & (paradas.ini <= b + JAN)]
        out.append((a, b, "NEUTRO", None) if len(cand) else (a, b, "FP", None))
    return out


if __name__ == "__main__":
    al = alarme()
    eps = AV.episodios(al)
    paradas = paradas_reais_2h()
    classif = classifica_regra_c(eps, paradas)

    n_tp = sum(1 for a, b, c, l in classif if c == "TP")
    n_fp = sum(1 for a, b, c, l in classif if c == "FP")
    n_ne = sum(1 for a, b, c, l in classif if c == "NEUTRO")
    m = AV.avalia(al, alvo, mask)
    meses = m["horas_op"] / 730.0
    h_fp = sum((b - a).total_seconds() / 3600 for a, b, c, l in classif if c == "FP")
    print(f"controle: TP={n_tp} FP={n_fp} NEUTRO={n_ne}  FP/mes={n_fp/meses:.3f}  "
          f"h/mes={h_fp/meses:.2f}  lead_med={m['lead_med']:.1f}h  (silencio={BLACKOUT_H}, kb={KB}, kv={KV})")

    t5 = g.loc[g.index >= T0, "T5_AVG_A"].resample("1D").median()
    parada_diaria = (~op.loc[op.index >= T0]).resample("1D").mean() > 0.5

    fig, ax = plt.subplots(figsize=(15, 5.0), facecolor="white")
    ax.set_facecolor("white")

    dias = parada_diaria.index
    em_parada = False
    ini_b = None
    for i, v_ in enumerate(parada_diaria.to_numpy()):
        if v_ and not em_parada:
            ini_b = dias[i]; em_parada = True
        elif not v_ and em_parada:
            ax.axvspan(ini_b, dias[i], color=STOP_COR, alpha=0.55, lw=0)
            em_parada = False
    if em_parada:
        ax.axvspan(ini_b, dias[-1], color=STOP_COR, alpha=0.55, lw=0)

    ax.plot(t5.index, t5.to_numpy(), color=LINHA, lw=1.0, zorder=3)

    for i, t in enumerate(alvo):
        ax.axvline(t, color=AMP, ls="--", lw=1.1, alpha=0.9, zorder=4)

    cores = {"TP": TP_COR, "FP": FP_COR, "NEUTRO": NEU_COR}
    contagem_dias = {"TP": 0, "FP": 0, "NEUTRO": 0}
    for a, b, c, lead_h in classif:
        dias_ep = pd.date_range(a.floor("D"), b.floor("D"), freq="1D")
        for d in dias_ep:
            if d not in t5.index or pd.isna(t5.loc[d]):
                continue
            contagem_dias[c] += 1
            ax.scatter([d], [t5.loc[d]], color=cores[c], s=42, zorder=6,
                      edgecolor="white", linewidth=0.6)

    h_leg = [
        plt.Rectangle((0, 0), 1, 1, color=STOP_COR, alpha=0.55, label="Máquina parada"),
        plt.Line2D([0], [0], color=LINHA, lw=1.2, label="T5_AVG_A (mediana diária)"),
        plt.Line2D([0], [0], color=AMP, ls="--", lw=1.2, label="falha real"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TP_COR, markersize=7,
                   label=f"antecipou falha ({contagem_dias['TP']} dias)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FP_COR, markersize=7,
                   label=f"falso positivo ({contagem_dias['FP']} dias)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NEU_COR, markersize=7,
                   label=f"antes de parada, não contado ({contagem_dias['NEUTRO']} dias)"),
    ]
    ax.legend(handles=h_leg, loc="upper left", fontsize=8.3, ncol=3, frameon=True,
             framealpha=0.92, edgecolor=RULE, bbox_to_anchor=(0.0, 1.08))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.tick_params(labelsize=8.5, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.set_ylabel("T5_AVG_A (°C)", fontsize=9.5, color=INK2)
    ax.set_xlabel("Tempo (data/hora)", fontsize=9.5, color=INK2)
    ax.grid(axis="y", color=RULE, lw=0.7)

    titulo = (f"Silêncio 12h (menor FP) — 4 sinais físicos, 16 meses — {n_tp}/8 falhas · "
             f"lead {m['lead_med']:.1f} h · {n_fp/meses:.2f} FP/mês · {h_fp/meses:.2f} h/mês "
             f"em alarme falso  (regra C)")
    fig.suptitle(titulo, fontsize=12.5, fontweight="bold", color=INK, y=1.06, x=0.02, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("fig_silencio12h.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("-> fig_silencio12h.png")
