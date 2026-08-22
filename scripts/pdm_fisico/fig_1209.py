#!/usr/bin/env python3
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import detector as D, avalia as A, rolante as RO, familias as F

T = pd.Timestamp("2025-12-09 08:36", tz="UTC")
A0, A1 = T - pd.Timedelta(days=12), T + pd.Timedelta(hours=8)
g = pd.read_parquet("grade2min.parquet")
q = pd.read_parquet("quente.parquet")["q"]
Z = pd.read_parquet("Z_rolante.parquet")
ev = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")

Zs = Z.ewm(halflife=int(pd.Timedelta(hours=8)/D.PAS), min_periods=1).mean()
r, _ = RO.limiar_rolante(Zs[[c for f in F.CONJ["mecanica"] for c in F.FAM[f]]], q, ev, 0.05, guarda_h=24)
al = A.sustenta((r > 3.0).fillna(False), 120) & q

w = slice(A0, A1)
ref = g[(g.index >= T-pd.Timedelta(days=40)) & (g.index < T-pd.Timedelta(days=10))]
ref = ref[q[ref.index]]

fig, ax = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True,
                       gridspec_kw=dict(height_ratios=[3, 2, 2], hspace=0.14))
cor = {"TV_355X_A": "#c0392b", "TV_355Y_A": "#e67e22", "TV_354X_A": "#8e44ad"}
for c, k in cor.items():
    s = g[c][w].where(q[w])
    ax[0].plot(s.index, s, lw=1.0, color=k, label=c[:-2])
    m = ref[c].median(); sd = (ref[c]-m).abs().median()*1.4826
    ax[0].axhspan(m-3*sd, m+3*sd, color=k, alpha=0.08)
    ax[0].axhline(m, color=k, ls=":", lw=0.8)
ax[0].set_ylabel("vibração")
ax[0].legend(loc="upper left", ncol=3, fontsize=9, frameon=False)
ax[0].set_title("TC-330.03A · trip por temperatura alta de mancal radial em 09/12/2025\n"
                "faixa sombreada = ±3σ da referência de 40 a 10 dias antes", fontsize=11, loc="left")

for c, k, lab in [("954005_624_TI_0305", "#2c3e50", "TI_0305 metal do mancal"),
                  ("954005_624_TI_0325", "#16a085", "TI_0325 óleo")]:
    s = g[c][w].where(q[w]); ax[1].plot(s.index, s, lw=1.0, color=k, label=lab)
ax[1].set_ylabel("°C"); ax[1].legend(loc="upper left", fontsize=9, frameon=False)

rr = r[w]
ax[2].plot(rr.index, rr, lw=0.9, color="#34495e")
ax[2].axhline(3.0, color="#c0392b", ls="--", lw=1.0, label="corte do detector")
aa = al[w]
ax[2].fill_between(aa.index, 0, ax[2].get_ylim()[1], where=aa.to_numpy(),
                   color="#c0392b", alpha=0.18, step="mid", label="alerta")
ax[2].set_ylabel("razão z/limiar"); ax[2].set_yscale("log")
ax[2].legend(loc="upper left", fontsize=9, frameon=False)

for a in ax:
    a.axvline(T, color="k", lw=1.4)
    a.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].annotate("trip", xy=(T, ax[0].get_ylim()[1]), xytext=(-28, -12),
               textcoords="offset points", fontsize=9, fontweight="bold")
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
ax[2].set_xlabel("2025 (UTC) — lacunas = máquina parada ou em partida")
fig.savefig("fig_1209.png", dpi=140, bbox_inches="tight")
print("ok", (al[(al.index>=T-pd.Timedelta(hours=48))&(al.index<T)]).sum()*2/60, "h de alerta na janela")
