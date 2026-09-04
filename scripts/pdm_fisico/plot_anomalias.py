#!/usr/bin/env python3
"""Serie temporal com os pontos de anomalia marcados.

Usa o cache de sinais (piso_fisico_cache.npz) e reimplementa mascara e pos-processamento
localmente -- importar `ablacao`/`reduz_fp` dispara varreduras completas no import e
levaria minutos so para carregar.

Tres faixas: T5 (carga, mostra as campanhas), TI_0305 (metal do mancal, o tag que dispara
5 dos 8 trips) e o maximo das 10 sondas de vibracao (o sinal que sustenta em 8 de 8).
Ponto de operacao: por sinal, degrau OU CUSUM (kappa=0,75, h=80, carga 0,25 na partida);
voto >=2; refratario 48 h; duracao minima 120 min.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET

T0 = pd.Timestamp("2025-01-01", tz="UTC")
PAS = pd.Timedelta("2min")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
SIN = ["t", "p", "sp", "vb"]
INK, MUTED, GRID = "#131a20", "#6b7885", "#e8ebed"
C1, C2, C3, VERM, CINZA = "#0f6e78", "#b8792a", "#2b6ca3", "#c0392b", "#eef0f1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "axes.labelcolor": "#3d4a55", "xtick.color": "#3d4a55",
                     "ytick.color": "#3d4a55", "text.color": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "figure.facecolor": "white",
                     "axes.facecolor": "white"})


def episodios(al, gap_h=2.0):
    """Mesma regra de avalia.episodios: funde episodios separados por < gap_h horas.
    Sem a fusao, o refratario e a contagem divergem do detector validado."""
    v = al.fillna(False).to_numpy()
    d = np.diff(np.r_[0, v.astype(int), 0])
    br = [(al.index[a], al.index[b-1])
          for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    if not br:
        return []
    out = [list(br[0])]
    for s_, e_ in br[1:]:
        if (s_ - out[-1][1]) <= pd.Timedelta(hours=gap_h):
            out[-1][1] = e_
        else:
            out.append([s_, e_])
    return [tuple(x) for x in out]


def main():
    g = pd.read_parquet("grade2min.parquet")
    idx = g.index
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    stable = op & (g["T5_AVG_A"] > 300)
    part = op & ~op.shift(fill_value=False)
    n_bl = int(pd.Timedelta(DET.BLACKOUT) / pd.Timedelta(C.GRID))
    black = part.rolling(n_bl, min_periods=1).max().astype(bool)
    sel = idx >= T0
    mask = (stable & ~black) & sel
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    alvo = list(fal[fal >= T0])

    z = np.load("piso_fisico_cache.npz")
    sp = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbz = np.full(len(idx), np.nan)
    vbz[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    out = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp, "vb": vbz}, index=idx)

    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    reset = ((~mask) | part).to_numpy()

    def cus(zz, h, carry=0.25):
        x = (zz - 0.75).fillna(0.0).to_numpy()
        S = np.empty(len(x)); acc = 0.0
        for i in range(len(x)):
            acc = acc*carry if reset[i] else max(0.0, acc + x[i]); S[i] = acc
        return S > h

    ON = {}
    for c in SIN:
        thr = BASE[c]*K[c]
        n = DET.SUSTAIN
        deg = ((E[c] > thr).astype(int).rolling(n, min_periods=n).sum() >= n)
        ON[c] = (deg | pd.Series(cus((E[c]/thr).clip(upper=20), 80), index=idx)) & mask
    voto = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask

    # refratario 48 h e duracao minima 120 min
    al = pd.Series(False, index=idx); bloq = None
    for a, b in episodios(voto):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
    al2 = pd.Series(False, index=idx)
    for a, b in episodios(al):
        if (b - a).total_seconds()/60 + 2 >= 120: al2.loc[a:b] = True
    al = al2

    eps = episodios(al & sel)
    jw = [(t - pd.Timedelta(hours=48), t) for t in alvo]
    det = sum(1 for t in alvo if any(a <= t and b >= t - pd.Timedelta(hours=48) for a, b in eps))
    fp = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
    meses = mask.sum()*2/60/730
    hfp = sum((b-a).total_seconds()/3600 + 2/60 for a, b in fp)
    print(f"{det}/{len(alvo)} paradas  |  {len(eps)} episodios, {len(fp)} FP  "
          f"({len(fp)/meses:.2f}/mes, {hfp/meses:.1f} h/mes)", flush=True)

    t5 = pd.to_numeric(g["T5_AVG_A"], errors="coerce")[sel]
    tm = pd.to_numeric(g["954005_624_TI_0305"], errors="coerce")[sel].where(lambda s: s.between(20, 200))
    vb = g[C.VIBRATION_TAGS].apply(pd.to_numeric, errors="coerce").max(axis=1)[sel]
    an = al[sel]; mk = mask[sel]; ii = idx[sel]

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.2), sharex=True, gridspec_kw=dict(hspace=0.10))
    off = ~mk.fillna(False); d = off.astype(int).diff().fillna(0)
    ini = list(ii[d == 1]); fim = list(ii[d == -1])
    if off.iloc[0]: ini = [ii[0]] + ini
    if off.iloc[-1]: fim = fim + [ii[-1]]
    for ax, (s, rot, cor) in zip(axes, [(t5, "T5_AVG_A — exaustão (°C)", C1),
                                        (tm, "TI_0305 — metal do mancal (°C)", C2),
                                        (vb, "máx. das 10 sondas TV_35* (µm)", C3)]):
        for a, b in zip(ini, fim): ax.axvspan(a, b, color=CINZA, lw=0, zorder=0)
        ax.plot(s.index, s.values, color=cor, lw=0.35, zorder=2)
        pts = s.where(an.reindex(s.index).fillna(False))
        ax.plot(pts.index, pts.values, ".", color=VERM, ms=1.8, zorder=4)
        for t in alvo: ax.axvline(t, color=INK, lw=0.9, zorder=5)
        ax.set_ylabel(rot, fontsize=7.6)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 820)
    axes[-1].xaxis.set_major_formatter(DateFormatter("%b/%y"))
    axes[0].legend(handles=[Line2D([], [], color=VERM, marker=".", ls="", ms=6, label="ponto anômalo"),
                            Line2D([], [], color=INK, lw=0.9, label=f"parada real ({len(alvo)})"),
                            Patch(facecolor=CINZA, label="fora da máscara")],
                   loc="upper left", ncol=3, fontsize=7.2, frameon=False,
                   handlelength=1.6, columnspacing=1.4, borderaxespad=0.4)
    fig.suptitle(f"TC-330.03A — série e anomalias detectadas   ·   {det}/{len(alvo)} paradas · "
                 f"{len(fp)/meses:.2f} FP/mês · {hfp/meses:.1f} h/mês", fontsize=10, y=0.965, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("fig_anomalias_serie.png", dpi=190)
    plt.close(fig); print("fig_anomalias_serie.png", flush=True)

    fig2, ax2 = plt.subplots(2, len(alvo), figsize=(14, 5.0), sharey="row")
    for k, t in enumerate(alvo):
        a, b = t - pd.Timedelta(hours=72), t + pd.Timedelta(hours=6)
        for r, (s, rot, cor) in enumerate([(tm, "mancal (°C)", C2), (vb, "vibração (µm)", C3)]):
            ax = ax2[r, k]; w = s.loc[a:b]
            mw = ~mk.loc[a:b].fillna(False); dd = mw.astype(int).diff().fillna(0)
            i2 = list(w.index[dd == 1]); f2 = list(w.index[dd == -1])
            if len(mw) and mw.iloc[0]: i2 = [w.index[0]] + i2
            if len(mw) and mw.iloc[-1]: f2 = f2 + [w.index[-1]]
            for xa, xb in zip(i2, f2): ax.axvspan(xa, xb, color=CINZA, lw=0, zorder=0)
            ax.plot(w.index, w.values, color=cor, lw=0.6, zorder=2)
            p = w.where(an.reindex(w.index).fillna(False))
            ax.plot(p.index, p.values, ".", color=VERM, ms=2.8, zorder=4)
            ax.axvline(t, color=INK, lw=1.0, zorder=5)
            ax.set_xticks([]); ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
            if k == 0: ax.set_ylabel(rot, fontsize=7.6)
            if r == 0: ax.set_title(f"{t:%d/%m/%y}", fontsize=8, color=INK)
    fig2.suptitle("Os 8 eventos — 72 h antes de cada parada", fontsize=10, y=0.97, color=INK)
    fig2.tight_layout(rect=(0, 0, 1, 0.94)); fig2.savefig("fig_anomalias_zoom.png", dpi=190)
    print("fig_anomalias_zoom.png", flush=True)


main()
