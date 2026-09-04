#!/usr/bin/env python3
"""Subir o piso de T5 da mascara: 300 degC (atual) ate 600 degC.

A mascara exige `RUNNING_A > 0,5 E T5_AVG_A > 300 degC`. Ja testamos TIRAR o filtro de T5
(fica pior: abre 48 h de transiente 5x mais densas em alarme e custa uma parada). Nunca
testamos SUBIR. A 300 degC a turbina ainda esta em rampa; a 500 degC ja esta em regime --
e 11 dos 15 falsos positivos estao a menos de 30 h de uma partida.

Subir o piso muda tres coisas ao mesmo tempo, e todas entram na conta:
  - o que e pontuavel (numerador e denominador de FP);
  - os meses de operacao usados na taxa;
  - o reset do CUSUM, que zera fora da mascara.

Usa o cache de sinais; nao refaz o walk-forward.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET

T0 = pd.Timestamp("2025-01-01", tz="UTC")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
SIN = ["t", "p", "sp", "vb"]
PISOS = [300, 350, 400, 450, 500, 550, 600]
JAN = pd.Timedelta(hours=48)


def episodios(al, gap_h=2.0):
    v = al.fillna(False).to_numpy()
    d = np.diff(np.r_[0, v.astype(int), 0])
    br = [(al.index[a], al.index[b-1])
          for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    if not br: return []
    out = [list(br[0])]
    for s, e in br[1:]:
        if (s - out[-1][1]) <= pd.Timedelta(hours=gap_h): out[-1][1] = e
        else: out.append([s, e])
    return [tuple(x) for x in out]


def main():
    g = pd.read_parquet("grade2min.parquet"); idx = g.index
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    t5 = pd.to_numeric(g["T5_AVG_A"], errors="coerce")
    part = op & ~op.shift(fill_value=False)
    n_bl = int(pd.Timedelta(DET.BLACKOUT) / pd.Timedelta(C.GRID))
    black = part.rolling(n_bl, min_periods=1).max().astype(bool)
    sel = idx >= T0
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    alvo = list(fal[fal >= T0])
    jw = [(t - JAN, t) for t in alvo]

    z = np.load("piso_fisico_cache.npz")
    sp = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbz = np.full(len(idx), np.nan)
    vbz[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    out = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp, "vb": vbz}, index=idx)

    print(f"{'piso T5':>9} {'h pontuaveis':>13} {'meses':>7} {'det':>6} {'eps':>5} {'FP/mes':>7} "
          f"{'h/mes':>7} {'duty':>6} {'lead':>6} {'precisao':>9}  perde")
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    L = []
    for piso in PISOS:
        mask = (op & (t5 > piso) & ~black) & sel
        meses = mask.sum()*2/60/730
        E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
             for c, h in HL.items()}
        reset = ((~mask) | part).to_numpy()

        def cus(zz, h=80, carry=0.25):
            x = (zz - 0.75).fillna(0.0).to_numpy()
            S = np.empty(len(x)); acc = 0.0
            for i in range(len(x)):
                acc = acc*carry if reset[i] else max(0.0, acc + x[i]); S[i] = acc
            return S > h
        ON = {}
        for c in SIN:
            thr = BASE[c]*K[c]; n = DET.SUSTAIN
            deg = ((E[c] > thr).astype(int).rolling(n, min_periods=n).sum() >= n)
            ON[c] = (deg | pd.Series(cus((E[c]/thr).clip(upper=20)), index=idx)) & mask
        voto = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
        al = pd.Series(False, index=idx); bloq = None
        for a, b in episodios(voto):
            if bloq is not None and a <= bloq: continue
            al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
        al2 = pd.Series(False, index=idx)
        for a, b in episodios(al):
            if (b - a).total_seconds()/60 + 2 >= 120: al2.loc[a:b] = True
        eps = episodios(al2 & sel)
        det = [f"{t:%Y-%m-%d}" for t in alvo
               if any(a <= t and b >= t - JAN for a, b in eps)]
        fp = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
        hfp = sum((b-a).total_seconds()/3600 + 2/60 for a, b in fp)
        leads = []
        for t in alvo:
            c_ = [a for a, b in eps if a <= t and b >= t - JAN]
            if c_: leads.append((t - min(c_)).total_seconds()/3600)
        prec = 100*(len(det)/meses)/((len(det)/meses) + len(fp)/meses) if fp else 100.0
        perd = sorted(set(alvo_s) - set(det))
        print(f"{piso:8d}C {mask.sum()*2/60:12.0f}h {meses:7.1f} {len(det):4d}/8 {len(eps):5d} "
              f"{len(fp)/meses:7.2f} {hfp/meses:7.1f} {100*hfp/meses/730:5.1f}% "
              f"{np.median(leads) if leads else float('nan'):6.1f} {prec:8.1f}%  "
              f"{', '.join(p[5:] for p in perd) if perd else '—'}", flush=True)
        L.append(dict(piso=piso, h=mask.sum()*2/60, det=len(det), eps=len(eps),
                      fp=len(fp)/meses, hm=hfp/meses,
                      lead=np.median(leads) if leads else np.nan, perde=",".join(perd)))
    pd.DataFrame(L).to_csv("t5_limiar.csv", index=False)


main()
