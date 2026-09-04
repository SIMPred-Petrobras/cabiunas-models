import numpy as np, pandas as pd, avalia as AV
from pos_processamento import cru, EW, BASE, sel, T0
from publica_clearml import GRID, SIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))

BLACKOUTS = ["6h", "9h", "12h", "18h", "24h"]
KB = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]
KV = [1.8, 2.2, 2.8]

def pos(voto, n_sin, refrat_h, dur_min):
    al = pd.Series(False, index=idx)
    bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=refrat_h)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds()/60 + 2 >= dur_min:
            fin.loc[a:b] = True
    return fin & sel

linhas = []
for bl in BLACKOUTS:
    n_bl = int(pd.Timedelta(bl) / pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    melhor = None
    for kb in KB:
        for kv in KV:
            K_ = {"t": kb, "p": kb, "sp": kb, "vb": kv}
            ON = {}
            for c in SIN:
                thr = BASE[c]*K_[c]
                E = EW[c].where(mask)
                deg = ((E > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
                cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(), reset) > H_CUSUM, index=idx)
                ON[c] = (deg | cu) & mask
            ns = sum(ON[c].astype(int) for c in SIN)
            v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
            al = pos(v, ns, REFRAT_H, DUR_MIN)
            m = AV.avalia(al, alvo, mask)
            if melhor is None or (m["det"], -m["h_fp_mes"]) > (melhor["det"], -melhor["h_fp_mes"]):
                melhor = dict(det=m["det"], eps=m["episodios"], fp_mes=m["fp_mes"],
                             h_fp_mes=m["h_fp_mes"], lead=m["lead_med"], kb=kb, kv=kv)
    linhas.append(dict(blackout=bl, **melhor))

T = pd.DataFrame(linhas)
print(T.to_string(index=False))

print("\n--- so os 8/8, ordenados por h_fp_mes ---")
oito = []
for bl in BLACKOUTS:
    n_bl = int(pd.Timedelta(bl) / pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    for kb in KB:
        for kv in KV:
            K_ = {"t": kb, "p": kb, "sp": kb, "vb": kv}
            ON = {}
            for c in SIN:
                thr = BASE[c]*K_[c]
                E = EW[c].where(mask)
                deg = ((E > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
                cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(), reset) > H_CUSUM, index=idx)
                ON[c] = (deg | cu) & mask
            ns = sum(ON[c].astype(int) for c in SIN)
            v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
            al = pos(v, ns, REFRAT_H, DUR_MIN)
            m = AV.avalia(al, alvo, mask)
            if m["det"] == 8:
                oito.append(dict(blackout=bl, kb=kb, kv=kv, eps=m["episodios"], fp_mes=m["fp_mes"],
                                 h_fp_mes=m["h_fp_mes"], lead=m["lead_med"]))
O = pd.DataFrame(oito).sort_values("h_fp_mes")
print(O.head(15).to_string(index=False))
