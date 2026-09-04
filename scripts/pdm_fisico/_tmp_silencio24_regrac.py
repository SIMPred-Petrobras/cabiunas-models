"""Silencio de 24h (a ideia do Francisco) medido pela REGRA C, no nosso detector.

Duas leituras:
  A) ponto de operacao FIXO (kb=1.7, kv=2.2) -- o efeito puro de alongar o silencio
  B) ponto de operacao REVARRIDO em cada blackout -- a comparacao justa, porque o
     Francisco naturalmente recalibraria o limiar junto
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import cru, EW, BASE, sel, T0
from publica_clearml import GRID, SIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))
paradas = paradas_reais_2h()

def pos(voto, refrat_h, dur_min, mask):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=refrat_h)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds()/60 + 2 >= dur_min: fin.loc[a:b] = True
    return fin & sel

def roda(bl, kb, kv):
    n_bl = int(pd.Timedelta(bl) / pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    K_ = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    ON = {}
    for c in SIN:
        thr = BASE[c]*K_[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
        cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(), reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, REFRAT_H, DUR_MIN, mask)
    m = AV.avalia(al, alvo, mask)
    eps = AV.episodios(al)
    cl = classifica_regra_c(eps, paradas)
    meses = m["horas_op"]/730.0
    n_tp = sum(1 for a,b,c,l in cl if c=="TP")
    n_fp = sum(1 for a,b,c,l in cl if c=="FP")
    n_ne = sum(1 for a,b,c,l in cl if c=="NEUTRO")
    h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
    return dict(det=m["det"], TP=n_tp, FP=n_fp, NEU=n_ne, fp_mes=n_fp/meses,
                h_mes=h_fp/meses, lead=m["lead_med"], meses=meses, kb=kb, kv=kv)

print("A) PONTO DE OPERACAO FIXO (kb=1.7, kv=2.2) -- efeito puro do silencio")
print(f"{'silencio':>9} {'det':>5} {'TP':>4} {'FP':>4} {'NEU':>4} {'FP/mes':>8} {'h/mes':>8} {'lead_h':>8} {'meses_op':>9}")
for bl in ["6h", "9h", "12h", "18h", "24h"]:
    r = roda(bl, 1.7, 2.2)
    marca = "  <<< ATUAL" if bl == "6h" else ""
    print(f"{bl:>9} {r['det']:5d} {r['TP']:4d} {r['FP']:4d} {r['NEU']:4d} {r['fp_mes']:8.3f} "
          f"{r['h_mes']:8.2f} {r['lead']:8.1f} {r['meses']:9.2f}{marca}")

print("\nB) PONTO REVARRIDO em cada silencio (kb x kv), criterio: max deteccao, depois min h/mes")
print(f"{'silencio':>9} {'kb':>5} {'kv':>5} {'det':>5} {'TP':>4} {'FP':>4} {'NEU':>4} {'FP/mes':>8} {'h/mes':>8} {'lead_h':>8}")
for bl in ["6h", "12h", "24h"]:
    melhor = None
    for kb in [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]:
        for kv in [1.8, 2.2, 2.8]:
            r = roda(bl, kb, kv)
            if melhor is None or (r["det"], -r["h_mes"]) > (melhor["det"], -melhor["h_mes"]):
                melhor = r
    print(f"{bl:>9} {melhor['kb']:5.1f} {melhor['kv']:5.1f} {melhor['det']:5d} {melhor['TP']:4d} "
          f"{melhor['FP']:4d} {melhor['NEU']:4d} {melhor['fp_mes']:8.3f} {melhor['h_mes']:8.2f} {melhor['lead']:8.1f}")

print("\nC) QUAIS eventos sobrevivem em cada silencio (ponto fixo 1.7/2.2)")
for bl in ["6h", "12h", "24h"]:
    n_bl = int(pd.Timedelta(bl) / pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    K_ = {"t":1.7,"p":1.7,"sp":1.7,"vb":2.2}; ON = {}
    for c in SIN:
        thr = BASE[c]*K_[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
        cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(), reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, REFRAT_H, DUR_MIN, mask)
    eps = AV.episodios(al)
    JAN = pd.Timedelta(hours=48); jw = [(t-JAN, t) for t in alvo]
    pegos = [t for t,(t0,t1) in zip(alvo,jw) if any(a<=t1 and b>=t0 for a,b in eps)]
    perd = [t for t in alvo if t not in pegos]
    print(f"\n  silencio {bl}: {len(pegos)}/8")
    print(f"    pegos:   {', '.join(f'{t:%d/%m/%Y}' for t in pegos)}")
    print(f"    perdidos:{', '.join(f'{t:%d/%m/%Y}' for t in perd) if perd else ' --'}")
