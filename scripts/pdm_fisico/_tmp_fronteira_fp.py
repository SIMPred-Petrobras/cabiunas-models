"""A FRONTEIRA COMPLETA do nosso detector na regra C.

Pergunta: se o criterio for MINIMO FALSO POSITIVO, quem ganha?
Constroi, para cada nivel de deteccao, o menor FP/mes e o menor h/mes que o
nosso detector alcanca -- varrendo limiar (kb,kv), silencio pos-partida,
refratario e duracao minima. Depois compara com a fronteira publicada do
Francisco (notebook 10) e com o ponto da Lara.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import cru, EW, BASE, sel, T0
from publica_clearml import GRID, SIN, KAPPA, H_CUSUM, HL
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
JAN = pd.Timedelta(hours=48); jw = [(t-JAN, t) for t in alvo]

KBS = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4, 3.0]
KVS = [1.8, 2.2, 2.8, 3.5]
BLKS = ["6h", "12h", "24h"]
REFRATS = [48, 96, 144]
DURS = [120, 240, 480]

def pos_(voto, mask, refrat_h, dur_min):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=refrat_h)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b-a).total_seconds()/60 + 2 >= dur_min: fin.loc[a:b] = True
    return fin & sel

linhas = []
for bl in BLKS:
    n_bl = int(pd.Timedelta(bl)/pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    for kb in KBS:
        for kv in KVS:
            K_ = {"t":kb,"p":kb,"sp":kb,"vb":kv}; ON = {}
            for c in SIN:
                thr = BASE[c]*K_[c]; E = EW[c].where(mask)
                deg = ((E > thr).astype(int).rolling(15, min_periods=15).sum() >= 15)
                cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(),
                                     reset) > H_CUSUM, index=idx)
                ON[c] = (deg | cu) & mask
            ns = sum(ON[c].astype(int) for c in SIN)
            v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
            for rf in REFRATS:
                for dm in DURS:
                    al = pos_(v, mask, rf, dm)
                    eps = AV.episodios(al)
                    if not eps: continue
                    m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
                    cl = classifica_regra_c(eps, paradas)
                    n_fp = sum(1 for a,b,c,l in cl if c=="FP")
                    h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
                    det = sum(any(a<=t1 and b>=t0 for a,b in eps) for t0,t1 in jw)
                    linhas.append(dict(bl=bl, kb=kb, kv=kv, rf=rf, dm=dm, det=det,
                                       fp_mes=n_fp/meses, h_mes=h_fp/meses,
                                       lead=m["lead_med"], meses=meses))
T = pd.DataFrame(linhas)
T.to_csv("_tmp_fronteira_fp.csv", index=False)
print(f"configuracoes avaliadas: {len(T)}\n")
print("NOSSA FRONTEIRA (regra C) -- por nivel de deteccao")
print("="*104)
print(f"{'nivel':>7} {'n_cfg':>7} {'menor FP/mes':>13} {'h desse pt':>11} | "
      f"{'menor h/mes':>12} {'FP desse pt':>12} {'config do menor h':>22}")
for nivel in range(8, 2, -1):
    s = T[T.det == nivel]
    if s.empty:
        print(f"{nivel:>6}/8 {'--':>7}   nenhuma configuracao"); continue
    a = s.sort_values("fp_mes").iloc[0]
    b = s.sort_values("h_mes").iloc[0]
    cfg = f"{b.bl}|kb{b.kb}|kv{b.kv}|rf{b.rf}|dm{b.dm}"
    print(f"{nivel:>6}/8 {len(s):>7} {a.fp_mes:13.3f} {a.h_mes:11.2f} | "
          f"{b.h_mes:12.2f} {b.fp_mes:12.3f} {cfg:>22}")
