"""Os pontos de baixo FP sao reais ou artefato de selecionar o minimo de 756?
Tres checagens: (1) quantos episodios o detector produz (detector quase-mudo?),
(2) robustez -- quantas configs do nivel ficam abaixo do limiar, (3) QUAIS
eventos sao pegos."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import EW, BASE, sel, T0
from publica_clearml import GRID, SIN, KAPPA, H_CUSUM
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

T = pd.read_csv("_tmp_fronteira_fp.csv")
g = pd.read_parquet("grade2min.parquet"); idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))
paradas = paradas_reais_2h(); JAN = pd.Timedelta(hours=48)
jw = [(t-JAN,t) for t in alvo]

print("1. ROBUSTEZ: distribuicao do FP/mes DENTRO de cada nivel (nao so o minimo)")
print("="*92)
print(f"{'nivel':>7} {'n':>5} {'min':>8} {'p25':>8} {'mediana':>9} {'max':>8}   "
      f"{'configs <=0,2':>14} {'<=0,5':>8}")
for n in range(8, 3, -1):
    s = T[T.det == n]
    if s.empty: continue
    print(f"{n:>6}/8 {len(s):>5} {s.fp_mes.min():8.3f} {s.fp_mes.quantile(.25):8.3f} "
          f"{s.fp_mes.median():9.3f} {s.fp_mes.max():8.3f}   "
          f"{(s.fp_mes<=0.2).sum():>8}/{len(s):<5} {(s.fp_mes<=0.5).sum():>4}/{len(s)}")

def detalha(bl, kb, kv, rf, dm, titulo):
    n_bl = int(pd.Timedelta(bl)/pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    K_ = {"t":kb,"p":kb,"sp":kb,"vb":kv}; ON = {}
    for c in SIN:
        thr = BASE[c]*K_[c]; E = EW[c].where(mask)
        deg = ((E>thr).astype(int).rolling(15,min_periods=15).sum()>=15)
        cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(),reset)>H_CUSUM, index=idx)
        ON[c] = (deg|cu)&mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns>=2, index=idx)&mask&(ON["sp"]|ON["vb"])
    al = pd.Series(False, index=idx); bloq=None
    for a,b in AV.episodios(v):
        if bloq is not None and a<=bloq: continue
        al.loc[a:b]=True; bloq=b+pd.Timedelta(hours=rf)
    fin = pd.Series(False,index=idx)
    for a,b in AV.episodios(al):
        if (b-a).total_seconds()/60+2>=dm: fin.loc[a:b]=True
    fin = fin & sel
    eps = AV.episodios(fin)
    cl = classifica_regra_c(eps, paradas)
    m = AV.avalia(fin, alvo, mask); meses=m["horas_op"]/730.0
    pegos = [t.strftime("%d/%m/%Y") for t,(t0,t1) in zip(alvo,jw)
             if any(a<=t1 and b>=t0 for a,b in eps)]
    n_fp = sum(1 for a,b,c,l in cl if c=="FP")
    print(f"\n  {titulo}")
    print(f"    config: silencio {bl}, kb={kb}, kv={kv}, refrat={rf}h, dur_min={dm}min")
    print(f"    episodios TOTAIS que o detector emite: {len(eps)}  "
          f"(TP={sum(1 for *_ ,c,_ in [(a,b,c,l) for a,b,c,l in cl] if c=='TP')}, "
          f"FP={n_fp}, NEUTRO={sum(1 for a,b,c,l in cl if c=='NEUTRO')})")
    print(f"    FP/mes={n_fp/meses:.3f}  lead={m['lead_med']:.1f}h  meses_vigiados={meses:.2f}")
    print(f"    eventos pegos: {', '.join(pegos)}")

print("\n\n2. OS PONTOS DE BAIXO FP -- sao detectores quase-mudos?")
print("="*92)
for n in (8,7,6,5,4):
    s = T[T.det==n]
    if s.empty: continue
    b = s.sort_values(["fp_mes","h_mes"]).iloc[0]
    detalha(b.bl, b.kb, b.kv, int(b.rf), int(b.dm), f"NIVEL {n}/8 -- menor FP")
