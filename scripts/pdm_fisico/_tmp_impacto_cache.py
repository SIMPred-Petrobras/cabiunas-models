"""A divergencia em t/p muda o detector? E de onde ela vem?"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV

a = dict(np.load("piso_fisico_cache.npz", allow_pickle=True).items())
b = dict(np.load("_tmp_cache_regerado.npz", allow_pickle=True).items())

print("MAGNITUDE DA DIVERGENCIA em t e p")
print("=" * 80)
for k in ("t", "p"):
    x, y = a[k].astype("float64"), b[k].astype("float64")
    fin = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x[fin] - y[fin])
    rel = d / np.maximum(np.abs(x[fin]), 1e-9)
    print(f"  {k}: n={fin.sum():,}  identicos={100*(d==0).mean():.2f}%  "
          f"mediana|dif|={np.median(d):.3g}  p99={np.percentile(d,99):.3g}  max={d.max():.3g}")
    print(f"     escala do sinal: mediana={np.median(x[fin]):.3f}  limiar=3.40  "
          f"dif relativa mediana={np.median(rel):.2e}")

print("\nO DETECTOR MUDA? (mesmo pipeline, os dois caches)")
print("=" * 80)
from publica_clearml import SIN, HL, BASE, K, KAPPA, H_CUSUM, SUSTAIN, REFRAT_H, DUR_MIN, T0, GRID, BLACKOUT
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

g = pd.read_parquet("grade2min.parquet"); idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
sel = idx >= T0
n_bl = int(pd.Timedelta(BLACKOUT)/pd.Timedelta(GRID))
mask = (estavel & ~part.rolling(n_bl, min_periods=1).max().astype(bool)) & sel
reset = ((~mask) | part).to_numpy()
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0])); paradas = paradas_reais_2h()

def roda(z, rot):
    spv = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Zv = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vb = np.full(len(idx), np.nan)
    vb[z["hot"]] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
    vb[~np.isfinite(vb)] = np.nan
    cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vb}, index=idx)
    EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}
    ON = {}
    for c in SIN:
        thr = BASE[c]*K[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(), reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pd.Series(False, index=idx); bloq = None
    for x, y in AV.episodios(v):
        if bloq is not None and x <= bloq: continue
        al.loc[x:y] = True; bloq = y + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for x, y in AV.episodios(al):
        if (y-x).total_seconds()/60 + 2 >= DUR_MIN: fin.loc[x:y] = True
    fin = fin & sel
    eps = AV.episodios(fin); m = AV.avalia(fin, alvo, mask); meses = m["horas_op"]/730.0
    cl = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _,_,c,_ in cl if c=="FP")
    h = sum((y-x).total_seconds()/3600 for x,y,c,_ in cl if c=="FP")
    print(f"  {rot:>22}: {m['det']}/8 · {len(eps)} episodios · {nfp/meses:.3f} FP/mes · "
          f"{h/meses:.2f} h/mes · lead {m['lead_med']:.1f}h")
    return fin

f1 = roda(a, "cache publicado")
f2 = roda(b, "cache regenerado")
print(f"\n  series de alarme identicas amostra a amostra: {'SIM' if f1.equals(f2) else 'NAO'}")
if not f1.equals(f2):
    print(f"  amostras divergentes: {int((f1 != f2).sum())} de {len(f1):,}")
