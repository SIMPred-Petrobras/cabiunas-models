"""Regenera o piso_fisico_cache.npz com o pacote restaurado e compara array por
array com o cache publicado. Se bater, a restauracao e fiel e o detector volta a
ser reprodutivel de ponta a ponta."""
import sys, os, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import piso_fisico as PF
from ablacao import canonico

NOVO = "_tmp_cache_regerado.npz"
if os.path.exists(NOVO):
    os.remove(NOVO)

antigo = {k: v for k, v in np.load("piso_fisico_cache.npz", allow_pickle=True).items()}
print(f"cache publicado: {len(antigo)} arrays -> {', '.join(sorted(antigo))}")

df = canonico()
falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
print(f"grade: {len(df):,} amostras · falhas: {len(falhas)}")

PF.CACHE = NOVO                       # desvia o guarda de cache
t0 = time.time()
novo = PF.pre(df, falhas)
print(f"\nregenerado em {time.time()-t0:.0f} s\n")

print("COMPARACAO ARRAY POR ARRAY")
print("=" * 84)
print(f"{'array':>10} {'shape':>16} {'iguais':>9} {'max|dif|':>12} {'NaN igual':>10}")
tudo_ok = True
for k in sorted(antigo):
    a, b = np.asarray(antigo[k], dtype="float64"), np.asarray(novo[k], dtype="float64")
    if a.shape != b.shape:
        print(f"{k:>10} {str(a.shape):>16}   SHAPE DIFERENTE -> {b.shape}")
        tudo_ok = False; continue
    na, nb = np.isnan(a), np.isnan(b)
    nan_ok = bool((na == nb).all())
    fin = ~na & ~nb
    dif = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
    ok = nan_ok and dif < 1e-9
    tudo_ok &= ok
    print(f"{k:>10} {str(a.shape):>16} {('SIM' if ok else 'NAO'):>9} {dif:12.3g} "
          f"{('sim' if nan_ok else 'NAO'):>10}")

print("\n" + "=" * 84)
print("RESTAURACAO FIEL -- o cache e regeneravel bit a bit" if tudo_ok
      else "DIVERGENCIA -- a restauracao NAO reproduz o cache publicado")
