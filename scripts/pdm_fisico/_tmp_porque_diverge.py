"""Por que as duas rotas divergem? Hipotese: o EWMA em amostra roda sobre serie
DESCONTINUA (so os 20.000 pontos estaveis selecionados), entao quase nao suaviza;
o EWMA real roda na grade contigua de 2 min e suaviza forte. Suavizar uma
distribuicao assimetrica a direita SOBE o corpo -- entao o mesmo limiar cai num
percentil muito mais baixo depois da suavizacao.

Se a hipotese estiver certa: cru -> percentil alto; suavizado -> percentil baixo,
nas DUAS rotas."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pos_processamento import cru, EW, mask, idx
from publica_clearml import SIN, BASE, K, HL

print(f"{'sinal':>6} {'k*base':>8} {'CRU (sem EWMA)':>18} {'EWMA real (contiguo)':>22}")
for c in SIN:
    thr = BASE[c] * K[c]
    v_cru = cru[c][mask].dropna()
    v_ew = EW[c].where(mask).dropna()
    p_cru = 100.0 * (v_cru < thr).mean()
    p_ew = 100.0 * (v_ew < thr).mean()
    print(f"{c:>6} {thr:8.2f} {p_cru:17.2f}% {p_ew:21.2f}%")

print("\nefeito da suavizacao no CORPO da distribuicao (mediana e p90):")
print(f"{'sinal':>6} {'mediana cru':>13} {'mediana EWMA':>14} {'p90 cru':>10} {'p90 EWMA':>11}")
for c in SIN:
    v_cru = cru[c][mask].dropna(); v_ew = EW[c].where(mask).dropna()
    print(f"{c:>6} {v_cru.median():13.3f} {v_ew.median():14.3f} "
          f"{np.percentile(v_cru,90):10.3f} {np.percentile(v_ew,90):11.3f}")

print("\nteste direto da hipotese do gap: EWMA sobre serie contigua vs. sobre a mesma")
print("serie com so os pontos mascarados (descontigua), no mesmo periodo:")
for c in ("t", "p"):
    s_full = cru[c]
    ew_cont = s_full.ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean()[mask].dropna()
    s_gap = cru[c][mask].dropna()
    ew_gap = s_gap.ewm(halflife=pd.Timedelta(HL[c]), times=s_gap.index).mean()
    thr = BASE[c]*K[c]
    print(f"  {c}: contiguo p{100*(ew_cont<thr).mean():.2f}   "
          f"descontiguo p{100*(ew_gap<thr).mean():.2f}   "
          f"cru p{100*(s_gap<thr).mean():.2f}")
