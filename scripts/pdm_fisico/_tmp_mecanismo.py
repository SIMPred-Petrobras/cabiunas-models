"""O que realmente entrega 09/12/2025 e 04/11/2025: o SENSOR ou o INTEGRADOR?

O Francisco olhou as 10 tags de vibracao no diagnostico de precursor (os "36
sensores" = 26 das familias + 10 TV_*) e mediu z no percentil 79-85 nas 48h
antes. Nos detectamos. Se o sinal cru esta no mesmo lugar nos dois, a diferenca
nao e o sensor -- e o que se faz com ele.

Separa, por episodio, qual mecanismo acendeu cada canal:
  degrau = limiar + SUSTAIN (15 amostras de 2min acima)  -> pega excursao FORTE
  CUSUM  = soma acumulada de (E/thr - kappa) > H_CUSUM    -> pega desvio FRACO e PERSISTENTE
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pos_processamento import EW, BASE, mask, idx, alvo, reset, cru
from publica_clearml import SIN, K, KAPPA, H_CUSUM, SUSTAIN
from blackout_curto import cusum

DEG, CU = {}, {}
for c in SIN:
    thr = BASE[c]*K[c]
    E = EW[c].where(mask)
    DEG[c] = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN) & mask
    CU[c] = pd.Series(cusum(((E/thr).clip(upper=20)-KAPPA).fillna(0.0).to_numpy(),
                            reset) > H_CUSUM, index=idx) & mask

EPS = {"09/12/2025": ("2025-12-08 09:02", "2025-12-08 23:02"),
       "04/11/2025": ("2025-11-03 21:32", "2025-11-04 03:52"),
       "27/02/2025": ("2025-02-21 08:42", "2025-02-27 08:36")}

print("MECANISMO QUE ACENDEU CADA CANAL, POR EPISODIO")
print("="*88)
for nome, (a, b) in EPS.items():
    a = pd.Timestamp(a, tz="UTC"); b = pd.Timestamp(b, tz="UTC")
    jan = (idx >= a) & (idx <= b)
    print(f"\n  episodio que antecipa {nome}   ({a:%d/%m %H:%M} -> {b:%d/%m %H:%M})")
    print(f"    {'canal':>6} {'so degrau':>11} {'so CUSUM':>10} {'ambos':>8}   "
          f"{'pico E/thr':>11}  quem entregou")
    for c in SIN:
        d = DEG[c].to_numpy()[jan]; u = CU[c].to_numpy()[jan]
        so_d = (d & ~u).mean()*100; so_u = (u & ~d).mean()*100; amb = (d & u).mean()*100
        pico = (EW[c].where(mask).to_numpy()[jan] / (BASE[c]*K[c]))
        pico = np.nanmax(pico) if np.isfinite(pico).any() else np.nan
        if so_d+so_u+amb == 0: quem = "-- apagado"
        elif so_u > 0 and so_d == 0: quem = "CUSUM sozinho (desvio fraco e persistente)"
        elif so_d > 0 and so_u == 0: quem = "degrau sozinho (excursao forte)"
        else: quem = "os dois"
        print(f"    {c:>6} {so_d:10.0f}% {so_u:9.0f}% {amb:7.0f}% {pico:11.2f}x  {quem}")

print("\n" + "="*88)
print("O SINAL CRU DE VIBRACAO NAS 48h ANTES -- em que percentil ele fica?")
print("(o Francisco mediu percentil 79-85 no diagnostico dele)")
print("="*88)
base = cru["vb"].where(mask).dropna()
for nome, (a, b) in EPS.items():
    t_alvo = pd.Timestamp({"09/12/2025":"2025-12-09 08:36","04/11/2025":"2025-11-04 06:22",
                           "27/02/2025":"2025-02-27 08:38"}[nome], tz="UTC")
    j48 = (idx >= t_alvo - pd.Timedelta("48h")) & (idx <= t_alvo) & mask.to_numpy()
    v = cru["vb"].to_numpy()[j48]; v = v[np.isfinite(v)]
    if not len(v): continue
    pmax = 100*(base < np.nanmax(v)).mean()
    pmed = 100*(base < np.nanmedian(v)).mean()
    print(f"  {nome}: vb cru nas 48h antes -> mediana no p{pmed:.0f}, maximo no p{pmax:.1f}")
