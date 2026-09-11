"""A TAXA BASE da nossa regra de associacao: quanto do 8/8 e o detector e
quanto e o tempo que ele passa aceso?

A nossa regra credita deteccao quando o alarme esta DE PE em algum instante de
[t-48h, t]. Se o detector passa uma fracao grande do tempo aceso, uma janela de
48 h contem alarme por acaso -- e o 8/8 seria fraco como evidencia. Este script
mede o nulo por permutacao: 8 janelas em instantes SORTEADOS (na mesma mascara
de operacao, para nao sortear dentro de blackout), sob as duas regras.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from plota_estilo_francisco import alarme

RNG = np.random.default_rng(20260904)
N = 20_000
JAN = pd.Timedelta(hours=48)

al = alarme()
eps = AV.episodios(al)
# ATENCAO: o indice e datetime64[us]; Timestamp.value e ns. Misturar os dois da
# um fator de 1000 e nenhuma janela intersecta nada. Tudo em us, via astype.
def us(ts):
    return np.asarray(pd.DatetimeIndex(ts).astype("int64"))

inis = us([a for a, _ in eps])
fins = us([b for _, b in eps])

# ---- duty cycle -------------------------------------------------------------
passo_h = (al.index[1] - al.index[0]).total_seconds() / 3600
n_on = int(al.fillna(False).to_numpy().astype(bool).sum())
horas_total = len(al) * passo_h
dur = np.array([(b - a).total_seconds() / 3600 for a, b in eps])
print("O NOSSO DETECTOR NO TEMPO")
print("=" * 72)
print(f"  janela avaliada          : {horas_total/24:8.1f} dias")
print(f"  horas com alarme de pe   : {n_on*passo_h:8.1f} h")
print(f"  DUTY CYCLE               : {100*n_on/len(al):8.2f} %")
print(f"  episodios                : {len(eps)}")
print(f"  duracao mediana          : {np.median(dur):8.2f} h")
print(f"  duracao media            : {dur.mean():8.2f} h")
print(f"  duracao maxima           : {dur.max():8.2f} h")
print(f"  acima de 48 h            : {(dur > 48).sum()} de {len(eps)}")

# ---- o nulo por permutacao --------------------------------------------------
# sorteia instantes na mascara de operacao, com folga de 48 h no inicio
elegivel = idx[(idx >= idx[0] + JAN) & mask.reindex(idx, fill_value=False)]
cand = us(elegivel)
jan_us = JAN.value // 1000

sorteio = RNG.choice(cand, size=(N, len(alvo)), replace=True)
t = sorteio[:, :, None]; t0 = t - jan_us
# de_pe: existe episodio que intersecta [t0, t]
d_pe = ((inis[None, None, :] <= t) & (fins[None, None, :] >= t0)).any(2).sum(1)
# inicio na janela
d_ini = ((inis[None, None, :] >= t0) & (inis[None, None, :] <= t)).any(2).sum(1)

print("\nO NULO -- 8 janelas de 48 h em instantes SORTEADOS (20.000 sorteios)")
print("=" * 72)
for rot, d, obs in [("nossa regra (de pe)", d_pe, 8), ("regra dele (inicio)", d_ini, 4)]:
    p = float((d >= obs).mean())
    print(f"  {rot:<22}: esperado {d.mean():.2f}/8  "
          f"(p10={np.percentile(d,10):.0f} p90={np.percentile(d,90):.0f})  "
          f"observado {obs}/8  ->  p = {p:.4f}"
          + ("  *** significativo" if p < 0.05 else "  <-- NAO significativo"))

# ---- conferencia aritmetica independente da permutacao ----------------------
horas_alarme = float(dur.sum())
p_pe = (horas_alarme + 48.0 * len(eps)) / horas_total     # medida de Lebesgue
p_ini = 48.0 * len(eps) / horas_total
from math import comb
def cauda(p, obs, n=8):
    return sum(comb(n, k) * p**k * (1-p)**(n-k) for k in range(obs, n+1))
print("\nCONFERENCIA -- binomial fechada, sem sorteio")
print("=" * 72)
print(f"  nossa regra : p_janela = {p_pe:.4f}  ->  P(>=8/8) = {cauda(p_pe, 8):.3e}")
print(f"  regra dele  : p_janela = {p_ini:.4f}  ->  P(>=4/8) = {cauda(p_ini, 4):.3e}")
