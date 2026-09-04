"""Verificacao independente da afirmacao "o limiar fixo ja e um percentil ~99,4".

A rota anterior (_tmp_percentil_diag.py) media o percentil na distribuicao EM
AMOSTRA da janela de ajuste. Ela tem duas fragilidades nao checadas:
  (a) a janela de ajuste e `estavel` mas NAO tem mascara de blackout -- contem os
      transientes de partida, que sao a cauda pesada;
  (b) o EWMA e aplicado sobre serie DESCONTINUA (so os 20.000 pontos estaveis),
      e suaviza diferente do EWMA real, que roda na grade contigua de 2 min.

Esta rota nao reimplementa nada: usa `z["t"]`/`z["p"]` do cache ja validado, o
EWMA real sobre a grade contigua e a mascara real do detector. Mede diretamente
a fracao do tempo SAUDAVEL em que cada canal fica acima do seu limiar -- que e
exatamente o complemento do percentil.

Tres definicoes de "saudavel", da mais frouxa para a mais estrita, para ver se a
resposta e robusta a essa escolha.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from pos_processamento import cru, EW, mask, idx, alvo, part, op
from publica_clearml import SIN, BASE, K, GRID, BLACKOUT

E = {c: EW[c].where(mask) for c in SIN}

# --- tres definicoes de "periodo saudavel" ---
jan_falha = pd.Series(False, index=idx)
for t in alvo:
    jan_falha |= (idx >= t - pd.Timedelta(days=7)) & (idx <= t + pd.Timedelta(days=7))

# partidas: janela pos-religamento alem do blackout de 6h ja aplicado na mascara
n24 = int(pd.Timedelta("24h") / pd.Timedelta(GRID))
pos24 = part.rolling(n24, min_periods=1).max().astype(bool)

defs = {
    "A: tudo que a mascara deixa passar": mask,
    "B: mascara, menos +-7d de cada falha": mask & ~jan_falha,
    "C: B, menos 24h pos-religamento": mask & ~jan_falha & ~pos24,
}

print("=" * 100)
print("FRACAO DO TEMPO SAUDAVEL ACIMA DO LIMIAR (= 100 - percentil), no sinal REAL")
print("=" * 100)
for nome, m in defs.items():
    n = int(m.sum())
    print(f"\n  {nome}   (n = {n:,} amostras = {n*2/60/24:.0f} dias)")
    print(f"    {'sinal':>6} {'k*base':>8} {'% acima':>10} {'percentil':>12}")
    for c in SIN:
        v = E[c][m].dropna()
        if not len(v):
            continue
        thr = BASE[c] * K[c]
        acima = 100.0 * (v >= thr).mean()
        print(f"    {c:>6} {thr:8.2f} {acima:9.3f}% {100-acima:11.3f}%")

print("\n" + "=" * 100)
print("CONTROLE CRUZADO: o mesmo numero pela rota em amostra (o que eu afirmei antes)")
print("=" * 100)
print("    t p99,37 · p p99,38 · sp p99,91 · vb p76,05   <- rota em amostra, janela de ajuste")
print("    (se as duas rotas discordarem muito, a afirmacao anterior nao se sustenta)")

print("\n" + "=" * 100)
print("DECOMPOSICAO: quanto do 'acima do limiar' esta dentro de episodio de alarme?")
print("   -- se a maior parte estiver, o tempo saudavel de fato fica abaixo")
print("=" * 100)
from plota_estilo_francisco import alarme
import avalia as AV
al = alarme()
em_alarme = pd.Series(False, index=idx)
for a, b in AV.episodios(al):
    em_alarme.loc[a:b] = True
m = defs["B: mascara, menos +-7d de cada falha"]
print(f"    {'sinal':>6} {'% acima (total)':>16} {'% acima e FORA de alarme':>26} {'percentil efetivo':>19}")
for c in SIN:
    v = E[c][m].dropna()
    thr = BASE[c] * K[c]
    tot = 100.0 * (v >= thr).mean()
    fora = 100.0 * ((v >= thr) & ~em_alarme[v.index]).mean()
    print(f"    {c:>6} {tot:15.3f}% {fora:25.3f}% {100-fora:18.3f}%")
