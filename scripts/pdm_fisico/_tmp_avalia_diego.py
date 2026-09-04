"""Duas ideias do relatorio EXP28/29 (Diego) medidas no NOSSO detector.

TESTE 1 -- o controle negativo que falta na Secao 5 do relatorio dele.
Ele cruza cada episodio "amarelo" contra o catalogo completo de 47 tags numa janela
de +-24h e conclui que 88,1% do que parecia FP e sinal real de outro alarme. O que o
relatorio NAO tem e o controle: que fracao de janelas ALEATORIAS tambem seria
"explicada" pelo mesmo criterio? Com ~3,4 alarmes/dia, uma janela de +-24h (48h de
largura) espera ~6,8 alarmes -- P(>=1) ~ 99,9%. Se o controle der ~99%, o 88,1% nao
mede enriquecimento, mede a densidade do catalogo.

TESTE 2 -- veto de sensor congelado (o item que ele diz ter trazido do Francisco).
Sensor travado num valor constante gera residuo de PCA crescente sem evento fisico.
Se os nossos FP coincidem com sensor congelado e os TP nao, isso transfere.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")

from plota_estilo_francisco import alarme, paradas_reais_2h, classifica_regra_c
from pos_processamento import idx, op, mask, g
from publica_clearml import VIBRATION_TAGS
import avalia as AV
from verdade import carrega_alarmes

rng = np.random.default_rng(20260901)
alarmes = carrega_alarmes(0)
alarmes = alarmes[(alarmes.ts >= idx.min()) & (alarmes.ts <= idx.max())]
ts_todos = pd.DatetimeIndex(sorted(alarmes.ts))
classif = classifica_regra_c(AV.episodios(alarme()), paradas_reais_2h())

print("=" * 100)
print("TESTE 1 -- CONTROLE NEGATIVO DA REGRA '+-24h CONTRA O CATALOGO COMPLETO'")
print("=" * 100)
print(f"catalogo: {alarmes['Tag Alarme'].nunique()} tags distintas, {len(alarmes)} ativacoes ACT")
dias = (idx.max() - idx.min()).total_seconds() / 86400
print(f"densidade: {len(alarmes)/dias:.2f} alarmes/dia sobre {dias/30:.1f} meses\n")

# instantes candidatos para sorteio: so quando a maquina esta operando (mesmo
# universo em que um episodio poderia nascer)
cand = idx[mask.to_numpy()]
N = 2000
sorteio = pd.DatetimeIndex(rng.choice(cand, size=N, replace=False))

print(f"{'janela':>9} {'nossos FP':>12} {'nossos TP':>12} {'ALEATORIO':>12}   leitura")
for jan_h in (1, 2, 4, 6, 12, 24):
    J = pd.Timedelta(hours=jan_h)
    def frac(instantes):
        n = sum(1 for t in instantes
                if ts_todos.searchsorted(t + J) > ts_todos.searchsorted(t - J))
        return n / len(instantes)
    f_fp = frac([a for a, b, c, l in classif if c == "FP"])
    f_tp = frac([a for a, b, c, l in classif if c == "TP"])
    f_rd = frac(sorteio)
    enr = (f_fp / f_rd) if f_rd > 0 else float("nan")
    nota = "informativo" if enr > 1.5 else ("SEM enriquecimento" if f_rd > 0.5 else "")
    print(f"  +-{jan_h:2d}h   {100*f_fp:10.0f}% {100*f_tp:11.0f}% {100*f_rd:11.0f}%   "
          f"enriquecimento {enr:.2f}x  {nota}")

print("\n" + "=" * 100)
print("TESTE 2 -- VETO DE SENSOR CONGELADO (>=30 min no mesmo valor)")
print("=" * 100)
cols_t = [c for c in g.columns if c.startswith("TC382") or c.startswith("T5_") or c.startswith("TI_")]
cols_v = [c for c in VIBRATION_TAGS if c in g.columns]
cols_p = [c for c in g.columns if c.startswith("PI_") or c.startswith("PDI") or c.startswith("PT")]
alvo_cols = cols_t + cols_v + cols_p
print(f"sensores checados: {len(alvo_cols)}  ({len(cols_t)} temp, {len(cols_v)} vib, {len(cols_p)} press)")

N_CONG = 15  # 15 amostras de 2 min = 30 min
cong = pd.DataFrame(index=idx)
for c in alvo_cols:
    v = g[c]
    igual = (v.diff() == 0)
    cong[c] = igual.rolling(N_CONG, min_periods=N_CONG).sum() >= N_CONG
n_cong = cong.sum(axis=1)
print(f"instantes com >=1 sensor congelado: {100*(n_cong>0)[mask].mean():.1f}% do tempo mascarado\n")

print(f"{'inicio':>16} {'classe':>7} {'dur_h':>7} {'% do episodio com sensor congelado':>36}  sensores")
for a, b, c, lead in classif:
    jan = (idx >= a) & (idx <= b)
    sub = cong.loc[jan]
    frac_cong = (sub.sum(axis=1) > 0).mean()
    quais = sorted(sub.columns[sub.any()].tolist())
    print(f"{a:%d/%m/%Y %H:%M} {c:>7} {(b-a).total_seconds()/3600:6.1f}h {100*frac_cong:34.0f}%  "
          f"{','.join(quais[:4])}{'...' if len(quais) > 4 else ''}")

print()
for classe in ("TP", "FP", "NEUTRO"):
    fr = []
    for a, b, c, lead in classif:
        if c != classe:
            continue
        jan = (idx >= a) & (idx <= b)
        fr.append((cong.loc[jan].sum(axis=1) > 0).mean())
    print(f"  {classe:>7}: fracao media do episodio com sensor congelado = {100*np.mean(fr):.1f}%  "
          f"(min {100*min(fr):.0f}%, max {100*max(fr):.0f}%)")
