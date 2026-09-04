"""Cada tag esta associada a parada REAL alem do acaso? Teste de Poisson.

A armadilha: uma tag que dispara muito vai cair perto de alguma parada por acaso.
Compara o observado com o esperado sob independencia, dado o numero de ativacoes
da tag, a janela [-1h,+30min] e as 64 paradas."""
import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from math import exp, factorial
from verdade import carrega_alarmes
from plota_estilo_francisco import paradas_reais_2h

alarmes = carrega_alarmes(0)
paradas = paradas_reais_2h()
n_par = len(paradas)
span_h = (alarmes.ts.max() - alarmes.ts.min()).total_seconds() / 3600
JAN_H = 1.5  # [-1h, +30min]

def p_poisson_ge(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    acc = sum(exp(-lam) * lam**i / factorial(i) for i in range(k))
    return max(0.0, 1.0 - acc)

linhas = []
for tag, grp in alarmes.groupby("Tag Alarme"):
    ts = pd.DatetimeIndex(grp.ts)
    obs = sum(1 for q in paradas.ini
              if ((ts >= q - pd.Timedelta(hours=1)) & (ts <= q + pd.Timedelta(minutes=30))).any())
    taxa_h = len(grp) / span_h
    p_uma = 1 - exp(-taxa_h * JAN_H)      # P(>=1 alarme numa janela qualquer)
    esp = n_par * p_uma
    linhas.append(dict(tag=tag, nivel=bool(grp.nivel.iloc[0]), n=len(grp),
                       obs=obs, esperado=esp, p=p_poisson_ge(obs, esp),
                       desc=grp["Descrição Alarme"].iloc[0]))
T = pd.DataFrame(linhas)
T = T[T.obs > 0].sort_values("p")
print(f"paradas reais: {n_par}   janela: {JAN_H}h   span: {span_h/24/30:.1f} meses\n")
print(f"{'tag':18s} {'niv':>4} {'n':>5} {'obs':>4} {'esper':>7} {'p':>9}  descricao")
for r in T.itertuples():
    sig = "  ***" if r.p < 0.01 else ("  *" if r.p < 0.05 else "")
    print(f"{r.tag:18s} {str(r.nivel)[:1]:>4} {r.n:5d} {r.obs:4d} {r.esperado:7.2f} {r.p:9.4f}  {r.desc[:44]}{sig}")
