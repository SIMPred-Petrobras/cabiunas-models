"""Inventario das tags de alarme: quem e protecao de maquina, quem e processo/suprimento,
e quem e nivel-1 (alerta) contra nivel-2 (trip).

MOTIVACAO. A anatomia achou que a parada de 02/04/2025 -- que hoje deixa o episodio de
31/03/2025 na caixa "neutro" -- foi precedida em 5 min por PAL_6240339 "Pressao Bx. Header
Oleo Lub.". Esse e o MESMO modo fisico dos trips de 04/11/2025 e 26/02/2026
("TRIP-Pr.Mt.Bx.Oleo Lub."), so que no primeiro estagio do alarme (PAL = alarme baixo)
em vez do segundo (TRIP = intertravamento).

Se o alvo so conta o estagio 2, o detector e penalizado justamente quando funciona:
avisou, o operador interveio, a maquina parou controlada e nunca chegou ao trip.
"""
from __future__ import annotations
import sys
import pandas as pd
sys.path.insert(0, ".")

from plota_estilo_francisco import paradas_reais_2h
from verdade import carrega_alarmes

alarmes = carrega_alarmes(0)
paradas = paradas_reais_2h()

print("=" * 104)
print("INVENTARIO DE TAGS DE ALARME (ACT)")
print("=" * 104)
inv = (alarmes.groupby(["Tag Alarme", "Descrição Alarme", "nivel"])
       .size().reset_index(name="n").sort_values("n", ascending=False))
print(f"{'tag':20s} {'n':>6} {'nivel':>6}  descricao")
for _, r in inv.iterrows():
    print(f"{r['Tag Alarme']:20s} {r['n']:6d} {str(r['nivel']):>6}  {r['Descrição Alarme']}")

print("\n" + "=" * 104)
print("QUANTAS PARADAS REAIS (>=2h) CADA TAG ACOMPANHA, em [-1h, +30min] da queda")
print("=" * 104)
print(f"total de paradas reais >=2h: {len(paradas)}")
linhas = []
for tag, grp in alarmes.groupby("Tag Alarme"):
    ts = pd.DatetimeIndex(grp.ts)
    n = 0
    for q in paradas.ini:
        if ((ts >= q - pd.Timedelta(hours=1)) & (ts <= q + pd.Timedelta(minutes=30))).any():
            n += 1
    linhas.append(dict(tag=tag, nivel=bool(grp.nivel.iloc[0]), n_alarmes=len(grp),
                       paradas_acompanhadas=n,
                       precisao=n / len(grp) if len(grp) else 0.0))
T = pd.DataFrame(linhas).sort_values("paradas_acompanhadas", ascending=False)
print(f"{'tag':20s} {'nivel':>6} {'n_alarm':>8} {'paradas':>8} {'precisao':>9}")
for r in T.itertuples():
    print(f"{r.tag:20s} {str(r.nivel):>6} {r.n_alarmes:8d} {r.paradas_acompanhadas:8d} {r.precisao:9.3f}")

print("\n" + "=" * 104)
print("AS 62 PARADAS REAIS, COM SEUS ALARMES -- quantas sao 'limpas' (nenhum alarme)?")
print("=" * 104)
n_lim = n_niv = n_out = 0
for q in paradas.ini:
    jan = alarmes[(alarmes.ts >= q - pd.Timedelta(hours=1)) & (alarmes.ts <= q + pd.Timedelta(minutes=30))]
    if len(jan) == 0:
        n_lim += 1
    elif jan.nivel.any():
        n_niv += 1
    else:
        n_out += 1
print(f"  sem alarme nenhum (parada limpa/programada): {n_lim}")
print(f"  com alarme de NIVEL (o nosso alvo de 8+1):   {n_niv}")
print(f"  so com alarme fora do regex de nivel:        {n_out}")

print("\n" + "=" * 104)
print("AS PARADAS 'SO COM ALARME FORA DO NIVEL' -- candidatas a alvo ampliado")
print("=" * 104)
for q, dur in zip(paradas.ini, paradas.dur_h):
    jan = alarmes[(alarmes.ts >= q - pd.Timedelta(hours=1)) & (alarmes.ts <= q + pd.Timedelta(minutes=30))]
    if len(jan) and not jan.nivel.any():
        tags = sorted(set(jan["Descrição Alarme"]))
        print(f"  {q:%d/%m/%Y %H:%M}  dur={dur:7.1f}h  {' | '.join(tags)}")
