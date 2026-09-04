import numpy as np, pandas as pd
from pos_processamento import idx, op

# metodo INDEPENDENTE: run-length encoding via groupby(cumsum das trocas), sem loop manual
g_id = (op != op.shift()).cumsum()
runs = pd.DataFrame({"tempo": idx, "op": op.values, "grupo": g_id.values})
resumo = runs.groupby("grupo").agg(ini=("tempo","first"), fim=("tempo","last"), op=("op","first"))
resumo["dur_h"] = (resumo.fim - resumo.ini).dt.total_seconds()/3600 + (2/60)  # +1 grade de 2min

paradas = resumo[(resumo.op == False) & (resumo.dur_h >= 2.0)]
print(f"paradas reais (>=2h), metodo independente (groupby): {len(paradas)}")

# compara com trips_completos.csv (fonte externa, ja existia antes desta sessao)
t = pd.read_csv("trips_completos.csv", parse_dates=["ini"])
reais_externo = t[t.parada_real == True]
print(f"\ntrips_completos.csv (fonte externa, Nov/25-Mar/26): {len(reais_externo)} paradas_real=True")
for r in reais_externo.itertuples():
    # procura essa mesma parada na nossa lista independente
    match = paradas[(paradas.ini - r.ini).abs() < pd.Timedelta(minutes=5)]
    if len(match):
        m = match.iloc[0]
        print(f"  externo: {r.ini}  dur={r.dur_h:.2f}h   <-> nosso: {m.ini}  dur={m.dur_h:.2f}h  "
              f"{'OK' if abs(m.dur_h - r.dur_h) < 0.1 else 'DIVERGE!'}")
    else:
        print(f"  externo: {r.ini}  dur={r.dur_h:.2f}h   <-> NAO ACHADO na nossa lista!")

print(f"\n--- as 6 paradas usadas na reclassificacao NEUTRO ---")
alvo_check = [pd.Timestamp("2025-01-08 14:58", tz="UTC"), pd.Timestamp("2025-01-21 08:34", tz="UTC"),
             pd.Timestamp("2025-04-02 05:18", tz="UTC"), pd.Timestamp("2025-06-18 14:24", tz="UTC"),
             pd.Timestamp("2025-10-21 05:24", tz="UTC"), pd.Timestamp("2026-04-04 08:56", tz="UTC")]
for t0 in alvo_check:
    match = paradas[(paradas.ini - t0).abs() < pd.Timedelta(minutes=5)]
    if len(match):
        m = match.iloc[0]
        print(f"  {t0}  -> confirmado, dur={m.dur_h:.2f}h")
    else:
        print(f"  {t0}  -> NAO CONFIRMADO no metodo independente!")
