"""PAL_6240339 e trip ou nao? Tres verificacoes independentes."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from verdade import carrega_alarmes
from plota_estilo_francisco import paradas_reais_2h
from pos_processamento import g, op, idx

alarmes = carrega_alarmes(0)
pd.set_option("display.width", 200)

print("=" * 100)
print("1. SEMANTICA DAS TAGS -- o que a propria base chama de TRIP")
print("=" * 100)
oleo = alarmes[alarmes["Descrição Alarme"].str.contains("Óleo|Oleo", case=False, na=False)]
inv = oleo.groupby(["Tag Alarme", "Descrição Alarme", "nivel"]).size().reset_index(name="n")
for _, r in inv.sort_values("n", ascending=False).iterrows():
    print(f"  {r['Tag Alarme']:18s} n={r['n']:4d}  nivel={str(r['nivel']):5s}  {r['Descrição Alarme']}")

print("\n" + "=" * 100)
print("2. NAS DUAS PARADAS NOVAS: algum tag de TRIP disparou? (janela larga +-12h)")
print("=" * 100)
for q in [pd.Timestamp("2025-04-02 05:18", tz="UTC"), pd.Timestamp("2025-11-24 09:22", tz="UTC")]:
    print(f"\n--- parada {q:%d/%m/%Y %H:%M} ---")
    jan = alarmes[(alarmes.ts >= q - pd.Timedelta(hours=12)) & (alarmes.ts <= q + pd.Timedelta(hours=12))]
    if not len(jan):
        print("    nenhum alarme em +-12h")
    for _, r in jan.iterrows():
        dt = (r.ts - q).total_seconds() / 3600
        print(f"    {dt:+7.2f}h  {r['Tag Alarme']:18s} nivel={str(r.nivel):5s} {r['Descrição Alarme']}")

print("\n" + "=" * 100)
print("3. A PARADA FOI ABRUPTA (trip) OU RAMPADA (parada controlada)?")
print("   compara com um trip conhecido (04/11/2025) e uma parada limpa (21/01/2025)")
print("=" * 100)
casos = [("02/04/2025 NOVO", "2025-04-02 05:18"),
         ("24/11/2025 NOVO", "2025-11-24 09:22"),
         ("04/11/2025 TRIP conhecido", "2025-11-04 06:22"),
         ("21/01/2025 parada limpa", "2025-01-21 08:34")]
for nome, ts in casos:
    q = pd.Timestamp(ts, tz="UTC")
    jan = g.loc[q - pd.Timedelta(hours=3):q + pd.Timedelta(minutes=30), ["T5_AVG_A", "RUNNING_A"]]
    t5 = jan["T5_AVG_A"].dropna()
    if len(t5) < 5:
        print(f"\n  {nome}: sem dado de T5")
        continue
    # taxa de queda de T5 nos 30 min antes da parada
    ante = t5.loc[:q]
    if len(ante) >= 16:
        d30 = ante.iloc[-1] - ante.iloc[-16]   # 15 passos de 2min = 30min
        d180 = ante.iloc[-1] - ante.iloc[0] if len(ante) >= 90 else float("nan")
    else:
        d30 = d180 = float("nan")
    print(f"\n  {nome}  (parada {q:%d/%m %H:%M})")
    print(f"    T5 3h antes = {ante.iloc[0]:6.1f} C   T5 30min antes = {ante.iloc[-16]:6.1f} C   "
          f"T5 na parada = {ante.iloc[-1]:6.1f} C")
    print(f"    variacao ultimos 30min = {d30:+7.1f} C     ultimas 3h = {d180:+7.1f} C")
