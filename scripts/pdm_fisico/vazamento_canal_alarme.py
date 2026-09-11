"""O canal 4 do Diego usa 5 tags de alarme da PLANTA como ENTRADA para prever
trips da planta. Ele proprio excluiu essas 5 tags da corroboracao do Passo 3
"para nao contar a mesma evidencia duas vezes" -- mas nao fez a mesma pergunta
das DETECCOES.

A pergunta: essas 5 tags fazem parte da propria cascata do trip? Se sim, o
canal 4 esta lendo a resposta, e a antecedencia media de 23,8 h nao e predicao.

Mesma classe de erro que a nossa [[vazamento-referencia-rolante]]: la a
referencia rolante absorvia as falhas e o 9/9 virava 8/9.
"""
import pandas as pd, numpy as np

TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
BASE = "/home/thallys/Documents/projeto-petrobras/wt-chico/dados_locais/alarmes_mapeados_colunas.csv"

a = pd.read_csv(BASE)
print("colunas:", list(a.columns)[:8])
col_t = next(c for c in a.columns if "time" in c.lower() or "data" in c.lower()
             or "stamp" in c.lower())
col_tag = next(c for c in a.columns if "tag" in c.lower() or "col" in c.lower())
a[col_t] = pd.to_datetime(a[col_t], utc=True, errors="coerce")
a = a.dropna(subset=[col_t])
print(f"catalogo: {len(a):,} linhas | {a[col_tag].nunique()} tags | "
      f"{a[col_t].min():%d/%m/%Y} a {a[col_t].max():%d/%m/%Y}\n")

fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = [t for t in fal if t >= pd.Timestamp("2024-02-01", tz="UTC")]

# quais das 5 tags existem no catalogo, por casamento flexivel
presentes = {}
todas = a[col_tag].astype(str).unique()
for t in TAGS:
    nucleo = t.replace("_AL", "").replace("PAL_", "").replace("PAH_", "") \
              .replace("PDAL_", "").replace("PI_", "").replace("TC382_", "")
    m = [u for u in todas if nucleo in u or t in u]
    presentes[t] = m
    print(f"  {t:<16} -> {len(m)} casamento(s): {m[:3]}")

print("\nQUANDO AS 5 TAGS DO CANAL 4 DISPARAM EM RELACAO AOS 8 TRIPS")
print("=" * 88)
print(f"{'trip':>12} {'tag mais proxima':>18} {'quando':>26} {'delta ate o trip':>18}")
print("-" * 88)
alvos_tags = [u for t in TAGS for u in presentes[t]]
sub = a[a[col_tag].astype(str).isin(alvos_tags)]
n_dentro_1h = n_dentro_24h = 0
for t in alvo:
    j = sub[(sub[col_t] >= t - pd.Timedelta("24h")) & (sub[col_t] <= t + pd.Timedelta("1h"))]
    if len(j) == 0:
        print(f"{t:%d/%m/%Y} {'--':>18} {'nenhum em [-24h,+1h]':>26} {'--':>18}")
        continue
    j = j.assign(d=(t - j[col_t]).dt.total_seconds() / 3600).sort_values("d")
    prox = j.iloc[(j["d"].abs()).argmin()]
    d = prox["d"]
    n_dentro_1h += bool(abs(d) <= 1.0); n_dentro_24h += 1
    print(f"{t:%d/%m/%Y} {str(prox[col_tag])[:18]:>18} "
          f"{prox[col_t]:%d/%m/%Y %H:%M} {d:15.2f} h")
print("-" * 88)
print(f"  trips com uma das 5 tags em [-24h, +1h] : {n_dentro_24h} de {len(alvo)}")
print(f"  trips com uma das 5 tags a menos de 1 h : {n_dentro_1h} de {len(alvo)}")
print("\n  -> tag que dispara junto com o trip nao e precursor, e a propria cascata;")
print("     o canal 4 entao vota 'sim' porque o trip ja comecou.")
