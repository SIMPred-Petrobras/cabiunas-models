"""Cruza os 20 'episodios de TRIP' do EXP10c (Diego, Secao 18) com o estado real da maquina.

Motivo: o relatorio reporta 25% de cobertura (5/20) nos alarmes de TRIP. Este script
verifica se os 20 episodios sao de fato eventos fisicos distintos. Reproduz o
declustering do relatorio (onsets dos 3 tags TRIP, gap <=30min, OOS >= 2025-07-01) e
marca, para cada episodio, se a maquina estava operando e se houve parada real >=2h.

Saida: eval_predictive_out/cruza_diego_trip.csv
"""
import pandas as pd, numpy as np, pathlib

CACHE = pathlib.Path.home()/".clearml/cache/storage_manager/datasets"
ALARMES = CACHE/"ds_d4c284df665e465d8492afd368837c8f/alarmes_selecionados_turbina_a.csv"
SENSORES = CACHE/"ds_d4c284df665e465d8492afd368837c8f/sensores_2024h2_2025_2026_30s.csv"
TAGS_TRIP = ["PALL_6240340", "TALL_6240325", "PALL_6240309"]
OOS = "2025-07-01"

def episodios_trip():
    al = pd.read_csv(ALARMES)
    al["t"] = pd.to_datetime(al["Data da Ocorrência"], errors="coerce")
    ons = al[al["t"].notna() & al["Status"].astype(str).str.startswith("ACT")]
    ts = sorted(ons.loc[ons["Tag Alarme"].isin(TAGS_TRIP) & (ons["t"] >= OOS), "t"])
    eps, ini, fim, n = [], ts[0], ts[0], 1
    for t in ts[1:]:
        if (t - fim).total_seconds() / 60 <= 30:
            fim, n = t, n + 1
        else:
            eps.append((ini, fim, n)); ini = fim = t; n = 1
    eps.append((ini, fim, n))
    return eps

def estado_maquina():
    r = pd.read_csv(SENSORES, usecols=["data_datetime", "RUNNING_A", "T5_AVG_A"], low_memory=False)
    r["t"] = pd.to_datetime(r["data_datetime"], errors="coerce")
    r = r.dropna(subset=["t"]).set_index("t")
    for c in ("RUNNING_A", "T5_AVG_A"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    d = r[["RUNNING_A", "T5_AVG_A"]].resample("2min").median()
    run = (d["RUNNING_A"] > 0.5).fillna(False)
    seg = (pd.DataFrame({"on": run, "g": run.ne(run.shift()).cumsum()})
             .reset_index().groupby("g").agg(on=("on", "first"), ini=("t", "first"), fim=("t", "last")))
    seg["dur_h"] = (seg["fim"] - seg["ini"]).dt.total_seconds() / 3600
    return run, d["T5_AVG_A"], seg[(~seg["on"]) & (seg["dur_h"] >= 2)]

def main():
    eps = episodios_trip()
    run, t5, paradas = estado_maquina()
    linhas = []
    for a, b, n in eps:
        jan = run.loc[a - pd.Timedelta("30min"): a + pd.Timedelta("30min")]
        estado = "parada" if not jan.any() else ("operando" if jan.all() else "transicao")
        m = (paradas["ini"] >= a - pd.Timedelta("1h")) & (paradas["ini"] <= a + pd.Timedelta("30min"))
        linhas.append(dict(inicio=a, fim=b, onsets=n, estado=estado,
                           t5_c=round(t5.loc[a - pd.Timedelta("10min"): a + pd.Timedelta("10min")].median(), 1),
                           parada_real=bool(m.any()),
                           parada_dur_h=round(paradas[m]["dur_h"].iloc[0], 1) if m.any() else np.nan))
    df = pd.DataFrame(linhas)
    out = pathlib.Path("eval_predictive_out/cruza_diego_trip.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nepisodios declusterizados: {len(df)}   com parada real >=2h: {int(df['parada_real'].sum())}")
    print(f"com a maquina ja parada no instante do alarme: {(df['estado'] == 'parada').sum()}")
    print(f"-> salvo em {out}")

if __name__ == "__main__":
    main()
