"""PIPELINE UNIFICADA FINAL -- consolida os 4 canais desenvolvidos nesta
investigacao em uma unica decisao (config recomendada de producao).

Canais:
  1. temperatura de mancal isolada (EXP33, OCSVM)   -- ja gateado
  2. vibracao de mancal isolada    (EXP34, OCSVM)   -- ja gateado
  3. pressao de oleo isolada       (EXP38, OCSVM)   -- ja gateado
  4. alarme de processo, SEM modelo (proximidade temporal a catalogo)

Combinacao: votacao >=N canais + filtro de duracao minima (45min) +
refratario de 48h. Nenhum modelo novo e treinado aqui -- reusa os
`is_anom_point` de EXP33/34/38 (ja com todos os portoes de producao:
filtro de duracao minima por canal, rampa de carga, degrau, veto de
sensor congelado) e adiciona o canal 100% estatistico de alarme. A
camada de decisao final (voto + filtro de duracao + refratario)
tambem nao e um modelo -- e uma camada de regras cujos parametros sao
escolhidos por varredura contra a regua rigorosa (nao treinados). O
filtro de duracao na votacao combinada (novo em 2026-09-04) elimina
FP residuais de coincidencia pontual entre canais ja individualmente
filtrados (ex.: temperatura+alarme concordando por so 30s) -- reduz
FP/mes de 6.59 para 2.88 mantendo 8/8.

Roda a varredura de MIN_VOTES em {2, 3} (de 4 canais) para decidir a
config final -- ver print "RESUMO DA VARREDURA" ao final.

Uso:
    PYTHONPATH=. python scripts/pipeline_unificada_final.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from clearml import Task, Dataset

from src.cnn1d_ae.scoring import (
    combine_channels_vote,
    apply_refractory,
    apply_min_duration_filter,
    group_alerts_into_episodes,
    classify_episodes_regua,
    compute_regua_metrics,
    compute_operational_period_days,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs_pipeline_unificada_final")
os.makedirs(OUT_DIR, exist_ok=True)

# --- canais 1, 2, 3: modelos ja treinados e gateados (EXP33/EXP34/EXP38) ---
# retreino completo em 2026-09-03 (ver docs/analise_automl_exp10.md) --
# task ids anteriores: eb46b8afc61c447bb773fe836a504773 / 806178140f38456cbb094c67320b7cec / d1b70f12e55040c9a38fb3cce0703f17
TID_TEMP = "805fbf34f99f4a889dbdcca7185f20a1"
KEY_TEMP_POINT = "mancal_temperatura_isolada/csv/point_anomalies_all.csv"
TID_VIB = "7815d2cf0d07491eb1c949d555cb5de7"
KEY_VIB_POINT = "mancal_vibracao_isolada/csv/point_anomalies_all.csv"
TID_OLEO = "18a61687eb78412ead48c9ce31109b67"
KEY_OLEO_POINT = "oleo_pressao_isolada/csv/point_anomalies_all.csv"

# --- canal 4: alarme de processo, SEM modelo ---
ALARM_CATALOG_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"
ALARM_CATALOG_FILE = "alarmes_selecionados_turbina_a.csv"
ALARM_CHANNEL_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
ALARM_WINDOW_HOURS = 24.0

# --- camada de decisao (regras, sem treino) ---
REFRACTORY_HOURS = 48.0
MIN_VOTES_GRID = [2, 3]

# Filtro de duracao minima aplicado na VOTACAO combinada (antes do
# refratario) -- reduz FP residuais de coincidencia pontual entre
# canais ja filtrados individualmente (ex.: temperatura+alarme por
# 30s). Verificado por varredura (2026-09-04, ver
# docs/analise_automl_exp10.md): plato seguro e 8/8 de 16 a 52min
# (14-15min tem um caso de fronteira instavel -- o TRIP de 11/04/2025,
# que ja e sabido disparar a 0,9h de um religamento com so 2 votos --
# mas volta a 8/8 logo acima); quebra permanente (perde >=1 TRIP) so a
# partir de 55min. 45min fica com margem folgada dos dois lados e
# reduz FP/mes de 6.59 para 2.88 (-56%).
MIN_VOTE_DURATION_MINUTES = 45.0

FALHAS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_francisco_lara", "alarmes_francisco_falhas.csv")


def alarm_channel_bool(index: pd.DatetimeIndex, alarm_times: pd.Series, window_hours: float) -> pd.Series:
    """Canal 4: True se um alarme de `alarm_times` ocorreu nas ultimas
    `window_hours` -- proximidade temporal direta, nenhum modelo."""
    times = np.sort(pd.DatetimeIndex(pd.Series(alarm_times).dropna()).values.astype("datetime64[ns]"))
    t_arr = index.values.astype("datetime64[ns]")
    out = np.zeros(len(t_arr), dtype=bool)
    if len(times) == 0:
        return pd.Series(out, index=index)
    pos = np.searchsorted(times, t_arr, side="right") - 1
    valid = pos >= 0
    dt_hours = (t_arr[valid] - times[pos[valid]]).astype("timedelta64[s]").astype(np.float64) / 3600.0
    out[valid] = dt_hours <= float(window_hours)
    return pd.Series(out, index=index)


def evaluate(decisao_final: pd.Series, op_state: pd.Series, ft: pd.Series):
    df_eval = pd.DataFrame({"is_anom_point": decisao_final.astype(int), "operational_state": op_state})
    dias_vigiados = compute_operational_period_days(df_eval)
    eps = group_alerts_into_episodes(df_eval["is_anom_point"], merge_gap_minutes=120.0)
    cls = classify_episodes_regua(eps, ft, df_eval["operational_state"], 48.0, 48.0, 2.0)
    metrics = compute_regua_metrics(cls, ft, dias_vigiados)
    return cls, metrics


def main() -> None:
    print("baixando is_anom_point dos tres modelos (EXP33 temperatura, EXP34 vibracao, EXP38 óleo)...", flush=True)
    t_temp = Task.get_task(task_id=TID_TEMP)
    p_temp = t_temp.artifacts[KEY_TEMP_POINT].get_local_copy()
    t_vib = Task.get_task(task_id=TID_VIB)
    p_vib = t_vib.artifacts[KEY_VIB_POINT].get_local_copy()
    t_oleo = Task.get_task(task_id=TID_OLEO)
    p_oleo = t_oleo.artifacts[KEY_OLEO_POINT].get_local_copy()

    df_temp = pd.read_csv(p_temp, index_col=0, parse_dates=True, low_memory=False)
    df_vib = pd.read_csv(p_vib, index_col=0, parse_dates=True, low_memory=False)
    df_oleo = pd.read_csv(p_oleo, index_col=0, parse_dates=True, low_memory=False)
    idx = df_temp.index.intersection(df_vib.index).intersection(df_oleo.index)
    print(f"indice comum aos 3 modelos: {len(idx)} amostras", flush=True)

    canal_temp = df_temp.loc[idx, "is_anom_point"].astype(bool)
    canal_vib = df_vib.loc[idx, "is_anom_point"].astype(bool)
    canal_oleo = df_oleo.loc[idx, "is_anom_point"].astype(bool)
    op_state = df_temp.loc[idx, "operational_state"]

    print("carregando catalogo de alarmes (canal 4, sem modelo)...", flush=True)
    root = Dataset.get(dataset_id=ALARM_CATALOG_DATASET_ID).get_local_copy()
    alarm = pd.read_csv(os.path.join(root, ALARM_CATALOG_FILE))
    alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
    alarm["Tag"] = alarm["Tag Alarme"]
    alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])
    alarm_times = alarm.loc[alarm["Tag"].isin(ALARM_CHANNEL_TAGS), "Data da Ocorrencia"]
    canal_alarme = alarm_channel_bool(idx, alarm_times, ALARM_WINDOW_HOURS)

    falhas = pd.read_csv(FALHAS_CSV)
    falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
    failure_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
    ft = failure_times.loc[(failure_times >= idx.min()) & (failure_times <= idx.max())]

    canais = {
        "temperatura": canal_temp,
        "vibracao": canal_vib,
        "oleo_pressao": canal_oleo,
        "alarme_processo": canal_alarme,
    }

    print("\n=== RESUMO DA VARREDURA (min_votes de 4 canais) ===")
    resultados = {}
    for min_votes in MIN_VOTES_GRID:
        voto = combine_channels_vote(canais, min_votes=min_votes)
        df_voto = pd.DataFrame({"is_anom_point": voto.astype(int)}, index=idx)
        voto_filtrado = apply_min_duration_filter(df_voto, MIN_VOTE_DURATION_MINUTES)["is_anom_point"].astype(bool)
        decisao_final = apply_refractory(voto_filtrado, refractory_minutes=REFRACTORY_HOURS * 60.0)
        cls, metrics = evaluate(decisao_final, op_state, ft)
        resultados[min_votes] = (decisao_final, cls, metrics)
        print(f"min_votes={min_votes}: falhas={metrics['falhas_detectadas']}/{metrics['n_falhas_catalogadas']}  "
              f"FP/mes={metrics['falso_positivo_por_mes']:.2f}  inconclusivo={metrics['n_episodios_inconclusivo']}")

    # escolhe a config final: prioriza 8/8, depois menor FP/mes
    candidatos = [(mv, m) for mv, (_, _, m) in resultados.items() if m["falhas_detectadas"] == 8]
    if candidatos:
        min_votes_final = min(candidatos, key=lambda x: x[1]["falso_positivo_por_mes"])[0]
    else:
        min_votes_final = max(resultados.keys(), key=lambda mv: resultados[mv][2]["falhas_detectadas"])
    print(f"\n>>> config final escolhida: min_votes={min_votes_final} de 4 canais (8/8 com menor FP/mes)")

    decisao_final, cls, metrics = resultados[min_votes_final]
    print("\n=== RESULTADO FINAL (pipeline unificada) ===")
    print(f"falhas detectadas: {metrics['falhas_detectadas']}/{metrics['n_falhas_catalogadas']}")
    print(f"FP/mes: {metrics['falso_positivo_por_mes']:.2f}")
    print(f"episodios inconclusivo: {metrics['n_episodios_inconclusivo']}")

    detectadas = sorted(pd.Timestamp(d) for d in cls.loc[cls["classe"] == "deteccao", "falha_associada"].dropna().unique())
    faltantes = [pd.Timestamp(f) for f in ft if not any(abs((pd.Timestamp(f) - d).total_seconds()) < 60 for d in detectadas)]
    print("\nfalhas detectadas:")
    for d in detectadas:
        print(" ", d)
    print("falhas nao detectadas:", faltantes)

    df_final = pd.DataFrame({
        "canal_temperatura": canal_temp.astype(int),
        "canal_vibracao": canal_vib.astype(int),
        "canal_oleo_pressao": canal_oleo.astype(int),
        "canal_alarme_processo": canal_alarme.astype(int),
        "is_anom_point": decisao_final.astype(int),
        "operational_state": op_state,
    }, index=idx)

    df_final.to_csv(os.path.join(OUT_DIR, "point_anomalies_final.csv"))
    cls.to_csv(os.path.join(OUT_DIR, "episodios_classificados.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            **metrics,
            "config": {
                "canais": ["temperatura (EXP33, OCSVM)", "vibracao (EXP34, OCSVM)",
                           "oleo_pressao (EXP38, OCSVM)", "alarme_processo (sem modelo)"],
                "alarm_channel_tags": ALARM_CHANNEL_TAGS,
                "alarm_window_hours": ALARM_WINDOW_HOURS,
                "min_votes": min_votes_final,
                "n_canais": 4,
                "min_vote_duration_minutes": MIN_VOTE_DURATION_MINUTES,
                "refractory_hours": REFRACTORY_HOURS,
                "temp_task_id": TID_TEMP,
                "vib_task_id": TID_VIB,
                "oleo_task_id": TID_OLEO,
            },
            "varredura_min_votes": {
                str(mv): {"falhas_detectadas": m["falhas_detectadas"], "falso_positivo_por_mes": m["falso_positivo_por_mes"]}
                for mv, (_, _, m) in resultados.items()
            },
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nartefatos salvos em {OUT_DIR}")


if __name__ == "__main__":
    main()
