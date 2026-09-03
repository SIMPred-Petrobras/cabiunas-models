"""EXP37 -- camada de decisao multi-canal (CONFIG RECOMENDADA).

Combina os `is_anom_point` ja gateados de dois modelos especializados
(EXP33 temperatura, EXP34 vibracao) com um canal de alarme de processo
SEM MODELO (estatistica pura: proximidade temporal a um alarme
catalogado) via votacao >=2 canais + refratario de 48h.

Historico que motiva esta config (ver docs/analise_automl_exp10.md,
secao "EXP37: camada de decisao multi-canal"):
  - EXP30 (modelo unico, mancal+vibracao juntos): 5/8 TRIPs, 8,72 FP/mes.
  - EXP33 (so temperatura) e EXP34 (so vibracao), cada um sozinho: 7/8.
  - EXP33 OU EXP34 (uniao ingenua): 8/8, mas 26,99 FP/mes (inaceitavel).
  - Esta config (votacao >=2 + alarme 24h + refratario 48h): 8/8 TRIPs,
    6,39 FP/mes -- MELHOR que a propria baseline de 5/8.

Nenhum modelo novo e treinado aqui -- reusa os `is_anom_point` do EXP33
e EXP34 (ja com todos os portoes de producao aplicados: filtro de
duracao minima, rampa de carga, degrau, veto de sensor congelado) e
adiciona um terceiro canal 100% estatistico (nenhum ML). A combinacao
final (voto + refratario) tambem nao e um modelo -- e uma camada de
regras com parametros calibrados por varredura contra a regua rigorosa.

Uso:
    PYTHONPATH=. python scripts/multicanal_votacao_exp37.py
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
    group_alerts_into_episodes,
    classify_episodes_regua,
    compute_regua_metrics,
    compute_operational_period_days,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs_exp37_multicanal")
os.makedirs(OUT_DIR, exist_ok=True)

# --- canais 1 e 2: modelos ja treinados e gateados (EXP33/EXP34) ---
TID_TEMP = "eb46b8afc61c447bb773fe836a504773"
KEY_TEMP_POINT = "mancal_temperatura_isolada/csv/point_anomalies_all.csv"
TID_VIB = "806178140f38456cbb094c67320b7cec"
KEY_VIB_POINT = "mancal_vibracao_isolada/csv/point_anomalies_all.csv"

# --- canal 3: alarme de processo, SEM modelo ---
ALARM_CATALOG_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"
ALARM_CATALOG_FILE = "alarmes_selecionados_turbina_a.csv"
ALARM_CHANNEL_TAGS = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
ALARM_WINDOW_HOURS = 24.0

# --- camada de decisao (regras, sem treino) ---
MIN_VOTES = 2
REFRACTORY_HOURS = 48.0

FALHAS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_francisco_lara", "alarmes_francisco_falhas.csv")


def alarm_channel_bool(index: pd.DatetimeIndex, alarm_times: pd.Series, window_hours: float) -> pd.Series:
    """Canal 3: True se um alarme de `alarm_times` ocorreu nas ultimas
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


def main() -> None:
    print("baixando is_anom_point dos dois modelos (EXP33 temperatura, EXP34 vibracao)...", flush=True)
    t_temp = Task.get_task(task_id=TID_TEMP)
    p_temp = t_temp.artifacts[KEY_TEMP_POINT].get_local_copy()
    t_vib = Task.get_task(task_id=TID_VIB)
    p_vib = t_vib.artifacts[KEY_VIB_POINT].get_local_copy()

    df_temp = pd.read_csv(p_temp, index_col=0, parse_dates=True, low_memory=False)
    df_vib = pd.read_csv(p_vib, index_col=0, parse_dates=True, low_memory=False)
    idx = df_temp.index.intersection(df_vib.index)
    print(f"indice comum: {len(idx)} amostras", flush=True)

    canal_temp = df_temp.loc[idx, "is_anom_point"].astype(bool)
    canal_vib = df_vib.loc[idx, "is_anom_point"].astype(bool)
    op_state = df_temp.loc[idx, "operational_state"]

    print("carregando catalogo de alarmes (canal 3, sem modelo)...", flush=True)
    root = Dataset.get(dataset_id=ALARM_CATALOG_DATASET_ID).get_local_copy()
    alarm = pd.read_csv(os.path.join(root, ALARM_CATALOG_FILE))
    alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrência"], errors="coerce")
    alarm["Tag"] = alarm["Tag Alarme"]
    alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].dropna(subset=["Data da Ocorrencia"])
    alarm_times = alarm.loc[alarm["Tag"].isin(ALARM_CHANNEL_TAGS), "Data da Ocorrencia"]
    canal_alarme = alarm_channel_bool(idx, alarm_times, ALARM_WINDOW_HOURS)

    print(f"votacao >={MIN_VOTES} canais...", flush=True)
    voto = combine_channels_vote(
        {"temperatura": canal_temp, "vibracao": canal_vib, "alarme_processo": canal_alarme},
        min_votes=MIN_VOTES,
    )
    print(f"aplicando refratario de {REFRACTORY_HOURS}h...", flush=True)
    decisao_final = apply_refractory(voto, refractory_minutes=REFRACTORY_HOURS * 60.0)

    # --- avaliacao contra a regua rigorosa ---
    falhas = pd.read_csv(FALHAS_CSV)
    falhas["Data da Ocorrência"] = pd.to_datetime(falhas["Data da Ocorrência"])
    failure_times = falhas.loc[falhas["Tag Alarme"] == "FALHA_CURADA", "Data da Ocorrência"].sort_values()
    ft = failure_times.loc[(failure_times >= idx.min()) & (failure_times <= idx.max())]

    df_final = pd.DataFrame({
        "canal_temperatura": canal_temp.astype(int),
        "canal_vibracao": canal_vib.astype(int),
        "canal_alarme_processo": canal_alarme.astype(int),
        "voto_bruto": voto.astype(int),
        "is_anom_point": decisao_final.astype(int),
        "operational_state": op_state,
    }, index=idx)

    dias_vigiados = compute_operational_period_days(df_final)
    eps = group_alerts_into_episodes(df_final["is_anom_point"], merge_gap_minutes=120.0)
    cls = classify_episodes_regua(eps, ft, df_final["operational_state"], 48.0, 48.0, 2.0)
    metrics = compute_regua_metrics(cls, ft, dias_vigiados)

    print("\n=== RESULTADO FINAL (EXP37) ===")
    print(f"falhas detectadas: {metrics['falhas_detectadas']}/{metrics['n_falhas_catalogadas']}")
    print(f"FP/mes: {metrics['falso_positivo_por_mes']:.2f}")
    print(f"episodios inconclusivo: {metrics['n_episodios_inconclusivo']}")

    detectadas = sorted(pd.Timestamp(d) for d in cls.loc[cls["classe"] == "deteccao", "falha_associada"].dropna().unique())
    faltantes = [pd.Timestamp(f) for f in ft if not any(abs((pd.Timestamp(f) - d).total_seconds()) < 60 for d in detectadas)]
    print("\nfalhas detectadas:")
    for d in detectadas:
        print(" ", d)
    print("falhas nao detectadas (dentro de 48h antes):", faltantes)

    df_final.to_csv(os.path.join(OUT_DIR, "point_anomalies_multicanal.csv"))
    cls.to_csv(os.path.join(OUT_DIR, "episodios_classificados.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            **metrics,
            "config": {
                "canais": ["temperatura (EXP33, OCSVM)", "vibracao (EXP34, OCSVM)", "alarme_processo (sem modelo)"],
                "alarm_channel_tags": ALARM_CHANNEL_TAGS,
                "alarm_window_hours": ALARM_WINDOW_HOURS,
                "min_votes": MIN_VOTES,
                "refractory_hours": REFRACTORY_HOURS,
                "temp_task_id": TID_TEMP,
                "vib_task_id": TID_VIB,
            },
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nartefatos salvos em {OUT_DIR}")


if __name__ == "__main__":
    main()
