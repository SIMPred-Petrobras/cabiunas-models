"""Retreino walk-forward mensal do EXP10c -- o modelo (ocsvm) ajuda mais
se for re-treinado todo mes com a janela de dados normais mais recente,
em vez do corte unico atual (treina uma vez antes de 2025-07-01, aplica
congelado nos 10 meses seguintes)?

MOTIVACAO: ver docs/analise_automl_exp10.md, secao "Diagnostico de
deriva temporal". Dois diagnosticos baratos (sem retreinar nada) nao
encontraram evidencia de degradacao no horizonte OOS atual (10 meses) --
nem no FP mensal, nem na distribuicao bruta das features. Mesmo sem
sintoma observado, o teste empirico direto (retreinar de verdade e medir)
e barato o suficiente pra vir na frente de qualquer suposicao.

ESCOPO: isola SO a variavel "cadencia de retreino do modelo" (ocsvm +
normalizacao + limiar de percentil, todos recomputados por mes, janela
EXPANSIVA -- cada mes usa TODO o historico normal anterior, nao so uma
janela movel fixa). Portoes de rampa/volatilidade, mascara operacional e
os limites de clipping de outlier ficam FIXOS nos mesmos valores do
EXP10c de producao -- re-selecionar esses por mes reabriria a mesma
questao de vies de selecao ja tratada em loeo_exp10c_portoes.py, fora do
escopo deste teste.

METODO:
1. Pre-processamento (build_group_dataframe, clip_outliers, mascara
   operacional, series de portao) calculado uma unica vez -- sao
   operacoes globais, independentes de qual mes esta sendo avaliado.
2. Para cada um dos 10 meses do periodo OOS (2025-07 a 2026-04):
   a. treina do zero em TODO o normal disponivel antes do inicio do mes
      (janela expansiva) -- normalizacao (center/scale) tambem
      recomputada nesse recorte.
   b. threshold no mesmo percentil (99.9) do config de referencia.
   c. score SO dos pontos daquele mes (nao a serie inteira -- mais
      barato, e so o que entra na avaliacao daquele mes).
   d. aplica mascara operacional + portoes fixos (ramp_max/vol_threshold
      da producao).
   e. mede hit/miss dos alarmes daquele mes e normal_alert_rate daquele
      mes.
3. Roda tambem o modelo CONGELADO (o mesmo do EXP10c: treinado uma vez
   antes do corte) avaliado mes a mes pela MESMA rotina, para comparacao
   pareada (mesma metodologia de agregacao dos dois lados).
4. Agrega os 10 meses de cada abordagem (hit_rate pooled, normal_alert_rate
   pooled) e compara.

Uso:
    PYTHONPATH=. python scripts/walkforward_exp10c_retrain_mensal.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from clearml import Task

from src.cnn1d_ae.config import PipelineConfig, update_cfg_from_dict
from src.cnn1d_ae.io import load_data
from src.cnn1d_ae.model import setup_gpu
from src.cnn1d_ae.preprocess import (
    build_sensor_dataframe,
    build_group_dataframe,
    build_exclusion_mask,
    clip_outliers,
    select_feature_columns,
    THERMAL_ARRAY_SPREAD_COL,
)
from src.cnn1d_ae.scoring import (
    build_operational_state,
    compute_load_ramp_gate,
    compute_volatility_index,
)
from src.cnn1d_ae.automl_models import fit_ocsvm, ocsvm_error

CONFIG_PATH = "configs/calibracao_v4_eq/test_grupo_exp10c_portao_volatilidade.json"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::walkforward_exp10c_retrain_mensal",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[WF] Carregando dados (io.load_data, mesma rotina do pipeline padrao)...", flush=True)
df_alarm, df_feat, df_raw, _time_report = load_data(cfg)

group = cfg.SENSOR_GROUPS[0]
group_name = group["name"]
sensors = list(group["sensors"])
target_sensor = group.get("target_sensor")
eval_sensors = list(group.get("eval_sensors") or sensors)

df_use, long_gap_mask = build_group_dataframe(cfg, df_feat, df_raw, sensors)
valid_sensors = [s for s in sensors if float(df_use[s].std()) >= cfg.MIN_STD]
sensors = valid_sensors
if target_sensor and target_sensor not in sensors:
    target_sensor = None

feature_cols = select_feature_columns(cfg, df_use, sensors)
if cfg.ENABLE_THERMAL_ARRAY_SPREAD and THERMAL_ARRAY_SPREAD_COL in df_use.columns:
    feature_cols += select_feature_columns(cfg, df_use, [THERMAL_ARRAY_SPREAD_COL])
df_use = df_use[feature_cols]

# --- estado operacional (fixo -- fora do escopo deste teste) ---
ref_sensor = cfg.OPERATIONAL_REF_SENSOR
if ref_sensor and ref_sensor not in sensors:
    df_ref, _ = build_sensor_dataframe(cfg, df_feat, df_raw, ref_sensor)
    ref_series = df_ref[ref_sensor]
else:
    ref_col = ref_sensor if (ref_sensor and ref_sensor in sensors) else sensors[0]
    ref_series = df_use[ref_col]
secondary_series = None
if cfg.OFF_TARGET_ABS_THRESHOLD is not None and target_sensor and target_sensor in df_use.columns:
    secondary_series = df_use[target_sensor]
state = build_operational_state(
    index=df_use.index, sensor_series=ref_series,
    off_value_quantile=cfg.OFF_VALUE_QUANTILE, off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
    off_long_min_hours=cfg.OFF_LONG_MIN_HOURS, transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
    transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
    secondary_series=secondary_series, secondary_off_abs_threshold=cfg.OFF_TARGET_ABS_THRESHOLD,
)

# --- portoes (fixos nos valores de producao -- fora do escopo deste teste) ---
load_gate_series = df_use[cfg.LOAD_GATE_SENSOR]
ramp_gate, _ = compute_load_ramp_gate(
    load_gate_series, cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES, cfg.LOAD_GATE_WINDOW_MINUTES
)
volatility_index = compute_volatility_index(df_use[cfg.VOLATILITY_GATE_SENSORS], cfg.VOLATILITY_GATE_WINDOW_MINUTES)

# --- alarmes ---
if "Tag" in df_alarm.columns:
    df_alarm_group = df_alarm.loc[df_alarm["Tag"].isin(eval_sensors)].copy()
else:
    df_alarm_group = df_alarm.copy()
df_alarm_group = df_alarm_group.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")
alarm_times = df_alarm_group["Data da Ocorrencia"]

exclude_alarm = build_exclusion_mask(df_use.index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
exclude = exclude_alarm.copy()
long_gap_mask = long_gap_mask.reindex(df_use.index).fillna(False)
if cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN:
    exclude = exclude | long_gap_mask
exclude = exclude | (state != "on")

df_normal = df_use.loc[~exclude].copy()
df_all = df_use.copy()

# clip de outlier GLOBAL (mesmo comportamento do EXP10c de producao --
# calculado sobre TODO o df_normal, nao so pre-corte -- ver nota no
# automl_pipeline.py; mantido fixo aqui de proposito, pra isolar so a
# variavel "cadencia de retreino do modelo").
df_normal = clip_outliers(df_normal, cfg)
df_all = clip_outliers(df_all, cfg)

near_alarm_mask = build_exclusion_mask(df_all.index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
on_arr_full = (state.reindex(df_all.index).fillna("on").values == "on")

ramp_at_point_full = ramp_gate.reindex(df_all.index, method="ffill").values.astype(float)
vol_at_point_full = volatility_index.reindex(df_all.index, method="ffill").values.astype(float)
ramp_at_point_full = np.nan_to_num(ramp_at_point_full, nan=-np.inf)
vol_at_point_full = np.nan_to_num(vol_at_point_full, nan=-np.inf)

df_alarm_eval_all = df_alarm_group.loc[
    df_alarm_group["Data da Ocorrencia"] >= pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
].reset_index(drop=True)
print(f"[WF] {len(df_alarm_eval_all)} alarmes OOS ({eval_sensors})", flush=True)

nu, gamma = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA
pct = cfg.AUTOML_THRESHOLD_PERCENTILES[0]
ramp_max, vol_thr = cfg.LOAD_GATE_RAMP_MAX, cfg.VOLATILITY_GATE_THRESHOLD


def fit_model(train_slice: pd.DataFrame):
    """Normaliza (stats do `train_slice`) e treina ocsvm. Retorna tudo o
    que `score_with_model` precisa pra pontuar qualquer recorte depois,
    sem re-treinar."""
    center = train_slice.mean(axis=0) if cfg.NORMALIZE_MODE.lower() == "zscore" else train_slice.median(axis=0)
    if cfg.NORMALIZE_MODE.lower() == "zscore":
        scale = train_slice.std(axis=0).replace(0, 1.0)
    else:
        q1, q3 = train_slice.quantile(0.25), train_slice.quantile(0.75)
        scale = (q3 - q1).replace(0, 1.0)

    x_train = ((train_slice - center) / scale).values.astype(np.float32)
    x_fit = x_train
    if cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES and len(x_train) > cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        idx = rng.choice(len(x_train), size=int(cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES), replace=False)
        x_fit = x_train[idx]
    clf = fit_ocsvm(x_fit, nu, gamma)
    train_err = ocsvm_error(clf, x_train)
    threshold = float(np.percentile(train_err, pct))
    return {"clf": clf, "center": center, "scale": scale, "threshold": threshold, "n_fit": len(x_fit)}


def score_with_model(model: dict, score_slice: pd.DataFrame) -> np.ndarray:
    x_score = ((score_slice - model["center"]) / model["scale"]).values.astype(np.float32)
    score_err = ocsvm_error(model["clf"], x_score)

    idx = score_slice.index
    pos = df_all.index.get_indexer(idx)
    on_arr = on_arr_full[pos]
    ramp_at = ramp_at_point_full[pos]
    vol_at = vol_at_point_full[pos]

    return (score_err > model["threshold"]) & on_arr & (ramp_at < ramp_max) & (vol_at <= vol_thr)


def eval_month(flags: np.ndarray, idx: pd.DatetimeIndex, month_start: pd.Timestamp, month_end: pd.Timestamp):
    win = pd.Timedelta(minutes=cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    alarms_m = df_alarm_eval_all.loc[
        (df_alarm_eval_all["Data da Ocorrencia"] >= month_start) & (df_alarm_eval_all["Data da Ocorrencia"] < month_end)
    ]
    hits = 0
    for t in alarms_m["Data da Ocorrencia"]:
        t0, t1 = t - win, t + win
        window_mask = (idx >= t0) & (idx <= t1)
        if flags[window_mask].any():
            hits += 1

    near_alarm_m = near_alarm_mask.reindex(idx).fillna(False).values
    normal_mask = ~near_alarm_m
    normal_flags = flags[normal_mask]
    n_normal = int(normal_flags.sum())
    n_total_normal = int(normal_mask.sum())

    return {
        "n_alarms": int(len(alarms_m)),
        "hits": int(hits),
        "n_normal_points": n_total_normal,
        "n_normal_anom": n_normal,
    }


# --- meses do periodo OOS ---
oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
data_end = df_all.index.max()
month_starts = pd.date_range(oos_start, data_end, freq="MS")
month_bounds = [(m, m + pd.DateOffset(months=1)) for m in month_starts]
print(f"[WF] {len(month_bounds)} meses no periodo OOS: {month_starts[0].date()} a {month_starts[-1].date()}", flush=True)

# --- baseline congelado: treina 1x antes do corte, reavalia mes a mes pela MESMA rotina ---
df_normal_frozen = df_normal.loc[df_normal.index < oos_start]
print(f"[WF] Treinando baseline congelado (n_normal_fit={len(df_normal_frozen)})...", flush=True)
frozen_model = fit_model(df_normal_frozen)
print(f"[WF] baseline congelado: threshold={frozen_model['threshold']:.6f} n_fit={frozen_model['n_fit']}", flush=True)

frozen_rows = []
walkforward_rows = []
for month_start, month_end in month_bounds:
    score_slice_idx = df_all.index[(df_all.index >= month_start) & (df_all.index < month_end)]
    score_slice = df_all.loc[score_slice_idx]

    flags_frozen = score_with_model(frozen_model, score_slice)
    res_frozen = eval_month(flags_frozen, score_slice_idx, month_start, month_end)
    res_frozen.update({"month": str(month_start.date())[:7], "threshold": frozen_model["threshold"],
                        "n_fit": frozen_model["n_fit"]})
    frozen_rows.append(res_frozen)

    df_normal_wf = df_normal.loc[df_normal.index < month_start]
    wf_model = fit_model(df_normal_wf)
    flags_wf = score_with_model(wf_model, score_slice)
    res_wf = eval_month(flags_wf, score_slice_idx, month_start, month_end)
    res_wf.update({"month": str(month_start.date())[:7], "threshold": wf_model["threshold"],
                    "n_fit": wf_model["n_fit"]})
    walkforward_rows.append(res_wf)

    print(f"[WF mes {month_start.date()}] congelado: hits={res_frozen['hits']}/{res_frozen['n_alarms']} "
          f"fp={res_frozen['n_normal_anom']}/{res_frozen['n_normal_points']}  |  "
          f"walk-forward (n_fit={wf_model['n_fit']}): hits={res_wf['hits']}/{res_wf['n_alarms']} "
          f"fp={res_wf['n_normal_anom']}/{res_wf['n_normal_points']}", flush=True)


def pool(rows):
    n_alarms = sum(r["n_alarms"] for r in rows)
    hits = sum(r["hits"] for r in rows)
    n_normal = sum(r["n_normal_points"] for r in rows)
    n_anom = sum(r["n_normal_anom"] for r in rows)
    return {
        "hit_rate": hits / n_alarms if n_alarms else None,
        "hits": hits, "n_alarms": n_alarms,
        "normal_alert_rate": n_anom / n_normal if n_normal else None,
        "n_normal_anom": n_anom, "n_normal_points": n_normal,
    }


pool_frozen = pool(frozen_rows)
pool_wf = pool(walkforward_rows)

print("\n[WF RESUMO] congelado (mesma rotina de avaliacao mensal): "
      f"hit_rate={pool_frozen['hit_rate']:.4f} ({pool_frozen['hits']}/{pool_frozen['n_alarms']})  "
      f"normal_alert_rate={pool_frozen['normal_alert_rate']:.5f}", flush=True)
print("[WF RESUMO] walk-forward mensal (janela expansiva): "
      f"hit_rate={pool_wf['hit_rate']:.4f} ({pool_wf['hits']}/{pool_wf['n_alarms']})  "
      f"normal_alert_rate={pool_wf['normal_alert_rate']:.5f}", flush=True)

report = {
    "group": group_name,
    "config": CONFIG_PATH,
    "months": [str(m.date())[:7] for m, _ in month_bounds],
    "frozen_monthly": frozen_rows,
    "walkforward_monthly": walkforward_rows,
    "frozen_pooled": pool_frozen,
    "walkforward_pooled": pool_wf,
}
out_path = os.path.join(OUTPUT_DIR, "walkforward_exp10c_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("walkforward_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
