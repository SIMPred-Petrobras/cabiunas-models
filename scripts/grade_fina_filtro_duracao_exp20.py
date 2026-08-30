"""Grade FINA e LITERAL (nao aproximada) de corte de duracao minima, em
volta de 6min, contra o modelo CONGELADO do EXP10c/EXP20.

MOTIVACAO: docs/analise_automl_exp10.md, secao "EXP20". A estimativa
original do filtro de 6min (90%/0,235%) foi feita por contagem/peso
sobre distribuicoes de duracao, nao uma simulacao literal. Rodando de
verdade (ordem de portoes corrigida), o resultado real foi 85%/0,189% --
2 alarmes a mais perdidos, um deles (2025-10-20) bem na fronteira do
corte de 6min. Este script varre uma grade fina reconstruindo a serie
de verdade a cada corte, usando a MESMA funcao de producao
(`scoring.apply_min_duration_filter`) -- sem nenhuma aproximacao.

METODO: reproduz o modelo de referencia (ocsvm, fit unico, threshold
p99,9) uma vez. Para cada corte candidato: aplica
`apply_min_duration_filter` (na mesma ordem da producao -- antes dos
portoes de rampa/volatilidade), depois os portoes, e reavalia
hit_rate/normal_alert_rate de verdade via `eval_alarm_hit_rate`/
`compute_normal_alert_rate` (as mesmas funcoes do pipeline).

Uso:
    PYTHONPATH=. python scripts/grade_fina_filtro_duracao_exp20.py
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
    normalize_train_only,
    select_feature_columns,
    THERMAL_ARRAY_SPREAD_COL,
)
from src.cnn1d_ae.scoring import (
    build_operational_state,
    compute_load_ramp_gate,
    compute_volatility_index,
    apply_load_gate,
    apply_volatility_gate,
    apply_min_duration_filter,
    eval_alarm_hit_rate,
    compute_normal_alert_rate,
)
from src.cnn1d_ae.automl_models import fit_ocsvm, ocsvm_error

CONFIG_PATH = "configs/calibracao_v4_eq/test_grupo_exp20_filtro_duracao.json"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

DURATION_GRID_MIN = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.9, 6.0, 6.1, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0]

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)
cfg.ENABLE_MIN_DURATION_FILTER = False  # aplicado manualmente aqui, na grade

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::grade_fina_filtro_duracao_exp20",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)
task.connect({"duration_grid_min": DURATION_GRID_MIN}, name="grid_params")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[GRADE] Carregando dados...", flush=True)
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

load_gate_series = df_use[cfg.LOAD_GATE_SENSOR]
ramp_gate, _ = compute_load_ramp_gate(
    load_gate_series, cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES, cfg.LOAD_GATE_WINDOW_MINUTES
)
volatility_index = compute_volatility_index(df_use[cfg.VOLATILITY_GATE_SENSORS], cfg.VOLATILITY_GATE_WINDOW_MINUTES)

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
df_normal = clip_outliers(df_normal, cfg)
df_all = clip_outliers(df_all, cfg)

oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
df_normal_fit = df_normal.loc[df_normal.index < oos_start]
df_normal_z, df_all_z, _c, _s = normalize_train_only(cfg, df_normal_fit, df_all)
x_normal = df_normal_z.values.astype(np.float32)
x_all = df_all_z.values.astype(np.float32)
all_index = df_all_z.index

on_arr = (state.reindex(all_index).fillna("on").values == "on")
near_alarm_mask = build_exclusion_mask(all_index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
eval_mask_arr = (all_index >= oos_start)

df_alarm_eval = df_alarm_group.loc[df_alarm_group["Data da Ocorrencia"] >= oos_start].reset_index(drop=True)
print(f"[GRADE] {len(df_alarm_eval)} alarmes OOS ({eval_sensors})", flush=True)

nu, gamma = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA
x_fit = x_normal
if cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES and len(x_normal) > cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    idx = rng.choice(len(x_normal), size=int(cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES), replace=False)
    x_fit = x_normal[idx]
print(f"[GRADE] Treinando ocsvm (n_fit={len(x_fit)})...", flush=True)
clf = fit_ocsvm(x_fit, nu, gamma)
train_err = ocsvm_error(clf, x_normal)
all_err = ocsvm_error(clf, x_all)

pct = cfg.AUTOML_THRESHOLD_PERCENTILES[0]
threshold = float(np.percentile(train_err, pct))
raw_flags = (all_err > threshold) & on_arr
print(f"[GRADE] threshold p{pct}={threshold:.6f}  base_flags={int(raw_flags.sum())}", flush=True)

# sanity check: sem filtro de duracao, so os portoes -- deve bater com a referencia conhecida
operational_state_col = np.where(on_arr, "on", "off")
df_point0 = pd.DataFrame(
    {"is_anom_point": raw_flags.astype(int), "operational_state": operational_state_col}, index=all_index
)
df_point0 = apply_load_gate(
    df_point0, load_gate_series, ramp_max=cfg.LOAD_GATE_RAMP_MAX, level_min=cfg.LOAD_GATE_LEVEL_MIN,
    ramp_halflife_minutes=cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES, window_minutes=cfg.LOAD_GATE_WINDOW_MINUTES,
)
df_point0 = apply_volatility_gate(df_point0, volatility_index, cfg.VOLATILITY_GATE_THRESHOLD)
eval0 = eval_alarm_hit_rate(df_alarm_eval, df_point0, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
fp0 = compute_normal_alert_rate(
    df_point0.loc[all_index[eval_mask_arr]], near_alarm_mask.loc[all_index[eval_mask_arr]]
)
print(f"[SANITY CHECK sem filtro de duracao] hit_rate={eval0['hit_rate']:.4f} "
      f"({eval0['alarms_with_detected_anomaly_in_window']}/{eval0['n_alarms']})  "
      f"normal_alert_rate={fp0:.5f}  (referencia: 0.9250 / 0.00350)", flush=True)

print(f"[GRADE] Varrendo {len(DURATION_GRID_MIN)} cortes de duracao (ordem correta: antes dos portoes)...",
      flush=True)
results = []
for min_dur in DURATION_GRID_MIN:
    df_point = pd.DataFrame(
        {"is_anom_point": raw_flags.astype(int), "operational_state": operational_state_col}, index=all_index
    )
    df_point = apply_min_duration_filter(df_point, min_dur)
    df_point = apply_load_gate(
        df_point, load_gate_series, ramp_max=cfg.LOAD_GATE_RAMP_MAX, level_min=cfg.LOAD_GATE_LEVEL_MIN,
        ramp_halflife_minutes=cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES, window_minutes=cfg.LOAD_GATE_WINDOW_MINUTES,
    )
    df_point = apply_volatility_gate(df_point, volatility_index, cfg.VOLATILITY_GATE_THRESHOLD)

    eval_stats = eval_alarm_hit_rate(df_alarm_eval, df_point, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    fp = compute_normal_alert_rate(
        df_point.loc[all_index[eval_mask_arr]], near_alarm_mask.loc[all_index[eval_mask_arr]]
    )
    results.append({
        "min_duration_min": min_dur, "hit_rate": eval_stats["hit_rate"],
        "hits": eval_stats["alarms_with_detected_anomaly_in_window"], "n_alarms": eval_stats["n_alarms"],
        "normal_alert_rate": fp,
    })
    print(f"[GRADE corte={min_dur}min] hit_rate={eval_stats['hit_rate']:.4f} "
          f"({eval_stats['alarms_with_detected_anomaly_in_window']}/{eval_stats['n_alarms']})  "
          f"normal_alert_rate={fp:.5f}  reducao_vs_ref={100*(1-fp/fp0):.1f}%", flush=True)

report = {
    "group": group_name,
    "reference_no_duration_filter": {"hit_rate": eval0["hit_rate"], "normal_alert_rate": fp0},
    "grid_results": results,
}
out_path = os.path.join(OUTPUT_DIR, "grade_fina_filtro_duracao_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("grade_fina_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
