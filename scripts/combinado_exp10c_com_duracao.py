"""Testa se o filtro de duracao minima (6min, achado em
analise_duracao_score_exp10c.py: -32,7% de FP por so 1 deteccao marginal)
combina bem com o retreino walk-forward mensal + veto de sensor
congelado (a combinacao "segura" D ja validada em
docs/analise_automl_exp10.md, secao "Combinacao dos ganhos validados").

MOTIVACAO: a mesma secao mostrou que um ganho "custo zero" contra o
modelo CONGELADO (limiares de portao) quebrou ao ser combinado com o
retreino mensal, porque a calibracao era especifica aquela distribuicao
de score. O filtro de duracao de 6min foi validado do mesmo jeito
(contra o modelo congelado) -- precisa do mesmo teste de composicao
antes de qualquer recomendacao.

DIFERENCA DE ESTRUTURA vs combinado_exp10c_final.py: o filtro de duracao
exige olhar EPISODIOS CONTINUOS, e o retreino walk-forward pontua um mes
de cada vez -- um precursor real poderia (em teoria) atravessar a
fronteira de um mes. Por isso aqui as flags de cada mes sao concatenadas
numa serie continua ANTES de fazer o RLE (run-length encoding) e aplicar
o filtro de duracao -- cada ponto usa o modelo do seu proprio mes (correto),
mas a continuidade de episodios entre meses fica preservada.

Uso:
    PYTHONPATH=. python scripts/combinado_exp10c_com_duracao.py
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
    eval_alarm_hit_rate,
)
from src.cnn1d_ae.automl_models import fit_ocsvm, ocsvm_error

CONFIG_PATH = "configs/calibracao_v4_eq/test_grupo_exp10c_portao_volatilidade.json"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

FROZEN_WINDOW_MIN = 5.0     # veto de sensor congelado (ja validado)
MIN_DURATION_MIN = 6.0      # filtro de duracao (achado nesta sessao)

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::combinado_exp10c_com_duracao",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)
task.connect({"frozen_window_min": FROZEN_WINDOW_MIN, "min_duration_min": MIN_DURATION_MIN}, name="combo_params")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[COMBO-DUR] Carregando dados...", flush=True)
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

raw_sensor_values = df_use[sensors].copy()  # pro veto de congelamento

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

near_alarm_mask = build_exclusion_mask(df_all.index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
on_arr_full = (state.reindex(df_all.index).fillna("on").values == "on")
ramp_at_point_full = np.nan_to_num(ramp_gate.reindex(df_all.index, method="ffill").values.astype(float), nan=-np.inf)
vol_at_point_full = np.nan_to_num(volatility_index.reindex(df_all.index, method="ffill").values.astype(float), nan=-np.inf)
PROD_RAMP_MAX, PROD_VOL_THR = cfg.LOAD_GATE_RAMP_MAX, cfg.VOLATILITY_GATE_THRESHOLD

# --- veto de sensor congelado, calculado uma unica vez sobre toda a serie ---
raw_sensor_values = raw_sensor_values.reindex(df_all.index)
diff_zero = (raw_sensor_values.diff() == 0)
dt_seconds = df_all.index.to_series().diff().dt.total_seconds().median()
if not np.isfinite(dt_seconds) or dt_seconds <= 0:
    dt_seconds = 30.0
w_samples = max(1, int(round((FROZEN_WINDOW_MIN * 60.0) / dt_seconds)))
frozen_any_full = pd.Series(False, index=df_all.index)
for s in sensors:
    sustained = diff_zero[s].rolling(w_samples, min_periods=w_samples).sum() >= w_samples
    frozen_any_full = frozen_any_full | sustained.fillna(False)
frozen_arr_full = frozen_any_full.values
print(f"[COMBO-DUR] veto de congelamento (W={FROZEN_WINDOW_MIN}min) calculado.", flush=True)

df_alarm_eval = df_alarm_group.loc[
    df_alarm_group["Data da Ocorrencia"] >= pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
].reset_index(drop=True)
print(f"[COMBO-DUR] {len(df_alarm_eval)} alarmes OOS ({eval_sensors})", flush=True)

nu, gamma = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA
pct = cfg.AUTOML_THRESHOLD_PERCENTILES[0]


def fit_model(train_slice: pd.DataFrame):
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


def raw_score(model: dict, score_slice: pd.DataFrame):
    x_score = ((score_slice - model["center"]) / model["scale"]).values.astype(np.float32)
    return ocsvm_error(model["clf"], x_score)


# --- meses do periodo OOS: monta a serie CONTINUA de flags (D = walk-forward + portoes de producao + veto) ---
oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
data_end = df_all.index.max()
month_starts = pd.date_range(oos_start, data_end, freq="MS")
month_bounds = [(m, m + pd.DateOffset(months=1)) for m in month_starts]
print(f"[COMBO-DUR] {len(month_bounds)} meses no periodo OOS", flush=True)

oos_index = df_all.index[df_all.index >= oos_start]
flags_d_full = np.zeros(len(oos_index), dtype=bool)
oos_pos0 = int(df_all.index.get_indexer([oos_index[0]])[0])

for month_start, month_end in month_bounds:
    score_slice_idx = df_all.index[(df_all.index >= month_start) & (df_all.index < month_end)]
    score_slice = df_all.loc[score_slice_idx]
    pos = df_all.index.get_indexer(score_slice_idx)

    df_normal_wf = df_normal.loc[df_normal.index < month_start]
    wf_model = fit_model(df_normal_wf)
    err_wf = raw_score(wf_model, score_slice)

    on_arr = on_arr_full[pos]
    ramp_at = ramp_at_point_full[pos]
    vol_at = vol_at_point_full[pos]
    frozen_at = frozen_arr_full[pos]
    flags = (err_wf > wf_model["threshold"]) & on_arr & (ramp_at < PROD_RAMP_MAX) & (vol_at <= PROD_VOL_THR) & (~frozen_at)

    out_pos = pos - oos_pos0
    flags_d_full[out_pos] = flags
    print(f"[COMBO-DUR mes {month_start.date()}] n_fit={wf_model['n_fit']} flags_ativas={int(flags.sum())}", flush=True)

print(f"[COMBO-DUR] serie D completa montada: {len(oos_index)} pontos, {int(flags_d_full.sum())} flags ativas", flush=True)


def eval_flags(flags: np.ndarray, idx: pd.DatetimeIndex, label: str):
    df_point = pd.DataFrame({"is_anom_point": flags.astype(int)}, index=idx)
    eval_stats = eval_alarm_hit_rate(df_alarm_eval, df_point, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    near_alarm_arr = near_alarm_mask.reindex(idx).fillna(False).values
    on_arr = on_arr_full[df_all.index.get_indexer(idx)]
    normal_mask = (~near_alarm_arr) & on_arr
    fp = float(flags[normal_mask].mean()) if normal_mask.any() else 0.0
    print(f"[COMBO-DUR RESULTADO {label}] hit_rate={eval_stats['hit_rate']:.4f} "
          f"({eval_stats['alarms_with_detected_anomaly_in_window']}/{eval_stats['n_alarms']})  "
          f"normal_alert_rate={fp:.5f}", flush=True)
    return {"hit_rate": eval_stats["hit_rate"], "hits": eval_stats["alarms_with_detected_anomaly_in_window"],
            "n_alarms": eval_stats["n_alarms"], "normal_alert_rate": fp}


result_d = eval_flags(flags_d_full, oos_index, "D (walk-forward + veto, sem filtro de duracao)")


def runs_from_bool(mask: np.ndarray):
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


runs = runs_from_bool(flags_d_full)
min_samples = max(1, int(round((MIN_DURATION_MIN * 60.0) / dt_seconds)))
flags_e_full = flags_d_full.copy()
n_removed_runs = 0
for (s, e) in runs:
    if (e - s) < min_samples:
        flags_e_full[s:e] = False
        n_removed_runs += 1
print(f"[COMBO-DUR] filtro de duracao ({MIN_DURATION_MIN}min = {min_samples} amostras): "
      f"{n_removed_runs}/{len(runs)} episodios removidos", flush=True)

result_e = eval_flags(flags_e_full, oos_index, f"E = D + filtro de duracao minima ({MIN_DURATION_MIN}min) [COMBO FINAL]")

report = {
    "group": group_name,
    "params": {"frozen_window_min": FROZEN_WINDOW_MIN, "min_duration_min": MIN_DURATION_MIN},
    "D_walkforward_veto": result_d,
    "E_walkforward_veto_duracao": result_e,
    "n_runs_before_filter": len(runs),
    "n_runs_removed_by_filter": n_removed_runs,
}
out_path = os.path.join(OUTPUT_DIR, "combinado_exp10c_com_duracao_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("combinado_duracao_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
