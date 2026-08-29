"""Combina os 3 ganhos independentes encontrados para o EXP10c, todos
documentados em docs/analise_automl_exp10.md:

  1. Retreino walk-forward mensal (janela expansiva) -- FP -19% sozinho.
  2. Limiares de portao mais agressivos (ramp_max=60, vol_threshold=0,21,
     achados no LOEO como o ponto que preserva os 14/14 preditivos com
     o menor FP na grade cheia) em vez dos valores de producao (100/0,39).
  3. Veto de sensor congelado (W=5min) -- FP -7,2% sozinho.

Cada um foi validado isoladamente contra a MESMA referencia (modelo
congelado, portoes de producao, sem veto: 92,50%/0,348%). A pergunta
aqui e se os efeitos se somam, se sobrepoem (mesmos pontos de FP sendo
suprimidos por mais de um mecanismo) ou se cancelam.

METODO: mesma rotina de avaliacao mensal do walk-forward (retreina o
ocsvm por mes, janela expansiva). Para cada mes, computa o score 4 vezes
com o MESMO modelo retreinado daquele mes, variando so a camada de
pos-processamento (waterfall, cada estagio soma o anterior):

  A. so walk-forward (portoes de producao 100/0,39, sem veto)
  B. + limiares novos (60/0,21)
  C. + veto de sensor congelado (5min)           <- combinacao final

Mais o baseline de hoje (modelo CONGELADO + portoes de producao, sem
veto) pela mesma rotina, para comparacao direta com o numero de
producao (92,50%/0,348%).

Uso:
    PYTHONPATH=. python scripts/combinado_exp10c_final.py
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

NEW_RAMP_MAX = 60.0
NEW_VOL_THR = 0.21
FROZEN_WINDOW_MIN = 5.0

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::combinado_exp10c_final",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)
task.connect({"new_ramp_max": NEW_RAMP_MAX, "new_vol_thr": NEW_VOL_THR,
              "frozen_window_min": FROZEN_WINDOW_MIN}, name="combo_params")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[COMBO] Carregando dados...", flush=True)
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

# valores brutos (pre-restricao a feature_cols) -- necessario pro veto de congelamento
raw_sensor_values = df_use[sensors].copy()

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

# --- veto de sensor congelado (W=5min, achado da secao anterior), calculado uma unica vez ---
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
print(f"[COMBO] veto de congelamento (W={FROZEN_WINDOW_MIN}min) calculado.", flush=True)

df_alarm_eval_all = df_alarm_group.loc[
    df_alarm_group["Data da Ocorrencia"] >= pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
].reset_index(drop=True)
print(f"[COMBO] {len(df_alarm_eval_all)} alarmes OOS ({eval_sensors})", flush=True)

nu, gamma = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA
pct = cfg.AUTOML_THRESHOLD_PERCENTILES[0]
PROD_RAMP_MAX, PROD_VOL_THR = cfg.LOAD_GATE_RAMP_MAX, cfg.VOLATILITY_GATE_THRESHOLD


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


def apply_layers(score_err: np.ndarray, threshold: float, pos: np.ndarray,
                  ramp_max: float, vol_thr: float, use_veto: bool) -> np.ndarray:
    on_arr = on_arr_full[pos]
    ramp_at = ramp_at_point_full[pos]
    vol_at = vol_at_point_full[pos]
    flags = (score_err > threshold) & on_arr & (ramp_at < ramp_max) & (vol_at <= vol_thr)
    if use_veto:
        flags = flags & (~frozen_arr_full[pos])
    return flags


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
    on_m = on_arr_full[df_all.index.get_indexer(idx)]
    normal_mask = (~near_alarm_m) & on_m
    normal_flags = flags[normal_mask]
    return {"n_alarms": int(len(alarms_m)), "hits": int(hits),
            "n_normal_points": int(normal_mask.sum()), "n_normal_anom": int(normal_flags.sum())}


def pool(rows):
    n_alarms = sum(r["n_alarms"] for r in rows)
    hits = sum(r["hits"] for r in rows)
    n_normal = sum(r["n_normal_points"] for r in rows)
    n_anom = sum(r["n_normal_anom"] for r in rows)
    return {"hit_rate": hits / n_alarms if n_alarms else None, "hits": hits, "n_alarms": n_alarms,
            "normal_alert_rate": n_anom / n_normal if n_normal else None,
            "n_normal_anom": n_anom, "n_normal_points": n_normal}


oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
data_end = df_all.index.max()
month_starts = pd.date_range(oos_start, data_end, freq="MS")
month_bounds = [(m, m + pd.DateOffset(months=1)) for m in month_starts]
print(f"[COMBO] {len(month_bounds)} meses no periodo OOS", flush=True)

df_normal_frozen = df_normal.loc[df_normal.index < oos_start]
print(f"[COMBO] Treinando baseline congelado de producao (n_normal_fit={len(df_normal_frozen)})...", flush=True)
frozen_model = fit_model(df_normal_frozen)

rows_prod_frozen = []   # producao hoje: modelo congelado + portoes 100/0.39 + sem veto
rows_A = []              # so walk-forward
rows_B = []              # walk-forward + limiares novos
rows_C = []              # walk-forward + limiares novos + veto
rows_D = []              # walk-forward + portoes de PRODUCAO (100/0.39) + veto -- sem trocar limiar

for month_start, month_end in month_bounds:
    score_slice_idx = df_all.index[(df_all.index >= month_start) & (df_all.index < month_end)]
    score_slice = df_all.loc[score_slice_idx]
    pos = df_all.index.get_indexer(score_slice_idx)

    err_frozen = raw_score(frozen_model, score_slice)
    flags_prod = apply_layers(err_frozen, frozen_model["threshold"], pos, PROD_RAMP_MAX, PROD_VOL_THR, use_veto=False)
    r_prod = eval_month(flags_prod, score_slice_idx, month_start, month_end)
    rows_prod_frozen.append(r_prod)

    df_normal_wf = df_normal.loc[df_normal.index < month_start]
    wf_model = fit_model(df_normal_wf)
    err_wf = raw_score(wf_model, score_slice)

    flags_a = apply_layers(err_wf, wf_model["threshold"], pos, PROD_RAMP_MAX, PROD_VOL_THR, use_veto=False)
    r_a = eval_month(flags_a, score_slice_idx, month_start, month_end)
    rows_A.append(r_a)

    flags_b = apply_layers(err_wf, wf_model["threshold"], pos, NEW_RAMP_MAX, NEW_VOL_THR, use_veto=False)
    r_b = eval_month(flags_b, score_slice_idx, month_start, month_end)
    rows_B.append(r_b)

    flags_c = apply_layers(err_wf, wf_model["threshold"], pos, NEW_RAMP_MAX, NEW_VOL_THR, use_veto=True)
    r_c = eval_month(flags_c, score_slice_idx, month_start, month_end)
    rows_C.append(r_c)

    flags_d = apply_layers(err_wf, wf_model["threshold"], pos, PROD_RAMP_MAX, PROD_VOL_THR, use_veto=True)
    r_d = eval_month(flags_d, score_slice_idx, month_start, month_end)
    rows_D.append(r_d)

    print(f"[COMBO mes {month_start.date()}] producao: {r_prod['hits']}/{r_prod['n_alarms']} "
          f"fp={r_prod['n_normal_anom']}/{r_prod['n_normal_points']}  |  "
          f"A(wf): {r_a['hits']}/{r_a['n_alarms']} fp={r_a['n_normal_anom']}/{r_a['n_normal_points']}  |  "
          f"B(+limiar): {r_b['hits']}/{r_b['n_alarms']} fp={r_b['n_normal_anom']}/{r_b['n_normal_points']}  |  "
          f"C(+veto): {r_c['hits']}/{r_c['n_alarms']} fp={r_c['n_normal_anom']}/{r_c['n_normal_points']}  |  "
          f"D(wf+veto,sem trocar limiar): {r_d['hits']}/{r_d['n_alarms']} "
          f"fp={r_d['n_normal_anom']}/{r_d['n_normal_points']}", flush=True)

pool_prod = pool(rows_prod_frozen)
pool_a = pool(rows_A)
pool_b = pool(rows_B)
pool_c = pool(rows_C)
pool_d = pool(rows_D)

print("\n[COMBO RESUMO]")
print(f"  Producao hoje (congelado, portoes 100/0.39, sem veto): "
      f"hit_rate={pool_prod['hit_rate']:.4f} ({pool_prod['hits']}/{pool_prod['n_alarms']})  "
      f"normal_alert_rate={pool_prod['normal_alert_rate']:.5f}")
print(f"  A. + walk-forward mensal: "
      f"hit_rate={pool_a['hit_rate']:.4f} ({pool_a['hits']}/{pool_a['n_alarms']})  "
      f"normal_alert_rate={pool_a['normal_alert_rate']:.5f}")
print(f"  B. + limiares novos (ramp={NEW_RAMP_MAX}/vol={NEW_VOL_THR}): "
      f"hit_rate={pool_b['hit_rate']:.4f} ({pool_b['hits']}/{pool_b['n_alarms']})  "
      f"normal_alert_rate={pool_b['normal_alert_rate']:.5f}")
print(f"  C. + veto de sensor congelado (W={FROZEN_WINDOW_MIN}min): "
      f"hit_rate={pool_c['hit_rate']:.4f} ({pool_c['hits']}/{pool_c['n_alarms']})  "
      f"normal_alert_rate={pool_c['normal_alert_rate']:.5f}")
print(f"  D. walk-forward + veto, SEM trocar limiar de portao (100/0.39) [COMBO SEGURO]: "
      f"hit_rate={pool_d['hit_rate']:.4f} ({pool_d['hits']}/{pool_d['n_alarms']})  "
      f"normal_alert_rate={pool_d['normal_alert_rate']:.5f}", flush=True)

report = {
    "group": group_name,
    "params": {"new_ramp_max": NEW_RAMP_MAX, "new_vol_thr": NEW_VOL_THR, "frozen_window_min": FROZEN_WINDOW_MIN},
    "months": [str(m.date())[:7] for m, _ in month_bounds],
    "producao_hoje_pooled": pool_prod,
    "A_walkforward_pooled": pool_a,
    "B_walkforward_limiares_pooled": pool_b,
    "C_combo_final_pooled": pool_c,
    "D_walkforward_veto_sem_limiar_pooled": pool_d,
    "producao_hoje_monthly": rows_prod_frozen,
    "A_monthly": rows_A,
    "B_monthly": rows_B,
    "C_monthly": rows_C,
    "D_monthly": rows_D,
}
out_path = os.path.join(OUTPUT_DIR, "combinado_exp10c_final_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("combinado_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
