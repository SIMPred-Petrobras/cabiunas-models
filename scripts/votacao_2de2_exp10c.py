"""Votacao 2-de-2 para o EXP10c: em vez de um unico ocsvm treinado sobre
os 12 sensores do grupo de uma vez, treina 2 modelos INDEPENDENTES --
familia termica (TC382_03_A, T5_AVG_A) e familia de vibracao (10 canais
TV_35*) -- e exige que as DUAS familias sinalizem anomalia (dentro de
uma janela de tolerancia) antes de contar como deteccao.

MOTIVACAO: ver ALARMES_POR_SENSOR_EFEITO_CASCATA.md -- um evento fisico
real dispara dezenas de tags em cascata porque varias grandezas reagem a
mesma causa raiz. Exigir que 2 FAMILIAS independentes concordem (nao so
tags do mesmo sinal) e um filtro fisico de ruido: um ruido isolado numa
familia (ex: vibracao) sem nenhum sinal termico correspondente e menos
provavel de ser um precursor real do que quando as duas concordam. E a
logica de arquitetura que a pipeline do Francisco usa (N-de-4 sinais),
nunca testada no EXP10c (que hoje e 1 modelo conjunto sobre as 12
colunas).

ESCOPO: mascara operacional e alarmes de treino compartilhados entre as
2 familias (mesmo estado ligado/desligado da maquina); os portoes de
rampa/volatilidade da producao sao aplicados por cima do resultado
combinado (nao dentro de cada familia) -- mesma separacao de camadas
(mascara -> modelo -> portao) que o resto do projeto ja usa. Varre so a
janela de tolerancia de votacao (`VOTING_WINDOW_GRID_MIN`, causal --
familia X "ativa" em t se sinalizou em algum ponto de [t-W, t]).

Uso:
    PYTHONPATH=. python scripts/votacao_2de2_exp10c.py
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
    eval_alarm_hit_rate,
)
from src.cnn1d_ae.automl_models import fit_ocsvm, ocsvm_error

CONFIG_PATH = "configs/calibracao_v4_eq/test_grupo_exp10c_portao_volatilidade.json"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

VOTING_WINDOW_GRID_MIN = [30, 60, 120, 240, 480, 720, 1440]

FAMILY_THERMAL = ["TC382_03_A", "T5_AVG_A"]
FAMILY_VIBRATION = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A",
                     "TV_353Y_A", "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::votacao_2de2_exp10c",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)
task.connect({"voting_window_grid_min": VOTING_WINDOW_GRID_MIN,
              "family_thermal": FAMILY_THERMAL, "family_vibration": FAMILY_VIBRATION}, name="voting_grid")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[VOTE] Carregando dados...", flush=True)
df_alarm, df_feat, df_raw, _time_report = load_data(cfg)

group = cfg.SENSOR_GROUPS[0]
group_name = group["name"]
eval_sensors = list(group.get("eval_sensors") or group["sensors"])
target_sensor_ref = group.get("target_sensor")

# --- estado operacional: compartilhado entre as 2 familias (mesma maquina) ---
sensors_all = list(group["sensors"])
df_use_all, _ = build_group_dataframe(cfg, df_feat, df_raw, sensors_all)
ref_sensor = cfg.OPERATIONAL_REF_SENSOR
if ref_sensor and ref_sensor not in sensors_all:
    df_ref, _ = build_sensor_dataframe(cfg, df_feat, df_raw, ref_sensor)
    ref_series = df_ref[ref_sensor]
else:
    ref_series = df_use_all[ref_sensor] if ref_sensor in df_use_all.columns else df_use_all[sensors_all[0]]
secondary_series = None
if cfg.OFF_TARGET_ABS_THRESHOLD is not None and target_sensor_ref and target_sensor_ref in df_use_all.columns:
    secondary_series = df_use_all[target_sensor_ref]
state = build_operational_state(
    index=df_use_all.index, sensor_series=ref_series,
    off_value_quantile=cfg.OFF_VALUE_QUANTILE, off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
    off_long_min_hours=cfg.OFF_LONG_MIN_HOURS, transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
    transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
    secondary_series=secondary_series, secondary_off_abs_threshold=cfg.OFF_TARGET_ABS_THRESHOLD,
)
ramp_gate, _ = compute_load_ramp_gate(
    df_use_all[cfg.LOAD_GATE_SENSOR], cfg.LOAD_GATE_RAMP_HALFLIFE_MINUTES, cfg.LOAD_GATE_WINDOW_MINUTES
)
volatility_index = compute_volatility_index(df_use_all[cfg.VOLATILITY_GATE_SENSORS], cfg.VOLATILITY_GATE_WINDOW_MINUTES)

if "Tag" in df_alarm.columns:
    df_alarm_group = df_alarm.loc[df_alarm["Tag"].isin(eval_sensors)].copy()
else:
    df_alarm_group = df_alarm.copy()
df_alarm_group = df_alarm_group.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")
alarm_times = df_alarm_group["Data da Ocorrencia"]

nu, gamma, pct = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA, cfg.AUTOML_THRESHOLD_PERCENTILES[0]
oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)


def fit_family(family_sensors: list[str], family_name: str):
    """Reproduz o pipeline do EXP10c (build->features->clip->normaliza->
    treina->score), so que restrito aos sensores desta familia. Mascara
    operacional (`state`) e alarmes de exclusao sao os MESMOS de sempre
    (compartilhados)."""
    df_use, long_gap_mask = build_group_dataframe(cfg, df_feat, df_raw, family_sensors)
    valid = [s for s in family_sensors if float(df_use[s].std()) >= cfg.MIN_STD]
    feature_cols = select_feature_columns(cfg, df_use, valid)
    df_use = df_use[feature_cols]

    exclude_alarm = build_exclusion_mask(df_use.index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    long_gap_mask = long_gap_mask.reindex(df_use.index).fillna(False)
    state_r = state.reindex(df_use.index).fillna("on")
    exclude = exclude_alarm.copy()
    if cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN:
        exclude = exclude | long_gap_mask
    exclude = exclude | (state_r != "on")

    df_normal = clip_outliers(df_use.loc[~exclude].copy(), cfg)
    df_all = clip_outliers(df_use.copy(), cfg)
    df_normal_fit = df_normal.loc[df_normal.index < oos_start]

    df_normal_z, df_all_z, _c, _s = normalize_train_only(cfg, df_normal_fit, df_all)
    x_normal = df_normal_z.values.astype(np.float32)
    x_all = df_all_z.values.astype(np.float32)
    all_index = df_all_z.index

    x_fit = x_normal
    if cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES and len(x_normal) > cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        idx = rng.choice(len(x_normal), size=int(cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES), replace=False)
        x_fit = x_normal[idx]
    print(f"[VOTE] Treinando familia '{family_name}' ({len(valid)} sensores, n_fit={len(x_fit)})...", flush=True)
    clf = fit_ocsvm(x_fit, nu, gamma)
    train_err = ocsvm_error(clf, x_normal)
    all_err = ocsvm_error(clf, x_all)
    threshold = float(np.percentile(train_err, pct))

    on_arr = (state_r.reindex(all_index).fillna("on").values == "on")
    raw_flags = (all_err > threshold) & on_arr
    return pd.Series(raw_flags, index=all_index)


flags_thermal = fit_family(FAMILY_THERMAL, "termica")
flags_vibration = fit_family(FAMILY_VIBRATION, "vibracao")

# alinha ambas ao mesmo indice (deveriam ja ser identicas -- mesma fonte/tempo)
common_index = flags_thermal.index.intersection(flags_vibration.index)
flags_thermal = flags_thermal.reindex(common_index).fillna(False)
flags_vibration = flags_vibration.reindex(common_index).fillna(False)
print(f"[VOTE] indice comum: {len(common_index)} pontos "
      f"(termica={len(flags_thermal)}, vibracao={len(flags_vibration)} antes do alinhamento)", flush=True)

# portoes de producao aplicados sobre o indice comum
ramp_at_point = np.nan_to_num(ramp_gate.reindex(common_index, method="ffill").values.astype(float), nan=-np.inf)
vol_at_point = np.nan_to_num(volatility_index.reindex(common_index, method="ffill").values.astype(float), nan=-np.inf)
not_blocked_ramp = ramp_at_point < cfg.LOAD_GATE_RAMP_MAX
not_blocked_vol = vol_at_point <= cfg.VOLATILITY_GATE_THRESHOLD

eval_mask_arr = (common_index >= oos_start)
state_full = state.reindex(common_index).fillna("on")
on_arr_full = (state_full.values == "on")
near_alarm_mask = build_exclusion_mask(common_index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
near_alarm_arr = near_alarm_mask.reindex(common_index).fillna(False).values
normal_mask_eval = eval_mask_arr & (~near_alarm_arr) & on_arr_full

df_alarm_eval = df_alarm_group.loc[df_alarm_group["Data da Ocorrencia"] >= oos_start].reset_index(drop=True)
win = pd.Timedelta(minutes=cfg.EXCLUDE_MINUTES_AROUND_ALARM)

EPISODE_GAP_MIN = 1440.0
gaps = df_alarm_eval["Data da Ocorrencia"].diff().dt.total_seconds().fillna(1e9) / 60.0
df_alarm_eval["ep_id"] = (gaps > EPISODE_GAP_MIN).cumsum()
episodes = sorted(df_alarm_eval["ep_id"].unique())
ep_windows = []
for ep in episodes:
    sub = df_alarm_eval.loc[df_alarm_eval["ep_id"] == ep, "Data da Ocorrencia"]
    t0, t1 = sub.min() - win, sub.max() + win
    start = int(common_index.searchsorted(t0, side="left"))
    end = int(common_index.searchsorted(t1, side="right"))
    ep_windows.append((start, end))


def normal_alert_rate(flags: np.ndarray) -> float:
    normal = flags[normal_mask_eval]
    return float(normal.mean()) if normal.size else 0.0


def episodes_hit(flags: np.ndarray):
    return [bool(flags[s:e].any()) for (s, e) in ep_windows]


# --- referencia: 1 modelo conjunto (igual ao EXP10c oficial) so pra ter o baseline no mesmo indice/rotina ---
df_use_ref, long_gap_ref = build_group_dataframe(cfg, df_feat, df_raw, sensors_all)
valid_ref = [s for s in sensors_all if float(df_use_ref[s].std()) >= cfg.MIN_STD]
feature_cols_ref = select_feature_columns(cfg, df_use_ref, valid_ref)
df_use_ref = df_use_ref[feature_cols_ref]
exclude_alarm_ref = build_exclusion_mask(df_use_ref.index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
long_gap_ref = long_gap_ref.reindex(df_use_ref.index).fillna(False)
state_ref = state.reindex(df_use_ref.index).fillna("on")
exclude_ref = exclude_alarm_ref | long_gap_ref | (state_ref != "on")
df_normal_ref = clip_outliers(df_use_ref.loc[~exclude_ref].copy(), cfg)
df_all_ref = clip_outliers(df_use_ref.copy(), cfg)
df_normal_fit_ref = df_normal_ref.loc[df_normal_ref.index < oos_start]
df_normal_z_ref, df_all_z_ref, _c, _s = normalize_train_only(cfg, df_normal_fit_ref, df_all_ref)
x_fit_ref = df_normal_z_ref.values.astype(np.float32)
if cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES and len(x_fit_ref) > cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    idx = rng.choice(len(x_fit_ref), size=int(cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES), replace=False)
    x_fit_ref = x_fit_ref[idx]
clf_ref = fit_ocsvm(x_fit_ref, nu, gamma)
threshold_ref = float(np.percentile(ocsvm_error(clf_ref, df_normal_z_ref.values.astype(np.float32)), pct))
all_err_ref = ocsvm_error(clf_ref, df_all_z_ref.values.astype(np.float32))
on_ref = (state_ref.reindex(df_all_z_ref.index).fillna("on").values == "on")
flags_single = pd.Series((all_err_ref > threshold_ref) & on_ref, index=df_all_z_ref.index).reindex(common_index).fillna(False).values
ramp_single = np.nan_to_num(ramp_gate.reindex(common_index, method="ffill").values.astype(float), nan=-np.inf) < cfg.LOAD_GATE_RAMP_MAX
vol_single = np.nan_to_num(volatility_index.reindex(common_index, method="ffill").values.astype(float), nan=-np.inf) <= cfg.VOLATILITY_GATE_THRESHOLD
flags_single_final = flags_single & ramp_single & vol_single

df_point_single = pd.DataFrame({"is_anom_point": flags_single_final.astype(int)}, index=common_index)
eval_single = eval_alarm_hit_rate(df_alarm_eval, df_point_single, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
fp_single = normal_alert_rate(flags_single_final)
print(f"[SANITY CHECK] modelo unico (referencia EXP10c) neste script: hit_rate={eval_single['hit_rate']:.4f} "
      f"({eval_single['alarms_with_detected_anomaly_in_window']}/{eval_single['n_alarms']})  "
      f"normal_alert_rate={fp_single:.5f}  (referencia: 0.9250 / 0.00350)", flush=True)
hits_single = episodes_hit(flags_single_final)
predictable_single = [i for i in range(len(episodes)) if hits_single[i]]

# --- votacao 2-de-2, varrendo a janela de tolerancia ---
print(f"[VOTE] Varrendo janela de votacao: {VOTING_WINDOW_GRID_MIN} min", flush=True)
results = []
for w_min in VOTING_WINDOW_GRID_MIN:
    active_thermal = flags_thermal.rolling(f"{w_min}min", min_periods=1).max().astype(bool).values
    active_vibration = flags_vibration.rolling(f"{w_min}min", min_periods=1).max().astype(bool).values
    combined = active_thermal & active_vibration
    final_flags = combined & not_blocked_ramp & not_blocked_vol

    hits = episodes_hit(final_flags)
    fp = normal_alert_rate(final_flags)
    n_hit_of_single_predictable = sum(1 for i in predictable_single if hits[i])
    results.append({
        "window_min": w_min, "fp": fp,
        "n_hit_of_single_predictable": n_hit_of_single_predictable,
        "n_single_predictable": len(predictable_single),
    })
    print(f"[VOTE w={w_min}min] fp={fp:.5f} ({fp/fp_single*100:.1f}% do fp do modelo unico)  "
          f"episodios={n_hit_of_single_predictable}/{len(predictable_single)} (referencia do modelo unico)", flush=True)

report = {
    "group": group_name,
    "single_model_reference": {"hit_rate": eval_single["hit_rate"], "normal_alert_rate": fp_single,
                                "n_predictable_episodes": len(predictable_single), "n_episodes": len(episodes)},
    "grid_results": results,
}
out_path = os.path.join(OUTPUT_DIR, "votacao_2de2_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("votacao_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
