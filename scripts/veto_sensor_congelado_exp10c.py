"""Veto de sensor congelado para o EXP10c: sensor travado (leitura
literalmente sem mudar por N minutos -- falha de instrumento/comunicacao,
nao sinal real) deveria suprimir o alarme, do mesmo jeito que os portoes
de rampa/volatilidade ja suprimem por manobra de carga legitima.

MOTIVACAO: ver docs/analise_pca_monitoramento_sistema.md, secao "v9" /
"v10" -- a ideia ja foi tentada na REIMPLEMENTACAO manual da pipeline do
Francisco (PCA multi-sinal) e refutada, mas la foi testada JUNTO com uma
outra mudanca (faixa fisica fixa em vez de clip por quantil), nunca
isolada -- e nunca foi testada no EXP10c (arquitetura ocsvm + portoes),
que e uma pipeline diferente. Existe uma coluna pre-computada no dataset
bruto (`any_sensor_constant_run`) mas ela e True ~99,8% do tempo (calculada
sobre TODOS os ~40 tags do painel, nao so os 12 do grupo) -- inutil como
esta, entao a deteccao de "congelado" e refeita aqui, escopada aos 12
sensores do proprio grupo do EXP10c.

ESCOPO: aditivo em cima da config de referencia do EXP10c (modelo,
portoes de rampa/volatilidade, mascara operacional -- tudo igual e FIXO).
So a duracao da janela "congelado" (`FROZEN_WINDOW_GRID`) e varrida.

METODO: para cada sensor do grupo, calcula se o valor ficou literalmente
constante (diff==0) por toda uma janela trailing de W minutos; "congelado"
= OR entre os 12 sensores. Veto: suprime is_anom_point quando congelado
(mesma direcao dos outros portoes -- so remove, nunca adiciona deteccao).
Varre W numa grade pequena e reporta, pra cada W, hit_rate (nos 14
episodios "predizíveis" ja identificados) e normal_alert_rate.

Uso:
    PYTHONPATH=. python scripts/veto_sensor_congelado_exp10c.py
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

FROZEN_WINDOW_GRID_MIN = [5, 10, 15, 20, 30, 45, 60, 90, 120]

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::veto_sensor_congelado_exp10c",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)
task.connect({"frozen_window_grid_min": FROZEN_WINDOW_GRID_MIN}, name="veto_grid")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[VETO] Carregando dados...", flush=True)
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

# guarda os valores BRUTOS dos sensores do grupo antes de restringir
# df_use as colunas de feature -- e disso que o veto de congelamento
# precisa (nao das derivadas).
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

oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
df_normal_fit = df_normal.loc[df_normal.index < oos_start]
df_normal_z, df_all_z, _c, _s = normalize_train_only(cfg, df_normal_fit, df_all)
x_normal = df_normal_z.values.astype(np.float32)
x_all = df_all_z.values.astype(np.float32)
all_index = df_all_z.index

eval_mask_arr = (all_index >= oos_start)
state_full = state.reindex(all_index).fillna("on")
on_arr = (state_full.values == "on")
near_alarm_mask = build_exclusion_mask(all_index, alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
near_alarm_arr = near_alarm_mask.reindex(all_index).fillna(False).values
normal_mask_eval = eval_mask_arr & (~near_alarm_arr) & on_arr

df_alarm_eval = df_alarm_group.loc[df_alarm_group["Data da Ocorrencia"] >= oos_start].reset_index(drop=True)
print(f"[VETO] {len(df_alarm_eval)} alarmes OOS ({eval_sensors})", flush=True)

nu, gamma = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA
x_fit = x_normal
if cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES and len(x_normal) > cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    idx = rng.choice(len(x_normal), size=int(cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES), replace=False)
    x_fit = x_normal[idx]
print(f"[VETO] Treinando ocsvm (n_fit={len(x_fit)})...", flush=True)
clf = fit_ocsvm(x_fit, nu, gamma)
train_err = ocsvm_error(clf, x_normal)
all_err = ocsvm_error(clf, x_all)

pct = cfg.AUTOML_THRESHOLD_PERCENTILES[0]
threshold = float(np.percentile(train_err, pct))
raw_flags = (all_err > threshold)
base_flags = raw_flags & on_arr

ramp_at_point = np.nan_to_num(ramp_gate.reindex(all_index, method="ffill").values.astype(float), nan=-np.inf)
vol_at_point = np.nan_to_num(volatility_index.reindex(all_index, method="ffill").values.astype(float), nan=-np.inf)
not_blocked_ramp = ramp_at_point < cfg.LOAD_GATE_RAMP_MAX
not_blocked_vol = vol_at_point <= cfg.VOLATILITY_GATE_THRESHOLD
flags_reference = base_flags & not_blocked_ramp & not_blocked_vol

win = pd.Timedelta(minutes=cfg.EXCLUDE_MINUTES_AROUND_ALARM)


def normal_alert_rate(flags: np.ndarray) -> float:
    normal = flags[normal_mask_eval]
    return float(normal.mean()) if normal.size else 0.0


df_point_ref = pd.DataFrame({"is_anom_point": flags_reference.astype(int)}, index=all_index)
eval_ref = eval_alarm_hit_rate(df_alarm_eval, df_point_ref, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
fp_ref = normal_alert_rate(flags_reference)
print(f"[SANITY CHECK] reproducao EXP10c local: hit_rate={eval_ref['hit_rate']:.4f} "
      f"({eval_ref['alarms_with_detected_anomaly_in_window']}/{eval_ref['n_alarms']})  "
      f"normal_alert_rate={fp_ref:.5f}  (referencia: 0.9250 / 0.00350)", flush=True)

# --- episodios fisicos e "predizíveis" (mesmo criterio das secoes anteriores) ---
EPISODE_GAP_MIN = 1440.0
gaps = df_alarm_eval["Data da Ocorrencia"].diff().dt.total_seconds().fillna(1e9) / 60.0
df_alarm_eval["ep_id"] = (gaps > EPISODE_GAP_MIN).cumsum()
episodes = sorted(df_alarm_eval["ep_id"].unique())
ep_windows, ep_labels = [], []
for ep in episodes:
    sub = df_alarm_eval.loc[df_alarm_eval["ep_id"] == ep, "Data da Ocorrencia"]
    t0, t1 = sub.min() - win, sub.max() + win
    start = int(all_index.searchsorted(t0, side="left"))
    end = int(all_index.searchsorted(t1, side="right"))
    ep_windows.append((start, end))
    ep_labels.append(f"{sub.min()} -> {sub.max()}")


def episodes_hit(flags: np.ndarray):
    return [bool(flags[s:e].any()) for (s, e) in ep_windows]


hits_ref = episodes_hit(flags_reference)
predictable = [i for i in range(len(episodes)) if hits_ref[i]]
print(f"[VETO] {len(predictable)}/{len(episodes)} episodios detectados pela referencia (baseline p/ comparar)",
      flush=True)

# --- deteccao de sensor congelado: diff==0 sustentado por W minutos, em qualquer um dos 12 sensores ---
raw_sensor_values = raw_sensor_values.reindex(all_index)
diff_zero = (raw_sensor_values.diff() == 0)

dt_seconds = all_index.to_series().diff().dt.total_seconds().median()
if not np.isfinite(dt_seconds) or dt_seconds <= 0:
    dt_seconds = 30.0

print(f"[VETO] Varrendo grade de janela 'congelado': {FROZEN_WINDOW_GRID_MIN} min", flush=True)
results = []
for w_min in FROZEN_WINDOW_GRID_MIN:
    w_samples = max(1, int(round((w_min * 60.0) / dt_seconds)))
    frozen_any = pd.Series(False, index=all_index)
    for s in sensors:
        sustained = diff_zero[s].rolling(w_samples, min_periods=w_samples).sum() >= w_samples
        frozen_any = frozen_any | sustained.fillna(False)
    frozen_arr = frozen_any.values

    flags_veto = flags_reference & (~frozen_arr)
    hits = episodes_hit(flags_veto)
    fp = normal_alert_rate(flags_veto)
    n_lost = sum(1 for i in predictable if not hits[i])
    frac_frozen = float(frozen_arr[eval_mask_arr].mean())
    frac_fp_removed = 1.0 - (fp / fp_ref) if fp_ref else 0.0
    results.append({
        "window_min": w_min, "fp": fp, "n_lost_of_predictable": n_lost,
        "frac_time_frozen_oos": frac_frozen, "frac_fp_removed": frac_fp_removed,
    })
    print(f"[VETO w={w_min}min] fp={fp:.5f} ({frac_fp_removed*100:+.1f}% vs ref)  "
          f"episodios_perdidos={n_lost}/{len(predictable)}  "
          f"frac_tempo_congelado(OOS)={frac_frozen*100:.3f}%", flush=True)

zero_cost = [r for r in results if r["n_lost_of_predictable"] == 0]
best = min(zero_cost, key=lambda r: r["fp"]) if zero_cost else None
if best:
    print(f"\n[VETO RESUMO] melhor janela custo-zero: {best['window_min']}min  "
          f"fp={best['fp']:.5f} ({best['fp']/fp_ref*100:.1f}% do fp de referencia)", flush=True)
else:
    print("\n[VETO RESUMO] nenhuma janela testada preserva todos os episodios predizíveis.", flush=True)

report = {
    "group": group_name,
    "reference": {"hit_rate": eval_ref["hit_rate"], "normal_alert_rate": fp_ref,
                  "n_predictable_episodes": len(predictable), "n_episodes": len(episodes)},
    "grid_results": results,
    "best_zero_cost": best,
}
out_path = os.path.join(OUTPUT_DIR, "veto_sensor_congelado_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("veto_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
