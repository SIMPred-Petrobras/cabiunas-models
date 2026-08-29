"""LOEO (leave-one-event-out) dos limiares dos portoes de rampa e
volatilidade do EXP10c (`configs/calibracao_v4_eq/test_grupo_exp10c_portao_volatilidade.json`).

MOTIVACAO: ver docs/analise_automl_exp10.md, secao "Validacao por evento
fisico (2026-08-29)". Essa validacao anterior confirmou que o 92,5%/0,35%
do EXP10c nao e artefato de cascata (37/40 alarmes == 14/16 episodios
fisicos distintos), mas NAO testou se os 3 limiares de portao
(LOAD_GATE_RAMP_MAX=100, VOLATILITY_GATE_THRESHOLD=0.39,
OFF_TARGET_ABS_THRESHOLD=150) generalizam -- eles foram escolhidos por
"simulacao offline" que olhou o efeito em TODOS os eventos disponiveis de
uma vez (criterio documentado: "preserva 29/29 preditivos, minimiza FP").
Esse e exatamente o tipo de vies de selecao que o LOEO ja revelou na
pipeline do Francisco (66,7% do grid virou 62,5% no LOEO honesto).

ESCOPO: reproduz LOAD_GATE_RAMP_MAX e VOLATILITY_GATE_THRESHOLD (os 2
limiares com criterio de selecao "zero-cost" documentado). Nao mexe em
OFF_TARGET_ABS_THRESHOLD (150 graus) -- esse veio de um piso fisico
observado num desligamento real especifico, nao de uma varredura de grade
contra os eventos de avaliacao, entao fica fora do escopo desta LOEO.

METODO:
1. Reproduz o pre-processamento e o treino do EXP10c (ocsvm, nu/gamma/
   percentil fixos) uma unica vez -- o modelo em si nao usa rotulos de
   alarme, entao nao ha necessidade de re-treinar por fold.
2. Calcula as series continuas dos 2 portoes (rampa de carga, indice de
   volatilidade) uma unica vez, com as JANELAS fixas (halflife/window do
   config) -- so os LIMIARES de corte sao re-selecionados por fold.
3. Varre uma grade (ramp_max x vol_threshold), aplica os portoes e
   verifica, para cada um dos 16 episodios fisicos OOS, se ele e
   detectado.
4. Identifica quais desses 16 episodios sao "preditivos" (o modelo SEM
   nenhum portao ja os detecta -- os 2 episodios de glitch de instrumento
   ficam de fora, sao estruturalmente nao detectaveis, nao e sobre isso
   que o LOEO deve responder).
5. LOEO de verdade: para cada episodio preditivo, escolhe o combo
   (ramp_max, vol_threshold) que preserva TODOS OS OUTROS episodios
   preditivos com o MENOR normal_alert_rate (mesmo criterio "custo zero"
   documentado, mas sem ver o episodio em questao) e verifica se esse
   combo ainda detecta o episodio retido.

Uso:
    PYTHONPATH=. python scripts/loeo_exp10c_portoes.py
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

EPISODE_GAP_MIN = 1440.0  # mesmo criterio da validacao por evento anterior

RAMP_MAX_GRID = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160,
                  180, 200, 250, 300, 400, 500, 700, 1000, 1.0e7]
VOL_THRESHOLD_GRID = sorted(set(
    [round(float(x), 3) for x in np.arange(0.05, 1.01, 0.02)] + [0.39, 1.0e7]
))

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::loeo_exp10c_portoes",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)
task.connect({
    "ramp_max_grid": RAMP_MAX_GRID,
    "vol_threshold_grid": VOL_THRESHOLD_GRID,
    "episode_gap_min": EPISODE_GAP_MIN,
}, name="loeo_grid")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[LOEO] Carregando dados (io.load_data, mesma rotina do pipeline padrao)...", flush=True)
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

# --- estado operacional (FIXO -- OFF_TARGET_ABS_THRESHOLD fora do escopo desta LOEO) ---
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

# --- series continuas dos portoes (JANELAS fixas -- so o limiar de corte sera varrido) ---
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

df_normal = clip_outliers(df_normal, cfg)
df_all = clip_outliers(df_all, cfg)

oos_start = pd.Timestamp(cfg.AUTOML_OOS_SPLIT_DATE)
df_normal_fit = df_normal.loc[df_normal.index < oos_start]

df_normal_z, df_all_z, _normalize_center, _normalize_scale = normalize_train_only(cfg, df_normal_fit, df_all)
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
print(f"[LOEO] {len(df_alarm_eval)} alarmes OOS ({eval_sensors})", flush=True)

# --- treino ocsvm (fixo, nao depende de portao -- nao usa rotulo de alarme) ---
nu, gamma = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA
x_fit = x_normal
if cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES and len(x_normal) > cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    idx = rng.choice(len(x_normal), size=int(cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES), replace=False)
    x_fit = x_normal[idx]
print(f"[LOEO] Treinando ocsvm (nu={nu}, gamma={gamma}, n_fit={len(x_fit)})...", flush=True)
clf = fit_ocsvm(x_fit, nu, gamma)
train_err = ocsvm_error(clf, x_normal)
all_err = ocsvm_error(clf, x_all)

pct = cfg.AUTOML_THRESHOLD_PERCENTILES[0]
threshold = float(np.percentile(train_err, pct))
raw_flags = (all_err > threshold)
base_flags = raw_flags & on_arr
print(f"[LOEO] threshold p{pct}={threshold:.6f}  raw_flags={int(raw_flags.sum())}  "
      f"base_flags(pos-mascara operacional, pre-portao)={int(base_flags.sum())}", flush=True)

ramp_at_point = ramp_gate.reindex(all_index, method="ffill").values.astype(float)
vol_at_point = volatility_index.reindex(all_index, method="ffill").values.astype(float)
ramp_at_point = np.nan_to_num(ramp_at_point, nan=-np.inf)
vol_at_point = np.nan_to_num(vol_at_point, nan=-np.inf)

win = pd.Timedelta(minutes=cfg.EXCLUDE_MINUTES_AROUND_ALARM)


def normal_alert_rate(flags: np.ndarray) -> float:
    normal = flags[normal_mask_eval]
    return float(normal.mean()) if normal.size else 0.0


def apply_gates(ramp_max: float, vol_thr: float) -> np.ndarray:
    not_blocked_ramp = ramp_at_point < ramp_max
    not_blocked_vol = vol_at_point <= vol_thr
    return base_flags & not_blocked_ramp & not_blocked_vol


# --- sanity check: reproduz o EXP10c com os limiares fixos da referencia ---
flags_ref = apply_gates(cfg.LOAD_GATE_RAMP_MAX, cfg.VOLATILITY_GATE_THRESHOLD)
df_point_ref = pd.DataFrame({"is_anom_point": flags_ref.astype(int)}, index=all_index)
eval_ref = eval_alarm_hit_rate(df_alarm_eval, df_point_ref, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
fp_ref = normal_alert_rate(flags_ref)
print(f"[SANITY CHECK] reproducao EXP10c local: hit_rate={eval_ref['hit_rate']:.4f} "
      f"({eval_ref['alarms_with_detected_anomaly_in_window']}/{eval_ref['n_alarms']})  "
      f"normal_alert_rate={fp_ref:.5f}  (referencia da task remota: 0.9250 / 0.00350)", flush=True)

# --- episodios fisicos (mesmo criterio do doc: gap > 1440min = novo episodio) ---
gaps = df_alarm_eval["Data da Ocorrencia"].diff().dt.total_seconds().fillna(1e9) / 60.0
df_alarm_eval["ep_id"] = (gaps > EPISODE_GAP_MIN).cumsum()
episodes = sorted(df_alarm_eval["ep_id"].unique())
n_episodes = len(episodes)

ep_windows = []
ep_labels = []
for ep in episodes:
    sub = df_alarm_eval.loc[df_alarm_eval["ep_id"] == ep, "Data da Ocorrencia"]
    t0, t1 = sub.min() - win, sub.max() + win
    start = int(all_index.searchsorted(t0, side="left"))
    end = int(all_index.searchsorted(t1, side="right"))
    ep_windows.append((start, end))
    ep_labels.append(f"{sub.min()} -> {sub.max()} ({len(sub)} alarmes)")

print(f"[LOEO] {n_episodes} episodios fisicos distintos (gap>{EPISODE_GAP_MIN:.0f}min)", flush=True)


def episodes_hit(flags: np.ndarray) -> list[bool]:
    return [bool(flags[s:e].any()) for (s, e) in ep_windows]


hits_nogate = episodes_hit(base_flags)  # portoes totalmente abertos = so o modelo + mascara operacional
coverable = [i for i in range(n_episodes) if hits_nogate[i]]
non_coverable = [i for i in range(n_episodes) if not hits_nogate[i]]

print(f"[LOEO] episodios cobriveis pelo modelo (mesmo sem portao): {len(coverable)}/{n_episodes}", flush=True)
for i in non_coverable:
    print(f"[LOEO]   NAO cobrivel (fora do escopo do portao): {ep_labels[i]}", flush=True)

# --- grade completa (ramp_max x vol_threshold) ---
print(f"[LOEO] Varrendo grade {len(RAMP_MAX_GRID)}x{len(VOL_THRESHOLD_GRID)}="
      f"{len(RAMP_MAX_GRID) * len(VOL_THRESHOLD_GRID)} combinacoes...", flush=True)
grid_results = []
for ramp_max in RAMP_MAX_GRID:
    not_blocked_ramp = ramp_at_point < ramp_max
    for vol_thr in VOL_THRESHOLD_GRID:
        not_blocked_vol = vol_at_point <= vol_thr
        flags = base_flags & not_blocked_ramp & not_blocked_vol
        hits = episodes_hit(flags)
        fp = normal_alert_rate(flags)
        grid_results.append({"ramp_max": ramp_max, "vol_thr": vol_thr, "hits": hits, "fp": fp})
print(f"[LOEO] Grade calculada.", flush=True)

# checagem: o ponto atual (100/0.39) esta na fronteira de custo-zero da grade cheia?
full_visible = coverable
zero_cost_full = [r for r in grid_results if all(r["hits"][i] for i in full_visible)]
best_full = min(zero_cost_full, key=lambda r: r["fp"]) if zero_cost_full else None
if best_full is not None:
    print(f"[LOEO] Melhor combo (grade completa, preservando os {len(coverable)} preditivos): "
          f"ramp_max={best_full['ramp_max']} vol_thr={best_full['vol_thr']} fp={best_full['fp']:.5f}",
          flush=True)
else:
    print("[LOEO] nenhum combo da grade preserva todos os preditivos!", flush=True)

# --- LOEO de verdade ---
loeo_rows = []
for held_idx in coverable:
    visible = [i for i in coverable if i != held_idx]
    candidates = [r for r in grid_results if all(r["hits"][i] for i in visible)]
    if not candidates:
        chosen = {"ramp_max": None, "vol_thr": None, "hits": hits_nogate, "fp": normal_alert_rate(base_flags)}
        note = "nenhum combo preservou os outros preditivos -- fallback sem portao"
    else:
        chosen = min(candidates, key=lambda r: r["fp"])
        note = ""
    held_hit = bool(chosen["hits"][held_idx])
    loeo_rows.append({
        "held_out_episode": ep_labels[held_idx],
        "chosen_ramp_max": chosen["ramp_max"],
        "chosen_vol_thr": chosen["vol_thr"],
        "held_out_detected": held_hit,
        "resulting_fp": chosen["fp"],
        "resulting_hit_rate_all_coverable": float(np.mean([chosen["hits"][i] for i in coverable])),
        "note": note,
    })
    print(f"[LOEO fold] held-out={ep_labels[held_idx]}  chosen=(ramp={chosen['ramp_max']}, "
          f"vol={chosen['vol_thr']})  held_out_detected={held_hit}  fp={chosen['fp']:.5f}  {note}", flush=True)

n_lost = sum(1 for r in loeo_rows if not r["held_out_detected"])
loeo_hit_rate = 1.0 - (n_lost / len(coverable))
print(f"\n[LOEO RESUMO] {len(coverable) - n_lost}/{len(coverable)} episodios preditivos "
      f"sobrevivem ao LOEO (honesto) = {loeo_hit_rate*100:.1f}%  "
      f"(referencia com limiares fixos vendo tudo: {len(coverable)}/{len(coverable)} = 100%)", flush=True)

report = {
    "group": group_name,
    "config": CONFIG_PATH,
    "reference_reproduction": {
        "hit_rate": eval_ref["hit_rate"],
        "alarms_hit": eval_ref["alarms_with_detected_anomaly_in_window"],
        "n_alarms": eval_ref["n_alarms"],
        "normal_alert_rate": fp_ref,
    },
    "n_episodes_oos": n_episodes,
    "coverable_episodes": [ep_labels[i] for i in coverable],
    "non_coverable_episodes": [ep_labels[i] for i in non_coverable],
    "best_combo_full_visibility": best_full,
    "loeo_folds": loeo_rows,
    "loeo_summary": {
        "n_predictive_episodes": len(coverable),
        "n_lost_in_loeo": n_lost,
        "loeo_hit_rate": loeo_hit_rate,
        "fixed_threshold_hit_rate": 1.0,
    },
}
out_path = os.path.join(OUTPUT_DIR, "loeo_exp10c_portoes_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("loeo_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
