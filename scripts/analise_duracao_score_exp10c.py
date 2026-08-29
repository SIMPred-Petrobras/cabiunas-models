"""Inspecao de falso positivo do EXP10c: quanto tempo o score bruto do
modelo (antes de qualquer portao) fica CONTINUAMENTE acima do limiar,
comparando episodios que precedem um alarme real (TP, 37/40 batidos)
contra episodios de falso positivo residual (que sobrevivem aos portoes
e aparecem como alarme hoje)?

MOTIVACAO (pergunta do usuario): se os precursores reais tem uma
"assinatura" de duracao tipica (o score fica elevado por N
minutos/horas), e os falsos positivos residuais NAO tem esse
comportamento (sao mais curtos/transientes), duracao minima poderia ser
mais um filtro -- diferente do debounce por CONTAGEM DE PONTOS ja
testado e descartado no EXP7 (docs/analise_automl_exp10.md, "Caminho
testado e descartado -- debounce": na epoca, ANTES dos portoes de
mascara/rampa/volatilidade existirem, a duracao mediana do FP residual
(6 pontos/2,5min) se sobrepunha a duracao dos precursores mais curtos
(25 percentil, 6min) -- debounce por contagem nao separava os dois).
Vale re-testar no perfil de FP residual de HOJE (pos EXP10c, com todos
os portoes), que e muito menor e pode ter uma assinatura diferente.

METODO:
1. Reproduz o modelo de referencia do EXP10c (ocsvm, fit unico,
   threshold p99,9) -- sanity check contra 92,50%/0,348% conhecidos.
2. `raw_flags` = score > threshold, E estado operacional "on", SEM
   nenhum portao (rampa/volatilidade) -- e a serie continua de
   "score elevado", independente do que os portoes decidem suprimir
   depois.
3. Agrupa `raw_flags` em episodios continuos (RLE) e mede a duracao de
   cada um.
4. TP: para cada um dos 37 alarmes batidos, pega o episodio raw MAIS
   LONGO que se sobrepoe a janela +-1440min do alarme (representando o
   "sinal" que gerou a deteccao).
5. FP: episodios raw fora de qualquer janela de alarme (nem os 40 do
   grupo nem os alarmes near_alarm mais amplos ja usados no resto do
   projeto). Marca quais desses SOBREVIVEM aos portoes de producao
   (aparecem em `is_anom_point`=1 final, i.e., sao FP residual hoje) vs
   os que os portoes ja suprimem.
6. Compara as distribuicoes de duracao (TP vs FP-todos vs FP-residual)
   e testa um filtro de duracao minima (pegando o minimo dos 37 TP,
   filtro "custo zero" por construcao) pra ver quanto FP residual isso
   removeria.

Uso:
    PYTHONPATH=. python scripts/analise_duracao_score_exp10c.py
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

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg = update_cfg_from_dict(PipelineConfig(), cfg_dict)

task = Task.init(
    project_name=cfg.CLEARML_PROJECT_NAME,
    task_name="cnn1d-ae::analise_duracao_score_exp10c",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(cfg.CLEARML_DOCKER_IMAGE)
task.connect(cfg_dict)

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

setup_gpu()

print("[DUR] Carregando dados...", flush=True)
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
near_alarm_arr = near_alarm_mask.reindex(all_index).fillna(False).values
eval_mask_arr = (all_index >= oos_start)
normal_mask_eval = eval_mask_arr & (~near_alarm_arr) & on_arr

df_alarm_eval = df_alarm_group.loc[df_alarm_group["Data da Ocorrencia"] >= oos_start].reset_index(drop=True)
print(f"[DUR] {len(df_alarm_eval)} alarmes OOS ({eval_sensors})", flush=True)

nu, gamma = cfg.AUTOML_OCSVM_NU, cfg.AUTOML_OCSVM_GAMMA
x_fit = x_normal
if cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES and len(x_normal) > cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    idx = rng.choice(len(x_normal), size=int(cfg.AUTOML_OCSVM_MAX_TRAIN_SAMPLES), replace=False)
    x_fit = x_normal[idx]
print(f"[DUR] Treinando ocsvm (n_fit={len(x_fit)})...", flush=True)
clf = fit_ocsvm(x_fit, nu, gamma)
train_err = ocsvm_error(clf, x_normal)
all_err = ocsvm_error(clf, x_all)

pct = cfg.AUTOML_THRESHOLD_PERCENTILES[0]
threshold = float(np.percentile(train_err, pct))
raw_flags = (all_err > threshold) & on_arr   # SEM portao -- so score+mascara operacional

ramp_at_point = np.nan_to_num(ramp_gate.reindex(all_index, method="ffill").values.astype(float), nan=-np.inf)
vol_at_point = np.nan_to_num(volatility_index.reindex(all_index, method="ffill").values.astype(float), nan=-np.inf)
not_blocked_ramp = ramp_at_point < cfg.LOAD_GATE_RAMP_MAX
not_blocked_vol = vol_at_point <= cfg.VOLATILITY_GATE_THRESHOLD
flags_reference = raw_flags & not_blocked_ramp & not_blocked_vol  # COM portao -- producao de hoje

df_point_ref = pd.DataFrame({"is_anom_point": flags_reference.astype(int)}, index=all_index)
eval_ref = eval_alarm_hit_rate(df_alarm_eval, df_point_ref, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
fp_ref = float(flags_reference[normal_mask_eval].mean())
print(f"[SANITY CHECK] hit_rate={eval_ref['hit_rate']:.4f} "
      f"({eval_ref['alarms_with_detected_anomaly_in_window']}/{eval_ref['n_alarms']})  "
      f"normal_alert_rate={fp_ref:.5f}  (referencia: 0.9250 / 0.00350)", flush=True)


def runs_from_bool(mask: np.ndarray):
    """Retorna lista de (start_pos, end_pos_exclusive) dos episodios continuos onde mask=True."""
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


dt_seconds = all_index.to_series().diff().dt.total_seconds().median()
if not np.isfinite(dt_seconds) or dt_seconds <= 0:
    dt_seconds = 30.0


def run_duration_minutes(start: int, end: int) -> float:
    # end e exclusivo -- duracao = (end-1 - start) amostras + 1 amostra de largura
    return (end - start) * dt_seconds / 60.0


raw_runs = runs_from_bool(raw_flags)
print(f"[DUR] {len(raw_runs)} episodios continuos de score-acima-do-limiar (sem portao), "
      f"toda a serie 2024-2026", flush=True)

win = pd.Timedelta(minutes=cfg.EXCLUDE_MINUTES_AROUND_ALARM)

# --- TP: pro cada alarme batido, pega o maior episodio raw que se sobrepoe a janela ---
tp_durations = []
tp_details = []
for _, row in df_alarm_eval.iterrows():
    t = row["Data da Ocorrencia"]
    t0, t1 = t - win, t + win
    p0 = int(all_index.searchsorted(t0, side="left"))
    p1 = int(all_index.searchsorted(t1, side="right"))
    overlapping = [(s, e) for (s, e) in raw_runs if e > p0 and s < p1]
    if not overlapping:
        continue  # alarme perdido (nao e um dos 37 hits) -- nao entra na distribuicao TP
    best = max(overlapping, key=lambda se: se[1] - se[0])
    dur = run_duration_minutes(*best)
    tp_durations.append(dur)
    tp_details.append({"alarm": str(t), "duration_min": dur,
                        "run_start": str(all_index[best[0]]), "run_end": str(all_index[best[1] - 1])})

print(f"[DUR] {len(tp_durations)} alarmes com episodio raw associado (dos {eval_ref['alarms_with_detected_anomaly_in_window']} hits)", flush=True)

# --- FP: episodios raw fora de qualquer janela de alarme, em estado "on" ---
alarm_all_times = df_alarm_group["Data da Ocorrencia"]  # exclusao usa TODOS os alarmes (nao so OOS), mesmo criterio do projeto
fp_durations_all = []
fp_durations_residual = []
for (s, e) in raw_runs:
    mid = s  # usa o inicio do episodio pra checar "near_alarm" (criterio conservador: se o INICIO ja esta longe, conta como FP)
    if near_alarm_arr[s] or not eval_mask_arr[s]:
        continue  # dentro de janela de alarme, ou fora do periodo OOS -- nao e o "residual FP" que nos interessa
    dur = run_duration_minutes(s, e)
    fp_durations_all.append(dur)
    if flags_reference[s:e].any():
        fp_durations_residual.append(dur)

print(f"[DUR] {len(fp_durations_all)} episodios raw de FP (OOS, fora de janela de alarme), "
      f"dos quais {len(fp_durations_residual)} sobrevivem aos portoes hoje (FP residual)", flush=True)


def describe(name, arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        print(f"[DUR] {name}: vazio")
        return {}
    stats = {
        "n": len(arr), "mean": float(arr.mean()), "median": float(np.median(arr)),
        "min": float(arr.min()), "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)), "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)), "p90": float(np.percentile(arr, 90)),
    }
    print(f"[DUR] {name}: n={stats['n']} media={stats['mean']:.1f}min mediana={stats['median']:.1f}min "
          f"min={stats['min']:.1f} p10={stats['p10']:.1f} p25={stats['p25']:.1f} "
          f"p75={stats['p75']:.1f} p90={stats['p90']:.1f} max={stats['max']:.1f}", flush=True)
    return stats


stats_tp = describe("TP (precursor de alarme batido)", tp_durations)
stats_fp_all = describe("FP-todos (raw, fora de alarme, OOS)", fp_durations_all)
stats_fp_residual = describe("FP-residual (sobrevive aos portoes hoje)", fp_durations_residual)

# --- teste de filtro: duracao minima = o MENOR dos TP (custo zero por construcao) ---
if tp_durations:
    min_tp = min(tp_durations)
    n_removed = sum(1 for d in fp_durations_residual if d < min_tp)
    frac_removed = n_removed / len(fp_durations_residual) if fp_durations_residual else 0.0
    print(f"\n[DUR FILTRO] duracao minima = menor TP observado ({min_tp:.1f}min): "
          f"remove {n_removed}/{len(fp_durations_residual)} episodios de FP residual "
          f"({frac_removed*100:.1f}%) sem custar nenhum dos {len(tp_durations)} TP (por construcao)", flush=True)

    # varre uma pequena grade de duracao minima pra ver a curva completa
    print("\n[DUR FILTRO grade]")
    for thr_min in [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120]:
        tp_lost = sum(1 for d in tp_durations if d < thr_min)
        fp_removed = sum(1 for d in fp_durations_residual if d < thr_min)
        frac = fp_removed / len(fp_durations_residual) if fp_durations_residual else 0.0
        print(f"  min_duracao={thr_min}min: TP perdidos={tp_lost}/{len(tp_durations)}  "
              f"FP-residual removido={fp_removed}/{len(fp_durations_residual)} ({frac*100:.1f}%)", flush=True)

report = {
    "group": group_name,
    "reference": {"hit_rate": eval_ref["hit_rate"], "normal_alert_rate": fp_ref},
    "tp_durations_min": tp_durations,
    "tp_details": tp_details,
    "fp_durations_all_min": fp_durations_all,
    "fp_durations_residual_min": fp_durations_residual,
    "stats_tp": stats_tp,
    "stats_fp_all": stats_fp_all,
    "stats_fp_residual": stats_fp_residual,
}
out_path = os.path.join(OUTPUT_DIR, "analise_duracao_score_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
task.upload_artifact("duracao_report_json", artifact_object=out_path)
print(f"\n[DONE] Relatorio salvo em: {out_path}", flush=True)
