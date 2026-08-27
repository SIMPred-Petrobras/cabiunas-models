"""Monitoramento multivariado via PCA com retreino mensal (walk-forward).

Script standalone (nao integrado ao automl_pipeline.py -- ver
docs/analise_pca_monitoramento_sistema.md pra motivacao e resultado).
Usa todos os sensores brutos disponiveis + PCA + ocsvm/iforest (mesmos
fitters do resto do projeto), retreinando mes a mes em janela expansiva,
avaliado contra o catalogo inteiro de alarmes.

Uso:
    PYTHONPATH=. python scripts/pca_monitoramento_sistema/pca_walkforward.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from clearml import Task, Dataset

from src.cnn1d_ae.automl_models import fit_ocsvm, ocsvm_error, fit_isolation_forest, isolation_forest_error
from src.cnn1d_ae.scoring import map_seq_to_point_anomalies, eval_alarm_hit_rate, build_operational_state
from src.cnn1d_ae.preprocess import build_exclusion_mask

CLEARML_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"  # "Cabiunas full 2024-2026 30s"
CLEARML_PROJECT_NAME = "TesteMLCab"
CLEARML_DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

task = Task.init(
    project_name=CLEARML_PROJECT_NAME,
    task_name="pca-walkforward::monitoramento_sistema_multivariado",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(CLEARML_DOCKER_IMAGE)

print("Resolvendo dataset ClearML...", flush=True)
_dataset_root = Dataset.get(dataset_id=CLEARML_DATASET_ID).get_local_copy()
RAW_CSV = os.path.join(_dataset_root, "sensores_full_2024_2026_30s.csv")
ALARM_CSV = os.path.join(_dataset_root, "alarmes_selecionados_turbina_a.csv")

ALL_SENSORS = [
    "954005_624_TI_0325", "954005_624_PI_0315", "954005_624_PI_0319",
    "954005_624_PI_0340", "954005_624_PI_0339", "954005_624_PDI_0317",
    "TC382_03_A", "T5_AVG_A", "TC382_02_A", "954005_624_PDI_0302",
    "TC382_05_A", "954005_624_TI_0315", "954005_624_TI_0317", "TC382_06_A",
    "TC382_01_A", "TC382_04_A", "954005_624_PDIT_0305", "954005_624_TI_0305",
    "954005_624_TI_0307", "954005_624_TI_0303", "TV_355Y_A", "TV_353X_A",
    "TV_352X_A", "954005_624_PI_0307", "954005_624_PI_0308", "TV_353Y_A",
    "TV_355X_A", "TV_351Y_A", "TV_354Y_A", "PI_5134001", "TV_351X_A",
    "954005_624_TI_0301", "954005_624_PDI_0338", "954005_624_PDI_0301",
    "TV_354X_A", "TV_352Y_A",
]
print(f"n_sensores_brutos={len(ALL_SENSORS)}", flush=True)

HORIZONS_MIN = {"3h": 360, "12h": 1440, "48h": 5760}  # em pontos de 30s
EXCLUDE_MIN_ALARM = 1440  # 24h, mesmo padrao do resto do projeto
DEBOUNCE = 6
PCT_THRESHOLD = 99.0
VARIANCE_TARGET = 0.90
MAX_COMPONENTS = 20
OCSVM_MAX_TRAIN = 50000
RANDOM_SEED = 42
BURN_IN_MONTHS = 2

_config = {
    "n_sensores_brutos": len(ALL_SENSORS),
    "horizons_min": HORIZONS_MIN,
    "exclude_min_alarm": EXCLUDE_MIN_ALARM,
    "debounce": DEBOUNCE,
    "pct_threshold": PCT_THRESHOLD,
    "variance_target": VARIANCE_TARGET,
    "max_components": MAX_COMPONENTS,
    "ocsvm_max_train": OCSVM_MAX_TRAIN,
    "random_seed": RANDOM_SEED,
    "burn_in_months": BURN_IN_MONTHS,
    "clearml_dataset_id": CLEARML_DATASET_ID,
}
task.connect(_config, name="pca_walkforward_config")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

print("Lendo dados brutos...", flush=True)
cols = ["data_datetime", "RUNNING_A"] + ALL_SENSORS
df = pd.read_csv(RAW_CSV, usecols=cols, low_memory=False)
df["data_datetime"] = pd.to_datetime(df["data_datetime"], errors="coerce")
for c in ["RUNNING_A"] + ALL_SENSORS:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
df = df.dropna(subset=["data_datetime"]).drop_duplicates(subset=["data_datetime"]).sort_values("data_datetime").reset_index(drop=True)
df = df.set_index("data_datetime")
print("shape bruto:", df.shape, flush=True)

print("Construindo features de volatilidade multi-horizonte (3h/12h/48h)...", flush=True)
feature_cols = list(ALL_SENSORS)
for label, win in HORIZONS_MIN.items():
    block = df[ALL_SENSORS].rolling(win, min_periods=win // 2).std()
    block.columns = [f"{c}__std_{label}" for c in ALL_SENSORS]
    df = pd.concat([df, block.astype("float32")], axis=1)
    feature_cols += list(block.columns)
    print(f"  ...{label} ok", flush=True)

print("n_features_total (antes do PCA):", len(feature_cols), flush=True)

print("Lendo catalogo de alarmes (todos os tags)...", flush=True)
alarm = pd.read_csv(ALARM_CSV)
if "Data da Ocorrência" in alarm.columns and "Data da Ocorrencia" not in alarm.columns:
    alarm["Data da Ocorrencia"] = alarm["Data da Ocorrência"]
if "Tag Alarme" in alarm.columns and "Tag" not in alarm.columns:
    alarm["Tag"] = alarm["Tag Alarme"]
alarm = alarm[alarm["Status"].astype(str).str.startswith("ACT")].copy()
alarm["Data da Ocorrencia"] = pd.to_datetime(alarm["Data da Ocorrencia"], errors="coerce")
alarm = alarm.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")
all_alarm_times = alarm["Data da Ocorrencia"]
print("total eventos ACT no catalogo:", len(alarm), " tags distintos:", alarm["Tag"].nunique(), flush=True)

near_any_alarm = build_exclusion_mask(df.index, all_alarm_times, EXCLUDE_MIN_ALARM)

print("Calculando estado operacional (on/off/transiente)...", flush=True)
operational_state = build_operational_state(
    index=df.index, sensor_series=df["RUNNING_A"],
    off_value_quantile=0.05, off_abs_threshold=0.5, off_long_min_hours=4.0,
    transient_padding_minutes=60, transient_diff_quantile=0.99,
)
is_on = (operational_state == "on")
print("fracao 'on':", is_on.mean(), flush=True)

months = pd.period_range(df.index.min(), df.index.max(), freq="M")
print("total de meses no periodo:", len(months), flush=True)

flags_ocsvm = pd.Series(0, index=df.index, dtype="int8")
flags_iforest = pd.Series(0, index=df.index, dtype="int8")
scored_mask = pd.Series(False, index=df.index)

for i, m in enumerate(months):
    if i < BURN_IN_MONTHS:
        continue
    month_start = m.start_time
    month_end = m.end_time + pd.Timedelta(seconds=30)

    train_idx = (df.index < month_start) & is_on.values & (~near_any_alarm)
    eval_idx = (df.index >= month_start) & (df.index < month_end)
    n_train, n_eval = int(train_idx.sum()), int(eval_idx.sum())
    if n_train < 2000 or n_eval < 100:
        print(f"[{m}] pulado (n_train={n_train} n_eval={n_eval})", flush=True)
        continue

    X_train_raw = df.loc[train_idx, feature_cols].values
    X_eval_raw = df.loc[eval_idx, feature_cols].values

    col_mean = np.nanmean(X_train_raw, axis=0)
    col_std = np.nanstd(X_train_raw, axis=0)
    col_std[col_std < 1e-6] = 1.0
    X_train = np.nan_to_num((X_train_raw - col_mean) / col_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval = np.nan_to_num((X_eval_raw - col_mean) / col_std, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp = min(MAX_COMPONENTS, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, VARIANCE_TARGET) + 1)
    k = max(2, min(k, n_comp))
    X_train_pca = X_train_pca[:, :k]
    X_eval_pca = pca.transform(X_eval)[:, :k]

    # ocsvm (kernel RBF) nao escala pra centenas de milhares de linhas --
    # mesmo limite de subamostragem usado no resto do projeto
    # (AUTOML_OCSVM_MAX_TRAIN_SAMPLES=50000).
    if X_train_pca.shape[0] > OCSVM_MAX_TRAIN:
        rng = np.random.default_rng(RANDOM_SEED)
        sub_idx = rng.choice(X_train_pca.shape[0], size=OCSVM_MAX_TRAIN, replace=False)
        X_train_ocsvm = X_train_pca[sub_idx]
    else:
        X_train_ocsvm = X_train_pca
    clf = fit_ocsvm(X_train_ocsvm, nu=0.05, gamma="scale")
    train_err_o = ocsvm_error(clf, X_train_ocsvm)
    eval_err_o = ocsvm_error(clf, X_eval_pca)
    thr_o = np.percentile(train_err_o, PCT_THRESHOLD)
    flags_ocsvm.loc[eval_idx] = (eval_err_o > thr_o).astype("int8")

    iso = fit_isolation_forest(X_train_pca, contamination=0.05, n_estimators=200, random_state=RANDOM_SEED)
    train_err_i = isolation_forest_error(iso, X_train_pca)
    eval_err_i = isolation_forest_error(iso, X_eval_pca)
    thr_i = np.percentile(train_err_i, PCT_THRESHOLD)
    flags_iforest.loc[eval_idx] = (eval_err_i > thr_i).astype("int8")

    scored_mask.loc[eval_idx] = True
    print(f"[{m}] n_train={n_train} n_eval={n_eval} k_pca={k} (var={cum_var[k-1]:.2f}) "
          f"flag_ocsvm={flags_ocsvm.loc[eval_idx].sum()} flag_iforest={flags_iforest.loc[eval_idx].sum()}", flush=True)

print("\nAplicando debounce e avaliando contra o catalogo completo...", flush=True)

for name, flags in [("ocsvm", flags_ocsvm), ("iforest", flags_iforest)]:
    point_df = map_seq_to_point_anomalies(
        flags.values, df.index, time_steps=1,
        point_rule="all_of_window", point_window=DEBOUNCE, point_min_count=DEBOUNCE,
    )
    point_df["is_anom_point"] = (
        point_df["is_anom_point"].astype(int)
        & scored_mask.reindex(point_df.index).fillna(False).astype(int)
        & is_on.reindex(point_df.index).fillna(False).astype(int)
    )
    print(f"  [{name}] pontos anomalos totais (pos mascara operacional): {point_df['is_anom_point'].sum()}"
          f" de {int(scored_mask.sum())} pontos avaliados", flush=True)

    scored_alarms = alarm[(alarm["Data da Ocorrencia"] >= months[BURN_IN_MONTHS].start_time)]
    fp_mask = (~near_any_alarm) & scored_mask & is_on
    fp_rate = point_df.loc[fp_mask.values, "is_anom_point"].mean()

    tag_results = []
    for tag, g in scored_alarms.groupby("Tag"):
        stats = eval_alarm_hit_rate(g, point_df, EXCLUDE_MIN_ALARM)
        if stats["n_alarms"] >= 3:
            tag_results.append({"tag": tag, **stats})
    tag_df = pd.DataFrame(tag_results).sort_values("hit_rate", ascending=False)

    print(f"\n=== modelo {name} ===")
    print(f"FP geral (on, longe de qualquer alarme): {fp_rate*100:.3f}%")
    print(f"Tags com >=3 alarmes genuinos no periodo avaliado: {len(tag_df)}")
    print(tag_df.to_string(index=False))
    csv_path = os.path.join(OUTPUT_DIR, f"resultado_{name}_por_tag.csv")
    tag_df.to_csv(csv_path, index=False)

    task.get_logger().report_scalar(title="fp_rate_geral", series=name, value=float(fp_rate) * 100, iteration=0)
    task.get_logger().report_scalar(
        title="hit_rate_medio_entre_tags", series=name, value=float(tag_df["hit_rate"].mean()) * 100, iteration=0
    )
    task.upload_artifact(name=f"resultado_{name}_por_tag", artifact_object=csv_path)

print("\nOK - fim do script", flush=True)
task.mark_completed(status_message="Monitoramento PCA walk-forward concluido com sucesso.")
task.close()
