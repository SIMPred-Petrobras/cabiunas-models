"""v4: mesma base do v2 (pre-processamento real + avaliacao por
episodio + janela de treino EXPANSIVA, que venceu o teste v2-vs-v3),
corrigindo um bug encontrado nos logs do v2/v3: o corte
`k = min(k, MAX_COMPONENTS)` fazia o PCA travar sempre em
MAX_COMPONENTS=20 componentes, MESMO quando isso nao alcancava a meta
de 90% de variancia explicada (na pratica ficava estacionado em
60-71% em todos os meses -- ver `docs/analise_pca_monitoramento_sistema.md`).
Ou seja, o teto artificial vencia o criterio de variancia antes dele
ser satisfeito, ao contrario do que a metodologia documentada dizia.

Correcao: MAX_COMPONENTS sobe de 20 para 150 (ainda bem menor que as
594 features de entrada, pra manter o PCA rapido), e o script agora
avisa explicitamente quando mesmo com o novo teto a meta de 90% nao e
atingida, em vez de mascarar isso silenciosamente.

Ver docs/analise_pca_monitoramento_sistema.md.

Uso:
    PYTHONPATH=. python scripts/pca_monitoramento_sistema/pca_walkforward_v4_pca_variancia_real.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from clearml import Task, Dataset

from src.cnn1d_ae.config import PipelineConfig
from src.cnn1d_ae.preprocess import (
    build_group_dataframe, select_feature_columns, clip_outliers,
    normalize_train_only, build_exclusion_mask,
)
from src.cnn1d_ae.automl_models import fit_ocsvm, ocsvm_error, fit_isolation_forest, isolation_forest_error
from src.cnn1d_ae.scoring import map_seq_to_point_anomalies, eval_alarm_hit_rate, build_operational_state

CLEARML_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"  # "Cabiunas full 2024-2026 30s"
CLEARML_PROJECT_NAME = "TesteMLCab"
CLEARML_DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

task = Task.init(
    project_name=CLEARML_PROJECT_NAME,
    task_name="pca-walkforward::monitoramento_sistema_v4_pca_variancia_real",
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
VIBRATION_SENSORS = [s for s in ALL_SENSORS if s.startswith("TV_")]

EXCLUDE_MIN_ALARM = 1440
DEBOUNCE = 6
PCT_THRESHOLD = 99.0
VARIANCE_TARGET = 0.90
MAX_COMPONENTS = 150  # v2/v3 tinham 20, que travava antes de atingir VARIANCE_TARGET (bug corrigido aqui)
OCSVM_MAX_TRAIN = 50000
RANDOM_SEED = 42
BURN_IN_MONTHS = 2

# TESTE_RAPIDO: se setado, so processa alguns meses (smoke test antes do full run)
FAST_MONTHS_LIMIT = int(os.environ.get("FAST_MONTHS_LIMIT", "0")) or None

_config = {
    "n_sensores_brutos": len(ALL_SENSORS),
    "derived_rolling_windows": [12, 120, 480, 2880],
    "texture_sensors": VIBRATION_SENSORS,
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
task.connect(_config, name="pca_walkforward_v4_config")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

cfg = PipelineConfig()
cfg.TIME_COL = "data_datetime"
cfg.TRAIN_SOURCE = "raw"
cfg.ENABLE_DERIVED_FEATURES = True
cfg.DERIVED_ROLLING_WINDOWS = [12, 120, 480, 2880]
cfg.TEXTURE_SENSORS = VIBRATION_SENSORS
cfg.OUTLIER_MODE = "quantile"
cfg.OUTLIER_Q_LOW = 0.001
cfg.OUTLIER_Q_HIGH = 0.999
cfg.NORMALIZE_MODE = "zscore"
cfg.INTERPOLATE_LIMIT = 3

print("Lendo dados brutos...", flush=True)
cols = [cfg.TIME_COL, "RUNNING_A"] + ALL_SENSORS
df_raw = pd.read_csv(RAW_CSV, usecols=cols, low_memory=False)
df_raw[cfg.TIME_COL] = pd.to_datetime(df_raw[cfg.TIME_COL], errors="coerce")
for c in ["RUNNING_A"] + ALL_SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").astype("float32")
df_raw = df_raw.dropna(subset=[cfg.TIME_COL]).drop_duplicates(subset=[cfg.TIME_COL]).sort_values(cfg.TIME_COL).reset_index(drop=True)
running_a_raw = df_raw.set_index(cfg.TIME_COL)["RUNNING_A"]
print("shape bruto:", df_raw.shape, flush=True)

print("Construindo features derivadas (build_group_dataframe -- pipeline real)...", flush=True)
df_use, long_gap_mask = build_group_dataframe(cfg, df_raw, df_raw, ALL_SENSORS)
feature_cols = select_feature_columns(cfg, df_use, ALL_SENSORS)
print("n_features_total (antes do PCA):", len(feature_cols), flush=True)
df_use = df_use[feature_cols]

running_a = running_a_raw.reindex(df_use.index).ffill().bfill()

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

near_any_alarm = build_exclusion_mask(df_use.index, all_alarm_times, EXCLUDE_MIN_ALARM)

print("Calculando estado operacional (on/off/transiente)...", flush=True)
operational_state = build_operational_state(
    index=df_use.index, sensor_series=running_a,
    off_value_quantile=0.05, off_abs_threshold=0.5, off_long_min_hours=4.0,
    transient_padding_minutes=60, transient_diff_quantile=0.99,
)
is_on = (operational_state == "on")
print("fracao 'on':", is_on.mean(), flush=True)

months = pd.period_range(df_use.index.min(), df_use.index.max(), freq="M")
print("total de meses no periodo:", len(months), flush=True)

flags_ocsvm = pd.Series(0, index=df_use.index, dtype="int8")
flags_iforest = pd.Series(0, index=df_use.index, dtype="int8")
scored_mask = pd.Series(False, index=df_use.index)

n_scored = 0
for i, m in enumerate(months):
    if i < BURN_IN_MONTHS:
        continue
    if FAST_MONTHS_LIMIT and n_scored >= FAST_MONTHS_LIMIT:
        print(f"[{m}] parado (FAST_MONTHS_LIMIT={FAST_MONTHS_LIMIT})", flush=True)
        break

    month_start = m.start_time
    month_end = m.end_time + pd.Timedelta(seconds=30)

    train_idx = (df_use.index < month_start) & is_on.values & (~near_any_alarm)
    eval_idx = (df_use.index >= month_start) & (df_use.index < month_end)
    n_train, n_eval = int(train_idx.sum()), int(eval_idx.sum())
    if n_train < 2000 or n_eval < 100:
        print(f"[{m}] pulado (n_train={n_train} n_eval={n_eval})", flush=True)
        continue

    df_train = df_use.loc[train_idx]
    df_eval = df_use.loc[eval_idx]

    # mesma sequencia exata do resto do projeto: clip_outliers (cada df
    # com seus proprios quantis, igual automl_pipeline.py) + normalize
    # treino-apenas.
    df_train = clip_outliers(df_train, cfg)
    df_eval = clip_outliers(df_eval, cfg)
    df_train_z, df_eval_z, _, _ = normalize_train_only(cfg, df_train, df_eval)

    X_train = np.nan_to_num(df_train_z.values, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval = np.nan_to_num(df_eval_z.values, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp = min(MAX_COMPONENTS, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, VARIANCE_TARGET) + 1)
    k = max(2, min(k, n_comp))
    if cum_var[n_comp - 1] < VARIANCE_TARGET:
        print(f"[{m}] AVISO: mesmo com todos os {n_comp} componentes disponiveis, "
              f"variancia explicada so chega a {cum_var[n_comp-1]:.2f} (< meta {VARIANCE_TARGET}) "
              f"-- MAX_COMPONENTS ainda esta limitando o resultado", flush=True)
    X_train_pca = X_train_pca[:, :k]
    X_eval_pca = pca.transform(X_eval)[:, :k]

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
    n_scored += 1
    print(f"[{m}] n_train={n_train} n_eval={n_eval} k_pca={k} (var={cum_var[k-1]:.2f}) "
          f"flag_ocsvm={flags_ocsvm.loc[eval_idx].sum()} flag_iforest={flags_iforest.loc[eval_idx].sum()}", flush=True)

print("\nAplicando debounce e avaliando contra o catalogo completo...", flush=True)

first_scored_month = months[BURN_IN_MONTHS].start_time
for name, flags in [("ocsvm", flags_ocsvm), ("iforest", flags_iforest)]:
    point_df = map_seq_to_point_anomalies(
        flags.values, df_use.index, time_steps=1,
        point_rule="all_of_window", point_window=DEBOUNCE, point_min_count=DEBOUNCE,
    )
    point_df["is_anom_point"] = (
        point_df["is_anom_point"].astype(int)
        & scored_mask.reindex(point_df.index).fillna(False).astype(int)
        & is_on.reindex(point_df.index).fillna(False).astype(int)
    )
    print(f"  [{name}] pontos anomalos totais (pos mascara operacional): {point_df['is_anom_point'].sum()}"
          f" de {int(scored_mask.sum())} pontos avaliados", flush=True)

    scored_alarms = alarm[(alarm["Data da Ocorrencia"] >= first_scored_month)]
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
    csv_path = os.path.join(OUTPUT_DIR, f"resultado_v4_{name}_por_tag.csv")
    tag_df.to_csv(csv_path, index=False)
    task.get_logger().report_scalar(title="fp_rate_geral_pct_pontos", series=name, value=float(fp_rate) * 100, iteration=0)
    task.get_logger().report_scalar(
        title="hit_rate_medio_entre_tags", series=name, value=float(tag_df["hit_rate"].mean()) * 100, iteration=0
    )
    task.upload_artifact(name=f"resultado_v4_{name}_por_tag", artifact_object=csv_path)

    # --- avaliacao por EPISODIO (nao por tag individual) -----------------
    # Muitos tags disparam juntos pro mesmo evento fisico (confirmado:
    # ate 46 dos 47 tags do catalogo juntos em minutos). Contar cada tag
    # separadamente infla o hit_rate agregado. Agrupa o catalogo inteiro
    # em episodios (gap>60min = novo episodio) e avalia deteccao por
    # episodio distinto, alem de FP em episodios/mes (mesma unidade da
    # referencia externa: "0,94 FP/mes").
    ep_gaps = scored_alarms["Data da Ocorrencia"].diff().dt.total_seconds().fillna(99999) / 60.0
    ep_id = (ep_gaps > 60).cumsum()
    episodes = scored_alarms.groupby(ep_id)["Data da Ocorrencia"].agg(["min", "max"])
    n_episodes = len(episodes)
    hits = 0
    for _, row in episodes.iterrows():
        t0, t1 = row["min"] - pd.Timedelta(minutes=EXCLUDE_MIN_ALARM), row["max"] + pd.Timedelta(minutes=EXCLUDE_MIN_ALARM)
        if point_df.loc[t0:t1, "is_anom_point"].sum() > 0:
            hits += 1
    print(f"\n[{name}] avaliacao por EPISODIO (nao por tag): {hits}/{n_episodes} episodios distintos detectados "
          f"({hits/n_episodes*100:.1f}% se n_episodes>0)" if n_episodes else "sem episodios no periodo avaliado")

    # FP em episodios/mes: agrupa os pontos anomalos (fora de qualquer
    # janela de alarme) em episodios e conta por mes avaliado.
    fp_points = point_df.loc[fp_mask.values]
    fp_idx = fp_points.index[fp_points["is_anom_point"] == 1]
    if len(fp_idx):
        fp_gaps = fp_idx.to_series().diff().dt.total_seconds().fillna(99999) / 60.0
        fp_ep_id = (fp_gaps > 60).cumsum()
        n_fp_episodes = fp_idx.to_series().groupby(fp_ep_id).ngroups
    else:
        n_fp_episodes = 0
    n_scored_months_final = n_scored
    fp_per_month = n_fp_episodes / n_scored_months_final if n_scored_months_final else float("nan")
    print(f"[{name}] FP em episodios: {n_fp_episodes} episodios distintos em {n_scored_months_final} meses avaliados "
          f"-> {fp_per_month:.2f} FP/mes (mesma unidade da referencia externa)")

    task.get_logger().report_scalar(
        title="hit_rate_por_episodio_pct", series=name,
        value=(hits / n_episodes * 100) if n_episodes else 0.0, iteration=0,
    )
    task.get_logger().report_scalar(title="fp_episodios_por_mes", series=name, value=float(fp_per_month), iteration=0)

print("\nOK - fim do script v4", flush=True)
task.mark_completed(status_message="Monitoramento PCA walk-forward v4 (PCA com variancia corrigida) concluido com sucesso.")
task.close()
