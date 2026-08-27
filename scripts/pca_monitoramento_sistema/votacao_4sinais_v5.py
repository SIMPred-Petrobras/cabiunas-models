"""v5: tenta reproduzir a arquitetura de votacao do Francisco
(branch `feat/pdm-deteccao-4sinais`, ver docs/analise_pca_monitoramento_sistema.md)
na nossa pipeline -- em vez de UM modelo PCA sobre todos os sensores juntos
(v2/v3/v4), usa QUATRO sinais independentes, agrupados por subsistema
fisico, e exige VOTACAO (>=N de 4 simultaneos) para disparar o alarme
final:

  1. `temperatura`  -- multivariado (14 sensores de temperatura/exaustao),
                       PCA + iforest (nosso AutoML, no lugar do autoencoder
                       dele -- "devemos usar automl como estavamos").
  2. `pressao_oleo` -- multivariado (12 sensores de pressao/oleo/selagem),
                       mesma receita.
  3. `mancal_spread`-- univariado: TI_0305 menos a mediana dos 3 mancais
                       irmaos (TI_0301/0303/0307). Robusto (mediana/MAD),
                       nao reusa o `thermal_array_spread` (desvio-padrao do
                       array) ja existente no projeto porque aqui queremos
                       o DESVIO COM SINAL de UM mancal especifico em
                       relacao aos outros, nao a dispersao geral do grupo.
  4. `selagem_z`    -- univariado: PDIT_0305 isolado, mesmo z-robusto.
                       Existe porque a familia pressao_oleo (12 sensores)
                       dilui o sinal desse sensor especifico na media.

Cada sinal e treinado/pontuado com o mesmo walk-forward mensal expansivo
que venceu o teste v2-vs-v3 (ver doc), usando o pre-processamento real do
projeto (clip_outliers, features derivadas, normalize_train_only) para os
dois sinais multivariados. Vibracao fica de fora dos 4 sinais (mesma
decisao do Francisco: deriva sem falha associada).

Pos-processamento: cada sinal tem seu proprio debounce/sustentacao (30 min,
igual ao `sustain_min` dele) antes da votacao -- a votacao em si soma
quantos dos 4 sinais estao "ligados" no mesmo instante e compara com
VOTE_MIN. Reporta tambem VOTE_MIN=3 e 4 (sensibilidade), que nao custam
retreino extra (e so mudar o limiar sobre a contagem de votos ja calculada).

Ver docs/analise_pca_monitoramento_sistema.md.

Uso:
    PYTHONPATH=. python scripts/pca_monitoramento_sistema/votacao_4sinais_v5.py
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
from src.cnn1d_ae.automl_models import fit_isolation_forest, isolation_forest_error
from src.cnn1d_ae.scoring import map_seq_to_point_anomalies, eval_alarm_hit_rate, build_operational_state

CLEARML_DATASET_ID = "a97ba56ba14840fbb1125c2a82f883c9"  # "Cabiunas full 2024-2026 30s"
CLEARML_PROJECT_NAME = "TesteMLCab"
CLEARML_DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
REMOTE_QUEUE = "default"
RUN_REMOTE = os.getenv("RUN_REMOTE", "true").lower() != "false"
OUTPUT_DIR = os.path.dirname(__file__)

task = Task.init(
    project_name=CLEARML_PROJECT_NAME,
    task_name="pca-walkforward::monitoramento_sistema_v5_votacao_4sinais",
    output_uri=True,
    reuse_last_task_id=False,
)
task.set_base_docker(CLEARML_DOCKER_IMAGE)

print("Resolvendo dataset ClearML...", flush=True)
_dataset_root = Dataset.get(dataset_id=CLEARML_DATASET_ID).get_local_copy()
RAW_CSV = os.path.join(_dataset_root, "sensores_full_2024_2026_30s.csv")
ALARM_CSV = os.path.join(_dataset_root, "alarmes_selecionados_turbina_a.csv")

# Mesmo agrupamento por familia fisica usado por ele (config.py da branch
# feat/pdm-deteccao-4sinais), com os nomes de tag do NOSSO dataset (identicos).
TEMPERATURE_SENSORS = [
    "954005_624_TI_0325", "954005_624_TI_0315", "954005_624_TI_0317",
    "954005_624_TI_0305", "954005_624_TI_0307", "954005_624_TI_0303",
    "954005_624_TI_0301",
    "TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A",
    "TC382_06_A", "T5_AVG_A",
]
PRESSURE_SENSORS = [
    "954005_624_PI_0315", "954005_624_PI_0319", "954005_624_PI_0340",
    "954005_624_PI_0339", "954005_624_PDI_0317", "954005_624_PDI_0302",
    "954005_624_PDIT_0305", "954005_624_PI_0307", "954005_624_PI_0308",
    "PI_5134001", "954005_624_PDI_0338", "954005_624_PDI_0301",
]
MANCAL_TARGET = "954005_624_TI_0305"
MANCAL_SIBLINGS = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0307"]
SELAGEM_TARGET = "954005_624_PDIT_0305"

ALL_NEEDED_SENSORS = sorted(set(TEMPERATURE_SENSORS) | set(PRESSURE_SENSORS))

EXCLUDE_MIN_ALARM = 1440
SUSTAIN_MINUTES = 30  # "sustain_min" do colega
DEBOUNCE = int(SUSTAIN_MINUTES * 60 / 30)  # 60 amostras a 30s de cadencia
Z_THRESHOLD = 3.0  # limiar do sinal univariado (early_warning dele usa 3.0)
PCT_THRESHOLD = 99.0
VARIANCE_TARGET = 0.90
MAX_COMPONENTS = 50  # familias tem so 12-14 sensores brutos (~170-200 features), nao precisa do teto de 150 do v4
RANDOM_SEED = 42
BURN_IN_MONTHS = 2
VOTE_MIN_PRIMARY = 2  # pedido do usuario: votacao 2-de-4

FAST_MONTHS_LIMIT = int(os.environ.get("FAST_MONTHS_LIMIT", "0")) or None

_config = {
    "families": {"temperatura": TEMPERATURE_SENSORS, "pressao_oleo": PRESSURE_SENSORS},
    "mancal_target": MANCAL_TARGET, "mancal_siblings": MANCAL_SIBLINGS,
    "selagem_target": SELAGEM_TARGET,
    "sustain_minutes": SUSTAIN_MINUTES,
    "z_threshold": Z_THRESHOLD,
    "pct_threshold": PCT_THRESHOLD,
    "variance_target": VARIANCE_TARGET,
    "max_components": MAX_COMPONENTS,
    "burn_in_months": BURN_IN_MONTHS,
    "vote_min_primary": VOTE_MIN_PRIMARY,
    "exclude_min_alarm": EXCLUDE_MIN_ALARM,
    "clearml_dataset_id": CLEARML_DATASET_ID,
}
task.connect(_config, name="votacao_4sinais_v5_config")

if RUN_REMOTE and task.running_locally():
    task.get_logger().report_text(f"Enqueuing task for remote execution on queue: {REMOTE_QUEUE}")
    task.execute_remotely(queue_name=REMOTE_QUEUE, exit_process=True)

cfg = PipelineConfig()
cfg.TIME_COL = "data_datetime"
cfg.TRAIN_SOURCE = "raw"
cfg.ENABLE_DERIVED_FEATURES = True
cfg.DERIVED_ROLLING_WINDOWS = [12, 120, 480, 2880]
cfg.TEXTURE_SENSORS = []  # vibracao fora -- nenhum sensor destas 2 familias usa textura
cfg.OUTLIER_MODE = "quantile"
cfg.OUTLIER_Q_LOW = 0.001
cfg.OUTLIER_Q_HIGH = 0.999
cfg.NORMALIZE_MODE = "zscore"
cfg.INTERPOLATE_LIMIT = 3

print("Lendo dados brutos...", flush=True)
cols = [cfg.TIME_COL, "RUNNING_A"] + ALL_NEEDED_SENSORS
df_raw = pd.read_csv(RAW_CSV, usecols=cols, low_memory=False)
df_raw[cfg.TIME_COL] = pd.to_datetime(df_raw[cfg.TIME_COL], errors="coerce")
for c in ["RUNNING_A"] + ALL_NEEDED_SENSORS:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").astype("float32")
df_raw = df_raw.dropna(subset=[cfg.TIME_COL]).drop_duplicates(subset=[cfg.TIME_COL]).sort_values(cfg.TIME_COL).reset_index(drop=True)
running_a_raw = df_raw.set_index(cfg.TIME_COL)["RUNNING_A"]
print("shape bruto:", df_raw.shape, flush=True)

print("Construindo familia 'temperatura' (14 sensores)...", flush=True)
df_temp_use, _ = build_group_dataframe(cfg, df_raw, df_raw, TEMPERATURE_SENSORS)
feat_temp = select_feature_columns(cfg, df_temp_use, TEMPERATURE_SENSORS)
print("  n_features temperatura:", len(feat_temp), flush=True)
df_temp_feat = df_temp_use[feat_temp]
mancal_target_raw = df_temp_use[MANCAL_TARGET]
mancal_siblings_raw = df_temp_use[MANCAL_SIBLINGS]
mancal_spread = mancal_target_raw - mancal_siblings_raw.median(axis=1)

print("Construindo familia 'pressao_oleo' (12 sensores)...", flush=True)
df_press_use, _ = build_group_dataframe(cfg, df_raw, df_raw, PRESSURE_SENSORS)
feat_press = select_feature_columns(cfg, df_press_use, PRESSURE_SENSORS)
print("  n_features pressao_oleo:", len(feat_press), flush=True)
df_press_feat = df_press_use[feat_press]
selagem_raw = df_press_use[SELAGEM_TARGET]

FULL_INDEX = df_temp_use.index
assert FULL_INDEX.equals(df_press_use.index), "indices das 2 familias divergiram -- investigar"

running_a = running_a_raw.reindex(FULL_INDEX).ffill().bfill()

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

near_any_alarm = build_exclusion_mask(FULL_INDEX, all_alarm_times, EXCLUDE_MIN_ALARM)

print("Calculando estado operacional (on/off/transiente)...", flush=True)
operational_state = build_operational_state(
    index=FULL_INDEX, sensor_series=running_a,
    off_value_quantile=0.05, off_abs_threshold=0.5, off_long_min_hours=4.0,
    transient_padding_minutes=60, transient_diff_quantile=0.99,
)
is_on = (operational_state == "on")
print("fracao 'on':", is_on.mean(), flush=True)

months = pd.period_range(FULL_INDEX.min(), FULL_INDEX.max(), freq="M")
print("total de meses no periodo:", len(months), flush=True)

SIGNAL_NAMES = ["temperatura", "pressao_oleo", "mancal_spread", "selagem_z"]
raw_flags = {name: pd.Series(0, index=FULL_INDEX, dtype="int8") for name in SIGNAL_NAMES}
scored_mask = pd.Series(False, index=FULL_INDEX)

n_scored = 0
for i, m in enumerate(months):
    if i < BURN_IN_MONTHS:
        continue
    if FAST_MONTHS_LIMIT and n_scored >= FAST_MONTHS_LIMIT:
        print(f"[{m}] parado (FAST_MONTHS_LIMIT={FAST_MONTHS_LIMIT})", flush=True)
        break

    month_start = m.start_time
    month_end = m.end_time + pd.Timedelta(seconds=30)

    train_idx = (FULL_INDEX < month_start) & is_on.values & (~near_any_alarm)
    eval_idx = (FULL_INDEX >= month_start) & (FULL_INDEX < month_end)
    n_train, n_eval = int(train_idx.sum()), int(eval_idx.sum())
    if n_train < 2000 or n_eval < 100:
        print(f"[{m}] pulado (n_train={n_train} n_eval={n_eval})", flush=True)
        continue

    # --- sinais multivariados (temperatura, pressao_oleo): PCA + iforest ---
    for fam_name, df_feat in [("temperatura", df_temp_feat), ("pressao_oleo", df_press_feat)]:
        df_train = df_feat.loc[train_idx]
        df_eval = df_feat.loc[eval_idx]
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
            print(f"[{m}][{fam_name}] AVISO: variancia so chega a {cum_var[n_comp-1]:.2f} com {n_comp} componentes", flush=True)
        X_train_pca = X_train_pca[:, :k]
        X_eval_pca = pca.transform(X_eval)[:, :k]

        iso = fit_isolation_forest(X_train_pca, contamination=0.05, n_estimators=200, random_state=RANDOM_SEED)
        train_err = isolation_forest_error(iso, X_train_pca)
        eval_err = isolation_forest_error(iso, X_eval_pca)
        thr = np.percentile(train_err, PCT_THRESHOLD)
        raw_flags[fam_name].loc[eval_idx] = (eval_err > thr).astype("int8")

    # --- sinais univariados (mancal_spread, selagem_z): z-robusto (mediana/MAD) ---
    for sig_name, raw_series in [("mancal_spread", mancal_spread), ("selagem_z", selagem_raw)]:
        train_vals = raw_series.loc[train_idx]
        med = float(train_vals.median())
        mad = float((train_vals - med).abs().median()) * 1.4826
        mad = max(mad, 1e-6)
        z_eval = (raw_series.loc[eval_idx] - med) / mad
        raw_flags[sig_name].loc[eval_idx] = (z_eval.abs() > Z_THRESHOLD).astype("int8")

    scored_mask.loc[eval_idx] = True
    n_scored += 1
    flag_counts = {name: int(raw_flags[name].loc[eval_idx].sum()) for name in SIGNAL_NAMES}
    print(f"[{m}] n_train={n_train} n_eval={n_eval} flags_brutos={flag_counts}", flush=True)

print("\nAplicando debounce (sustentacao de 30min) por sinal...", flush=True)
point_dfs = {}
for name in SIGNAL_NAMES:
    pdf = map_seq_to_point_anomalies(
        raw_flags[name].values, FULL_INDEX, time_steps=1,
        point_rule="all_of_window", point_window=DEBOUNCE, point_min_count=DEBOUNCE,
    )
    pdf["is_anom_point"] = (
        pdf["is_anom_point"].astype(int)
        & scored_mask.reindex(pdf.index).fillna(False).astype(int)
        & is_on.reindex(pdf.index).fillna(False).astype(int)
    )
    point_dfs[name] = pdf
    print(f"  [{name}] pontos anomalos sustentados: {pdf['is_anom_point'].sum()} de {int(scored_mask.sum())} pontos avaliados", flush=True)

vote_count = sum(point_dfs[name]["is_anom_point"] for name in SIGNAL_NAMES)

first_scored_month = months[BURN_IN_MONTHS].start_time
scored_alarms = alarm[(alarm["Data da Ocorrencia"] >= first_scored_month)]
fp_mask = (~near_any_alarm) & scored_mask & is_on

ep_gaps_cat = scored_alarms["Data da Ocorrencia"].diff().dt.total_seconds().fillna(99999) / 60.0
ep_id_cat = (ep_gaps_cat > 60).cumsum()
episodes_cat = scored_alarms.groupby(ep_id_cat)["Data da Ocorrencia"].agg(["min", "max"])
n_episodes_cat = len(episodes_cat)


def evaluate_series(name: str, is_anom_point: pd.Series):
    point_df_local = pd.DataFrame({"is_anom_point": is_anom_point})
    fp_rate = point_df_local.loc[fp_mask.values, "is_anom_point"].mean()

    tag_results = []
    for tag, g in scored_alarms.groupby("Tag"):
        stats = eval_alarm_hit_rate(g, point_df_local, EXCLUDE_MIN_ALARM)
        if stats["n_alarms"] >= 3:
            tag_results.append({"tag": tag, **stats})
    tag_df = pd.DataFrame(tag_results).sort_values("hit_rate", ascending=False) if tag_results else pd.DataFrame()

    hits = 0
    for _, row in episodes_cat.iterrows():
        t0 = row["min"] - pd.Timedelta(minutes=EXCLUDE_MIN_ALARM)
        t1 = row["max"] + pd.Timedelta(minutes=EXCLUDE_MIN_ALARM)
        if point_df_local.loc[t0:t1, "is_anom_point"].sum() > 0:
            hits += 1
    hit_rate_episodio = (hits / n_episodes_cat * 100) if n_episodes_cat else 0.0

    fp_points = point_df_local.loc[fp_mask.values]
    fp_idx = fp_points.index[fp_points["is_anom_point"] == 1]
    if len(fp_idx):
        fp_gaps = fp_idx.to_series().diff().dt.total_seconds().fillna(99999) / 60.0
        fp_ep_id = (fp_gaps > 60).cumsum()
        n_fp_episodes = fp_idx.to_series().groupby(fp_ep_id).ngroups
    else:
        n_fp_episodes = 0
    fp_per_month = n_fp_episodes / n_scored if n_scored else float("nan")

    print(f"\n=== {name} ===")
    print(f"FP geral (% pontos, on, longe de alarme): {fp_rate*100:.3f}%")
    print(f"hit_rate medio entre tags (n={len(tag_df)}): {tag_df['hit_rate'].mean()*100:.2f}%" if len(tag_df) else "hit_rate medio: sem tags")
    print(f"hit_rate por episodio: {hits}/{n_episodes_cat} ({hit_rate_episodio:.1f}%)")
    print(f"FP episodios/mes: {n_fp_episodes} em {n_scored} meses -> {fp_per_month:.2f}/mes")

    return {
        "fp_rate_pct": float(fp_rate) * 100,
        "hit_rate_medio_tags_pct": float(tag_df["hit_rate"].mean() * 100) if len(tag_df) else 0.0,
        "hit_rate_episodio_pct": hit_rate_episodio,
        "fp_episodios_por_mes": float(fp_per_month),
        "tag_df": tag_df,
    }


print("\n\n########## RESULTADO POR SINAL INDIVIDUAL (antes da votacao) ##########")
for name in SIGNAL_NAMES:
    res = evaluate_series(name, point_dfs[name]["is_anom_point"])
    task.get_logger().report_scalar(title="fp_rate_geral_pct_pontos", series=f"sinal_{name}", value=res["fp_rate_pct"], iteration=0)
    task.get_logger().report_scalar(title="hit_rate_medio_entre_tags", series=f"sinal_{name}", value=res["hit_rate_medio_tags_pct"], iteration=0)
    task.get_logger().report_scalar(title="hit_rate_por_episodio_pct", series=f"sinal_{name}", value=res["hit_rate_episodio_pct"], iteration=0)
    task.get_logger().report_scalar(title="fp_episodios_por_mes", series=f"sinal_{name}", value=res["fp_episodios_por_mes"], iteration=0)
    if len(res["tag_df"]):
        csv_path = os.path.join(OUTPUT_DIR, f"resultado_v5_sinal_{name}_por_tag.csv")
        res["tag_df"].to_csv(csv_path, index=False)
        task.upload_artifact(name=f"resultado_v5_sinal_{name}_por_tag", artifact_object=csv_path)

print("\n\n########## RESULTADO DA VOTACAO (>=N de 4 sinais simultaneos) ##########")
for vote_min in [2, 3, 4]:
    is_anom_vote = (vote_count >= vote_min).astype(int)
    res = evaluate_series(f"votacao_{vote_min}de4", is_anom_vote)
    series_label = f"votacao_{vote_min}de4"
    task.get_logger().report_scalar(title="fp_rate_geral_pct_pontos", series=series_label, value=res["fp_rate_pct"], iteration=0)
    task.get_logger().report_scalar(title="hit_rate_medio_entre_tags", series=series_label, value=res["hit_rate_medio_tags_pct"], iteration=0)
    task.get_logger().report_scalar(title="hit_rate_por_episodio_pct", series=series_label, value=res["hit_rate_episodio_pct"], iteration=0)
    task.get_logger().report_scalar(title="fp_episodios_por_mes", series=series_label, value=res["fp_episodios_por_mes"], iteration=0)
    if len(res["tag_df"]):
        csv_path = os.path.join(OUTPUT_DIR, f"resultado_v5_{series_label}_por_tag.csv")
        res["tag_df"].to_csv(csv_path, index=False)
        task.upload_artifact(name=f"resultado_v5_{series_label}_por_tag", artifact_object=csv_path)

print("\nOK - fim do script v5 (votacao 4 sinais)", flush=True)
task.mark_completed(status_message="Votacao 4 sinais v5 (reproducao da arquitetura do Francisco) concluida com sucesso.")
task.close()
