"""
CNN-1D Autoencoder — pipeline modular uni/multivariada
Versão: v6-ngp

Novidades em relação à v5-auditado:

  [1] --regime_col  — escolha do regime operacional em runtime
        --regime_col RUNNING_A    (padrão, igual v5)
        --regime_col NGP_A        (usa velocidade da turbina GG)
        --regime_col NGP_A --ngp_min 60  (NGP >= 60% como steady state)

  [2] Scoring em 3 níveis físicos
      Para cada ponto anômalo detectado pelo modelo, o CSV de saída
      inclui colunas adicionais:
        physical_level  → "normal" | "alerta" | "critico" | "sem_limite"
        valor_bruto     → valor real do sensor no instante
        ngp_no_instante → NGP no instante (se disponível)

      Isso permite separar:
        - Anomalias abaixo de H (detectadas pelo modelo, não pelo alarme)
        - Anomalias entre H e HH (alerta — o alarme já cobre, mas o
          modelo pode antecipar)
        - Anomalias acima de HH (crítico — confirmação dupla)

  [3] Figura de série com faixas físicas
      O gráfico da série inclui faixas coloridas H e HH do sensor,
      quando disponíveis.

MODOS DE USO:

  # Univariado com RUNNING_A (padrão):
  python CNN1D_AE_universal.py --sensor TC382_03_A

  # Univariado com NGP:
  python CNN1D_AE_universal.py --sensor TC382_03_A \\
      --regime_col NGP_A --ngp_min 60

  # Multivariado T5 com NGP:
  python CNN1D_AE_universal.py --sensor T5_AVG_A \\
      --mode multivariate \\
      --context_sensors TC382_03_A TC382_02_A TC382_05_A TC382_01_A TC382_04_A \\
      --input_features zscore+grad --epochs 20 --max_trials 10 \\
      --time_steps 240 --batch_size 512 \\
      --regime_col NGP_A --ngp_min 60

  # Vibração com RUNNING_A (TV_* sempre treina em todos os estados):
  python CNN1D_AE_universal.py --sensor TV_353X_A \\
      --mode multivariate --auto_context --n_context 4 \\
      --input_features zscore+grad

  [4] Integração ClearML (opcional)
      Rastreia automaticamente hiperparâmetros, métricas, figuras e
      o modelo treinado no servidor ClearML (https://cica.tail4d7f36.ts.net).

        --clearml_project  "Turbina_33003A"   nome do projeto no ClearML
        --clearml_task     "T5_NGP60_multi"   nome da task (experimento)
        --no_clearml                           desativa o ClearML se presente

      Se --clearml_project não for informado, o ClearML NÃO é ativado
      e o script roda exatamente como antes (retrocompatível).

      O que é rastreado automaticamente:
        - Todos os argumentos CLI como hiperparâmetros
        - Loss curve por época (train + val)
        - MAE steady e transição
        - Thresholds calibrados
        - Métricas de avaliação (hit_rate, lead_time, fp_per_day)
        - Contagem de anomalias por nível físico (normal/alerta/critico)
        - Figuras geradas (loss_curve, threshold_histograms, series)
        - Modelo salvo como artefato
        - evaluation_report.json como artefato

Dependência: sensor_preprocessing.py v6-ngp na mesma pasta.
ClearML opcional: pip install clearml
"""

import os, sys, json, argparse, warnings, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt

# ── ClearML (opcional) ────────────────────────────────────
try:
    from clearml import Task as ClearMLTask, Logger as ClearMLLogger
    _CLEARML_AVAILABLE = True
except ImportError:
    _CLEARML_AVAILABLE = False


# =========================================================
# ARGS
# =========================================================
parser = argparse.ArgumentParser()

# ── Sensor e modo ─────────────────────────────────────────
parser.add_argument("--sensor",          required=True)
parser.add_argument("--mode",            default="univariate",
                    choices=["univariate", "multivariate"])
parser.add_argument("--context_sensors", nargs="+", default=None)
parser.add_argument("--auto_context",    action="store_true")
parser.add_argument("--n_context",       type=int,   default=4)
parser.add_argument("--min_corr",        type=float, default=0.5)
parser.add_argument("--input_features",  default="zscore",
                    choices=["zscore", "zscore+grad", "raw"])

# ── Regime operacional — NOVO v6 ─────────────────────────
parser.add_argument("--regime_col", default=None,
                    help="Variável de regime: RUNNING_A (padrão) ou NGP_A. "
                         "Se omitido, usa o valor definido em sensor_preprocessing.py")
parser.add_argument("--ngp_min",    type=float, default=None,
                    help="NGP mínimo %% para steady state (padrão: valor do preprocessing)")

# ── Hiperparâmetros ───────────────────────────────────────
parser.add_argument("--time_steps",        type=int,   default=60)
parser.add_argument("--stride",            type=int,   default=1)
parser.add_argument("--max_trials",        type=int,   default=10)
parser.add_argument("--epochs",            type=int,   default=20)
parser.add_argument("--batch_size",        type=int,   default=1024)
parser.add_argument("--patience",          type=int,   default=6)
parser.add_argument("--thresh_steady",     type=float, default=99.0)
parser.add_argument("--thresh_transition", type=float, default=98.2)
parser.add_argument("--val_frac",          type=float, default=0.10)
parser.add_argument("--output_dir",        default=None)

# ── ClearML ───────────────────────────────────────────────
parser.add_argument("--clearml_project", default=None,
                    help="Nome do projeto no ClearML. Se omitido, ClearML não é ativado.")
parser.add_argument("--clearml_task",    default=None,
                    help="Nome da task no ClearML.")
parser.add_argument("--clearml_dataset_id", default=None,
                    help="ID do dataset CABIUNAS_DATA no ClearML (6ee8bb959e6948d1873ab3687fa91c65). "
                         "Se informado, baixa os dados automaticamente do servidor.")
parser.add_argument("--no_clearml",      action="store_true",
                    help="Força desativação do ClearML.")
parser.add_argument("--clearml_run_remote", action="store_true",
                    help="Enfileira a task no ClearML Agent para execução remota.")
parser.add_argument("--clearml_queue",   default="default",
                    help="Fila do ClearML Agent para execução remota (default: 'default').")
parser.add_argument("--clearml_docker",
                    default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
                    help="Imagem Docker para execução remota via ClearML Agent.")
args = parser.parse_args()

SENSOR     = args.sensor
MODE       = args.mode
TIME_STEPS = args.time_steps
STRIDE     = args.stride
FEAT_MODE  = args.input_features


# =========================================================
# CLEARML — inicialização opcional
# =========================================================

# Auto-detecta clearml.conf na raiz do projeto (evita exportar CLEARML_CONFIG_FILE)
if not os.getenv("CLEARML_CONFIG_FILE"):
    _conf_candidate = pathlib.Path(__file__).parent / "clearml.conf"
    if _conf_candidate.is_file():
        os.environ["CLEARML_CONFIG_FILE"] = str(_conf_candidate)

_clearml_task   = None
_clearml_logger = None

_use_clearml = (
    _CLEARML_AVAILABLE
    and args.clearml_project is not None
    and not args.no_clearml
)

if _use_clearml:
    _task_name = args.clearml_task or (
        f"{args.sensor}_{args.mode}_{args.input_features}"
        + (f"_ngp{int(args.ngp_min)}" if args.ngp_min else "")
        + (f"_regime{args.regime_col}" if args.regime_col and args.regime_col != "RUNNING_A" else "")
    )
    print(f"\n[CLEARML] Inicializando task: '{_task_name}' no projeto '{args.clearml_project}'")
    _clearml_task = ClearMLTask.init(
        project_name=args.clearml_project,
        task_name=_task_name,
        output_uri=True,
        reuse_last_task_id=False,
    )
    # connect() registra todos os args como hiperparâmetros E sobrescreve vars(args)
    # com os valores armazenados na task ao rodar remotamente via ClearML Agent.
    _clearml_task.connect(vars(args), name="CLI_args")
    _clearml_logger = _clearml_task.get_logger()

    # Re-extrai globals com os valores possivelmente atualizados pelo ClearML
    SENSOR     = args.sensor
    MODE       = args.mode
    TIME_STEPS = args.time_steps
    STRIDE     = args.stride
    FEAT_MODE  = args.input_features

    # Execução remota: enfileira a task e encerra o processo local
    if args.clearml_run_remote and _clearml_task.running_locally():
        _clearml_task.set_base_docker(args.clearml_docker)
        _clearml_task.get_logger().report_text(
            f"Enqueuing task for remote execution on queue: {args.clearml_queue}"
        )
        _clearml_task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    print(f"[CLEARML] Task ID: {_clearml_task.id}")
    print(f"[CLEARML] URL: {_clearml_task.get_output_log_web_page()}")
elif args.clearml_project and not _CLEARML_AVAILABLE:
    print("[CLEARML] WARN: clearml não instalado. Rode: pip install clearml")
elif args.no_clearml:
    print("[CLEARML] Desativado via --no_clearml")
else:
    print("[CLEARML] Não ativado (use --clearml_project para ativar)")


# =========================================================
# CARREGA PREPROCESSING v6
# =========================================================
import importlib.util

here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "sensor_preprocessing", here / "sensor_preprocessing.py"
)
prep_mod = importlib.util.module_from_spec(spec)

# Passa configuração para sensor_preprocessing.py via variáveis de ambiente.
# O módulo lê os valores com os.getenv() antes de redefinir as constantes,
# portanto a injeção via atributo de módulo NÃO funciona (exec_module sobrescreveria).
if args.regime_col is not None:
    os.environ["REGIME_COL"] = args.regime_col
if args.ngp_min is not None:
    os.environ["NGP_MIN_PCT"] = str(args.ngp_min)
if args.clearml_dataset_id is not None:
    os.environ["CLEARML_DATASET_ID"] = args.clearml_dataset_id

spec.loader.exec_module(prep_mod)

sensors                     = prep_mod.EXPORTS["sensors"]
sensor_profiles             = prep_mod.EXPORTS["sensor_profiles"]
normalization               = prep_mod.EXPORTS["normalization"]
sensor_classes              = prep_mod.EXPORTS["sensor_classes"]
df_raw                      = prep_mod.EXPORTS["df"]
df_all_z                    = prep_mod.EXPORTS["df_all_z"]
df_normal_z_map             = prep_mod.EXPORTS["df_normal_per_sensor"]
running_bin                 = prep_mod.EXPORTS["running_bin"]
regime_bin                  = prep_mod.EXPORTS["regime_bin"]
regime_label                = prep_mod.EXPORTS["regime_label"]
ngp_series                  = prep_mod.EXPORTS["ngp_series"]
ngp_available               = prep_mod.EXPORTS["ngp_available"]
alarm_mask                  = prep_mod.EXPORTS["alarm_mask"]
grad_masks                  = prep_mod.EXPORTS["grad_masks"]
get_suppression             = prep_mod.EXPORTS["suppression_mask_fn"]
get_multi_normal_mask       = prep_mod.EXPORTS["get_multi_normal_mask"]
get_feature_array           = prep_mod.EXPORTS["get_feature_array"]
get_context_sensors_by_corr = prep_mod.EXPORTS["get_context_sensors_by_correlation"]
get_transition_mask_fn      = prep_mod.EXPORTS["get_transition_mask"]
get_physical_level          = prep_mod.EXPORTS["get_physical_level"]
sensor_limits               = prep_mod.EXPORTS["sensor_limits"]

TRANSITION_MASK_MINUTES_DEFAULT = prep_mod.TRANSITION_MASK_MINUTES_DEFAULT
EXCLUDE_ALARM_MINUTES           = prep_mod.EXCLUDE_ALARM_MINUTES
REGIME_COL_USED = prep_mod.EXPORTS["regime_col_used"]


# ─── Validações ──────────────────────────────────────────
if SENSOR not in sensors:
    print(f"[ERROR] Sensor '{SENSOR}' não encontrado.")
    sys.exit(1)

sensor_class = sensor_classes.get(SENSOR, "default")
if sensor_class == "binary":
    print(f"[SKIP] '{SENSOR}' é binary. Sem modelo.")
    sys.exit(0)

_sprof = sensor_profiles.get(SENSOR, {})
sensor_transition_mask    = get_transition_mask_fn(SENSOR)
SENSOR_TRANSITION_MINUTES = _sprof.get("transition_mask_minutes",
                                       TRANSITION_MASK_MINUTES_DEFAULT)
lim_sensor = sensor_limits.get(SENSOR, {})


# =========================================================
# RESOLUÇÃO DOS SENSORES DE CONTEXTO
# =========================================================
if MODE == "multivariate":
    if args.auto_context:
        context_sensors = get_context_sensors_by_corr(
            SENSOR, n_top=args.n_context, min_corr=args.min_corr
        )
        if not context_sensors:
            print("[WARN] Nenhum contexto encontrado. Falling back univariado.")
            MODE = "univariate"
            all_sensors = [SENSOR]
        else:
            all_sensors = [SENSOR] + context_sensors
    elif args.context_sensors:
        invalid = [s for s in args.context_sensors if s not in sensors]
        if invalid:
            print(f"[ERROR] Sensores inválidos: {invalid}")
            sys.exit(1)
        context_sensors = args.context_sensors
        all_sensors     = [SENSOR] + [s for s in context_sensors if s != SENSOR]
    else:
        print("[ERROR] --mode multivariate requer --context_sensors ou --auto_context.")
        sys.exit(1)
else:
    all_sensors     = [SENSOR]
    context_sensors = []

TARGET_IDX          = all_sensors.index(SENSOR)
N_SENSORS           = len(all_sensors)
FEATURES_PER_SENSOR = 2 if FEAT_MODE == "zscore+grad" else 1
N_FEATURES          = N_SENSORS * FEATURES_PER_SENSOR

regime_suffix = (f"_ngp{int(prep_mod.NGP_MIN_PCT)}"
                 if REGIME_COL_USED != "RUNNING_A" else "")
suffix = f"_{MODE}"
if FEAT_MODE != "zscore":
    suffix += f"_{FEAT_MODE.replace('+', '_')}"
suffix += regime_suffix

OUTPUT_DIR = args.output_dir or f"OUTPUT_{SENSOR}_v6{suffix}"
for sub in ["tuner", "best_model", "figs", "csv"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

print(f"\n{'='*68}")
print(f"  Sensor-alvo  : {SENSOR}")
print(f"  Classe       : {sensor_class}  |  Grupo: {_sprof.get('group','default')}")
print(f"  Regime       : {regime_label}  (col={REGIME_COL_USED})")
print(f"  Modo         : {MODE}  |  Features: {FEAT_MODE}")
print(f"  Sensores     : {all_sensors}")
print(f"  n_features   : {N_FEATURES}  |  TIME_STEPS: {TIME_STEPS} "
      f"({TIME_STEPS * 30 // 60} min)  |  Trans: {SENSOR_TRANSITION_MINUTES}min")
H_val  = lim_sensor.get("H",  "N/D")
HH_val = lim_sensor.get("HH", "N/D")
L_val  = lim_sensor.get("L",  "N/D")
print(f"  Limiares     : L={L_val}  H={H_val}  HH={HH_val}  "
      f"[{lim_sensor.get('unit', '')}]")
print(f"  Output       : {OUTPUT_DIR}")
print(f"{'='*68}\n")

# Log configuração do sensor no ClearML
if _use_clearml:
    _clearml_task.connect({
        "sensor":                   SENSOR,
        "sensor_class":             sensor_class,
        "sensor_group":             _sprof.get("group", "default"),
        "regime_col":               REGIME_COL_USED,
        "regime_label":             regime_label,
        "ngp_available":            ngp_available,
        "all_sensors":              all_sensors,
        "n_features":               N_FEATURES,
        "TIME_STEPS":               TIME_STEPS,
        "transition_mask_minutes":  SENSOR_TRANSITION_MINUTES,
        "H_limiar":                 lim_sensor.get("H", "N/D"),
        "HH_limiar":                lim_sensor.get("HH", "N/D"),
        "unit":                     lim_sensor.get("unit", ""),
    }, name="sensor_config")


# =========================================================
# GPU
# =========================================================
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] {len(gpus)} GPU(s).")
    else:
        print("[GPU] Rodando em CPU.")
except Exception as e:
    print(f"[GPU] {e}")


# =========================================================
# DADOS DE TREINO
# =========================================================
if MODE == "multivariate":
    normal_mask = get_multi_normal_mask(all_sensors)
else:
    normal_mask = None

feat_df = get_feature_array(all_sensors, mode=FEAT_MODE)

target_col_names = [c for c in feat_df.columns if c.startswith(f"{SENSOR}__")]
target_col_idxs  = [feat_df.columns.get_loc(c) for c in target_col_names]

if MODE == "univariate":
    normal_series = df_normal_z_map[SENSOR]
    feat_normal   = feat_df.loc[normal_series.index]
else:
    feat_normal = feat_df.loc[normal_mask]

feat_all = feat_df.values.astype(np.float32)

print(f"[DATA] feat_df shape: {feat_df.shape}")
print(f"[DATA] Treino: {len(feat_normal):,} pts ({100*len(feat_normal)/len(feat_df):.1f}%)")


# =========================================================
# SEQUÊNCIAS
# =========================================================
def make_sequences(arr_2d: np.ndarray, time_steps: int, stride: int) -> np.ndarray:
    if arr_2d.ndim == 1:
        arr_2d = arr_2d[:, np.newaxis]
    idx = np.arange(0, len(arr_2d) - time_steps + 1, stride)
    out = np.stack([arr_2d[i:i + time_steps] for i in idx], axis=0)
    return out.astype(np.float32)


feat_normal_arr = feat_normal.values.astype(np.float32)
x_train_full    = make_sequences(feat_normal_arr, TIME_STEPS, STRIDE)

n_val   = int(np.floor(args.val_frac * len(x_train_full)))
x_train = x_train_full[:len(x_train_full) - n_val]
x_val   = x_train_full[len(x_train_full) - n_val:]
print(f"[SEQ]  x_train={x_train.shape}  x_val={x_val.shape}")


# =========================================================
# LOSS FOCAL
# =========================================================
def make_target_mse(target_idxs):
    idxs = tf.constant(target_idxs, dtype=tf.int32)
    def target_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(
            tf.gather(y_true, idxs, axis=-1) -
            tf.gather(y_pred, idxs, axis=-1)
        ))
    target_mse.__name__ = "target_mse"
    return target_mse

loss_fn = make_target_mse(target_col_idxs) if MODE == "multivariate" else "mse"


# =========================================================
# HYPERMODEL
# =========================================================
def build_cnn1d_ae(hp: kt.HyperParameters):
    f1      = hp.Choice("filters_1", [8, 16, 32, 64])
    f2      = hp.Choice("filters_2", [8, 16, 32, 64])
    k1      = hp.Choice("kernel_1",  [3, 5, 7, 9])
    k2      = hp.Choice("kernel_2",  [3, 5, 7, 9])
    dropout = hp.Float( "dropout",   0.0, 0.4, step=0.1)
    lr      = hp.Choice("lr",        [1e-4, 3e-4, 1e-3, 3e-3])
    s1      = hp.Choice("stride_1",  [1, 2])
    s2      = hp.Choice("stride_2",  [1, 2])

    inp = keras.Input(shape=(TIME_STEPS, N_FEATURES))
    x   = layers.Conv1D(f1, k1, padding="same", strides=s1, activation="relu")(inp)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x   = layers.Conv1D(f2, k2, padding="same", strides=s2, activation="relu")(x)
    x   = layers.Conv1DTranspose(f2, k2, padding="same", strides=s2, activation="relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x   = layers.Conv1DTranspose(f1, k1, padding="same", strides=s1, activation="relu")(x)
    out = layers.Conv1DTranspose(N_FEATURES, 3, padding="same")(x)

    model = keras.Model(inp, out, name="cnn1d_ae")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# =========================================================
# TUNER
# =========================================================
print("\n[TUNER] Buscando hiperparâmetros...")

_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience,
        mode="min", restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=max(2, args.patience // 2), min_lr=1e-6),
]

tuner = kt.RandomSearch(
    hypermodel=build_cnn1d_ae,
    objective=kt.Objective("val_loss", direction="min"),
    max_trials=args.max_trials,
    executions_per_trial=1,
    directory=os.path.join(OUTPUT_DIR, "tuner"),
    project_name=f"trials_{SENSOR}",
    overwrite=True,
)

tuner.search(
    x_train, x_train,
    validation_data=(x_val, x_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=_callbacks,
    verbose=1,
)

best_hp    = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]

trial_rows = []
for t in tuner.oracle.get_best_trials(num_trials=args.max_trials):
    row = {"trial_id": t.trial_id, "val_loss": t.score}
    row.update(t.hyperparameters.values)
    trial_rows.append(row)
pd.DataFrame(trial_rows).sort_values("val_loss").to_csv(
    os.path.join(OUTPUT_DIR, "csv", "trials_ranking.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "best_model", "best_hp.json"), "w") as f:
    json.dump(best_hp.values, f, indent=2)

print(f"\n[TRAIN] Refit final (max {args.epochs * 3} épocas)...")
history = best_model.fit(
    x_train, x_train,
    validation_data=(x_val, x_val),
    epochs=args.epochs * 3,
    batch_size=args.batch_size,
    callbacks=_callbacks,
    verbose=1,
)

model_path = os.path.join(OUTPUT_DIR, "best_model", "model.keras")
best_model.save(model_path)
print(f"[SAVE] Modelo: {model_path}")

# Upload modelo como artefato ClearML
if _use_clearml:
    _clearml_task.upload_artifact("model_keras", model_path)
    _clearml_task.upload_artifact("best_hp", os.path.join(OUTPUT_DIR, "best_model", "best_hp.json"))
    # Log hiperparâmetros do melhor trial
    _clearml_task.connect(best_hp.values, name="best_hyperparameters")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history.get("loss",     []), label="train_loss")
ax.plot(history.history.get("val_loss", []), label="val_loss")
ax.set_title(f"Loss — {SENSOR} [{MODE}/{FEAT_MODE}/{regime_label}]")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figs", "loss_curve.png"), dpi=150, bbox_inches="tight")

# Log loss por época no ClearML
if _use_clearml:
    for i, (tl, vl) in enumerate(zip(
        history.history.get("loss", []),
        history.history.get("val_loss", [])
    )):
        _clearml_logger.report_scalar("loss", "train", value=tl, iteration=i)
        _clearml_logger.report_scalar("loss", "val",   value=vl, iteration=i)
    _clearml_logger.report_matplotlib_figure(
        title="loss_curve", series="refit", figure=fig, iteration=0
    )

plt.close(fig)


# =========================================================
# HELPER DE ERRO
# =========================================================
def reconstruction_error(x_true, x_pred):
    err = np.abs(x_pred[:, :, target_col_idxs] - x_true[:, :, target_col_idxs])
    return err.mean(axis=(1, 2))


# =========================================================
# THRESHOLD DUPLO
# =========================================================
print("\n[THRESH] Calibrando thresholds...")

x_tr_pred     = best_model.predict(x_train_full, batch_size=args.batch_size, verbose=0)
tr_mae        = reconstruction_error(x_train_full, x_tr_pred)
thresh_steady = float(np.percentile(tr_mae, args.thresh_steady))
print(f"[THRESH] steady  (p{args.thresh_steady:.0f}): {thresh_steady:.6f}")

if _use_clearml:
    _clearml_logger.report_scalar("threshold", "steady",     value=thresh_steady,     iteration=0)
    _clearml_logger.report_single_value("thresh_steady", thresh_steady)

all_idx    = feat_df.index
x_all      = make_sequences(feat_all, TIME_STEPS, STRIDE)
x_all_pred = best_model.predict(x_all, batch_size=args.batch_size, verbose=0)
mae_all    = reconstruction_error(x_all, x_all_pred)

trans_aligned = sensor_transition_mask.reindex(all_idx, fill_value=False)
seq_is_trans  = np.array([trans_aligned.iloc[i] for i in range(len(mae_all))], dtype=bool)
mae_trans     = mae_all[seq_is_trans]

if len(mae_trans) > 100:
    thresh_transition = float(np.percentile(mae_trans, args.thresh_transition))
    print(f"[THRESH] transition (p{args.thresh_transition:.0f}): {thresh_transition:.6f} "
          f"({len(mae_trans):,} seqs)")
else:
    thresh_transition = thresh_steady * 3.0
    print(f"[THRESH] Fallback transition = steady × 3.0 = {thresh_transition:.6f}")

if _use_clearml:
    _clearml_logger.report_scalar("threshold", "transition", value=thresh_transition, iteration=0)
    _clearml_logger.report_single_value("thresh_transition", thresh_transition)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(tr_mae, bins=60, color="#1D9E75", edgecolor="none")
axes[0].axvline(thresh_steady, color="red", lw=1.5, ls="--",
                label=f"p{args.thresh_steady:.0f}={thresh_steady:.4f}")
axes[0].set_title(f"MAE steady — {SENSOR}"); axes[0].legend()
if len(mae_trans) > 0:
    axes[1].hist(mae_trans, bins=60, color="#D85A30", edgecolor="none")
    axes[1].axvline(thresh_transition, color="red", lw=1.5, ls="--",
                    label=f"p{args.thresh_transition:.0f}={thresh_transition:.4f}")
    axes[1].set_title(f"MAE transição — {SENSOR}"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figs", "threshold_histograms.png"),
            dpi=150, bbox_inches="tight")
if _use_clearml:
    _clearml_logger.report_matplotlib_figure(
        title="threshold_histograms", series="MAE", figure=fig, iteration=0
    )
plt.close(fig)


# =========================================================
# SCORING COM SUPRESSÃO + NÍVEL FÍSICO
# =========================================================
print("\n[SCORE] Aplicando flags com supressão universal...")

suppress_vec    = get_suppression(SENSOR).reindex(all_idx, fill_value=True)
anom_seq_steady = mae_all > thresh_steady
anom_seq_trans  = mae_all > thresh_transition
anom_seq_final  = np.where(seq_is_trans, anom_seq_trans, anom_seq_steady)

anomalous_idx = []
for data_idx in range(TIME_STEPS - 1, len(all_idx) - TIME_STEPS + 1):
    if suppress_vec.iloc[data_idx]:
        continue
    covering = anom_seq_final[data_idx - TIME_STEPS + 1: data_idx + 1]
    if np.all(covering):
        anomalous_idx.append(data_idx)

anomalous_idx   = np.array(anomalous_idx, dtype=int)
anom_timestamps = all_idx[anomalous_idx] if len(anomalous_idx) > 0 else pd.DatetimeIndex([])

print(f"[SCORE] Pontos anômalos: {len(anom_timestamps):,} "
      f"({100 * len(anom_timestamps) / len(all_idx):.3f}%)")


# =========================================================
# NÍVEL FÍSICO POR PONTO ANÔMALO  ← NOVO v6
# =========================================================
raw_vals_sensor = prep_mod.EXPORTS["df"][SENSOR]

def _get_ngp_at(ts):
    if not ngp_available:
        return np.nan
    try:
        return float(ngp_series.loc[ts])
    except Exception:
        return np.nan

if len(anom_timestamps) > 0:
    anom_raw_vals = raw_vals_sensor.reindex(anom_timestamps, method="nearest").values
    anom_ngp_vals = np.array([_get_ngp_at(t) for t in anom_timestamps])
    anom_levels   = np.array([
        get_physical_level(SENSOR, float(v), ngp=float(n) if not np.isnan(n) else None)
        for v, n in zip(anom_raw_vals, anom_ngp_vals)
    ])
else:
    anom_raw_vals = np.array([])
    anom_ngp_vals = np.array([])
    anom_levels   = np.array([])

# Contagem por nível
level_counts = {}
for lv in ["normal", "alerta", "critico", "sem_limite"]:
    level_counts[lv] = int((anom_levels == lv).sum()) if len(anom_levels) > 0 else 0

print(f"[SCORE] Nível físico das anomalias detectadas:")
print(f"  normal:     {level_counts['normal']:,} pts")
print(f"  alerta:     {level_counts['alerta']:,} pts  (entre H e HH)")
print(f"  critico:    {level_counts['critico']:,} pts  (acima de HH)")
print(f"  sem_limite: {level_counts['sem_limite']:,} pts")

if _use_clearml:
    _clearml_logger.report_single_value("total_anom_points",  len(anom_timestamps))
    _clearml_logger.report_single_value("pct_anom",           round(100*len(anom_timestamps)/len(all_idx), 4))
    _clearml_logger.report_single_value("anom_normal",        level_counts["normal"])
    _clearml_logger.report_single_value("anom_alerta",        level_counts["alerta"])
    _clearml_logger.report_single_value("anom_critico",       level_counts["critico"])
    _clearml_logger.report_single_value("anom_sem_limite",    level_counts["sem_limite"])


# =========================================================
# SALVAR CSVs
# =========================================================
gmask_s = grad_masks[SENSOR].reindex(all_idx, fill_value=False)

df_seq = pd.DataFrame({
    "seq_start_time": all_idx[:len(mae_all)],
    "mae_seq":        mae_all,
    "is_transition":  seq_is_trans.astype(int),
    "thresh_used":    np.where(seq_is_trans, thresh_transition, thresh_steady),
    "is_anom_seq":    anom_seq_final.astype(int),
    "is_anom_raw":    anom_seq_steady.astype(int),
})
df_seq.to_csv(os.path.join(OUTPUT_DIR, "csv", "sequence_scores.csv"), index=False)

df_pt = pd.DataFrame(index=all_idx)
df_pt["running_bin"]        = running_bin.reindex(all_idx, fill_value=0).astype(int)
df_pt["regime_bin"]         = regime_bin.reindex(all_idx, fill_value=0).astype(int)
df_pt["is_transition"]      = trans_aligned.astype(int)
df_pt["is_grad_suppressed"] = gmask_s.astype(int)
df_pt["is_suppressed"]      = suppress_vec.astype(int)
df_pt["is_anom_point"]      = 0
df_pt["valor_bruto"]        = raw_vals_sensor.reindex(all_idx)
df_pt["physical_level"]     = "normal"

if ngp_available:
    df_pt["ngp"]            = ngp_series.reindex(all_idx)

if len(anom_timestamps) > 0:
    df_pt.loc[anom_timestamps, "is_anom_point"]  = 1
    df_pt.loc[anom_timestamps, "physical_level"] = anom_levels

df_pt.to_csv(os.path.join(OUTPUT_DIR, "csv", "point_anomalies.csv"))
print(f"[CSV]  Salvo em {OUTPUT_DIR}/csv/")


# =========================================================
# VISUALIZAÇÃO
# =========================================================
print("[VIZ]  Gerando gráfico da série...")

raw_idx  = prep_mod.EXPORTS["df"].index

def _paint_spans(ax, idx, mask_bool, color, alpha, label):
    in_span = False
    t_start = t_prev = None
    lbl = label
    gap = pd.Timedelta(minutes=3)
    for t in idx[mask_bool.reindex(idx, fill_value=False).values]:
        if not in_span:
            t_start = t; in_span = True
        elif t_prev is not None and (t - t_prev) > gap:
            ax.axvspan(t_start, t_prev, alpha=alpha, color=color, label=lbl)
            lbl = "_nolegend_"; t_start = t
        t_prev = t
    if in_span and t_start is not None:
        ax.axvspan(t_start, t_prev, alpha=alpha, color=color, label=lbl)

n_panels = 2 if ngp_available else 1
fig, axes = plt.subplots(n_panels, 1, figsize=(18, 5 * n_panels),
                         sharex=True if n_panels > 1 else False)
if n_panels == 1:
    axes = [axes]

ax = axes[0]

_paint_spans(ax, all_idx,
             running_bin.reindex(all_idx, fill_value=0) == 0,
             color="#888780", alpha=0.08, label="Parada (RUNNING≠1)")
_paint_spans(ax, all_idx, trans_aligned,
             color="#D85A30", alpha=0.10,
             label=f"Transição ±{SENSOR_TRANSITION_MINUTES} min")
_paint_spans(ax, all_idx, gmask_s,
             color="#EF9F27", alpha=0.14, label="Gradiente abrupto")

raw_plot = raw_vals_sensor.reindex(all_idx)
ax.plot(raw_idx, raw_vals_sensor.values, lw=0.7,
        color="#185FA5", label=f"{SENSOR} (bruto)", zorder=2)

# Faixas físicas H e HH
H_v  = lim_sensor.get("H")
HH_v = lim_sensor.get("HH")
L_v  = lim_sensor.get("L")
if H_v is not None:
    ax.axhline(H_v,  color="#FF7F0E", lw=1.0, ls="--", alpha=0.8,
               label=f"H={H_v} {lim_sensor.get('unit','')}")
if HH_v is not None:
    ax.axhline(HH_v, color="#D62728", lw=1.0, ls="--", alpha=0.8,
               label=f"HH={HH_v} {lim_sensor.get('unit','')}")
if L_v is not None:
    ax.axhline(L_v,  color="#1FA05F", lw=1.0, ls="--", alpha=0.8,
               label=f"L={L_v} {lim_sensor.get('unit','')}")

# Anomalias coloridas por nível
color_map = {"normal": "#A32D2D", "alerta": "#FF7F0E",
             "critico": "#D62728", "sem_limite": "#888780"}
if len(anom_timestamps) > 0:
    for lv, col in color_map.items():
        mask_lv = anom_levels == lv
        if mask_lv.sum() > 0:
            ts_lv = anom_timestamps[mask_lv]
            vs_lv = anom_raw_vals[mask_lv]
            ax.scatter(ts_lv, vs_lv, s=10, color=col, zorder=5,
                       label=f"Anomalia {lv} ({mask_lv.sum()})")

ctx_label = (f"ctx=[{', '.join(context_sensors[:2])}"
             f"{'...' if len(context_sensors) > 2 else ''}]") if context_sensors else ""
ax.set_title(
    f"{SENSOR} — CNN-1D AE v6-ngp  [{MODE}/{FEAT_MODE}]  "
    f"regime={regime_label}  {ctx_label}  (thresh={thresh_steady:.4f})"
)
ax.set_ylabel(f"Valor [{lim_sensor.get('unit', 'u.a.')}]")
ax.legend(loc="upper right", fontsize=7, ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Painel NGP (se disponível)
if ngp_available and n_panels > 1:
    ax2 = axes[1]
    ax2.plot(ngp_series.index, ngp_series.values, lw=0.7,
             color="#2CA02C", label="NGP_A (%)")
    ax2.axhline(prep_mod.NGP_MIN_PCT, color="red", lw=1.0, ls="--",
                label=f"NGP_min={prep_mod.NGP_MIN_PCT:.0f}%")
    ax2.set_ylabel("NGP (%)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

fig.autofmt_xdate()
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "figs", f"series_{SENSOR}.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
if _use_clearml:
    _clearml_logger.report_matplotlib_figure(
        title=f"series_{SENSOR}", series="anomalias", figure=fig, iteration=0
    )
plt.close(fig)
print(f"[VIZ]  {fig_path}")


# =========================================================
# AVALIAÇÃO vs ALARMES
# =========================================================
print("\n[EVAL] Avaliando detecção vs alarmes (dois níveis)...")

df_alarm_eval = getattr(prep_mod, "df_alarm", None)
if df_alarm_eval is None or len(df_alarm_eval) == 0:
    df_alarm_eval = pd.DataFrame(columns=["data_datetime", "Tag Alarme", "Status"])

win = pd.Timedelta(minutes=EXCLUDE_ALARM_MINUTES)


def _eval_alarm_set(alarms_df, label):
    if len(alarms_df) == 0:
        return {"n_alarms": 0, "hits": 0, "hit_rate": None,
                "lead_time_median_min": None, "lead_time_mean_min": None,
                "lead_time_min_min": None, "lead_time_max_min": None,
                "anom_with_alarm": 0, "anom_no_alarm": int(len(anom_timestamps))}

    hits_local = 0
    lead_times = []

    for _, row in alarms_df.iterrows():
        t  = row["data_datetime"]
        wdow = df_pt.loc[(df_pt.index >= t - win) & (df_pt.index <= t + win), "is_anom_point"]
        if wdow.sum() > 0:
            hits_local += 1
            before = df_pt.loc[(df_pt.index >= t - win) & (df_pt.index < t), "is_anom_point"]
            if before.sum() > 0:
                first_anom = before[before == 1].index[0]
                lead_times.append((t - first_anom).total_seconds() / 60.0)

    anom_in_win = sum(
        1 for ts in anom_timestamps
        if abs(pd.to_datetime(alarms_df["data_datetime"].values) - ts).min() <= win
    )
    n  = len(alarms_df)
    hr = hits_local / n if n > 0 else float("nan")

    msg = (f"  [{label}] alarmes={n} | hits={hits_local} | "
           f"hit_rate={hr:.1%}" if not np.isnan(hr) else
           f"  [{label}] alarmes={n} | hits={hits_local} | hit_rate=N/A")
    print(msg)
    if lead_times:
        print(f"  [{label}] lead_time: mediana={np.median(lead_times):.0f}min "
              f"| média={np.mean(lead_times):.0f}min "
              f"| range=[{min(lead_times):.0f}, {max(lead_times):.0f}]min")

    return {
        "n_alarms": n, "hits": hits_local,
        "hit_rate": round(float(hr), 4) if not np.isnan(hr) else None,
        "lead_time_median_min": float(np.median(lead_times)) if lead_times else None,
        "lead_time_mean_min":   float(np.mean(lead_times))   if lead_times else None,
        "lead_time_min_min":    float(min(lead_times))        if lead_times else None,
        "lead_time_max_min":    float(max(lead_times))        if lead_times else None,
        "anom_with_alarm": anom_in_win,
        "anom_no_alarm":   int(len(anom_timestamps)) - anom_in_win,
    }


has_tag    = "Tag Alarme" in df_alarm_eval.columns
has_status = "Status"     in df_alarm_eval.columns

alarms_direct = (df_alarm_eval[
    (df_alarm_eval["Status"] == "ACT/UNACK") &
    (df_alarm_eval["Tag Alarme"] == SENSOR)
].copy() if (has_tag and has_status) else
    df_alarm_eval[df_alarm_eval["Tag Alarme"] == SENSOR].copy() if has_tag else
    pd.DataFrame(columns=df_alarm_eval.columns))

print(f"\n[EVAL] Nível 1 — tag exata '{SENSOR}': {len(alarms_direct)} alarmes")
eval_direct = _eval_alarm_set(alarms_direct, "tag_exata")

alarms_equip = (df_alarm_eval[df_alarm_eval["Status"] == "ACT/UNACK"].copy()
                if has_status else df_alarm_eval.copy())
print(f"\n[EVAL] Nível 2 — todos ACT/UNACK: {len(alarms_equip)}")
eval_equip = _eval_alarm_set(alarms_equip, "equipamento")

alarm_mask_s = alarm_mask.reindex(all_idx, fill_value=False)
safe_mask    = (~suppress_vec) & (~alarm_mask_s)
df_safe      = df_pt.loc[safe_mask]
days_safe    = ((df_safe.index[-1] - df_safe.index[0]).total_seconds() / 86400.0
                if len(df_safe) > 1 else 0.0)
fp_per_day   = (df_safe["is_anom_point"].sum() / days_safe
                if days_safe > 0 else float("nan"))


# =========================================================
# RELATÓRIO CONSOLIDADO
# =========================================================
eval_report = {
    "sensor":            SENSOR,
    "sensor_class":      sensor_class,
    "sensor_group":      _sprof.get("group", "default"),
    "version":           "v6-ngp",
    "regime_col":        REGIME_COL_USED,
    "regime_label":      regime_label,
    "ngp_available":     ngp_available,
    "mode":              MODE,
    "input_features":    FEAT_MODE,
    "all_sensors":       all_sensors,
    "n_features":        N_FEATURES,
    "TIME_STEPS":        TIME_STEPS,
    "transition_mask_minutes": SENSOR_TRANSITION_MINUTES,
    "thresh_steady":     thresh_steady,
    "thresh_transition": thresh_transition,
    "physical_limits":   {k: str(v) for k, v in lim_sensor.items()},

    "total_anom_points": int(len(anom_timestamps)),
    "pct_anom":          round(100 * len(anom_timestamps) / len(all_idx), 4),
    "anom_por_nivel":    level_counts,

    "tag_direta":  {
        "n_alarms": eval_direct["n_alarms"], "hits": eval_direct["hits"],
        "hit_rate": eval_direct["hit_rate"],
        "lead_time_median_min": eval_direct["lead_time_median_min"],
        "lead_time_mean_min":   eval_direct["lead_time_mean_min"],
        "lead_time_min_min":    eval_direct["lead_time_min_min"],
        "lead_time_max_min":    eval_direct["lead_time_max_min"],
        "anom_with_alarm": eval_direct["anom_with_alarm"],
        "anom_no_alarm":   eval_direct["anom_no_alarm"],
    },
    "equipamento": {
        "n_alarms": eval_equip["n_alarms"], "hits": eval_equip["hits"],
        "hit_rate": eval_equip["hit_rate"],
        "lead_time_median_min": eval_equip["lead_time_median_min"],
        "lead_time_mean_min":   eval_equip["lead_time_mean_min"],
        "lead_time_min_min":    eval_equip["lead_time_min_min"],
        "lead_time_max_min":    eval_equip["lead_time_max_min"],
        "anom_with_alarm": eval_equip["anom_with_alarm"],
        "anom_no_alarm":   eval_equip["anom_no_alarm"],
    },

    "fp_per_day_safe":  round(float(fp_per_day), 4) if not np.isnan(fp_per_day) else None,
    "safe_period_days": round(days_safe, 2),
    "best_hyperparameters": best_hp.values,
}

with open(os.path.join(OUTPUT_DIR, "csv", "evaluation_report.json"), "w", encoding="utf-8") as f:
    json.dump(eval_report, f, indent=2, ensure_ascii=False)

# ── ClearML: log métricas finais e fechar task ────────────
if _use_clearml:
    # Upload artefatos finais
    _clearml_task.upload_artifact(
        "evaluation_report",
        os.path.join(OUTPUT_DIR, "csv", "evaluation_report.json")
    )
    _clearml_task.upload_artifact(
        "point_anomalies",
        os.path.join(OUTPUT_DIR, "csv", "point_anomalies.csv")
    )
    _clearml_task.upload_artifact(
        "trials_ranking",
        os.path.join(OUTPUT_DIR, "csv", "trials_ranking.csv")
    )
    _clearml_task.upload_artifact(
        "sequence_scores",
        os.path.join(OUTPUT_DIR, "csv", "sequence_scores.csv")
    )

    # Métricas de avaliação (tag direta)
    if eval_direct["hit_rate"] is not None:
        _clearml_logger.report_single_value("hit_rate_tag_direta",      eval_direct["hit_rate"])
    if eval_direct["lead_time_median_min"] is not None:
        _clearml_logger.report_single_value("lead_time_median_min",     eval_direct["lead_time_median_min"])

    # Métricas de avaliação (equipamento)
    if eval_equip["hit_rate"] is not None:
        _clearml_logger.report_single_value("hit_rate_equipamento",     eval_equip["hit_rate"])

    # FP por dia
    if not np.isnan(fp_per_day):
        _clearml_logger.report_single_value("fp_per_day_safe",          fp_per_day)

    # Resumo como scalar por nível
    for lv, cnt in level_counts.items():
        _clearml_logger.report_scalar("anomalias_por_nivel", lv, value=cnt, iteration=0)

    print(f"\n[CLEARML] Task finalizada.")
    print(f"[CLEARML] Acesse: {_clearml_task.get_output_log_web_page()}")
    _clearml_task.mark_completed(status_message="Pipeline v6-ngp concluída com sucesso.")
    _clearml_task.close()

print(f"\n{'='*68}")
print(f"  [DONE] {SENSOR}  —  v6-ngp")
print(f"  Regime:        {regime_label}")
print(f"  Modo:          {MODE} / {FEAT_MODE}")
print(f"  Sensores:      {all_sensors}")
print(f"  Modelo:        {model_path}")
print(f"  Anomalias:     {len(anom_timestamps):,} pts ({eval_report['pct_anom']:.3f}%)")
print(f"  Níveis:        normal={level_counts['normal']} | "
      f"alerta={level_counts['alerta']} | critico={level_counts['critico']}")
print(f"  FP/dia (safe): {fp_per_day:.4f}" if not np.isnan(fp_per_day) else "  FP/dia: N/A")
hr_d = f"{eval_direct['hit_rate']:.1%}" if eval_direct['hit_rate'] is not None else "N/A"
lm_d = f"{eval_direct['lead_time_median_min']:.0f}min" if eval_direct['lead_time_median_min'] else "N/A"
hr_e = f"{eval_equip['hit_rate']:.1%}" if eval_equip['hit_rate'] is not None else "N/A"
lm_e = f"{eval_equip['lead_time_median_min']:.0f}min" if eval_equip['lead_time_median_min'] else "N/A"
print(f"  [tag direta]   n={eval_direct['n_alarms']:3d} | hits={eval_direct['hits']:3d} | "
      f"hit_rate={hr_d:>6s} | lead_med={lm_d}")
print(f"  [equipamento]  n={eval_equip['n_alarms']:3d} | hits={eval_equip['hits']:3d} | "
      f"hit_rate={hr_e:>6s} | lead_med={lm_e}")
print(f"{'='*68}\n")
