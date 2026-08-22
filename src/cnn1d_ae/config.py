from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union


@dataclass
class PipelineConfig:
    # Inputs
    FEATURES_CSV: str = "data/features_TodosSensores.csv"
    RAW_CSV: str = "data/sensores_filtrados_2025.csv"
    ALARM_CSV: str = "data/ocorrencia_alarmes_sensores_2025.csv"
    TIME_COL: str = "data_datetime"

    # =========================
    # POLITICA TEMPORAL (UTC-first)
    # =========================
    SOURCE_TZ: str = "America/Sao_Paulo"
    TARGET_TZ: str = "UTC"
    APPLY_HOUR_SHIFT: bool = False
    SHIFT_HOURS: int = 0
    LOG_TIME_AUDIT_SAMPLES: int = 3

    # Recorte opcional do inicio/fim da serie usada por toda a pipeline
    # (aplicado logo apos o parsing de tempo, antes de qualquer split
    # treino/OOS). None = usa a serie inteira disponivel no arquivo.
    DATA_START_DATE: Optional[str] = None
    DATA_END_DATE: Optional[str] = None

    # =========================
    # MODO DE EXECUCAO
    # =========================
    MODE: str = "operacional"  # "operacional" | "local"

    # =========================
    # SAIDAS
    # =========================
    OUTPUT_DIR_TEMPLATE: str = "OUTPUT_CNN1D_AE_{sensor}"
    OUTPUT_ROOT: str = ""

    # Fonte do treino
    TRAIN_SOURCE: str = "raw"  # "raw" ou "feat"

    # =========================
    # PREPROCESSAMENTO
    # =========================
    EXCLUDE_MINUTES_AROUND_ALARM: int = 1440
    INTERPOLATE_LIMIT: int = 3
    EXCLUDE_LONG_GAPS_FROM_TRAIN: bool = True
    ENABLE_DERIVED_FEATURES: bool = False
    DERIVED_ROLLING_WINDOW: int = 12
    # Lista opcional de janelas (em nº de amostras) para gerar features
    # derivadas em multiplas escalas de tempo (ex: [12, 120, 480, 2880] =
    # 6min/1h/4h/24h a 30s de resolucao). Quando definida, substitui
    # DERIVED_ROLLING_WINDOW (que continua sendo o fallback de janela unica).
    # Ver docs/analise_automl_exp7_planejamento.md.
    DERIVED_ROLLING_WINDOWS: Optional[List[int]] = None
    # Subconjunto opcional de `sensors` (por grupo) que recebe features de
    # "textura" (kurtosis/skew/crest, ver TEXTURE_MIN_WINDOW em
    # preprocess.py) quando ENABLE_DERIVED_FEATURES=true. None = todos os
    # sensores recebem (comportamento anterior, EXP7 item 2). Definido para
    # restringir textura a sensores onde a motivacao fisica de
    # "impulsividade" se aplica (ex: vibracao) e evitar aplicar o mesmo
    # tratamento, sem base fisica, a sensores de temperatura/pressao
    # adicionados so como contexto correlacionado. Ver
    # docs/analise_automl_exp9_planejamento.md (EXP9c).
    TEXTURE_SENSORS: Optional[List[str]] = None

    # Features causais de deteccao de mudanca de regime (CUSUM + z-score de
    # linha de base local), complementares ao threshold por percentil
    # global. Ver docs/analise_automl_exp7_planejamento.md (item 3).
    ENABLE_CHANGEPOINT_FEATURES: bool = False
    CHANGEPOINT_SHORT_WINDOW: int = 120   # amostras (1h a 30s)
    CHANGEPOINT_LONG_WINDOW: int = 2880   # amostras (24h a 30s)
    CHANGEPOINT_CUSUM_K: float = 0.5      # folga (slack) em desvios-padrao locais

    # Feature de desbalanceamento entre sondas fisicamente redundantes (ex:
    # os 6 termopares TC382_0X_A do mesmo anel de exaustao): desvio-padrao
    # entre as leituras do array a cada instante, tratado como um
    # pseudo-sensor (recebe o mesmo tratamento multi-escala/textura que
    # qualquer sensor de `sensors`, via `select_feature_columns`). Ver
    # docs/analise_automl_exp9_planejamento.md.
    ENABLE_THERMAL_ARRAY_SPREAD: bool = False
    THERMAL_ARRAY_SENSORS: Optional[List[str]] = None

    OUTLIER_MODE: str = "none"  # "none" | "quantile" | "mad"
    OUTLIER_Q_LOW: float = 0.005
    OUTLIER_Q_HIGH: float = 0.995
    OUTLIER_MAD_K: float = 6.0
    NORMALIZE_MODE: str = "zscore"  # "zscore" | "robust"

    # Sequencias
    TIME_STEPS: int = 60
    STRIDE: int = 1

    # Split
    VAL_FRAC: float = 0.10
    SHUFFLE_TRAIN: bool = False
    SPLIT_MODE: str = "temporal"  # "temporal" | "random"

    # Tuning
    MAX_TRIALS: int = 10
    EXECUTIONS_PER_TRIAL: int = 1
    EPOCHS: int = 20
    BATCH_SIZE: int = 1024
    PATIENCE: int = 6

    # Threshold
    THRESH_MODE: str = "p99"  # "max_train" | "p95" | "p97" | "p99" | "p99_5" | "target_rate" | "mean_std" | "robust_mad"
    TARGET_ANOMALY_RATE: float = 0.01
    THRESH_STD_K: float = 3.0  # "mean_std": media + K*desvio_padrao | "robust_mad": mediana + K*1.4826*MAD

    # Regra de ponto
    POINT_RULE: str = "k_of_window"  # "all_of_window" | "k_of_window"
    POINT_WINDOW: int = 60
    POINT_MIN_COUNT: int = 3

    # Mascara operacional (reduz falso positivo em liga/desliga e off)
    ENABLE_OPERATIONAL_MASK: bool = False
    OFF_VALUE_QUANTILE: float = 0.05
    OFF_ABS_THRESHOLD: Optional[float] = None
    OFF_LONG_MIN_HOURS: float = 24.0
    TRANSIENT_PADDING_MINUTES: int = 20
    # Segundo criterio de "off", independente de OPERATIONAL_REF_SENSOR:
    # quando definido, o proprio sensor-alvo (`target_sensor` do grupo)
    # sendo <= este valor tambem conta como off (OR'd com o criterio do
    # sensor de referencia). Motivado por um desligamento real em que
    # RUNNING_A ficou ~1 o tempo todo enquanto TC382_03_A caiu para nivel
    # ambiente -- ver build_operational_state em scoring.py.
    OFF_TARGET_ABS_THRESHOLD: Optional[float] = None
    TRANSIENT_DIFF_QUANTILE: float = 0.99

    # Portao de rampa de carga (suprime falso positivo durante manobra, sem
    # mexer no threshold). LOAD_GATE_SENSOR e a coluna raw usada como proxy de
    # carga (ex: "T5_AVG_A"); bloqueia is_anom_point quando a rampa desse
    # sensor ultrapassa LOAD_GATE_RAMP_MAX (unidade/hora) numa janela trailing
    # de LOAD_GATE_WINDOW_MINUTES. LOAD_GATE_LEVEL_MIN=0 desativa a condicao
    # de nivel minimo.
    ENABLE_LOAD_GATE: bool = False
    LOAD_GATE_SENSOR: Optional[str] = None
    LOAD_GATE_RAMP_HALFLIFE_MINUTES: float = 120.0
    LOAD_GATE_WINDOW_MINUTES: float = 360.0
    LOAD_GATE_RAMP_MAX: float = 80.0
    LOAD_GATE_LEVEL_MIN: float = 0.0

    # Portao de volatilidade (complementar ao portao de rampa): suprime
    # falso positivo quando um indice de volatilidade multivariado
    # (desvio-padrao movel medio entre VOLATILITY_GATE_SENSORS, tipicamente
    # os canais de vibracao) ultrapassa VOLATILITY_GATE_THRESHOLD -- diferente
    # do portao de rampa (que reage a taxa de variacao do NIVEL de um unico
    # proxy), este reage ao NIVEL da variabilidade em si, o que combina
    # melhor com o padrao fisico observado (vibracao fica mais "ruidosa" e
    # PERMANECE assim durante toda a manobra, nao so na subida). Ver
    # docs/analise_automl_exp9_planejamento.md.
    ENABLE_VOLATILITY_GATE: bool = False
    VOLATILITY_GATE_SENSORS: Optional[List[str]] = None
    VOLATILITY_GATE_WINDOW_MINUTES: float = 60.0
    VOLATILITY_GATE_THRESHOLD: Optional[float] = None

    # Reprodutibilidade
    RANDOM_SEED: int = 42

    # Selecao de sensores
    SENSOR_LIST: Optional[List[str]] = None
    SENSOR_EXCLUDE: Optional[List[str]] = None
    SENSOR_REGEX: Optional[str] = None

    # Grupos de sensores fisicamente conectados.
    # Cada grupo pode ter overrides: time_steps, stride, thresh_mode,
    # target_anomaly_rate, point_window, point_min_count.
    # Exemplo:
    #   [{"name": "bomba_01",
    #     "sensors": ["P_ent_B01", "P_sai_B01", "T_oleo_B01"],
    #     "time_steps": 360}]
    SENSOR_GROUPS: Optional[List[Dict[str, Any]]] = None

    # =========================
    # FONTE DE DADOS EXTRA
    # =========================
    # CSV adicional no dataset ClearML (ou caminho local) cujas colunas novas
    # são mescladas no df_raw principal.  Útil quando sensores de interesse
    # (ex: NGP_A) estão num arquivo separado do RAW_CSV principal.
    EXTRA_RAW_CSV: Optional[str] = None

    # Sensor de referência para detectar estado operacional (liga/desliga).
    # Quando definido, substitui o próprio sensor-alvo em build_operational_state.
    # Exemplos: "RUNNING_A" (arquivo novo) ou "NGP_A" (arquivo antigo via EXTRA_RAW_CSV).
    OPERATIONAL_REF_SENSOR: Optional[str] = None

    # =========================
    # AUTOML (busca de modelo, adaptado da pipeline transpetro_modelos da Lara
    # — ver analise_automl_lara.md). Quando ENABLE_AUTOML=true, roda um search
    # sobre AUTOML_MODELS x AUTOML_THRESHOLD_PERCENTILES x AUTOML_DEBOUNCE_GRID
    # para cada grupo de SENSOR_GROUPS, ranqueado por composite_score
    # (deteccao de alarme real, na nossa propria janela — nao os 30 dias/
    # eventos nao curados da pipeline original). Substitui o caminho CNN-1D;
    # nao roda sensores individuais fora de SENSOR_GROUPS.
    # =========================
    ENABLE_AUTOML: bool = False
    AUTOML_MODELS: Optional[List[str]] = None  # default: ["dense", "ocsvm", "iforest"]
    AUTOML_THRESHOLD_PERCENTILES: Optional[List[float]] = None  # default: [95, 97, 99, 99.5]
    AUTOML_DEBOUNCE_GRID: Optional[List[int]] = None  # pontos consecutivos exigidos; default: [1]
    AUTOML_FP_PENALTY: float = 2.0
    AUTOML_MIN_DETECTION_RATE: float = 0.3

    AUTOML_DENSE_LAYERS: Optional[List[int]] = None  # default: [256, 128]
    AUTOML_DENSE_DROPOUT: float = 0.0
    AUTOML_DENSE_LR: float = 1e-3
    AUTOML_DENSE_EPOCHS: int = 50
    AUTOML_DENSE_BATCH_SIZE: int = 256
    AUTOML_DENSE_PATIENCE: int = 10

    AUTOML_OCSVM_NU: float = 0.05
    AUTOML_OCSVM_GAMMA: str = "scale"
    # Grades opcionais de hiperparametro para ocsvm: quando definidas,
    # substituem o par unico AUTOML_OCSVM_NU/AUTOML_OCSVM_GAMMA por um
    # produto cartesiano de (nu, gamma) re-treinados e ranqueados junto no
    # mesmo automl_ranking.csv. Ver docs/analise_automl_exp9_planejamento.md
    # (item 3).
    AUTOML_OCSVM_NU_GRID: Optional[List[float]] = None
    AUTOML_OCSVM_GAMMA_GRID: Optional[List[Any]] = None
    # Limite de amostras usadas para *ajustar* o OneClassSVM (RBF escala mal
    # com n). None = sem limite (usa x_normal inteiro).
    AUTOML_OCSVM_MAX_TRAIN_SAMPLES: Optional[int] = 50000

    AUTOML_IFOREST_CONTAMINATION: float = 0.05
    AUTOML_IFOREST_N_ESTIMATORS: int = 100

    # Validacao out-of-sample: quando definido (ex: "2025-05-01"), o treino
    # (ajuste do modelo, estatisticas de normalizacao e percentil de threshold)
    # usa APENAS dados com timestamp anterior a essa data; hit_rate,
    # normal_alert_rate e composite_score sao calculados so no periodo >= essa
    # data (alarmes e pontos que o modelo nunca viu). None = sem split, avalia
    # no mesmo periodo do treino (numero apenas indicativo, nao validado).
    AUTOML_OOS_SPLIT_DATE: Optional[str] = None

    # Mesmo mecanismo de split OOS acima, mas para o pipeline CNN1D-AE
    # sequencial (run_one_sensor/run_one_group) -- campo separado porque as
    # duas pipelines sao configuradas/rodadas independentemente (uma tem
    # ENABLE_AUTOML=false). Sem isso, o CNN1D-AE avalia hit_rate/
    # normal_alert_rate contra TODOS os alarmes do periodo carregado, sem
    # nenhum holdout temporal -- nao comparavel com a disciplina OOS do
    # AutoML (ver docs/analise_automl_exp5.md).
    OOS_SPLIT_DATE: Optional[str] = None

    # Checagem de variancia de semente: apos escolher o melhor trial, se o
    # modelo vencedor for "iforest" (unico com randomness controlada
    # diretamente por RANDOM_SEED nesta pipeline), re-treina o mesmo
    # model_type/threshold_percentile/debounce com N seeds extras
    # (RANDOM_SEED+1 .. RANDOM_SEED+N) e registra hit_rate/normal_alert_rate
    # de cada um em calibration_report.json["seed_sweep"]. 0/None = desliga.
    AUTOML_SEED_SWEEP_N: int = 0

    # =========================
    # SUPERVISIONADO (EXP7/EXP8 item 4 -- classificador de alerta precoce,
    # em vez de detector de anomalia nao-supervisionado. Reusa as mesmas
    # features do AutoML (select_feature_columns) mas otimiza diretamente
    # "existe alarme nas proximas PREDICTION_HORIZON_HOURS?" usando os
    # alarmes reais como rotulo. Ver docs/analise_automl_exp7_planejamento.md
    # e src/cnn1d_ae/supervised_pipeline.py.
    # =========================
    ENABLE_SUPERVISED: bool = False
    PREDICTION_HORIZON_HOURS: float = 24.0
    SUPERVISED_PROBA_THRESHOLDS: Optional[List[float]] = None  # default: [0.2..0.8]
    SUPERVISED_N_ESTIMATORS: int = 300
    SUPERVISED_MAX_DEPTH: Optional[int] = None
    SUPERVISED_MIN_SAMPLES_LEAF: int = 5
    # Subamostra negativos pro fit (mantem todos os positivos) -- RandomForest
    # com class_weight="balanced" nao precisa disso pra corretude estatistica,
    # so por velocidade com centenas de milhares de pontos "on".
    SUPERVISED_MAX_TRAIN_SAMPLES: Optional[int] = 200000

    # Execucao em lote
    OVERWRITE: bool = False
    MIN_STD: float = 1e-8
    N_WORKERS: int = 1

    # ClearML
    CLEARML_PROJECT_NAME: str = "TesteMLCab"
    CLEARML_DATASET_NAME: str = "Cabiunas 2025"
    CLEARML_DATASET_ID: str = ""
    USE_CLEARML_DATASET: bool = True
    CLEARML_DOCKER_IMAGE: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    RUN_REMOTE: bool = False
    REMOTE_QUEUE: str = "default"


def cfg_to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
    return asdict(cfg)


def update_cfg_from_dict(cfg: PipelineConfig, d: Dict[str, Any]) -> PipelineConfig:
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
