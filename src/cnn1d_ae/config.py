from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


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
    # Deprecated: substituido por ENABLE_ROLLING_FEATURES / ENABLE_SPECTRAL_FEATURES / ENABLE_CONTEXT_FEATURES.
    ENABLE_DERIVED_FEATURES: bool = False
    DERIVED_ROLLING_WINDOW: int = 12
    ENABLE_ROLLING_FEATURES: bool = True
    ROLLING_WINDOW: Optional[int] = None  # None = usa TIME_STEPS
    ENABLE_SPECTRAL_FEATURES: bool = False
    SPECTRAL_WINDOW: Optional[int] = None  # None = usa TIME_STEPS
    SPECTRAL_STRIDE: Optional[int] = None  # None = usa STRIDE
    SENSOR_TYPE: str = "temperature"  # "temperature" | "pressure" | "vibration"
    ENABLE_CONTEXT_FEATURES: bool = False
    CONTEXT_COLS: List[str] = field(default_factory=list)

    # Features de tendência: slope + desvio do baseline de longo prazo
    # Captura deriva direcional que rolling mean/std não vê.
    ENABLE_TREND_FEATURES: bool = False
    TREND_SLOPE_WINDOW: Optional[int] = None  # None = usa TIME_STEPS
    BASELINE_HOURS: float = 24.0              # janela do baseline de longo prazo (horas)
    OUTLIER_MODE: str = "none"  # "none" | "quantile" | "mad"
    OUTLIER_Q_LOW: float = 0.005
    OUTLIER_Q_HIGH: float = 0.995
    OUTLIER_MAD_K: float = 6.0
    NORMALIZE_MODE: str = "zscore"  # "zscore" | "robust"

    # Detecção e remoção de valores sentinela (ex: -40.5°C = thermocouple aberto)
    # Aplicado antes da interpolação; valores fora do range físico viram NaN
    SENTINEL_MODE: str = "none"  # "none" | "clip"
    SENTINEL_LOW: Optional[float] = None   # ex: -10.0 para sensores de temperatura
    SENTINEL_HIGH: Optional[float] = None  # ex: 900.0 para sensores de temperatura

    # Filtro de Hampel: remove spikes isolados (não remove rampas sustentadas)
    # Janela efetiva = 2*HAMPEL_WINDOW + 1 pontos; limiar = HAMPEL_SIGMA * 1.4826 * MAD
    ENABLE_HAMPEL_FILTER: bool = False
    HAMPEL_WINDOW: int = 5     # pontos de cada lado da janela central
    HAMPEL_SIGMA: float = 3.0  # equivalente a 3σ

    # Normalização somente sobre períodos de operação estável
    # Evita distorção do z-score pela distribuição bimodal ON/OFF
    NORMALIZE_ON_STABLE_ONLY: bool = False
    STABLE_ON_GRADIENT_QUANTILE: float = 0.95  # pontos com grad <= quantile são "estáveis"

    # Exclusão de X minutos após cada startup do conjunto de treino
    # Cobre a rampa de temperatura que o modelo não deve aprender como "normal"
    EXCLUDE_STARTUP_MINUTES: int = 0

    # Coluna de estado operacional direta (ex: "RUNNING_A").
    # Quando definida, substitui a detecção por limiar: ON = coluna > 0.5.
    # Se None, usa OFF_ABS_THRESHOLD / OFF_VALUE_QUANTILE como antes.
    RUNNING_COL: Optional[str] = None

    # Exclusão de runs de forward-fill upstream do treino.
    # Quando o dado de entrada é pré-interpolado com last-value-carried-forward,
    # runs de N pontos idênticos consecutivos durante ON são artefatos, não dados reais.
    # Excluí-los evita que o AE aprenda plateaus artificiais como padrão normal
    # e pare de flagrar esses plateaus como anomalias durante a inferência.
    EXCLUDE_CONSTANT_RUNS: bool = False
    CONSTANT_RUN_MIN_LENGTH: int = 3  # mínimo de pontos iguais para considerar forward-fill

    # Sequencias
    # Exploratorio: TIME_STEPS=48, EPOCHS=10, PATIENCE=2, MAX_TRIALS=10.
    # Calibracao final recomendada: definir CONTEXT_HOURS conforme janela fisica desejada,
    # EPOCHS=50-100, PATIENCE=8-12 e MAX_TRIALS=30-50.
    TIME_STEPS: int = 60
    CONTEXT_HOURS: Optional[float] = None
    TIME_STEPS_TOLERANCE: float = 0.05
    REQUIRE_CONTEXT_MATCH: bool = False
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
    THRESH_MODE: str = "p99"  # "max_train" | "p95" | "p97" | "p99" | "p99_5" | "target_rate" | "alarm_f2"
    TARGET_ANOMALY_RATE: float = 0.01

    # Threshold semi-supervisionado (THRESH_MODE="alarm_f2")
    ALARM_F2_TARGET_RECALL: float = 0.65   # meta mínima de recall por incidente (gap > ALARM_F2_INCIDENT_GAP_HOURS)
    ALARM_F2_MAX_FP_PER_DAY: float = 15.0  # FP/dia máximo tolerado
    ALARM_F2_INCIDENT_GAP_HOURS: float = 4.0  # gap mínimo (h) para definir incidentes distintos

    # Threshold adaptativo mensal (recalibra sem retreinar)
    ADAPTIVE_THRESHOLD_MODE: str = "none"       # "none" | "monthly"
    ADAPTIVE_THRESHOLD_PERCENTILE: float = 99.0

    # Detecção em dois níveis: warning (sensível) + alarm (confirmado)
    ENABLE_WARN_LEVEL: bool = False
    WARN_POINT_MIN_COUNT: int = 5
    WARN_TARGET_ANOMALY_RATE: float = 0.01

    # Avaliacao
    EVAL_WINDOW_MINUTES: Optional[int] = None

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
    TRANSIENT_DIFF_QUANTILE: float = 0.99

    # Buffer assimétrico da máscara operacional: só mascara o ramp-up pós-partida
    # (off->on), preservando a janela pré-desligamento (on->off, possível trip por falha).
    # Recupera recall sem reintroduzir FP de partida.
    OPERATIONAL_MASK_ASYMMETRIC: bool = True

    # Plot: mostrar só os alarmes da própria variável plotada (Tag == sensor),
    # em vez de todos os alarmes. Avaliação (hit_rate) segue usando todos.
    PLOT_ALARMS_PER_VARIABLE: bool = True

    # Reprodutibilidade
    RANDOM_SEED: int = 42

    # Selecao de sensores
    SENSOR_LIST: Optional[List[str]] = None
    SENSOR_EXCLUDE: Optional[List[str]] = None
    SENSOR_REGEX: Optional[str] = None

    # Modelo multivariado conjunto: treina UM AE com todos os SENSOR_LIST como canais
    MULTIVARIATE_JOINT: bool = False

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
