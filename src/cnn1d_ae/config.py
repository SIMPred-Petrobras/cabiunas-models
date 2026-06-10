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
    THRESH_MODE: str = "p99"  # "max_train" | "p95" | "p97" | "p99" | "p99_5" | "target_rate"
    TARGET_ANOMALY_RATE: float = 0.01

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
