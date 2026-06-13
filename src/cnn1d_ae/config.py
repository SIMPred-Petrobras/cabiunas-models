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
    # Remoção de common-mode: substitui o valor do sensor pelo resíduo contra a
    # média leave-one-out dos irmãos do grupo (ex: os 6 TC382). Mantém univariado
    # (sem OOM), neutraliza deriva/carga compartilhada e deixa o AE enxergar a
    # DIVERGÊNCIA de um sensor vs os vizinhos = sinal de anomalia local.
    # Aplicado no início do feature engineering (antes do rolling).
    ENABLE_COMMON_MODE_REMOVAL: bool = False
    COMMON_MODE_GROUP: list = field(default_factory=list)

    EXCLUDE_MINUTES_AROUND_ALARM: int = 1440
    # Exclusão assimétrica em torno do alarme (sobrepõe AROUND quando definidos).
    # Motivação: alarmes HI/HIHI disparam em carga alta; uma janela simétrica larga
    # (ex: ±48h) remove do treino o regime de carga alta normal, fazendo o AE
    # subajustar carga alta → falsos positivos concentrados lá. O ideal é excluir
    # generosamente DEPOIS (evento + recuperação) e pouco ANTES, preservando o
    # entorno de carga alta normal. None => usa EXCLUDE_MINUTES_AROUND_ALARM (simétrico).
    EXCLUDE_MINUTES_BEFORE_ALARM: Optional[int] = None
    EXCLUDE_MINUTES_AFTER_ALARM: Optional[int] = None
    TRAIN_SKIP_CONDITIONS: list = field(default_factory=list)
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

    # Máscara de spike de gradiente no treino (ideia do pipeline v4 externo).
    # Exclui do treino ±GRADIENT_SPIKE_SUPPRESS_MINUTES ao redor de cada spike
    # de gradiente por sensor (spike = grad > μ + GRADIENT_SPIKE_STD_MULT × σ,
    # calibrado em steady state). Limpa a distribuição "normal" aprendida pelo AE
    # removendo transições abruptas locais que contaminariam as sequências de treino.
    ENABLE_GRADIENT_SPIKE_MASK: bool = False
    GRADIENT_SPIKE_STD_MULT: float = 8.0
    GRADIENT_SPIKE_SUPPRESS_MINUTES: int = 60
    # Calibra o threshold de gradiente excluindo ±N min ao redor de transições ON/OFF.
    # Refina o steady state: evita que rampas de partida/parada distorçam μ/σ.
    # 0 = desabilitado (usa só ON state como antes).
    GRADIENT_SPIKE_TRANSITION_MINUTES: int = 0
    # Suprime anomaly_seq durante janelas de spike de gradiente no scoring.
    # Reduz FP causados por transientes abruptos sem afetar recall (spikes são curtos).
    GRADIENT_SPIKE_SUPPRESS_SCORING: bool = False
    # Lista de sensores onde scoring suppression é aplicada. None = todos os sensores.
    # Permite restringir a supressão a sensores de vibração/pressão (TV_*, PI_*)
    # sem penalizar termopares (TC_*, T5_*) onde spikes são a assinatura de falha.
    GRADIENT_SPIKE_SUPPRESS_SENSORS: Optional[List[str]] = None

    # Filtra SENSOR_LIST automaticamente para sensores com eventos de alarme no CSV.
    # Evita treinar sensores sem ground truth de anomalia.
    SENSOR_FILTER_HAS_ALARMS: bool = False

    # Corte temporal para treino (OOS evaluation).
    # Quando definido (ex: "2025-12-31"), df_normal é filtrado até essa data,
    # garantindo que dados posteriores só entrem no scoring, nunca no treino.
    TRAIN_END_DATE: Optional[str] = None

    # Coluna de estado operacional direta (ex: "RUNNING_A").
    # Quando definida, substitui a detecção por limiar: ON = coluna > RUNNING_THRESHOLD.
    # Se None, usa OFF_ABS_THRESHOLD / OFF_VALUE_QUANTILE como antes.
    RUNNING_COL: Optional[str] = None

    # Limiar para considerar "ON" quando RUNNING_COL é um sinal contínuo
    # (ex: NGP_A em % de rotação do gerador de gás → ON = NGP_A > 50).
    # Default 0.5 preserva o comportamento legado de coluna binária (RUNNING_A > 0.5).
    RUNNING_THRESHOLD: float = 0.5

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

    # Política de alarme no nível da sequência (decide anomaly_seq a partir do erro).
    # "threshold" = erro > threshold ponto a ponto (legado).
    # "cusum"     = detector de deriva sustentada (imune a spikes isolados); usa
    #               mu/sigma do erro de TREINO. Reduz FP que são picos curtos.
    ALARM_POLICY: str = "threshold"
    CUSUM_K: float = 0.5   # folga (em unidades de sigma)
    CUSUM_H: float = 5.0   # limiar do acumulador que dispara
    # Debounce (on-delay): exige N sequências consecutivas anômalas antes de confirmar.
    # 0/1 = desligado. Aplicado depois da política de alarme, antes do mapeamento p/ ponto.
    DEBOUNCE_POINTS: int = 0

    # Threshold por banda de regime de carga. O erro de reconstrução do AE depende
    # da carga: regimes raros (ex: carga alta) têm baseline de erro maior, gerando
    # FP com um threshold global único. Calibra um threshold (THRESH_MODE) separado
    # por banda no erro de TREINO, usando o sinal de RUNNING_COL (ex: NGP_A) como
    # proxy de carga. Bandas com poucas amostras caem para o threshold global.
    ENABLE_REGIME_BAND_THRESHOLD: bool = False
    REGIME_BANDS: list = field(default_factory=list)  # bordas, ex: [90, 94] → ≤90 / 90-94 / >94
    REGIME_BAND_MIN_SAMPLES: int = 150

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

    # Arquitetura do modelo: 'per_sensor' (N AEs univariados, OR-de-quantile) ou
    # 'multivariate' (1 AE com N canais). Validado em produção (in-sample, 2025):
    # per_sensor entrega +3pp recall e 5x menos FA/dia que multivariate (recall
    # 0.69 vs 0.66, FA/dia 0.01 vs 0.05 a H=8h). Manter como default.
    MODEL_MODE: str = "per_sensor"
    # Arquitetura do autoencoder: "cnn1d" (padrão) ou "gru"
    MODEL_ARCH: str = "cnn1d"
    # Hiperparâmetros do backend per_sensor (cada AE é mínimo: 1 canal, latente ~15)
    PER_SENSOR_F1: int = 4
    PER_SENSOR_F2: int = 1
    PER_SENSOR_S1: int = 2
    PER_SENSOR_S2: int = 2
    PER_SENSOR_EPOCHS: int = 15

    # Camada preditiva (Fase 2): EWMA sobre o health-index + curva lead-time x falso-alarme.
    # Substitui "hit_rate@threshold_único" como métrica oficial — robusta à variância de tuning.
    ENABLE_PREDICTIVE_LAYER: bool = True
    # Lista vazia = todas as prioridades (todos os alarmes sao considerados reais
    # pela operacao). Filtre por prioridade so se houver uma curadoria explicita.
    PREDICTIVE_INCIDENT_PRIORITY: List[str] = field(default_factory=list)
    PREDICTIVE_HORIZONS_HOURS: List[float] = field(
        default_factory=lambda: [8.0, 24.0, 72.0]
    )
    PREDICTIVE_EWMA_HALF_LIFE_HOURS: float = 4.0
    # Override de half-life por sensor (default usa o global acima). O half-life
    # ideal depende do tipo de evento: deriva sustentada (térmica, ex: TC382_03)
    # pede hl longo (~4h); dips breves (UNDER de termopar, ex: TC382_04) pedem hl
    # curto (~0.5h) senão a EWMA os alisa. Ex: {"TC382_04_A": 0.5}.
    PREDICTIVE_EWMA_HALF_LIFE_HOURS_PER_SENSOR: dict = field(default_factory=dict)
    PREDICTIVE_ALERT_DEBOUNCE_HOURS: float = 8.0
    PREDICTIVE_FA_BUDGET_PER_DAY: float = 1.0

    # Sistema tier de alertas (per_sensor mode): 3 niveis baseados em q e k votacao.
    # Calibrado por experimentacao (improve_voting_high_q.py): warn=monitoramento,
    # alarm=inspecao urgente, critical=parada sugerida.
    ENABLE_TIER_ALERTS: bool = True
    # Calibracao revisada (2026-05-29): WARN exigia 1 sensor em q=0.70 e fazia
    # 80% do tempo → operacao acostuma e ignora. Nova regra exige coordenacao
    # multi-sensor (que e o sinal fisico real de degradacao).
    WARN_Q: float = 0.80
    WARN_K: int = 2
    ALARM_Q: float = 0.90
    ALARM_K: int = 2
    CRITICAL_Q: float = 0.95
    CRITICAL_K: int = 1

    # Reprodutibilidade
    RANDOM_SEED: int = 42

    # Selecao de sensores
    SENSOR_LIST: Optional[List[str]] = None
    SENSOR_EXCLUDE: Optional[List[str]] = None
    SENSOR_REGEX: Optional[str] = None

    # Modelo multivariado conjunto: treina UM AE com todos os SENSOR_LIST como canais
    MULTIVARIATE_JOINT: bool = False

    # Sensor-alvo do grupo multivariado: o AE reconstrói TODOS os canais, mas o
    # threshold e a decisão de anomalia usam SÓ o MAE deste canal. Detecta quebra
    # de correlação do alvo vs os vizinhos (ex: T5_AVG_A com TC382_03_A de contexto).
    # None = comportamento legado (MAE combinado entre canais).
    TARGET_SENSOR: Optional[str] = None

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
