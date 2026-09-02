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

    # Features no estilo do detector de 4 sinais de mecanismo conhecido
    # (DOC_EQUIPE/ROTEIRO_APRESENTACAO_TC33003A.pdf) -- adicionadas como
    # colunas extras ao MESMO modelo unico (nao substituem a arquitetura,
    # so enriquecem a representacao): "e se alimentarmos o modelo com os
    # dados brutos MAIS essa leva de informacao afinada?" em vez de trocar
    # o modelo por 4 sinais votando. Ver preprocess.py:_build_bearing_spread
    # e :_build_vibration_envelope.
    #
    # Spread de mancal: sensor-alvo (ex: TI_0305) MENOS a mediana dos
    # irmaos (outros termopares de mancal), depois z-robusto rolante --
    # mede DIVERGENCIA de um mancal especifico, nao temperatura absoluta.
    # Assimetrico (tem um alvo definido) e diferente de
    # ENABLE_THERMAL_ARRAY_SPREAD (desvio-padrao SIMETRICO de um array,
    # sem alvo).
    ENABLE_BEARING_SPREAD: bool = False
    BEARING_SPREAD_TARGET_SENSOR: Optional[str] = None
    BEARING_SPREAD_SIBLING_SENSORS: Optional[List[str]] = None
    BEARING_SPREAD_WINDOW_MINUTES: float = 24000.0  # 400h, mesma referencia rolante do detector de 4 sinais

    # Envelope de vibracao: MAXIMO do z-robusto rolante entre os canais de
    # vibracao -- pensado para expor um precursor fraco (achado do
    # Thallys: sinal real em p80 da distribuicao saudavel, some se diluido
    # dentro de um score multivariado com limiar unico em p99,9).
    ENABLE_VIBRATION_ENVELOPE: bool = False
    VIBRATION_ENVELOPE_SENSORS: Optional[List[str]] = None
    VIBRATION_ENVELOPE_WINDOW_MINUTES: float = 24000.0  # 400h

    OUTLIER_MODE: str = "none"  # "none" | "quantile" | "mad"
    OUTLIER_Q_LOW: float = 0.005
    OUTLIER_Q_HIGH: float = 0.995
    OUTLIER_MAD_K: float = 6.0
    NORMALIZE_MODE: str = "zscore"  # "zscore" | "robust"
    # Restringe as estatisticas de normalizacao (center/scale) ao periodo
    # operacional ("on") do treino, em vez de todo df_normal_fit -- exige
    # ENABLE_OPERATIONAL_MASK=true. EXP13 (k=7,0): parada/partida ocupa ~42%
    # do treino, inflando o desvio-padrao global de TC382_03_A de ~51 (so
    # 'on') para ~323 -- um desvio real de +115C dentro da operacao vira
    # z-score 1,22 em vez de 2,22, ficando abaixo do ruido normal do sinal e
    # invisivel ao reconstruction error. Ver docs/analise_cnn1dae_exp13.md.
    NORMALIZE_ON_STATE_ONLY: bool = False

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

    # Bloqueio gradual dos portoes (load/volatilidade): em vez de zerar
    # is_anom_point de forma binaria durante toda a janela bloqueada, um
    # ponto cujo MAE bruto ultrapassa threshold*GATE_ESCAPE_MULTIPLIER
    # "escapa" do bloqueio. Motivado pelo EXP13 (episodio 2026-01-29: MAE
    # 65% acima do threshold normal ficou completamente escondido por 4h
    # porque load_gate E volatility_gate bloquearam ao mesmo tempo, sendo a
    # elevacao de volatilidade fisicamente real, nao artefato de calculo --
    # ver docs/analise_cnn1dae_exp13.md). None ou <=1.0 = desligado
    # (comportamento binario de sempre, nenhum config existente e afetado).
    GATE_ESCAPE_MULTIPLIER: Optional[float] = None

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

    # EWMA no score CONTINUO antes do limiar de percentil -- nao debounce no
    # flag binario depois (que e o que o pipeline ja faz via AUTOML_DEBOUNCE_GRID/
    # map_seq_to_point_anomalies). Ideia portada da branch feat/pdm-deteccao-4sinais
    # (colega Francisco -- ver docs/analise_pca_monitoramento_sistema.md e
    # ALARMES_POR_SENSOR_EFEITO_CASCATA.md): suaviza train_err/all_err com uma
    # media movel exponencial baseada em TEMPO real (pandas .ewm(times=...),
    # nao contagem de amostras -- importante porque df_normal tem buracos,
    # janelas de alarme/desligamento excluidas) ANTES de calcular o percentil
    # do treino e comparar o score de avaliacao com ele. Um pico isolado de
    # 1-2 pontos, que cruzaria o limiar bruto na hora, pode nunca cruzar o
    # limiar depois de suavizado -- reduz falso positivo sem depender so do
    # debounce pos-flag. False (default) = comportamento anterior inalterado,
    # nenhum config existente e afetado.
    ENABLE_SCORE_EWMA: bool = False
    SCORE_EWMA_HALFLIFE: str = "30min"

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

    # Checagem de variancia de semente pro CNN1D-AE (run_one_group):
    # re-treina a MESMA arquitetura (best_hp do tuner) com N seeds extras,
    # medindo o quanto hit_rate/normal_alert_rate variam so por causa da
    # aleatoriedade de inicializacao/treino -- mesmo mecanismo/motivacao do
    # AUTOML_SEED_SWEEP_N (ver analise_automl_lara.md secao 2), campo
    # separado pelo mesmo motivo de OOS_SPLIT_DATE. 0 = desliga.
    SEED_SWEEP_N: int = 0

    # Checagem de variancia de semente: apos escolher o melhor trial, se o
    # modelo vencedor for "iforest" (unico com randomness controlada
    # diretamente por RANDOM_SEED nesta pipeline), re-treina o mesmo
    # model_type/threshold_percentile/debounce com N seeds extras
    # (RANDOM_SEED+1 .. RANDOM_SEED+N) e registra hit_rate/normal_alert_rate
    # de cada um em calibration_report.json["seed_sweep"]. 0/None = desliga.
    AUTOML_SEED_SWEEP_N: int = 0

    # Veto de sensor congelado: suprime is_anom_point quando QUALQUER
    # sensor do grupo fica com leitura literalmente constante (diff()==0)
    # por uma janela sustentada de FROZEN_SENSOR_VETO_WINDOW_MINUTES --
    # indica falha de instrumento/comunicacao, nao sinal real. Mesma
    # camada dos outros portoes (rampa/volatilidade): so remove deteccao,
    # nunca adiciona. Validado no EXP10c: W=5min reduz normal_alert_rate
    # em ~7,2% sem custar nenhuma deteccao real (ver
    # docs/analise_automl_exp10.md, secao "Veto de sensor congelado").
    # False (default) = comportamento anterior inalterado.
    ENABLE_FROZEN_SENSOR_VETO: bool = False
    FROZEN_SENSOR_VETO_WINDOW_MINUTES: float = 5.0

    # Anotacao de contexto de alerta contra um catalogo MAIS AMPLO de
    # alarmes (ex: 47 tags de alarmes_selecionados_turbina_a.csv -- nao
    # o ALARM_CSV usado na avaliacao oficial, que costuma ser um
    # subconjunto curado). PURAMENTE INFORMATIVO: adiciona as colunas
    # alert_catalog_tag/alert_catalog_time/alert_catalog_distance_h/
    # alert_confidence ao point_anomalies_all.csv SEM alterar
    # is_anom_point, hit_rate ou normal_alert_rate. Implementa a
    # estrategia "classificar por confianca, corroborar em tempo de
    # alerta" (nao suprimir dentro do treino/score) -- ver
    # docs/analise_automl_exp10.md, secoes "Cruzamento com catalogo
    # completo" (88,1% do residuo "amarelo" era sinal real de outro
    # alarme) e "Supressao cirurgica baseada em mecanismo: testada
    # offline e REJEITADA" (suprimir dentro da pipeline custaria 2 dos
    # 8 TRIPs reais -- anotar em vez de suprimir tem risco zero porque
    # nao decide nada, so da contexto pro operador). False (default) =
    # comportamento anterior inalterado (nenhuma coluna nova).
    #
    # ALERT_CONTEXT_WINDOW_HOURS default = 2.0 (nao mais 24.0): revisao
    # externa da equipe (relatorio DOC_EQUIPE/RELATORIO_DECISAO_DETECTOR.pdf)
    # apontou, e validamos com controle negativo (compute_catalog_enrichment_control),
    # que em +-24h ate 71,7% de instantes ALEATORIOS ja caem perto de algum
    # alarme do catalogo (catalogo denso) -- o "88,1%/97,6% explicado" medido
    # naquela janela e so 1,36x de enriquecimento sobre o acaso, nao a
    # descoberta forte que parecia. +-2h tem denominador muito mais estreito
    # e enriquecimento genuino. Ver docs/analise_automl_exp10.md, secao
    # "Controle negativo do enriquecimento por catalogo".
    ENABLE_ALERT_CATALOG_CONTEXT: bool = False
    ALERT_CONTEXT_CATALOG_CSV: Optional[str] = None
    ALERT_CONTEXT_WINDOW_HOURS: float = 2.0
    ALERT_CONTEXT_CONTROL_N_SAMPLES: int = 5000

    # Retreino walk-forward mensal: em vez de um unico ajuste (modelo +
    # normalizacao + limiar de percentil) sobre todo o normal anterior a
    # AUTOML_OOS_SPLIT_DATE, re-treina do zero a cada
    # WALKFORWARD_RETRAIN_FREQ (alias de frequencia do pandas, ex "MS" =
    # inicio de mes) usando janela EXPANSIVA (todo o normal disponivel
    # antes daquele ponto), pontuando so o proximo periodo com o modelo
    # daquele momento. Portoes/mascara operacional continuam FIXOS nos
    # valores do config -- so a cadencia de retreino do modelo muda.
    # Exige AUTOML_OOS_SPLIT_DATE definido (walk-forward so faz sentido
    # dentro do periodo de avaliacao OOS). Validado no EXP10c: reduz
    # normal_alert_rate em ~19% mantendo o mesmo hit_rate (ver
    # docs/analise_automl_exp10.md, secao "Retreino walk-forward mensal").
    # CUIDADO: um limiar/filtro adicional calibrado contra um UNICO
    # modelo (congelado) nao e garantidamente seguro sob walk-forward --
    # a mesma secao do doc mostra 2 tentativas (limiares de portao mais
    # agressivos, filtro de duracao) que pareciam "custo zero" contra o
    # modelo congelado e quebraram (perderam deteccoes reais) quando
    # combinadas com retreino mensal, porque a distribuicao de score
    # muda a cada retreino. Antes de adicionar qualquer novo
    # limiar/filtro dependente de score/duracao, revalidar especificamente
    # com ENABLE_WALKFORWARD_RETRAIN=true, nao So contra o modelo
    # congelado. False (default) = comportamento anterior inalterado.
    ENABLE_WALKFORWARD_RETRAIN: bool = False
    WALKFORWARD_RETRAIN_FREQ: str = "MS"

    # Filtro de duracao minima: suprime episodios continuos de
    # is_anom_point mais curtos que MIN_DURATION_FILTER_MINUTES --
    # precursores reais tendem a persistir muito mais tempo que ruido
    # residual (mediana 49,5min vs 2,5min no EXP10c congelado). Aplicado
    # logo apos a mascara operacional, ANTES dos portoes de rampa/
    # volatilidade/veto de congelamento -- aplicar depois deles mediria a
    # duracao de episodios ja fragmentados (derrubou hit_rate de 90% pra
    # 47,5% num primeiro teste, bug de ordem ja corrigido). Validado
    # contra o modelo CONGELADO: 6min reduz normal_alert_rate em ~32,7%
    # custando so 1 deteccao ja marginal (um unico ponto de 30s) de 37 --
    # ver docs/analise_automl_exp10.md, secao "Duracao do score: TP vs FP
    # residual".
    #
    # INCOMPATIVEL COM ENABLE_WALKFORWARD_RETRAIN: 3 tentativas
    # independentes (limiar fixo, adaptativo sobre o treino expansivo,
    # adaptativo sobre janela recente de 60d) mostraram que duracao
    # calibrada contra retreino mensal derruba o hit_rate de 92,5% pra
    # 15-40% -- nao e questao de calibracao, e estrutural (o score de
    # varios meses tem comportamento de "quase-limiar" mais persistente
    # que o do modelo congelado, tanto em ruido quanto em precursor
    # real). `run_automl_group` levanta erro se as duas flags estiverem
    # ligadas juntas. False (default) = comportamento anterior inalterado.
    ENABLE_MIN_DURATION_FILTER: bool = False
    MIN_DURATION_FILTER_MINUTES: float = 6.0

    # Tags de alarme EXTRAS (fora de `eval_sensors`) cuja janela de
    # +-EXTRA_NEAR_ALARM_WINDOW_MINUTES tambem e excluida do denominador
    # de normal_alert_rate -- NAO afeta treino nem hit_rate/eval_sensors,
    # so reconhece que um ponto marcado anomalo perto de um evento real
    # de OUTRO sensor (nao um dos avaliados) nao deveria contar como
    # falso alerta "do nada". Motivado por achado empirico no EXP20: 72%
    # dos pontos oficialmente contados como FP coincidiam (+-24h) com um
    # alarme de algum dos outros 45 tags do catalogo (principalmente
    # pressao -- PI_6240319_AL/PAL_6240315/PDAL_6240302), e a maioria dos
    # que sobram isolados ainda mostra pressao se movendo 2-3 desvios-
    # padrao acima do normal, so que abaixo do limiar oficial de alarme.
    # Ver docs/analise_automl_exp10.md, secao "Portao de pressao (EXP21)".
    #
    # Janela PROPOSITALMENTE curta (default 120min = +-2h, nao os
    # +-1440min usados pros 40 alarmes raros do target): esses 3 tags de
    # pressao ocorrem ~1x/dia no periodo OOS -- uma janela de +-24h
    # cobriria 60% do tempo todo (checado empiricamente), inviabilizando
    # a metrica; +-2h cobre ~8,5%, proporcional a escala de tempo real
    # dos episodios residuais que motivaram este portao (a maioria dura
    # minutos, nao horas). None (default) = comportamento anterior
    # inalterado, nenhum config existente e afetado.
    EXTRA_NEAR_ALARM_TAGS: Optional[List[str]] = None
    EXTRA_NEAR_ALARM_WINDOW_MINUTES: float = 120.0

    # Portao de MUDANCA DE NIVEL (step-change): complementa o portao de
    # rampa (que reage a TAXA de variacao suavizada por EWMA) -- pensado
    # pra pegar degraus de carga quase instantaneos (temperatura E
    # vibracao mudando de patamar juntas em poucos minutos), rapido
    # demais pro portao de rampa reagir (a suavizacao de 15min dilui a
    # taxa percebida bem na hora da transicao). Indice = |media curta -
    # media longa| / (desvio longo + eps) do mesmo proxy de carga
    # (LOAD_GATE_SENSOR) -- mesma matematica do "localz" ja usado como
    # FEATURE do modelo em _build_changepoint_features (preprocess.py),
    # aqui aplicado como PORTAO (suprime deteccao, nao alimenta o
    # modelo). Suprime is_anom_point quando o indice ultrapassa
    # STEP_CHANGE_THRESHOLD. Validado no EXP22: threshold=1.5 (janelas
    # 5min/60min) reduz normal_alert_rate em ~54,5% sem custar nenhuma
    # das 36 deteccoes do EXP21 -- ver docs/analise_automl_exp10.md,
    # secao "Portao de mudanca de nivel (EXP22)". False (default) =
    # comportamento anterior inalterado.
    ENABLE_STEP_CHANGE_GATE: bool = False
    STEP_CHANGE_GATE_SENSOR: Optional[str] = None  # None = usa LOAD_GATE_SENSOR
    STEP_CHANGE_SHORT_WINDOW_MINUTES: float = 5.0
    STEP_CHANGE_LONG_WINDOW_MINUTES: float = 60.0
    STEP_CHANGE_THRESHOLD: float = 1.5

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
