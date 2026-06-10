# Guia de Configuração — Pipeline CNN-1D Autoencoder

Este documento explica cada parâmetro do arquivo de configuração JSON usado para rodar o pipeline de detecção de anomalias em sensores industriais.

---

## Modo de execução

```json
"MODE": "local",
"N_WORKERS": 1
```

| Campo | O que faz |
|---|---|
| `MODE` | `"local"` roda na sua máquina; `"operacional"` usa caminhos de produção |
| `N_WORKERS` | Quantos sensores rodam em paralelo. Use `1` para depurar; valores maiores exigem mais RAM pois cada processo carrega todos os dados |

---

## Arquivos de entrada

```json
"FEATURES_CSV": "features_record_2025_tzm3.csv",
"RAW_CSV": "serie_consolidada_2025_interpolated_antigo.csv",
"ALARM_CSV": "alarmes_record_2025_tags_modelo.csv",
"EXTRA_RAW_CSV": null
```

| Campo | O que faz |
|---|---|
| `RAW_CSV` | Arquivo principal com as séries temporais dos sensores. **Este arquivo determina a fonte de dados de treino.** Para usar `NGP_A` como referência, aponte para o arquivo antigo. Para usar `RUNNING_A`, aponte para o arquivo novo |
| `FEATURES_CSV` | Arquivo com features derivadas (médias, FFT, etc.). Usado quando `TRAIN_SOURCE: "feat"` |
| `ALARM_CSV` | Registro histórico de alarmes/ocorrências. Usado para excluir janelas ao redor de falhas conhecidas do treino e para calcular o *hit rate* de detecção |
| `EXTRA_RAW_CSV` | Arquivo CSV adicional cujas colunas novas são mescladas no `RAW_CSV`. Use `null` quando os dados vêm de uma fonte única. **Nunca misture fontes quando `OPERATIONAL_REF_SENSOR` está configurado** |

---

## Fuso horário

```json
"TIME_COL": "data_datetime",
"SOURCE_TZ": "America/Sao_Paulo",
"TARGET_TZ": "UTC",
"APPLY_HOUR_SHIFT": true,
"SHIFT_HOURS": -3,
"LOG_TIME_AUDIT_SAMPLES": 3
```

| Campo | O que faz |
|---|---|
| `TIME_COL` | Nome da coluna de data/hora no CSV |
| `SOURCE_TZ` | Fuso horário em que os dados foram gravados (Brasília = `"America/Sao_Paulo"`) |
| `TARGET_TZ` | Fuso horário de destino interno. Usar `"UTC"` evita ambiguidades de horário de verão |
| `APPLY_HOUR_SHIFT` | `true` aplica uma correção manual de horas **após** a conversão de fuso. Necessário para arquivos com offset incorreto na gravação |
| `SHIFT_HOURS` | Horas a adicionar (negativo = subtrair). `-3` corrige arquivos gravados em UTC mas rotulados como horário local |
| `LOG_TIME_AUDIT_SAMPLES` | Quantas linhas imprimir no log para conferência visual da conversão. `0` desativa |

---

## Saídas

```json
"OUTPUT_ROOT": "runs_test_grupo_T5_NGP",
"OUTPUT_DIR_TEMPLATE": "OUTPUT_CNN1D_AE_{sensor}"
```

| Campo | O que faz |
|---|---|
| `OUTPUT_ROOT` | Pasta raiz onde todos os resultados serão salvos |
| `OUTPUT_DIR_TEMPLATE` | Subpasta por sensor/grupo. `{sensor}` é substituído pelo nome do sensor ou grupo |

Estrutura gerada por sensor/grupo:
```
OUTPUT_ROOT/
  OUTPUT_CNN1D_AE_T5_temperatura/
    best_model/   → model.keras + best_hyperparameters.json
    tuner/        → tentativas do KerasTuner
    figs/         → gráficos de perda, histograma MAE, série com anomalias
    csv/          → scores, anomalias por ponto, relatório de calibração
```

---

## Seleção de sensores e grupos

```json
"SENSOR_LIST": null,
"SENSOR_EXCLUDE": null,
"SENSOR_REGEX": null,

"SENSOR_GROUPS": [
  {
    "name": "T5_temperatura",
    "sensors": ["T5_AVG_A", "TC382_03_A"],
    "time_steps": 180,
    "thresh_mode": "target_rate",
    "target_anomaly_rate": 0.003,
    "point_window": 180,
    "point_min_count": 15
  }
]
```

### Seleção individual (sem grupos)

| Campo | O que faz |
|---|---|
| `SENSOR_LIST` | Lista explícita de sensores a processar. `null` = todos os sensores do arquivo |
| `SENSOR_EXCLUDE` | Lista de sensores a ignorar |
| `SENSOR_REGEX` | Filtro por expressão regular no nome da coluna. Ex: `"^TC382"` seleciona todos TC382 |

### Grupos de sensores (`SENSOR_GROUPS`)

Agrupa sensores **fisicamente conectados** para treinar um único autoencoder multicanal. O modelo aprende as correlações entre os canais — uma anomalia que quebra a relação física entre dois sensores é detectada com maior confiança do que analisando cada sensor isoladamente.

| Campo no grupo | O que faz |
|---|---|
| `name` | Identificador do grupo (usado no nome da pasta de saída) |
| `sensors` | Lista dos sensores que compõem o grupo. Devem vir todos do mesmo `RAW_CSV` |
| `time_steps` | **Override local**: janela temporal deste grupo em número de amostras. Com dados a 30s, `180` = 90 minutos de histórico por janela de treino |
| `thresh_mode` | **Override local**: como calcular o limiar de anomalia (ver seção Threshold abaixo) |
| `target_anomaly_rate` | **Override local**: taxa alvo de anomalias sobre os dados de treino |
| `point_window` | **Override local**: janela de votação para confirmar anomalia pontual |
| `point_min_count` | **Override local**: mínimo de sequências anômalas dentro da janela para confirmar o ponto |

> Sensores não incluídos em nenhum grupo continuam sendo processados individualmente com os parâmetros globais.

---

## Pré-processamento

```json
"TRAIN_SOURCE": "raw",
"EXCLUDE_MINUTES_AROUND_ALARM": 1440,
"INTERPOLATE_LIMIT": 3,
"EXCLUDE_LONG_GAPS_FROM_TRAIN": true,
"OUTLIER_MODE": "quantile",
"OUTLIER_Q_LOW": 0.001,
"OUTLIER_Q_HIGH": 0.999,
"NORMALIZE_MODE": "zscore"
```

| Campo | O que faz |
|---|---|
| `TRAIN_SOURCE` | `"raw"` treina sobre o `RAW_CSV`; `"feat"` usa o `FEATURES_CSV` |
| `EXCLUDE_MINUTES_AROUND_ALARM` | Exclui do treino uma janela de N minutos antes e depois de cada alarme registrado. `1440` = 24 horas. Evita que o modelo aprenda comportamento anômalo como "normal" |
| `INTERPOLATE_LIMIT` | Preenche até N amostras consecutivas ausentes por interpolação linear. Buracos maiores que isso são marcados como gaps longos |
| `EXCLUDE_LONG_GAPS_FROM_TRAIN` | `true` remove os gaps longos (buracos maiores que `INTERPOLATE_LIMIT`) do treino |
| `OUTLIER_MODE` | `"none"` não trata outliers; `"quantile"` clippa pelos percentis definidos abaixo; `"mad"` usa desvio absoluto mediano |
| `OUTLIER_Q_LOW` / `OUTLIER_Q_HIGH` | Percentis de corte para o modo `"quantile"`. `0.001` / `0.999` = corta apenas os 0,1% extremos de cada lado |
| `NORMALIZE_MODE` | `"zscore"` normaliza por média e desvio padrão; `"robust"` usa mediana e IQR (mais robusto a outliers). A normalização usa **apenas os dados de treino** para evitar vazamento de informação |

---

## Sequências temporais

```json
"TIME_STEPS": 48,
"STRIDE": 1
```

| Campo | O que faz |
|---|---|
| `TIME_STEPS` | Comprimento da janela deslizante em amostras (parâmetro global). Dentro de cada grupo o valor é sobrescrito pelo `time_steps` do grupo |
| `STRIDE` | Passo entre janelas consecutivas. `1` = máximo de sobreposição; valores maiores reduzem o dataset e aceleram o treino |

---

## Divisão treino/validação

```json
"VAL_FRAC": 0.1,
"SHUFFLE_TRAIN": false,
"SPLIT_MODE": "temporal"
```

| Campo | O que faz |
|---|---|
| `VAL_FRAC` | Fração dos dados normais reservada para validação. `0.1` = 10% |
| `SPLIT_MODE` | `"temporal"` separa os últimos N% como validação (preserva a ordem do tempo); `"random"` embaralha antes de dividir |
| `SHUFFLE_TRAIN` | `false` mantém a ordem temporal no treino. Recomendado para séries temporais |

---

## Hiperparâmetros e treino

```json
"MAX_TRIALS": 5,
"EXECUTIONS_PER_TRIAL": 1,
"EPOCHS": 20,
"BATCH_SIZE": 256,
"PATIENCE": 5
```

| Campo | O que faz |
|---|---|
| `MAX_TRIALS` | Número de combinações de hiperparâmetros que o KerasTuner vai testar. Mais trials = melhor modelo, mais tempo |
| `EXECUTIONS_PER_TRIAL` | Quantas vezes treinar cada combinação de hiperparâmetros (para estabilidade). `1` é suficiente para testes |
| `EPOCHS` | Número máximo de épocas por treino. O EarlyStopping normalmente para antes |
| `BATCH_SIZE` | Tamanho do lote para o gradiente. `256` é um bom equilíbrio entre velocidade e estabilidade |
| `PATIENCE` | Épocas sem melhora na `val_loss` antes de parar. `5` com `EPOCHS: 20` garante que o modelo para cedo quando converge |

---

## Threshold de anomalia

```json
"THRESH_MODE": "target_rate",
"TARGET_ANOMALY_RATE": 0.003
```

| Campo | O que faz |
|---|---|
| `THRESH_MODE` | Como calcular o limiar sobre o erro de reconstrução (MAE) do treino: `"p99"` = percentil 99; `"p99_5"` = percentil 99,5; `"max_train"` = máximo do treino; `"target_rate"` = ajusta o limiar para que exatamente a fração `TARGET_ANOMALY_RATE` dos dados de treino seja marcada como anômala |
| `TARGET_ANOMALY_RATE` | Taxa alvo. `0.003` = 0,3% das janelas de treino — usado apenas quando `THRESH_MODE: "target_rate"` |

---

## Regra de anomalia pontual

```json
"POINT_RULE": "k_of_window",
"POINT_WINDOW": 48,
"POINT_MIN_COUNT": 5
```

O pipeline detecta anomalias em **sequências** (janelas de `time_steps` amostras). Para converter isso em **pontos no tempo**, aplica uma votação deslizante.

| Campo | O que faz |
|---|---|
| `POINT_RULE` | `"k_of_window"` marca um ponto como anômalo se pelo menos `POINT_MIN_COUNT` das últimas `POINT_WINDOW` sequências forem anômalas; `"all_of_window"` exige que todas sejam |
| `POINT_WINDOW` | Tamanho da janela de votação em número de sequências |
| `POINT_MIN_COUNT` | Mínimo de votos para confirmar anomalia pontual. Com `POINT_WINDOW: 180` e `POINT_MIN_COUNT: 15`, exige que ~8% da janela seja anômala — reduz falsos positivos |

---

## Máscara operacional

```json
"OPERATIONAL_REF_SENSOR": "NGP_A",
"ENABLE_OPERATIONAL_MASK": true,
"OFF_VALUE_QUANTILE": 0.05,
"OFF_ABS_THRESHOLD": 5.0,
"OFF_LONG_MIN_HOURS": 4.0,
"TRANSIENT_PADDING_MINUTES": 30,
"TRANSIENT_DIFF_QUANTILE": 0.99
```

A máscara operacional usa um sensor de referência para identificar quando o equipamento está fora de operação, excluindo esses períodos do treino e das anomalias detectadas.

| Campo | O que faz |
|---|---|
| `OPERATIONAL_REF_SENSOR` | Sensor usado como indicador de estado operacional. `"NGP_A"` (potência/carga do compressor) ou `"RUNNING_A"` (sinal digital liga/desliga). Deve vir do mesmo arquivo `RAW_CSV` |
| `ENABLE_OPERATIONAL_MASK` | `true` ativa a máscara; `false` desativa (treina com todos os dados) |
| `OFF_ABS_THRESHOLD` | Valor absoluto abaixo do qual o equipamento é considerado desligado. **Tem precedência sobre `OFF_VALUE_QUANTILE`.** Para `NGP_A`, `5.0` separa "desligado" (≈0) de "operando" (≈87–90) com margem de segurança |
| `OFF_VALUE_QUANTILE` | Alternativa ao `OFF_ABS_THRESHOLD`: usa o percentil N do sensor como limiar. Ignorado se `OFF_ABS_THRESHOLD` não for `null` |
| `OFF_LONG_MIN_HOURS` | Período desligado com duração superior a este valor é marcado como `off_longo` (parada programada). Abaixo disso é `off_curto` (parada rápida) |
| `TRANSIENT_PADDING_MINUTES` | Janela em minutos ao redor de cada evento de liga/desliga marcada como `transiente`. `30` minutos protege o treino dos transitórios de partida do compressor |
| `TRANSIENT_DIFF_QUANTILE` | Variações bruscas acima deste percentil também são marcadas como `transiente`, independentemente de liga/desliga |

**Estados possíveis por ponto:**
- `on` → operação normal → incluído no treino e nas anomalias
- `off_longo` → parada longa → excluído do treino, anomalias mascaradas
- `off_curto` → parada breve → excluído do treino, anomalias mascaradas
- `transiente` → partida/parada → excluído do treino, anomalias mascaradas

---

## ClearML

```json
"OVERWRITE": true,
"RUN_REMOTE": true,
"REMOTE_QUEUE": "default",
"CLEARML_PROJECT_NAME": "TesteMLCab",
"CLEARML_DATASET_ID": "424e5b589e13402d9d95371a317e85c9",
"USE_CLEARML_DATASET": true
```

| Campo | O que faz |
|---|---|
| `OVERWRITE` | `true` re-treina mesmo que já exista um modelo salvo; `false` pula sensores já processados |
| `RUN_REMOTE` | `true` envia a task para o worker ClearML (execução remota/GPU); `false` roda localmente |
| `REMOTE_QUEUE` | Nome da fila ClearML onde a task será enfileirada. `"default"` é a fila padrão do agente |
| `CLEARML_PROJECT_NAME` | Nome do projeto no ClearML onde a task e os resultados serão registrados |
| `CLEARML_DATASET_ID` | ID do dataset ClearML com os CSVs de entrada. O ID `424e5b589e13402d9d95371a317e85c9` é a versão que inclui o arquivo antigo (`serie_consolidada_2025_interpolated_antigo.csv`) |
| `USE_CLEARML_DATASET` | `true` baixa os dados do ClearML; `false` usa os caminhos locais em `RAW_CSV`, `FEATURES_CSV` e `ALARM_CSV` |

---

## Reprodutibilidade

```json
"RANDOM_SEED": 42
```

Semente aleatória usada em todas as operações não determinísticas (divisão treino/validação, inicialização de pesos, KerasTuner). Manter o mesmo valor garante resultados reproduzíveis entre execuções.
