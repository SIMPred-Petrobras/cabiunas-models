# Pré-processamento do Dado de Entrada — CNN-1D Autoencoder

**Módulos envolvidos:** `src/cnn1d_ae/io.py`, `src/cnn1d_ae/preprocess.py`, `src/cnn1d_ae/sequences.py`  
**Orquestração:** `src/cnn1d_ae/pipeline.py → run_one_group()`

---

## Visão Geral do Fluxo

O pipeline transforma séries temporais brutas de sensores industriais em tensores 3-D prontos para o autoencoder. Todas as etapas seguem um princípio central: **as estatísticas de normalização são calculadas exclusivamente sobre dados normais (sem alarmes), mas a inferência é executada sobre toda a série**, incluindo períodos anômalos. Isso garante que o modelo aprenda apenas o comportamento esperado e que anomalias elevem o erro de reconstrução.

```
CSV bruto
   │
   ▼
[1] Carregamento e Normalização Temporal
   │  Parsing de timestamps, conversão de fuso, shift opcional
   │
   ▼
[2] Construção do DataFrame do Grupo
   │  Alinha múltiplos sensores no mesmo índice temporal
   │  Detecta e mascara gaps longos (> INTERPOLATE_LIMIT amostras)
   │  Interpola lacunas curtas
   │
   ▼
[3] Máscara de Exclusão de Alarmes
   │  Exclui janela de ±EXCLUDE_MINUTES_AROUND_ALARM em torno de cada alarme
   │  Union com máscara de gaps longos
   │
   ▼
[4] Clipping de Outliers
   │  Aplica sobre df_normal (dados sem alarme) e df_all (série completa)
   │  Modo quantile ou MAD — calcula limites apenas em df_normal
   │
   ▼
[5] Normalização train-only
   │  Calcula center/scale em df_normal → aplica em df_normal e df_all
   │  Modo zscore (µ, σ) ou robust (mediana, IQR)
   │
   ▼
[6] Criação de Sequências Deslizantes
   │  make_sequences(values, TIME_STEPS, STRIDE)
   │  Saída: (N_seq, TIME_STEPS, N_features) — float32
   │
   ▼
[7] Divisão Treino / Validação
   │  train_val_split com modo temporal (preserva ordem cronológica)
   │
   ▼
Modelo CNN-1D Autoencoder
   Input shape: (batch, TIME_STEPS, N_features)
```

---

## Fase 1 — Carregamento e Normalização Temporal

**Módulo:** `io.py → load_data()`, `_parse_datetime_smart()`, `_to_utc_indexed_series()`

### O que é feito
Os três arquivos CSV (alarmes, features, raw) são lidos e seus timestamps convertidos para UTC naive.

### Como é feito

```python
# 1. Detecta automaticamente o formato do timestamp
#    Razão da amostra com DD/MM/YYYY vs YYYY-MM-DD → escolhe dayfirst
parsed = _parse_datetime_smart(df[col])

# 2. Localiza no fuso de origem, converte para UTC
converted = out.dt.tz_localize(SOURCE_TZ, nonexistent="shift_forward", ambiguous="NaT")
converted = converted.dt.tz_convert(TARGET_TZ)

# 3. Shift adicional para datasets com offset de captura
if APPLY_HOUR_SHIFT:
    converted += pd.Timedelta(hours=SHIFT_HOURS)

# 4. Remove tzinfo para compatibilidade (UTC naive)
df[col] = converted.dt.tz_localize(None)
```

### Por que é feito

Os dados de campo da Cabiunas são gerados no fuso `America/Sao_Paulo` (UTC-3 no inverno, UTC-2 no verão), mas podem chegar sem anotação de fuso ou com offsets incorretos dependendo do sistema de aquisição. A conversão para UTC naive garante:

- **Consistência entre datasets:** alarmes e séries temporais sempre comparáveis no mesmo referencial.
- **Horário de verão:** `nonexistent="shift_forward"` evita `NaT` na transição DST.
- **Datasets `tzm3`:** o parâmetro `SHIFT_HOURS: -3` corrige uma captura histórica com offset adicional.

### Auditoria de integridade

Após o carregamento, o pipeline verifica e loga:
- Monotonicidade do índice temporal (`is_monotonic_increasing`)
- Quantidade de timestamps duplicados
- Cobertura temporal (início, fim, total de linhas)

---

## Fase 2 — Construção do DataFrame do Grupo

**Módulo:** `preprocess.py → build_group_dataframe()`

### O que é feito
Os N sensores do grupo são extraídos do CSV, alinhados no mesmo índice temporal e lacunas curtas são preenchidas por interpolação.

### Como é feito

```python
# Extrai apenas as colunas dos sensores do grupo
df_use = source_df[[TIME_COL] + list(sensors)].copy()

# Para cada sensor: detecta gaps longos ANTES de interpolar
for s in sensors:
    df_use[s] = pd.to_numeric(df_use[s], errors="coerce")
    lgm = _long_gap_mask(df_use[s], INTERPOLATE_LIMIT)
    long_gap_union = lgm | long_gap_union   # OR conservador

# Define índice temporal e interpola lacunas curtas
df_use = df_use.set_index(TIME_COL).sort_index()
for s in sensors:
    df_use[s] = df_use[s].interpolate(limit=INTERPOLATE_LIMIT, limit_direction="both")
    df_use[s] = df_use[s].ffill().bfill()
```

**Detecção de gaps longos:**
```python
def _long_gap_mask(series, interpolate_limit):
    missing = series.isna()
    # Identifica grupos consecutivos de NaN
    grp = missing.ne(missing.shift()).cumsum()
    run_len = missing.groupby(grp).transform("sum")
    # Gap longo = ausência contínua > INTERPOLATE_LIMIT amostras
    return missing & (run_len > interpolate_limit)
```

### Por que é feito

**Interpolação curta (`INTERPOLATE_LIMIT = 3`):** lacunas de 1-3 amostras (30s–1.5 min a 30s/amostra) são tipicamente falhas de comunicação, não ausência real de sinal. Interpolação linear preserva a continuidade sem distorcer o comportamento do sensor.

**Máscara de gaps longos (OR entre sensores):** uma sequência deslizante que cruzar um gap longo em qualquer canal terá valores artificialmente interpolados naquele canal, comprometendo o treino. A máscara union é conservadora por design: se o canal `TC382_03_A` esteve ausente por 2 horas, nenhuma sequência desse período entra no treino, mesmo que `T5_AVG_A` estivesse presente. Isso evita que o modelo aprenda padrões de preenchimento artificial.

**`ffill().bfill()` como fallback:** após `interpolate`, garante que não reste nenhum `NaN` residual nas bordas do dataset, onde `limit_direction="both"` não alcança.

---

## Fase 3 — Máscara de Exclusão de Alarmes

**Módulo:** `preprocess.py → build_exclusion_mask()`

### O que é feito
Uma máscara booleana é construída marcando como `True` todos os instantes dentro de uma janela de `±EXCLUDE_MINUTES_AROUND_ALARM` ao redor de cada alarme registrado. Esta máscara é unida à máscara de gaps longos.

### Como é feito

```python
def build_exclusion_mask(index, alarm_times, minutes):
    exclude = pd.Series(False, index=index)
    delta = pd.Timedelta(minutes=minutes)
    for t in alarm_times.values:
        t0 = pd.Timestamp(t) - delta
        t1 = pd.Timestamp(t) + delta
        exclude.loc[(index >= t0) & (index <= t1)] = True
    return exclude

# Pipeline:
exclude_alarm = build_exclusion_mask(index, alarm_times, EXCLUDE_MINUTES_AROUND_ALARM)
exclude = exclude_alarm | long_gap_mask     # union final

df_normal = df_use.loc[~exclude]    # dados para TREINO
df_all    = df_use.copy()           # dados para INFERÊNCIA (inclui tudo)
```

### Por que é feito

O autoencoder é um **modelo generativo de comportamento normal**: ele aprende a reconstruir fielmente o sinal quando opera dentro dos padrões esperados. Se períodos de falha entrarem no treino, o modelo aprende também a reconstruir comportamentos anômalos, reduzindo o erro de reconstrução nesses momentos e cegando o detector.

**Janela bilateral (`±EXCLUDE_MINUTES_AROUND_ALARM`):** o alarme é registrado *após* a detecção humana ou do sistema SCADA, que pode levar minutos ou horas após o início real da anomalia. A janela bidirecional captura tanto o precursor da falha quanto a persistência do evento. O valor padrão de 1440 min (24h) é conservador para alarmes do tipo vibração/temperatura em equipamentos rotativos, onde a degradação pode se arrastar por dias.

**`df_normal` vs `df_all` — separação fundamental:**

| DataFrame | Conteúdo | Uso |
|---|---|---|
| `df_normal` | Apenas dados sem alarme e sem gap longo | Calcular estatísticas (clipping, normalização), gerar sequências de treino |
| `df_all` | Série completa, inclui alarmes | Inferência — gerar scores MAE em toda a série para detectar anomalias |

Essa separação é a garantia de que o modelo nunca "vê" dados anômalos durante o aprendizado, mas o detector avalia 100% do histórico.

---

## Fase 4 — Clipping de Outliers

**Módulo:** `preprocess.py → clip_outliers()`

### O que é feito
Valores extremos são truncados em limites calculados a partir de `df_normal`.

### Como é feito

**Modo `quantile` (padrão):**
```python
q_low  = df_normal.quantile(OUTLIER_Q_LOW)    # ex: 0.001 → percentil 0.1%
q_high = df_normal.quantile(OUTLIER_Q_HIGH)   # ex: 0.999 → percentil 99.9%
df.clip(lower=q_low, upper=q_high, axis=1)
```

**Modo `mad` (Median Absolute Deviation):**
```python
med = df_normal.median()
mad = (df_normal - med).abs().median().replace(0, 1e-9)
low  = med - OUTLIER_MAD_K * 1.4826 * mad
high = med + OUTLIER_MAD_K * 1.4826 * mad
```
O fator `1.4826` torna o MAD consistente com o desvio padrão de uma distribuição normal.

### Por que é feito

Sensores industriais produzem picos espúrios por: saturação de sensor, falha de comunicação, reset de CLP. Esses picos inflam o `σ` da normalização z-score subsequente, comprimindo toda a distribuição normal para uma faixa estreita próxima de zero. O resultado seria sequências com variância artificialmente baixa, prejudicando a capacidade do modelo de discriminar comportamentos.

O clipping é aplicado tanto em `df_normal` (preservando o treino limpo) quanto em `df_all` com **os limites de `df_normal`**. Isso é intencional: picos anômalos reais em `df_all` também são truncados, o que pode parecer que esconde anomalias, mas na prática o modelo aprende a reconstruir a distribuição normal dos valores — mesmo que o pico seja truncado, a persistência temporal do comportamento desviante ainda eleva o MAE.

---

## Fase 5 — Normalização Train-Only

**Módulo:** `preprocess.py → normalize_train_only()`

### O que é feito
Os dados são centrados e escalados usando estatísticas calculadas exclusivamente em `df_normal`. Os mesmos parâmetros são aplicados em `df_all`.

### Como é feito

**Modo `zscore` (padrão):**
```python
center = df_normal.mean(axis=0)               # µ por coluna
scale  = df_normal.std(axis=0).replace(0, 1.0) # σ por coluna (protege divisão por zero)

df_normal_z = (df_normal - center) / scale
df_all_z    = (df_all    - center) / scale
```

**Modo `robust`:**
```python
center = df_normal.median(axis=0)             # mediana
scale  = (Q75 - Q25).replace(0, 1.0)         # IQR
```

### Por que é feito

**Por que normalizar?**  
O CNN-1D autoencoder usa ativações `relu` e otimizador Adam com taxa de aprendizado fixa. Sem normalização, canais com magnitudes muito diferentes (ex: temperatura 600°C vs pressão 0.5 bar) dominam a função de perda MSE e o gradiente. A normalização per-feature coloca todos os canais na mesma escala de contribuição.

**Por que apenas com dados normais?**  
Se `df_all` fosse incluído no cálculo de `µ` e `σ`, picos anômalos elevariam a média e o desvio, fazendo com que o modelo "esperasse" eventos anômalos como parte da distribuição normal. Ao usar apenas `df_normal`, o modelo aprende que valores normalizados próximos de zero são o padrão — qualquer desvio substancial no dado real se traduz em valor normalizado alto (em módulo), elevando o erro de reconstrução.

**Proteção `replace(0, 1.0)`:** sensores com sinal constante (std=0) teriam divisão por zero. O replace para 1.0 preserva o valor original centrado, sem amplificar nem suprimir — matematicamente correto, pois um sinal constante não carrega informação variacional.

**`zscore` vs `robust`:**
- `zscore`: adequado quando a distribuição de operação normal é aproximadamente gaussiana.
- `robust`: preferível quando há assimetria ou quando outliers são frequentes mesmo após clipping — o IQR é insensível a extremos.

---

## Fase 6 — Criação de Sequências Deslizantes

**Módulo:** `sequences.py → make_sequences()`

### O que é feito
A série temporal 2D `(N_pontos, N_features)` é transformada em um tensor 3D de janelas deslizantes `(N_sequências, TIME_STEPS, N_features)`.

### Como é feito

```python
def make_sequences(values_2d, time_steps, stride):
    n = len(values_2d)
    out = []
    for i in range(0, n - time_steps + 1, stride):
        out.append(values_2d[i : i + time_steps])   # fatia (TIME_STEPS, N_features)
    return np.stack(out, axis=0).astype(np.float32)
```

**Exemplo com `TIME_STEPS=180`, `STRIDE=1`, `N_features=2`:**
```
Ponto t=0:   sequência [0..179]   → janela 0
Ponto t=1:   sequência [1..180]   → janela 1
...
Ponto t=N-180: sequência [N-180..N-1] → janela N-180

Total de sequências: N - TIME_STEPS + 1
Tensor de saída: (N-TIME_STEPS+1, 180, 2)
```

### Por que é feito

O autoencoder é treinado para **reconstruir janelas temporais completas**, não pontos isolados. Isso captura dependências temporais: padrões de rampa, oscilação, comportamento periódico. Uma anomalia que se manifesta como alteração de padrão (não necessariamente de amplitude) só é detectável em contexto de janela.

**`TIME_STEPS = 180` (30 minutos a 10s/amostra):**  
Janela suficiente para capturar:
- Ciclos térmicos típicos de equipamentos industriais (aquecimento/resfriamento)
- Oscilações de processo com período de minutos
- Transientes de partida que não devem ser confundidos com anomalias

**`STRIDE = 1`:**  
Stride unitário maximiza o número de amostras de treino e garante resolução temporal de 10s na detecção. Strides maiores reduzem o conjunto de treino e a resolução temporal de scoring.

**`float32`:** TensorFlow opera nativamente em float32 na GPU. Converter neste ponto evita casts implícitos no grafo computacional, reduzindo consumo de memória e latência.

---

## Fase 7 — Divisão Treino / Validação

**Módulo:** `sequences.py → train_val_split()`

### O que é feito
O tensor de sequências normais é dividido em treino e validação.

### Como é feito

**Modo `temporal` (padrão e recomendado):**
```python
n_val   = int(floor(VAL_FRAC * N_total))    # ex: 10% final
n_train = N_total - n_val

x_train = x[:n_train]    # primeiras N_train sequências (cronologicamente)
x_val   = x[n_train:]    # últimas N_val sequências
```

### Por que é feito

**Por que não shuffle?**  
Sequências consecutivas compartilham `TIME_STEPS - 1` pontos (com STRIDE=1). Embaralhar aleatoriamente criaria data leakage: sequências de validação teriam pontos em comum com sequências de treino. O split temporal respeita a independência entre conjuntos.

**Por que 10% no final (`VAL_FRAC=0.1`)?**  
O modelo precisa generalizar para dados futuros. A validação no final cronológico testa exatamente isso — o modelo nunca "olha para frente" durante o treino. Além disso, deriva de dados do período mais recente, potencialmente mais relevante para calibrar o threshold.

---

## Interface com o Modelo — Formato do Tensor

### Entrada do autoencoder

```python
# Gerado por make_sequences sobre df_normal_z (treino)
x_train: np.ndarray  shape=(N_train, TIME_STEPS, N_features)  dtype=float32
x_val:   np.ndarray  shape=(N_val,   TIME_STEPS, N_features)  dtype=float32

# Gerado por make_sequences sobre df_all_z (inferência)
x_all:   np.ndarray  shape=(N_all,   TIME_STEPS, N_features)  dtype=float32
```

### Chamada do modelo

```python
# Treino (KerasTuner search)
tuner.search(x_train, x_train,                 # input = target (autoencoder)
             validation_data=(x_val, x_val),
             epochs=EPOCHS, batch_size=BATCH_SIZE)

# Inferência — toda a série
x_all_pred = model.predict(x_all, batch_size=BATCH_SIZE, verbose=0)

# Erro de reconstrução por sequência e por canal
abs_err_all = np.abs(x_all_pred - x_all)      # (N_all, TIME_STEPS, N_features)
mae_seq_all = np.mean(abs_err_all, axis=(1,2)) # (N_all,) — MAE global
mae_per_ch  = np.mean(abs_err_all, axis=1)    # (N_all, N_features) — MAE por canal
```

### Por que `x_train == target` no autoencoder?

O objetivo do autoencoder é minimizar o erro de reconstrução do próprio input. A função de perda é:

```
L = MSE(x_pred, x)  =  (1/T·F) Σ_t Σ_f (x_pred[t,f] - x[t,f])²
```

Quando o modelo é exposto a um padrão que nunca viu durante o treino (anomalia), ele não consegue reconstruí-lo fielmente, resultando em MAE elevado. O threshold `τ` é definido no `TARGET_ANOMALY_RATE`-percentil da distribuição de MAE de treino — qualquer sequência com MAE > τ é sinalizada como anômala.

---

## Parâmetros de Configuração Relevantes

| Parâmetro | Valor típico | Fase | Efeito |
|---|---|---|---|
| `SOURCE_TZ` | `"UTC"` | 1 | Fuso de origem dos timestamps |
| `APPLY_HOUR_SHIFT` | `false` | 1 | Ativa shift manual de horas |
| `SHIFT_HOURS` | `-3` | 1 | Horas de offset adicional |
| `INTERPOLATE_LIMIT` | `3` | 2 | Máx. amostras consecutivas a interpolar |
| `EXCLUDE_MINUTES_AROUND_ALARM` | `1440` | 3 | Janela bilateral de exclusão (min) |
| `EXCLUDE_LONG_GAPS_FROM_TRAIN` | `true` | 3 | Inclui gaps longos na máscara |
| `OUTLIER_MODE` | `"quantile"` | 4 | Método de clipping |
| `OUTLIER_Q_LOW / Q_HIGH` | `0.001 / 0.999` | 4 | Percentis de clipping |
| `NORMALIZE_MODE` | `"zscore"` | 5 | Método de normalização |
| `TIME_STEPS` | `180` | 6 | Comprimento da janela temporal |
| `STRIDE` | `1` | 6 | Passo entre janelas consecutivas |
| `VAL_FRAC` | `0.1` | 7 | Fração de validação |
| `SPLIT_MODE` | `"temporal"` | 7 | Estratégia de split |
| `BATCH_SIZE` | `256` | Modelo | Amostras por step de gradiente |

---

## Considerações de Design e Trade-offs

### Conservadorismo da Máscara de Exclusão
A union OR entre alarmes e gaps (vs intersection AND) é uma escolha deliberadamente conservadora. Em cenários industriais, é preferível ter **menos dados de treino com alta confiabilidade** do que mais dados com possível contaminação anômala. A consequência é que sensores com alta taxa de alarmes podem ter pouco material de treino — mitigado pelo `MIN_STD` check que descarta sensores com sinal constante.

### Normalização vs Estacionariedade
A normalização z-score assume implicitamente que a série de operação normal é estacionária (µ e σ constantes no tempo). Em sensores industriais sujeitos a sazonalidade ou degradação lenta, µ e σ derivam ao longo do tempo. O modo `robust` (mediana/IQR) é menos sensível a isso, mas não elimina o problema. Para séries fortemente não-estacionárias, considerar normalização por janela deslizante ou re-calibração periódica do modelo.

### Sequências Sobrepostas e Volume de Treino
Com `STRIDE=1` e `TIME_STEPS=180`, cada ponto da série de treino aparece em até 180 sequências diferentes. Isso cria forte correlação entre sequências adjacentes, que o KerasTuner e o EarlyStopping tratam adequadamente via `val_loss` no conjunto temporal separado. O benefício é maximizar o número de exemplos de treino, crítico para modelos profundos com muitos parâmetros.
