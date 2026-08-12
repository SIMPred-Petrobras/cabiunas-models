# Análise da pipeline AutoML da Lara (`transpetro_modelos`)

Documento de estudo antes de reproduzirmos (com correções) a abordagem de AutoML
da Lara dentro da nossa própria pipeline (`cnn1d_ae`). Objetivo: registrar o que
a pipeline dela faz, o que já validamos sobre os resultados dela no Cabiúnas, e
os pontos que achamos problemáticos — para não repetir os mesmos erros quando
formos construir a nossa versão.

Fontes:
- Notebook `automl_model_optimized.ipynb` (raiz de `~/REPO_CABIUNAS/`, gerado pela Lara)
- Repositório `github.com/SIMPred-Petrobras/transpetro-models`, branch `Lara`
  (clonado para leitura em `/tmp/.../scratchpad/transpetro-models`, não é
  dependência deste repo)
- Task ClearML `efe4884dc12c4a179765e3ffe2579a03` (projeto `Transpetro`,
  `automl_cabiunas_2025_2026_balanced`, criada por `lara@lara` em 12/08/2026)

## 1. O que existe na pipeline dela

### 1.1 Framework: `transpetro_modelos`

Pacote separado (não é este repo), usado para vários equipamentos Transpetro
além do Cabiúnas (`B-8801C`, `B-90001A`, `B-24001B`, `B-4064A`, `B-3403C`,
`B-5501B`). Cada equipamento tem um `EquipmentConfig` em
`src/transpetro_modelos/config.py`, com preprocessamento, features e datas de
falha próprios.

Config do Cabiúnas (`config.py:240-254`, id `cabiunas_2025_2026`):
- Filtro de "máquina ligada": `RUNNING_A == 1` + remoção de 10 min de
  transientes em cada liga/desliga (`pre_split_steps`)
- 35 features selecionadas (`select_features`), incluindo `TC382_03_A`,
  `T5_AVG_A` e vários sensores `954005_624_*` (pressão/temperatura) que a
  nossa pipeline ainda não usa
- Preset de preprocessamento `baseline_raw_no_common`
- **Sem `failure_date`/`failure_events` fixos no config** — os eventos são
  carregados dinamicamente (ver 1.3)

### 1.2 AutoML (`scripts/automl.py`)

Faz grid/random search sobre:
- **Modelo**: `dense` (autoencoder denso simples, PyTorch), `lstm`
  (autoencoder sequencial), `ocsvm` (One-Class SVM), `iforest` (Isolation
  Forest)
- **Preset** de preprocessamento: `baseline_raw_no_common`,
  `moving_average_raw_no_common`
- **`threshold_percentile`**: percentil do erro de reconstrução no treino
  (equivalente ao nosso `THRESH_MODE=p95/p97/...`)
- **`debounce_consecutive`**: ver 1.4

Rodou **249 trials** para o Cabiúnas. Cada trial é ranqueado por
`composite_score` (ver 1.5). O melhor trial (`best_trial`) foi:

| campo | valor |
|---|---|
| model | `dense` |
| preset | `baseline_raw_no_common` |
| threshold_percentile | 99.0 |
| debounce_consecutive | 6 |
| seq_len | 24 |
| dense_layers | 256,128 |
| prefailure_alert_rate | 23,1% |
| normal_alert_rate | 4,2% |
| composite_score | 0,696 |

### 1.3 Origem dos "eventos" de falha (`automl.py:85-106`)

```python
def load_failure_events_by_equipment(alarm_file, ..., equipment_col="Tag Alarme", date_col="data"):
    df = pd.read_csv(alarm_file)  # ou excel
    ...
    return {equip_id: sorted(group[date_col].tolist()) for equip_id, group in df.groupby(equipment_col)}
```

Os eventos usados no `compute_balanced_score_multi_failure` **não são
incidentes curados** — são todas as linhas de um CSV de alarmes, agrupadas por
tag. O melhor trial do Cabiúnas foi avaliado contra **368 eventos**, dos quais
314 tinham amostras na janela de pré-falha.

Essa função é definida mas **não é chamada em nenhum lugar do
`automl.py`/`transpetro_modelos` commitado no branch `Lara`** — ou foi chamada
de um notebook não commitado, ou de um script ad-hoc. Não temos o código exato
que gerou a lista de 368 eventos para o Cabiúnas especificamente.

### 1.4 Debounce (`training/evaluate.py:181-198`)

```python
def apply_debounce(scores, consecutive=1):
    rolling_count = scores["is_anomaly"].astype(int).rolling(consecutive, min_periods=consecutive).sum()
    return (rolling_count >= consecutive).fillna(False)
```

Isso exige que **todos** os últimos N pontos estejam acima do threshold — é o
equivalente ao nosso `POINT_RULE="all_of_window"` com `POINT_WINDOW=N`, **não**
ao `k_of_window` (que já é mais flexível que o dela, porque permite `k < N`).

### 1.5 `composite_score` (`training/evaluate_multi_failure.py:10-98`)

```python
balanced_score = prefailure_alert_rate - false_positive_penalty * (normal_alert_rate ** 2)   # penalty=2.0
if prefailure_alert_rate < min_prefailure_rate:                                              # default 0.5
    balanced_score -= (min_prefailure_rate - prefailure_alert_rate) * 2.0
composite_score = clip((balanced_score + false_positive_penalty) / (1.0 + false_positive_penalty), 0, 1)
```

`prefailure_alert_rate` é a **média (macro) da taxa de alerta por evento**, não
uma taxa micro (soma de alertas / soma de amostras) — cada evento pesa igual
independente de quantas amostras tem, inclusive eventos com 0 amostras entram
com taxa 0 na média.

`prefailure_days` (janela de crédito antes do evento) tem **default de 30
dias** na assinatura da função — ver seção 3.1 sobre por que isso importa.

## 2. O que já validamos sobre o resultado dela no Cabiúnas

Reconstruindo o `balanced_score` do melhor trial (0,0892) a partir de
`prefailure_alert_rate=0,2309` e `normal_alert_rate=0,0423`:

```
0,2309 - 2,0*(0,0423²) = 0,2274   (sem penalidade de min_prefailure_rate)
0,0892 medido → faltam 0,138 de penalidade
(min_prefailure_rate - 0,2309) * 2,0 = 0,138  →  min_prefailure_rate ≈ 0,30
```

Ou seja, o `min_prefailure_rate` efetivamente usado no run foi ~0,30 (não o
default de 0,5, mas também não zero). **O próprio melhor trial dos 249 fica
abaixo desse piso e é penalizado** — é o menos ruim do lote, não um resultado
que bateu a meta que ela mesma definiu.

Recalculando em nível de incidente (pelo menos 1 alerta em algum ponto da
janela pré-falha do evento, métrica mais parecida com o nosso `hit_rate`):

```
261 detectados / 314 eventos com amostra = 83,1%
```

Esse número (83,1%) é mais generoso que o `prefailure_alert_rate` (23,1%,
macro-média por evento) porque exige só 1 acerto na janela inteira, não uma
fração alta de cobertura.

**Variância de seed**: duas linhas do ranking com os *mesmos* hiperparâmetros
(`dense / baseline_raw_no_common / p99 / debounce 6`) deram resultados bem
diferentes:

| | prefailure_alert_rate | normal_alert_rate | composite_score |
|---|---|---|---|
| trial A | 23,1% | 4,2% | 0,696 |
| trial B (mesmos hparams) | 17,2% | 0,58% | 0,638 |

Isso é consistente com o que o Thallys já tinha documentado para o
TC382_03_A (`~±27pp` de ruído de semente) — reforça que **um único run não é
prova de nada**, precisa de múltiplas sementes pra confiar no número.

## 3. O que achamos que não está certo (pontos críticos)

### 3.1 Janela de crédito de 30 dias é muito generosa

`prefailure_days=30` credita como "detecção" qualquer alerta disparado até um
**mês inteiro** antes do evento. O nosso horizonte operacional (e o do
Thallys) é de **8 horas** — a diferença de escala é de ~90x. Os percentuais
dela (23% macro / 83% por-incidente) **não são comparáveis** aos nossos
(`hit_rate` do exp4 = 38,3%) nem aos do Thallys (recall 86,2% @ FA/dia 0,103)
sem reprocessar na mesma régua temporal.

### 3.2 Os "368 eventos" provavelmente não são incidentes reais e distintos

Vêm de agrupar **todas as linhas de um CSV de alarme por tag**
(`load_failure_events_by_equipment`), sem os filtros que o Thallys aplicou no
relatório de 10/08 (só HI/HIHI, onset→OK, máquina ligada, remove alarme
"fantasma" <500°C — ver commit `6465d42` no branch `backup/thallys`). É
provável que o conjunto inclua alarmes menores/repetidos que inflam a
contagem de "eventos" e tornam a métrica mais fácil de "acertar por
proximidade" dentro da janela de 30 dias.

**Não sabemos ao certo** — o código que gerou a lista de eventos para o
Cabiúnas especificamente não está commitado no branch `Lara`. Isso é uma
pergunta em aberto pra ela.

### 3.3 `normal_alert_rate` de 4,2% não é uma vitória de FP

O melhor trial ainda dispara falso alarme em 4,2% das amostras do período
"normal" — isso não é obviamente melhor que o nosso ponto de operação atual
(exp4: `anomaly_rate_points_per_day` calculado diretamente, threshold
`mean_std`). Precisaria converter pra a mesma unidade (alertas/dia) pra
comparar de verdade.

### 3.4 O ajuste fino do notebook (`y=2, k=15, n=20`, FP=0%) foi validado em
**um único evento**

As células 6-8 do notebook (`automl_model_optimized.ipynb`) fazem um grid
search de `y` (threshold = média + y·std) × debounce **só sobre o evento F1**
(17/01/2025 22:55) — não sobre os 368 eventos do AutoML original. O resultado
"FP=0,00%, pré-falha=28,1%" é sedutor mas foi otimizado e validado no mesmo
ponto de dados. É exatamente o padrão de overfitting que o Thallys evitou
deliberadamente no relatório dele (fixou o critério de promoção *antes* de
rodar o sweep, e teve pontos que pareciam ótimos mas reprovavam no OOS).

### 3.5 Não reproduzível neste ambiente hoje

O pacote `transpetro_modelos` não está instalado aqui e não há
`pyproject`/venv que o disponibilize neste repo. Clonamos o branch `Lara` só
pra leitura (`/tmp/.../scratchpad/transpetro-models`, descartável). Não existe
também, no branch dela, nenhum notebook commitado especificamente para
`cabiunas_2025_2026` (só para os outros equipamentos) — o notebook que temos
foi rodado no ambiente dela e só consome artifacts prontos do ClearML.

## 4. O que vale aproveitar

- **Modelo DENSE (autoencoder denso simples, não sequencial-conv)** como
  alternativa ao CNN-1D atual — vale testar na nossa pipeline com nossa
  própria avaliação (hit_rate sobre alarmes HI/HIHI, horizonte curto), não
  como comparação direta com os números dela.
- Mecânica do `composite_score` (penalizar FP quadraticamente + piso mínimo de
  recall) é uma ideia razoável de função-objetivo para um grid/AutoML search
  — poderíamos adaptar para escolher automaticamente entre nossos próprios
  `THRESH_MODE`/`POINT_RULE`/`POINT_WINDOW`/`POINT_MIN_COUNT` em vez de
  escolher manualmente como fizemos do exp1 ao exp5.
- Os 35 sensores extras do config dela (`954005_624_TI_*`, `PI_*`, `PDI_*`)
  são candidatos a entrar no nosso `SENSOR_GROUPS` — ainda não avaliamos se
  ajudam.

## 5. Próximos passos (quando formos implementar)

1. Perguntar pra Lara: (a) qual CSV/tag(s) geraram os 368 eventos, e se
   incluem alarmes menores; (b) se ela sabia que `min_prefailure_rate` não foi
   atingido pelo melhor trial.
2. Reproduzir a ideia do AutoML **dentro da nossa pipeline**: mesma busca de
   modelo/preset/threshold, mas com nossa avaliação já validada (`hit_rate`
   sobre alarmes HI/HIHI reais, horizonte de poucas horas, sem crédito de
   30 dias) — isso corrige os pontos 3.1-3.4 de uma vez.
3. Testar modelo DENSE como alternativa ao CNN-1D-AE atual, mesma régua de
   avaliação, para comparação justa.
4. Fazer isso em uma branch nova (não `feat_diego_mult`), já que é uma
   abordagem de busca de modelo diferente do que estamos fazendo agora
   (grupo T5 + load gate).
