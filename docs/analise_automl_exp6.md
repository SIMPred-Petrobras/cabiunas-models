# Análise de Experimentos — AutoML EXP6 (vibração de mancais)

Continuação de `docs/analise_automl_exp5.md`. O EXP5 terminou com um candidato
validado (`TC382_03_A_univariado` + `iforest`, 65% hit rate OOS, sem variância
de semente). O EXP6 muda de base de dados para explorar se sinais de vibração
melhoram a detecção/antecedência para `TC382_03_A` e `T5_AVG_A`.

## Nova fonte de dados

Dataset ClearML **`Cabiunas full 2024-2026 30s`** (`TesteMLCab`, ID
`a97ba56ba14840fbb1125c2a82f883c9`), diferente do usado no EXP1-EXP5:

- `sensores_full_2024_2026_30s.csv`: 2024-01-01 a 2026-04-30 (2,45M linhas,
  30s), 39 colunas de sensor, completude 99,5-100%.
- `alarmes_selecionados_turbina_a.csv`: 2022-01-04 a 2026-04-18, 6.851 linhas
  (pares onset/clear ACT+INACT), 47 tags, **sem duplicação** (diferente do
  arquivo usado no EXP1-5, que tinha cada linha 2x).

### O que ganhamos / perdemos em relação ao dataset antigo

**Ganhos:**
- Período 2,5x maior (10 meses -> 2,5 anos de sensores; alarmes desde 2022)
- **Vibração disponível**: `TV_35[1-5][X|Y]_A` = vibração X/Y de cada um dos
  5 mancais da turbina (confirmado pela descrição dos alarmes, ex:
  `TV_351X_A` = "TC_33003A - Vibração X Mancal 1"). Não existia no dataset
  antigo.
- Arquivo de alarmes sem duplicação e com eventos onset/clear distintos.

**Perdas:**
- `NGP_A` (nosso `OPERATIONAL_REF_SENSOR` de sempre) não existe neste
  arquivo -> trocado por `RUNNING_A`.
- Sensores que não migraram: `NPT_A`, `STD_FLOW_A`, `NCPSR_A`,
  `TM_TORQUE_A`, `BATERY_A`/`PLCBAT_LO_A`/`VDC24BC_AL_A`, `FI_0201`/`FI_0311`,
  `ZI_0301`/`ZI_0302`, diversos `TI_`/`PI_` extras.

## Correções feitas no código antes de rodar

1. **`io.py`**: normaliza `"Tag Alarme"` -> `"Tag"` e filtra o arquivo de
   alarmes para manter só linhas `Status` iniciando em `"ACT"` (onset) —
   o arquivo novo tem pares onset/clear que duplicariam o `n_alarms` se
   não filtrados.
2. **`io.py`**: novo `DATA_START_DATE`/`DATA_END_DATE` no config, pra
   recortar o início/fim da série usada por toda a pipeline (antes só
   existia corte de fim via `AUTOML_OOS_SPLIT_DATE`).
3. **`automl_pipeline.py`**: removida a trava que proibia
   `OPERATIONAL_REF_SENSOR="RUNNING_A"` — era específica da fonte de dados
   antiga (`RUNNING_A` pouco confiável lá). Confirmado com dado real que,
   após a limpeza padrão (`to_numeric` + interpolate/ffill/bfill) já
   existente em `build_sensor_dataframe`, `RUNNING_A` fica utilizável
   (só 1,49% de valores sujos tipo `{'Name': 'No Data', ...}`, resolvidos
   pelo preenchimento).
4. **`automl_pipeline.py`**: novo campo de grupo `eval_sensors` — permite
   sensores entrarem só como *feature* (vibração) sem que seus próprios
   alarmes contem no `hit_rate`/`composite_score`, que fica restrito a
   `["TC382_03_A", "T5_AVG_A"]`.
5. **`preprocess.py`**: `build_group_dataframe` agora aplica
   `ENABLE_DERIVED_FEATURES` (rolling median/std + delta por sensor) —
   antes só existia no caminho de sensor único.
6. **`automl_pipeline.py`**: bug corrigido onde `df_use = df_use[sensors]`
   descartava as colunas derivadas logo após criá-las.
7. **`automl_pipeline.py`**: `ocsvm` agora limita o treino a
   `AUTOML_OCSVM_MAX_TRAIN_SAMPLES` (50k, subamostra aleatória) — com o
   dataset maior, o fit em ~700k+ pontos seria impraticável (RBF-SVM escala
   ~O(n²)-O(n³)). Scoring continua sobre os dados inteiros.
8. **Threshold operacional**: `OFF_ABS_THRESHOLD` mudado de `5.0` (fazia
   sentido pra `NGP_A`, sensor de rotação contínuo) para `0.5` (limiar
   binário correto pra `RUNNING_A` ∈ {0,1}). Validado com dado real: sem
   essa troca, `is_off = s <= 5.0` marcaria **100% da série como "off"**
   silenciosamente, já que RUNNING_A nunca passa de 1.

## Config `test_grupo_exp6_vibracao.json`

| Parâmetro | Valor |
|---|---|
| Grupo | `TC382_T5_vibracao_mancais`: `TC382_03_A`, `T5_AVG_A` + 10 canais `TV_` (vibração, 5 mancais x X/Y) |
| `eval_sensors` | `["TC382_03_A", "T5_AVG_A"]` (só esses contam pra hit_rate) |
| Janela de fit | `DATA_START_DATE=2024-07-01` até `AUTOML_OOS_SPLIT_DATE=2025-07-01` (12 meses) |
| Janela de avaliação OOS | 2025-07-01 até o fim da série (2026-04-30, ~10 meses) |
| `OPERATIONAL_REF_SENSOR` | `RUNNING_A` (`OFF_ABS_THRESHOLD=0.5`) |
| `ENABLE_DERIVED_FEATURES` | `true` (rolling median/std + delta, janela 12 = 6min) |
| Grid | 7 percentis x 6 debounces x 3 modelos = 126 trials |
| `AUTOML_DENSE_LAYERS` | `[256, 128]`, dropout 0.1, 100 epochs, patience 15 |
| `AUTOML_IFOREST_N_ESTIMATORS` | 200 |
| `AUTOML_OCSVM_MAX_TRAIN_SAMPLES` | 50.000 |

### Validação local (sem treinar modelo) antes de submeter

- Alarmes onset de `TC382_03_A`+`T5_AVG_A`: **363 no total** (253 + 110),
  **152 na janela de fit**, **40 na janela OOS** — o dobro da amostra OOS
  que tínhamos no EXP5 (20).
- `RUNNING_A` pós-limpeza: 1,49% viraram NaN e foram preenchidos; estado
  operacional final = 68,3% on / 30,2% off_longo / 1,3% transiente / 0,2%
  off_curto — distribuição sã, compatível com o que víamos com `NGP_A`.

## Resultado

_(preencher após a run — task ClearML: TBD)_
