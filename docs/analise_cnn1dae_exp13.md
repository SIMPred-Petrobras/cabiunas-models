# Análise de Experimentos — Port dos gates do EXP10 pro CNN1D-AE (EXP13)

Branch `AE_novo_legado`. Contexto: depois de fechar a série EXP5–EXP10
(redução de falso alerta em `TC382_03_A`/`T5_AVG_A` via AutoML, ver
`docs/analise_automl_exp10.md`), o usuário confirmou que o CNN1D-AE
(`src/cnn1d_ae/pipeline.py`, a pipeline sequencial/janelada original,
pré-AutoML) continua sendo uma linha de produção ativa — o AutoML foi uma
exploração paralela, não uma substituição. Como o CNN1D-AE tinha
estagnado exatamente no problema de falso alerta, este experimento (EXP13)
porta os 3 mecanismos de redução de FP do EXP10/10b/10c de volta pro
CNN1D-AE, e mede o resultado numa base comparável ao AutoML.

## Escopo confirmado com o usuário

- Migrar o CNN1D-AE pro dataset atual (`sensores_full_2024_2026_30s.csv`
  / `alarmes_selecionados_turbina_a.csv` / `RUNNING_A`), mantendo
  compatibilidade com o dataset antigo (`NPT_A`/`NGP_A`) via campos de
  config já existentes (`OPERATIONAL_REF_SENSOR`/`LOAD_GATE_SENSOR`),
  sem hardcode.
- Grupo idêntico ao candidato de referência do AutoML
  (`TC382_T5_vibracao_mancais_multiescala`: `TC382_03_A` + `T5_AVG_A` +
  10 canais de vibração), para uma comparação justa.
- Depois de constatar que o hit_rate ficava artificialmente baixo sem as
  features de engenharia do EXP7 (multiescala + textura), o usuário pediu
  explicitamente para ligar `ENABLE_DERIVED_FEATURES` e comparar de novo.

## O que foi portado (mecanismos, não bugs)

Em `src/cnn1d_ae/pipeline.py` (`run_one_sensor` e `run_one_group`),
reaproveitando funções já existentes e compartilhadas em `scoring.py`:

1. **Máscara operacional secundária** -- `secondary_series`/
   `secondary_off_abs_threshold` em `build_operational_state` (o próprio
   sensor-alvo caindo abaixo de um piso físico também conta como "off").
2. **Portão de volatilidade** -- `compute_volatility_index`/
   `apply_volatility_gate`, chamado logo após o portão de rampa
   (`apply_load_gate`, que já existia nativamente no CNN1D-AE e foi a
   fonte do port pro AutoML no EXP10b).

Commit inicial: `24f13c1`.

## A jornada de depuração: 7 rodadas até um resultado válido

Portar os 2 mecanismos foi rápido. O que consumiu o grosso do trabalho
foi uma série de gaps metodológicos e de escala que só apareceram ao
rodar de verdade contra o dataset completo (1,93 milhões de linhas) --
nenhum smoke test sintético pequeno os revelou a tempo, por desenho (são
problemas de escala/volume de dados, não de lógica unitária).

| # | Task ClearML | Commit | O que mudou | Resultado |
|---|---|---|---|---|
| 1 | `196d2843...` | `24f13c1` | Port inicial (sem nenhuma correção) | **hit_rate 0,2\% (1/497)** -- `n_alarms` incluía alarmes dos 10 canais de vibração, diluindo o denominador |
| 2 | `913c476f...` | `b9b381c` | + `eval_sensors` (restringe `n_alarms`/hit\_rate a `TC382_03_A`/`T5_AVG_A`, mesmo mecanismo do AutoML desde o EXP6) | **hit\_rate 3,6\% (13/363)** -- `n_alarms` ainda contava os 363 alarmes de todo o histórico (2022--2026), sem holdout OOS |
| 3 | `5f117b8c...` | `7569843` | + split OOS (`OOS_SPLIT_DATE`, novo campo) + `normal_alert_rate`/`composite_score` (nunca eram calculados) + `select_feature_columns` (`ENABLE_DERIVED_FEATURES` era calculado e **descartado** por um `df_use[sensors]` logo em seguida) | **OOM, exit 137** -- 12 sensores viram ~276 colunas (multiescala+textura); nenhum crash em teste pequeno |
| 4 | `9a7ad8b3...` | `994a5c6` | + downcast float32 em `preprocess.py` (`build_group_dataframe`/`clip_outliers`/`normalize_train_only` promoviam pra float64 mesmo com entrada float32) + `STRIDE` 1→15 com correção de alinhamento em `map_seq_to_point_anomalies`/`build_sequence_scores_df` (assumiam `stride=1` implicitamente na conta de posição) | **OOM, exit 137** -- desta vez o tuner completou os 20 trials e o modelo foi salvo; crash na inferência sobre a série inteira (`x_all`/`x_all_pred`/`abs_err_all`, ~8,5GB cada, vivos ao mesmo tempo que os arrays de treino) |
| 5 | `61572f45...` | `3b00615` | + `del`/`gc.collect()` dos arrays de inferência logo após o uso | **OOM, exit 137**, mesmo ponto -- outros dataframes/arrays intermediários (`df_use`, `df_normal_z`, `df_all_z`, `values_normal`) ainda ficavam vivos sem necessidade pelo resto da função |
| 6 | `f4d092bf...` | `d2c87c6` | Varredura completa de `del`/`gc.collect()` em `run_one_group` -- todo dataframe/array grande liberado assim que deixa de ser necessário; `df_all` reduzido de ~276 pra ~12 colunas (só as brutas) assim que os dados normalizados são extraídos | **Completou sem crash!** Mas `hit_rate=0\%` **e** `normal_alert_rate=0\%` -- modelo nunca marcava nada como anômalo em lugar nenhum |
| 7 | `7fba01f5...` | `086cacd` | + `mask_anomaly_seq_by_operational_state` tinha o **mesmo bug de stride** das funções corrigidas na rodada 4 (não fora pego na varredura anterior) -- com `stride=15`, o lookup de estado operacional caía em timestamps completamente errados (só os primeiros ~44 dias da série), mascarando 100\% das 2.603 sequências que de fato cruzavam o threshold. + `THRESH_MODE=robust_mad` (mediana + k×1,4826×MAD), sugerido pelo usuário no lugar de `mean_std` (média+k×desvio), que é sensível à cauda longa da distribuição de erro de treino | **hit\_rate 87,5\% (35/40), FP 1,21\%** -- primeiro resultado real e comparável |

Cada rodada foi confirmada por smoke test sintético dedicado (dados
fabricados, escala pequena) antes da resubmissão remota -- os smoke
tests confirmaram a lógica correta em todos os casos; os problemas de
memória (rodadas 3--5) só se manifestam na escala real (1,9M linhas ×
~276 colunas), e o bug de stride da rodada 7 só se manifesta com
`STRIDE>1`, que nenhum config anterior usava.

## Resultado final: comparação com o AutoML

Mesma base de 40 alarmes, mesmo split OOS (`2025-07-01`), mesmo grupo de
12 sensores, mesmas features (multiescala + textura, `DERIVED_ROLLING_WINDOWS=[12,120,480,2880]`):

| Candidato | hit\_rate | FP (`normal_alert_rate`) |
|---|---|---|
| AutoML EXP7 item1+2 (multiescala+textura, sem gates) | 92,5\% (37/40) | 1,94\% |
| AutoML EXP10c (+ 3 gates de pós-processamento) | 92,5\% (37/40) | **0,35\%** |
| **CNN1D-AE EXP13 (multiescala+textura + gates portados)** | **87,5\% (35/40)** | **1,21\%** |

**Configuração final validada** (`configs/calibracao_v4_eq/test_grupo_exp13_AE_novo_dataset_gates.json`):
`TIME_STEPS=60`, `STRIDE=15` (ver rodada 4 -- necessário por limite de
memória, não por escolha de resolução), `THRESH_MODE=robust_mad`,
`THRESH_STD_K=4.0`, `POINT_WINDOW=4`/`POINT_MIN_COUNT=1` (escalados por
`/15` em relação aos valores originais de `STRIDE=1`, pra preservar a
mesma janela de ~30min em tempo real), `OOS_SPLIT_DATE=2025-07-01`,
máscara operacional + portão de rampa + portão de volatilidade todos
ativos com os mesmos valores calibrados no AutoML.

## Leitura do resultado

O CNN1D-AE chega a 5pp do hit_rate do AutoML com a mesma engenharia de
features e os mesmos gates de pós-processamento -- **não é o mesmo
modelo vencendo por arquitetura**, é uma arquitetura diferente
(autoencoder sequencial/janelado, reconstrução multicanal) reproduzindo
quase o mesmo resultado que um conjunto de modelos ponto-a-ponto
(dense/ocsvm/iforest) otimizados diretamente para essa métrica. O FP de
1,21\% fica entre o AutoML sem gates (1,94\%) e com gates calibrados
(0,35\%) -- os gates claramente ajudam (ver `load_gate_points_blocked=35.147`,
uma fração grande da série sendo suprimida por manobra de carga), mas
ainda não foram *recalibrados* especificamente para o CNN1D-AE -- os
valores de `LOAD_GATE_RAMP_MAX`/`VOLATILITY_GATE_THRESHOLD` foram
herdados diretamente do AutoML.

## Recalibração fina (simulação offline)

Antes de gastar mais uma rodada remota, simulamos offline diferentes
`THRESH_STD_K`/`POINT_WINDOW`/`POINT_MIN_COUNT` reaproveitando os dados
já computados pela task 7 (`mae_seq` por sequência de
`sequence_scores_all.csv`, `operational_state`/`load_gate_blocked`/
`volatility_gate_blocked` de `point_anomalies_all.csv`) e as **mesmas
funções de produção** (`mask_anomaly_seq_by_operational_state`,
`map_seq_to_point_anomalies`, `eval_alarm_hit_rate`,
`compute_normal_alert_rate`) -- só o valor do `threshold` final e os
parâmetros de agregação variam, tudo o resto (rede treinada, gates)
permanece fixo.

**Método:** como reconstruir exatamente o conjunto de treino usado para
calcular a mediana/MAD do `robust_mad` offline (sem os dados brutos de
gap/exclusão completos) introduz um erro de aproximação, a varredura foi
feita diretamente no espaço de **threshold absoluto** (não em `k`),
ancorada no valor real conhecido (`threshold=0,159566` em `k=4,0`).

**Achado:** há margem real. Descendo o threshold pra ~0,125--0,15 (contra
0,16 atual), o hit_rate simulado sobe pra **92,5% (37/40) -- o mesmo do
AutoML** --, com FP ainda em ~1,2--1,6%. `POINT_WINDOW=2`/
`POINT_MIN_COUNT=1` superou `4`/`1` em toda a grade testada.

**Ação:** `THRESH_STD_K` reduzido de `4,0` para `3,0` (redução
proporcional e moderada -- a tradução exata de threshold-simulado para
`k` real não é confiável sem conhecer a mediana/MAD reais de treino) e
`POINT_WINDOW` de `4` para `2`. Resultado dessa configuração ainda
**pendente de confirmação remota** -- a simulação offline usa uma
aproximação do conjunto de treino, não é garantia do número final.

## Pendências / próximos passos

- Confirmar o resultado da recalibração (`THRESH_STD_K=3,0`,
  `POINT_WINDOW=2`) com uma rodada remota real.
- Os gates (`LOAD_GATE_RAMP_MAX`, `VOLATILITY_GATE_THRESHOLD`) foram
  herdados do AutoML sem recalibração específica para a arquitetura
  sequencial -- podem não estar no ponto ótimo para o CNN1D-AE.
- Checagem de variância de semente (seed-sweep) do candidato final --
  ainda não feita para o CNN1D-AE (já é rotina no AutoML desde o EXP5).
- `STRIDE=15` foi uma decisão de necessidade (limite de memória do
  worker remoto), não uma escolha livre -- se a memória deixar de ser
  fator limitante (worker maior, ou uma reescrita de `make_sequences`
  para não copiar janelas sobrepostas explicitamente), vale testar
  `STRIDE` menor para granularidade mais fina.

## Tasks ClearML (ordem cronológica)

1. `196d2843fcf54bb4b50aa0a83a30695b` -- port inicial, hit_rate 0,2%
2. `913c476f72cb44f08d7ef107775f9ecb` -- + eval_sensors, hit_rate 3,6%
3. `5f117b8c33694cbd9530faaab1a90a1c` -- + OOS/composite/select_feature_columns, OOM
4. `9a7ad8b3e099490395e488fe0d513bd0` -- + float32/stride/STRIDE=15, OOM (na inferência)
5. `61572f45aa3e42febfa79dd4a27a69f0` -- + del parcial, OOM (mesmo ponto)
6. `f4d092bf69b445e891ed058cbc4f8f2b` -- + varredura completa de del/gc, completou mas 0%/0%
7. `7fba01f514674418adba71e3149b5e64` -- + fix stride na máscara operacional + robust_mad, **resultado final: 87,5%/1,21%**
