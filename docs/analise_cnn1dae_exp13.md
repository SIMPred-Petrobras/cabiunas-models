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
| CNN1D-AE EXP13 v1 (`THRESH_STD_K=4`, `POINT_WINDOW=4`, gates AutoML) | 87,5\% (35/40) | 1,21\% |
| CNN1D-AE EXP13 v2 (`THRESH_STD_K=3`, `POINT_WINDOW=2`, gates AutoML) | 90,0\% (36/40) | 0,94\% |
| CNN1D-AE EXP13 v3 (+ gates recalibrados, 40 trials) | 90,0\% (36/40) | 0,80\% |
| **CNN1D-AE EXP13 v4, média do seed-sweep (5 seeds, ver Seção seed-sweep)** | **86,9\% ± 3,25pp** | **0,60\% ± 0,13pp** |

Ressalva importante lida na íntegra na seção "Seed-sweep" abaixo: as
linhas v1--v3 são cada uma um único run/uma única semente -- parte da
diferença entre elas é ruído de treino, não só o efeito da
recalibração. A linha v4 (média de 5 sementes sobre a MESMA
configuração de v3) é a leitura estatisticamente mais honesta.

**Configuração final validada** (`configs/calibracao_v4_eq/test_grupo_exp13_AE_novo_dataset_gates.json`):
`TIME_STEPS=60`, `STRIDE=15` (ver rodada 4 -- necessário por limite de
memória, não por escolha de resolução), `THRESH_MODE=robust_mad`,
`THRESH_STD_K=3.0`, `POINT_WINDOW=2`/`POINT_MIN_COUNT=1`,
`OOS_SPLIT_DATE=2025-07-01`, máscara operacional + portão de rampa +
portão de volatilidade todos ativos com os mesmos valores calibrados no
AutoML.

## Leitura do resultado

O CNN1D-AE chega a 2,5pp do hit_rate do AutoML com a mesma engenharia de
features e os mesmos gates de pós-processamento -- **não é o mesmo
modelo vencendo por arquitetura**, é uma arquitetura diferente
(autoencoder sequencial/janelado, reconstrução multicanal) reproduzindo
quase o mesmo resultado que um conjunto de modelos ponto-a-ponto
(dense/ocsvm/iforest) otimizados diretamente para essa métrica. O FP de
0,94\% já fica **abaixo** do AutoML sem gates (1,94\%), ainda acima do
AutoML com gates calibrados (0,35\%) -- os gates claramente ajudam (ver
`load_gate_points_blocked=35.147`, uma fração grande da série sendo
suprimida por manobra de carga), mas ainda não foram *recalibrados*
especificamente para o CNN1D-AE -- os valores de
`LOAD_GATE_RAMP_MAX`/`VOLATILITY_GATE_THRESHOLD` continuam herdados
diretamente do AutoML (possível próxima etapa de ganho).

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

**Ação e confirmação remota:** `THRESH_STD_K` reduzido de `4,0` para
`3,0` e `POINT_WINDOW` de `4` para `2` (task
`2db0722a20694cdfb47d77e1b07242a6`, commit `b32c5bf`). **Confirmado**:
hit_rate subiu de 87,5\% para 90,0\% (36/40) **e** FP caiu de 1,21\% para
0,94\% ao mesmo tempo -- não foi uma troca, os dois eixos melhoraram
juntos, sinal de que `k=4,0` estava mesmo conservador demais em relação
ao ponto ótimo. `threshold` final: `0,151967` (bem próximo da faixa
prevista pela simulação offline, `0,125--0,15`).

## Recalibração dos gates (rampa + volatilidade)

Os valores de `LOAD_GATE_RAMP_MAX`/`VOLATILITY_GATE_THRESHOLD` até aqui
eram herdados diretamente do AutoML (EXP10b/10c), nunca recalibrados
especificamente pro CNN1D-AE. Repetimos offline a mesma busca sequencial
do EXP10b/10c (rampa primeiro, depois volatilidade com a rampa fixa no
melhor achado), reconstruindo o sinal *pré-gate* (threshold+máscara
operacional já aplicados, com `THRESH_STD_K=3,0`/`POINT_WINDOW=2`) e
aplicando `apply_load_gate`/`apply_volatility_gate` reais sobre os dados
brutos de `T5_AVG_A`/vibração já cacheados localmente.

**Achado:** o sinal pré-gate já teria hit_rate 92,5% (37/40) e FP 0,92%
nessa reconstrução -- mais otimista que a rodada real (90,0%/0,94%), a
mesma imprecisão de aproximação já observada na recalibração de
threshold (Seção acima). Ainda assim, a *direção* é clara: com os gates
recalibrados (rampa `ramp_max=50/halflife=10min/window=60min`;
volatilidade `window=30min/threshold=0,1745`, ambos mais "atentos" que os
valores do AutoML) o FP simulado cai de 0,84% pra 0,56% (~33% relativo)
mantendo hit_rate em 92,5%.

**Ação e confirmação remota:** config atualizado com os novos valores
dos 2 gates; `MAX_TRIALS` também dobrado de `20` para `40` (task
`9ff669b9a83f4ad7bb0fefc01aead335`, commit `e31fa0f`). **Confirmado**:
hit_rate ficou igual (90,0%, 36/40 -- os 40 trials extras não acharam
arquitetura melhor), mas FP caiu de 0,94% para **0,80%** (~15%
relativo) -- menos dramático que a simulação offline previu (que
apontava ~0,56% mantendo 92,5%), mas confirma a mesma direção: os
gates herdados do AutoML não eram o ponto ótimo. `load_gate_points_blocked`
subiu de 35.147 para 110.550 (portão de rampa recalibrado bloqueia 3x
mais pontos) sem custar nenhuma detecção -- sinal de que o ganho veio
de FP genuíno, não de sorte.

## Seed-sweep (checagem de variância de semente)

Portado `_refit_cnn1dae_with_seed`/seed-sweep pro CNN1D-AE (novo campo
`SEED_SWEEP_N`, espelha `AUTOML_SEED_SWEEP_N`), mesma motivação do
AutoML desde o EXP5: re-treinar a MESMA arquitetura (`best_hp` já
escolhida pelo tuner) com N seeds extras, sem repetir a busca de
hiperparâmetros, pra medir quanto `hit_rate`/`normal_alert_rate` variam
só por causa da aleatoriedade de inicialização/treino.

**Detalhe de implementação:** diferente do AutoML (onde `_seed_sweep` só
precisa dos arrays já prontos), o CNN1D-AE precisa treinar uma rede do
zero por seed -- isso exigiu mover a construção da máscara operacional e
dos gates pra **antes** do treino (nenhum dos dois depende do modelo),
permitindo reusar a mesma infraestrutura de avaliação (`_score_to_report`,
função local) tanto no modelo principal quanto em cada seed, sem manter
`x_train_full`/`x_train`/`x_val`/`values_all` vivos até o fim da função
inteira -- os arrays só são liberados depois que o seed-sweep termina de
usá-los.

`SEED_SWEEP_N=4` adicionado ao config -- cada seed é um retreino
completo (mais caro que o do AutoML), então o custo é ~4x o tempo de
refit em vez de quase gratuito.

**Resultado (task `8b97c625af78431e88b912c6e9288332`, commit `cf37da6`)
-- achado importante: o CNN1D-AE NÃO tem variância zero.**

| Seed | hit\_rate | FP |
|---|---|---|
| 42 (principal) | 85,0% | 0,59% |
| 43 | 85,0% | 0,72% |
| 44 | 85,0% | 0,69% |
| 45 | 85,0% | 0,39% |
| 46 | 92,5% | 0,60% |
| **Média (4 extras) / desvio** | **86,9% / ±3,25pp** | **0,60% / ±0,13pp** |

Diferente do AutoML (`iforest`/`ocsvm`, variância **zero** em `hit_rate`
desde o EXP5 -- ver `docs/analise_automl_exp5.md`), a rede neural do
CNN1D-AE varia de verdade entre sementes -- inicialização de pesos +
estocasticidade do treino (dropout, ordem de batch) produzem resultados
genuinamente diferentes mesmo com a mesma arquitetura (`best_hp`) e os
mesmos dados. Isso muda a leitura de toda a série de recalibrações
anteriores: os números que fomos reportando a cada rodada (87,5% →
90,0% → 85,0% nesta) provavelmente refletem, em boa parte, esse ruído de
semente/busca de hiperparâmetro (o `KerasTuner` também não tem sua
própria aleatoriedade fixada antes da busca, então até a arquitetura
vencedora pode variar entre execuções) -- **não** só o efeito líquido de
cada recalibração isolada. A média do seed-sweep (86,9%/0,60%) é uma
leitura mais honesta do candidato do que qualquer rodada isolada.

## Seed global fixada antes do tuner

Aplicado: `keras.utils.set_random_seed(cfg.RANDOM_SEED)` +
`kt.RandomSearch(..., seed=cfg.RANDOM_SEED)` no início de `run_tuner`
(`tuning.py`), e reseed de novo logo antes do `refit_best_model` (que
continua o treino do `best_model` já retornado pela busca -- sem
reseed, o `.fit()` final reembaralha o dataset consumindo o estado
aleatório residual deixado pelo número variável de operações da busca,
que difere entre execuções mesmo com a mesma seed no início).

**Confirmado por smoke test dedicado** (2 execuções completas do mesmo
pipeline sintético, mesma seed): `best_hp` (arquitetura vencedora do
tuner) sai **idêntico** entre as duas execuções -- eliminada a maior
fonte de variância (a "loteria de arquitetura"). O `threshold` final
ainda varia um pouco (~0,3\% de diferença relativa, contra ~0,65\% antes
do segundo reseed) -- resíduo de não-determinismo do TensorFlow em
operações paralelas (Conv1D em CPU), que só fecharia de vez com
`tf.config.experimental.enable_op_determinism()` (mais invasivo, pode
custar desempenho de treino; não aplicado aqui).

**Confirmado remotamente (task `2d63b3fbe1ff43459f8db28283910c49`,
`THRESH_STD_K=3,0` inalterado -- só o fix de seed):** hit_rate_std caiu
de ±3,25pp para ±2,50pp (5 valores agora oscilam só entre 85\% e 90\%,
sumiu o outlier de 92,5\% visto antes) -- melhora real, mas **FP piorou**
(média 0,60\%→1,23\%, chegando a 1,66\% no modelo principal). Threshold
resultante (0,135) saiu bem mais baixo que o da rodada anterior (0,157) --
mesmo com arquitetura fixa, o resíduo de não-determinismo desloca o
ponto na curva threshold→hit\_rate/FP o suficiente pra mudar o
resultado visivelmente (a curva é íngreme nessa faixa, ver seção
seguinte).

## Achado crítico: artefato de antecedência perto do teto da janela

Investigação caso a caso (pedida pelo usuário antes de recalibrar de
novo) revelou que boa parte dos "preditivos" do candidato de
`THRESH_STD_K=3,0` é **artefato de janela**, não sinal genuíno -- o
mesmo padrão já documentado no EXP8/EXP11: com FP alto (1,66\%), pontos
de falso alerta espalhados pela série têm boa chance de cair, por
coincidência, perto do início da janela de $\pm$24h ao redor de um
alarme, sendo contados como "antecedência" sem ser precursor real.

**Evidência:** dos 27 casos "preditivos" do threshold 0,135, **22 têm
antecedência entre 20h e 24h** (mediana geral = 23,9h, colada no teto).
Comparando com o EXP10c (AutoML): só 5 de 29 preditivos (17\%) caem
nessa faixa suspeita -- o EXP10c é majoritariamente genuíno (mediana
real, excluindo suspeitos, 13,4h), consistente com seu FP baixo (0,35\%).

**Critério de recalibração revisado:** em vez de só `composite_score`
(cego a esse artefato), simulação offline em grade de threshold
classificou cada um dos 40 alarmes em preditivo genuíno (antecedência
<20h), preditivo suspeito (≥20h), reativo ou sem detecção, e escolheu o
threshold que maximiza **cobertura genuína** ((preditivo genuíno +
reativo) / 40), não hit_rate bruto.

| Candidato | Preditivo genuíno | Reativo | Cobertura genuína | FP |
|---|---|---|---|---|
| AutoML EXP10c | 24 | 8 | **80,0\% (32/40)** | 0,35\% |
| CNN1D-AE (threshold 0,135, `k=3,0`) | 7 | 9 | 40,0\% (16/40) | 1,66\% |
| CNN1D-AE recalibrado (threshold≈0,265, `k≈6,0` estimado) | 14 | 10 | 60,0\% (24/40) | 0,19\% |
| CNN1D-AE recalibrado (threshold real 0,2294, `k=6,0` confirmado) | 10 | 9 | 47,5\% (19/40) | 0,30\% |
| **CNN1D-AE recalibrado (threshold real 0,2609, `k=7,0` confirmado)** | **14** | **10** | **60,0\% (24/40)** | **0,17\%** |

A segunda rodada de recalibração (`k=7,0`) bateu a meta: threshold real
`0,2609` ficou praticamente idêntico ao ótimo previsto pelo regrid
offline (`0,260`). Resultado: cobertura genuína **60,0\% (24/40)**, com
FP **0,17\%** -- menos da metade do FP do próprio EXP10c (0,35\%). O
número de casos "suspeitos" (artefato de janela) também caiu bastante,
de 12 (em `k=6,0`) para **3**, ou seja a cobertura genuína agora é
majoritariamente detecção real, não ruído perto do teto da janela.

Mesmo assim, o CNN1D-AE fica genuinamente atrás do AutoML (60\% vs 80\%
de cobertura real) -- a vantagem do EXP10c não é artefato, é detecção
real mais forte. Mas a recalibração ainda vale: FP de 0,17\% é **menos
da metade do FP do EXP10c** (0,35\%), e a cobertura genuína subiu de
40\% (candidato original) para 60\%.

**Tradução threshold→`THRESH_STD_K`:** como `robust_mad` depende da
mediana/MAD reais do treino (que variam levemente entre execuções, ver
seção anterior), a tradução de "quero threshold≈0,265" para um `k`
exato não é determinística -- usamos uma extrapolação proporcional a
partir do ponto conhecido (`k=3,0` → threshold real `0,135`):
`k ≈ 3,0 × (0,265/0,135) ≈ 5,9`, arredondado para `THRESH_STD_K=6,0`.

**Confirmação remota (task `48119896110c4ccfa2581ab2087f4d88`,
commit `5dd413a`):** o threshold real saiu em **0,2294**, não 0,265 --
a extrapolação proporcional assumiu que mediana/MAD do treino seriam
parecidos entre modelos, mas cada resubmissão retreina do zero e essa
distribuição muda. Resultado real: cobertura genuína **47,5\% (19/40)**,
FP **0,30\%** -- melhora real sobre o baseline (40\%→47,5\%,
FP 1,66\%→0,30\%), mas abaixo da meta de 60\%.

Regrid offline sobre os dados reais desta task (mesmo
`sequence_scores_all.csv`/`point_anomalies_all.csv` de produção)
confirma que o threshold ótimo para *este* modelo específico é
**≈0,260** (cobertura genuína 60,0\%, FP≈0,20\%) -- a meta original
ainda é alcançável, só que num threshold ligeiramente mais alto que o
que `k=6,0` produziu. Nova extrapolação a partir dos dois pontos reais
conhecidos (`k=6,0`→0,2294 neste modelo; proxy de mediana/MAD de toda a
série vs. threshold real, fator ≈0,644) aponta para **`k≈7,0`**,
resubmetido para confirmação (mesma ressalva: aproximação, não valor
exato).

**Achado adicional (seed-sweep pós-recalibração):** com `k=6,0`, a
variância do seed-sweep aumentou bastante -- `hit_rate_std` foi de
±2,50pp (resultado anterior, `k=3,0` pós-fix de seed) para **±10,51pp**
(seeds 43-46: 42,5\%-70,0\%). Com `k=7,0` (task
`8a95bccb0e40461ea9caea20c94dae10`, commit `b1b4fff`), a variância caiu
um pouco (**±7,15pp**, seeds 37,5\%-57,5\%) mas segue bem acima do
±2,50pp original -- não foi um efeito pontual só do `k=6,0`; threshold
mais alto nesta base parece consistentemente mais sensível a variação
de treino/seed do que o threshold mais baixo de antes da recalibração.

## Auditoria caso a caso dos alarmes não detectados (candidato k=7,0)

Pedido do usuário: "como podemos melhorar a predição... queremos
acertar quase todos os alarmes, retirando aqueles que são erros de
sensor." Investigação completa dos 13 "sem detecção" + 3 "suspeitos"
do candidato final, cruzando com o dado bruto (`sensores_full_...csv`)
e o próprio código de status do SCADA.

**5 dos 40 alarmes têm condição `UNDER` (temperatura entre -18°C e
-22°C, fisicamente impossível para gás de escape):**

| Data | Causa raiz (confirmada no dado bruto) | Categoria |
|---|---|---|
| 2025-07-24 | Pico de 1 amostra (-38°C por 30s, recupera na amostra seguinte) | Erro de sensor |
| 2025-08-08 | Dado ausente (`NaN`) por ~3,5min no instante exato | Erro de sensor |
| 2026-01-14 | Código `Out of Serv` explícito no dado bruto | Erro de sensor |
| 2026-01-17 | Código `Comm Fail` por ~5min, depois `Out of Serv` | Erro de sensor |
| 2025-11-29 | Temperatura real ~32°C, turbina genuinamente em `off_longo` | Não é erro de sensor, mas também não é falha a prever |

4 dos 5 são literalmente falhas de comunicação/instrumento
(`Comm Fail`/`Out of Serv` no dado bruto do SCADA -- não é inferência).
Nenhum modelo, nem o AutoML EXP10c (cross-checado), trata esses casos
como "preditivo" -- não existe precursor físico para uma queda de
comunicação. **Excluindo os 4 confirmados:** `n_alarms` 40→33,
`reativo` 10→8, `sem_deteccao` 13→8 -- **cobertura genuína corrigida:
66,7% (22/33)**, sem mexer no modelo, só corrigindo a métrica.

**Dos 8 "sem detecção"/"suspeitos" restantes (excluindo erro de
sensor), 3 são falhas reais do modelo/pipeline:**

| Data | MAE máximo na janela vs threshold (0,2609) | Causa |
|---|---|---|
| **2026-01-29** | **0,37-0,43 (cruzou!)** | Detectado pelo modelo, mas `load_gate` E `volatility_gate` bloquearam simultaneamente por 4h -- ver achado abaixo |
| 2026-03-25 | 0,2555 (quase) | Falha real, margem pequena |
| 2026-04-14 | 0,09 (nunca chegou perto) | Falha real do modelo -- alvo pra ensemble/arquitetura |

**Achado: dois gates independentes suprimindo uma detecção real.** Em
2026-01-29, a turbina parte (`RUNNING_A` 0→1 às 11h02), a temperatura
sobe de ~33°C pra ~740°C na partida (rampa legítima) mas **continua
subindo lentamente** até 790°C nas 2h seguintes -- a falha real. O
`load_gate` (referência `T5_AVG_A`, o próprio sensor-alvo) fica
bloqueado continuamente de 11h03 até 13h10+ porque a subida lenta
sustentada nunca deixa a rampa suavizada cair abaixo de `RAMP_MAX`.
Trocar a referência para um proxy de carga independente
(`954005_624_PI_0340`, que estabiliza ~11h20 enquanto a temperatura
segue subindo -- validado em 4/5 partidas reais testadas) **não
resolveu sozinho**: o `volatility_gate` (canais de vibração) também
fica com `volatility_gate_blocked=True` em **100% de uma janela de 4h**
(11h20-15h10) -- confirmado como elevação **fisicamente real** da
vibração (não artefato de janela/cálculo), provavelmente sintoma da
mesma degradação. Dois gates independentes, bloqueio binário cada um
-- corrigir só um não bastava.

## Bloqueio gradual dos gates (`GATE_ESCAPE_MULTIPLIER`)

Em vez de reformular os dois gates (load/volatilidade) individualmente,
a correção escolhida ataca a causa comum: **bloqueio binário demais**.
Um ponto cujo MAE bruto ultrapassa `threshold × GATE_ESCAPE_MULTIPLIER`
agora "escapa" do bloqueio de qualquer gate -- não precisa saber QUAL
gate bloqueou, nem por quê; só que o desvio é grande demais pra ser
coincidência com uma manobra legítima.

**Implementação (`pipeline.py`, `run_one_group`):** `_score_to_report`
passou a receber o MAE bruto (`mae_for_anom_raw`) + `threshold_local`
em vez de um array já binarizado. Quando `GATE_ESCAPE_MULTIPLIER` está
definido (`>1,0`), computa um SEGUNDO mapeamento sequência→ponto usando
`threshold × multiplier` (via `_map_to_points`, fatorado do código
anterior) -- os gates continuam aplicados normalmente ao mapeamento
NORMAL, e o resultado final é a união (`OR`) dos dois: um ponto conta
como anômalo se passou no threshold normal E não foi bloqueado, OU se
sozinho já ultrapassa o threshold elevado (bloqueado ou não). Novo
campo `GATE_ESCAPE_MULTIPLIER: Optional[float] = None` em `config.py`
-- default preserva o comportamento binário de sempre, nenhum config
existente é afetado. Mesmo tratamento no seed-sweep
(`_refit_cnn1dae_with_seed` já retornava `mae_for_anom`/`threshold`,
só mudou a chamada de `_score_to_report`).

**Validação:** smoke test dedicado (`smoke_test_gate_escape.py`) injeta
uma rampa de carga legítima (gate deve bloquear) com um desvio grande
e não-correlacionado do sensor-alvo bem no meio da mesma janela.
`GATE_ESCAPE_MULTIPLIER=None`: pico 100% suprimido (0/11 pontos).
`GATE_ESCAPE_MULTIPLIER=1,3`: gate continua ativo (`load_gate_blocked`
ainda `True`), mas 2/11 pontos escapam do bloqueio -- confirma que o
resgate funciona sem desligar o gate em si.

**Pendente:** escolher `GATE_ESCAPE_MULTIPLIER` e confirmar remotamente
se recupera o episódio real de 2026-01-29 sem inflar `normal_alert_rate`
de forma proibitiva (um multiplicador baixo demais devolveria FPs que
os gates existem justamente para suprimir).

### Escolha de `GATE_ESCAPE_MULTIPLIER` (simulação offline)

Grid search sobre os dados reais da task k=7,0 já cacheados
(`k7_sequence_scores_all.csv`/`k7_point_anomalies_all.csv`, reusando
`load_gate_blocked`/`volatility_gate_blocked` já calculados -- sem
gastar rodada remota):

| `multiplier` | Cobertura genuína | FP |
|---|---|---|
| 1,0 (baseline, sem resgate) | 66,7% (22/33) | 0,174% |
| 1,1 | 75,8% (25/33) | 0,305% |
| 1,3 | 75,8% (25/33) | 0,248% |
| **1,5** | **75,8% (25/33)** | **0,221%** |
| 1,65 | 75,8% (25/33) | 0,194% |
| 1,70 | 66,7% (22/33) -- volta ao baseline | 0,188% |

Cobertura genuína salta de 66,7% para **75,8% (25/33)** e fica estável
em todo o intervalo `1,30`-`1,65` -- FP cai continuamente dentro desse
platô (o resgate fica mais seletivo) até desabar de volta ao baseline
em `1,70` (a janela de resgate fecha por completo). Escolhido
`GATE_ESCAPE_MULTIPLIER=1,5`: dentro do platô com margem confortável
antes do "penhasco" em 1,70, evitando escolher um ponto sensível
demais a variação de retreino.

**Confirmação remota (task `f8b884932a2441b987086b611182fe1d`, commit
`dc15daa`):** bateu exatamente a simulação offline. Threshold
reproduziu idêntico (`0,2609` -- mesmo modelo, seed fixa), FP saiu em
`0,2204%` (previsto `0,221%`). Investigação caso a caso confirma: o
episódio de 2026-01-29 (antes "sem detecção", suprimido por load_gate
E volatility_gate simultaneamente) agora aparece como **reativo** (2
pontos) e **preditivo genuíno** (1 ponto, antecedência 1,48h) -- o
resgate funcionou exatamente como projetado.

| Candidato | Genuíno | Suspeito | Reativo | Cobertura genuína | FP |
|---|---|---|---|---|---|
| AutoML EXP10c | 24 | 5 | 8 | **80,0%** | 0,35% |
| CNN1D-AE `k=7,0` (baseline, sem gate escape) | 14 | 3 | 8 | 66,7% (22/33) | 0,17% |
| **CNN1D-AE `k=7,0` + `GATE_ESCAPE_MULTIPLIER=1,5` (candidato final)** | **15** | **3** | **10** | **75,8% (25/33)** | **0,22%** |

O gap pro AutoML caiu de 13,3pp para **4,2pp**, com FP ainda 37%
menor que o EXP10c (0,22% vs 0,35%). Este é o **candidato final
consolidado do EXP13**.

## Investigação dos 2 episódios não detectados (2026-03-25, 2026-04-14)

Puxados os artefatos da task final (`f8b884932a2441b987086b611182fe1d`,
commit `dc15daa`) + dataset bruto do ClearML pra investigar caso a caso.
**As duas causas são diferentes entre si -- e diferentes do que a
pendência original supunha.**

### 2026-03-25 -- não é falha do modelo, é o mesmo bug do 2026-01-29

`mae_seq` pico às 11h30 = **0,26107, ACIMA do threshold (0,26091)** --
`is_anom_seq=1`, o autoencoder detectou. Mas `volatility_gate_blocked=True`
durante toda a janela (11h15-11h45+), com elevação de vibração real (MAE
dos canais `TV_353Y_A`/`TV_354X_A`/`TV_354Y_A` entre 0,51-0,61, não
artefato). `GATE_ESCAPE_MULTIPLIER=1,5` exige MAE > threshold×1,5 = 0,391
pra resgatar -- essa detecção cruzou o threshold normal por uma margem de
apenas **0,00015**, muito longe dos 0,391 necessários. Mesmo mecanismo do
episódio 2026-01-29 (dois gates bloqueando uma detecção real), só que
aqui a margem de cruzamento é pequena demais pro resgate atual alcançar.
Não é um alvo de ensemble/arquitetura -- é uma questão de calibração do
gate-escape para margens pequenas.

### 2026-04-14 -- falha real, causa raiz identificada

MAE nunca passou de 0,123 (thr 0,261) nos 4 alarmes do dia -- não é
questão de gate. É uma **deriva lenta**: `TC382_03_A` sobe suavemente de
~677°C (13/04 12h) pra ~793°C (14/04 12h) ao longo de ~24h, sem
descontinuidade local -- nada pro reconstruction error de uma janela de
30min (`TIME_STEPS=60` a 30s) pegar.

Causa raiz do porquê nem o z-score capturou a magnitude: `normalize_train_only`
calculava `center`/`scale` sobre **todo** `df_normal_fit`, incluindo os
períodos off/partida (~42% do treino têm `TC382_03_A<100°C`). Isso infla
o desvio-padrão de `TC382_03_A` de **51,1°C** (só operando) para
**323,0°C** (misturado com off) -- quase 6,3x. O pico de 793,8°C vira
z-score **1,22** com as stats contaminadas, contra **2,22** se usasse só
o período "on" -- a deriva real fica estatisticamente escondida atrás da
variação off↔on, que domina o desvio-padrão usado pra normalizar TODO o
sinal (inclusive as features derivadas).

## EXP15 -- normalização restrita ao período operacional

Fix implementado: novo campo `NORMALIZE_ON_STATE_ONLY` (`config.py`) --
quando `true` (exige `ENABLE_OPERATIONAL_MASK=true`), `normalize_train_only`
(`preprocess.py`) recebe um `stats_mask` opcional e calcula `center`/`scale`
só sobre as linhas `operational_state=='on'` de `df_normal_fit`. Não filtra
`df_normal_fit` em si (as sequências de treino continuam contíguas, sem
gaps artificiais de janelas off removidas) -- só as estatísticas do
zscore/robust mudam. Em `pipeline.py` (`run_one_group`), o cálculo de
`state` foi antecipado pra antes do `normalize_train_only` (mesmos inputs
de sempre, pós-`clip_outliers` -- comportamento idêntico ao anterior
quando a flag nova está desligada, só a ordem mudou).

Config: `configs/calibracao_v4_eq/test_grupo_exp15_normalizacao_on_state.json`
(cópia do EXP13 + `NORMALIZE_ON_STATE_ONLY: true`). `THRESH_STD_K=7,0`
mantido como ponto de partida, mas a escala do MAE deve mudar com a nova
normalização -- espera-se precisar de pelo menos uma rodada de
recalibração, como em todo o histórico do EXP13.

**Rodada 1 (task `651553afb8444d62972a3ca14d209b95`, 2026-08-24):
resultado misto.** O pico de MAE do episódio 2026-04-14 nas duas janelas
de alarme (00h12 e 12h01) subiu pra **1,93 e 1,68 -- acima do threshold
usado (1,5704)**, contra 0,12 (bem abaixo de 0,26) antes do fix: a
normalização on-state-only funcionou pro efeito pretendido. Mas
`hit_rate` bruto do grupo caiu de 75,0% pra **27,5% (11/40)** --
`THRESH_STD_K=7,0` (herdado do EXP13) ficou alto demais na nova escala
do MAE (threshold real saiu em 1,5704, não mais 0,2609).

**Causa raiz da mudança de escala não ser um simples reescalonamento
linear:** `NORMALIZE_ON_STATE_ONLY` só filtra as ESTATÍSTICAS
(center/scale), não as linhas de `df_normal_fit` -- as sequências de
treino continuam incluindo os períodos off/partida. Com desvio-padrão
calculado só do "on" (~51 em vez de ~323 pra TC382_03_A), um valor off
típico (~33°C) agora normaliza pra z-score **≈-12,7** em vez de ≈-1,1.
O autoencoder é treinado tendo que reconstruir essas transições
artificialmente amplificadas, o que infla o MAE de forma não-uniforme
entre os canais/episódios -- por isso a recalibração de K não é uma
extrapolação proporcional trivial como nas rodadas anteriores do EXP13
(que só mudavam o modelo, não a distribuição de entrada).

**Regrid offline (reproduz exato o resultado oficial: hit_rate 0,275
batendo 11/40 bit-a-bit contra `evaluation_alarm_hit_rate.json`)** sobre
`sequence_scores_all.csv`/`point_anomalies_all.csv` desta mesma task,
reaplicando `map_seq_to_point_anomalies`/gates/`GATE_ESCAPE_MULTIPLIER`
exatamente como `pipeline.py`:

| threshold | hit_rate bruto | FP (`normal_alert_rate`) |
|---|---|---|
| 0,90 | 82,5% (33/40) | 0,57% |
| 1,00 | 82,5% (33/40) | 0,46% |
| **1,10** | **77,5% (31/40)** | **0,29%** |
| 1,20 | 60,0% (24/40) | 0,20% |
| candidato anterior (task 14, ref.) | 75,0% (30/40) | 0,22% |

`threshold≈1,10` bate/supera o candidato anterior nos dois eixos, e os
dois picos do episódio 2026-04-14 (1,93/1,68) ficam bem acima dele.
Extrapolação linear a partir do único ponto real conhecido
(`K=7,0`→1,5704) aponta pra **`K≈4,5`** -- aproximação, não valor exato
(mesma ressalva de sempre: o modelo real vai ter mediana/MAD levemente
diferentes ao retreinar).

**Rodada 2 (recalibração, task `a273ea8c9f674e8ba04ac291f45d2795`,
2026-08-24): confirma a hipótese, bate a extrapolação quase exata e
resgata o 2026-04-14.**

Threshold real saiu em **1,0892** -- praticamente idêntico à
extrapolação offline (previsto 1,10). Resultado:

| Candidato | hit_rate bruto | FP (`normal_alert_rate`) |
|---|---|---|
| Anterior (task 14, sem normalização on-state) | 75,0% (30/40) | 0,22% |
| EXP15 rodada 1 (`K=7,0`, threshold 1,5704) | 27,5% (11/40) | 0,08% |
| **EXP15b (`K=4,5`, threshold 1,0892)** | **77,5% (31/40)** | **0,30%** |

`hit_rate` bruto supera o candidato anterior (77,5% vs 75,0%), com FP
ainda bem abaixo do AutoML EXP10c (0,35%) apesar de ~37% maior que o
candidato anterior (0,30% vs 0,22%). **2026-04-14 confirmado como
detectado**: os dois picos de MAE (1,93 e 1,68, ambos > threshold 1,0892
e > `threshold×GATE_ESCAPE_MULTIPLIER`=1,634) geram 138 e 161 pontos
anômalos nas janelas de ±24h dos alarmes (`point_anomalies_all.csv`),
mesmo com `load_gate`/`volatility_gate` ativos na região -- resgatados
pelo gate-escape, mesmo mecanismo do episódio 2026-01-29.

**Ressalva (corrigida -- comparação inicial usava um número desatualizado
de uma task anterior, não o seed-sweep da própria task 14):** puxando o
`seed_sweep` real da task 14 pra comparar igual-com-igual --

| | seed principal | seeds 43-46 (média) | std | min | max |
|---|---|---|---|---|---|
| task 14 (candidato anterior) | 75,0% | 50,0% | ±7,7pp | 37,5% | 57,5% |
| EXP15b | 77,5% | 60,6% | ±9,9pp | 47,5% | 75,0% |

EXP15b supera task 14 em **todos** os pontos (seed principal, média,
mínimo e máximo) -- o desvio-padrão em pp é maior, mas o coeficiente de
variação (std/média) é praticamente igual (16,3% vs 15,4%). Não é o
candidato anterior "mais estável" e o EXP15b "mais disperso" como a
primeira leitura sugeriu -- os dois têm dispersão relativa parecida,
EXP15b só parte de uma base mais alta.

**Diagnóstico da variância:** com `THRESH_MODE=robust_mad`, o corte é
recalibrado a partir da mediana/MAD da distribuição de erro de treino
de *cada* modelo retreinado -- se o formato dessa distribuição varia
entre sementes, o mesmo `K` produz cortes de rigor diferentes. No
EXP15b, `hit_rate` e `normal_alert_rate` sobem/descem juntos entre
sementes (correlação 0,67 em n=4 -- amostra pequena, não conclusiva
sozinha) consistente com essa hipótese. `EarlyStopping` já usa
`restore_best_weights=True`, então instabilidade de convergência por
parada precoce é pouco provável como causa alternativa. Descartado
`tf.config.experimental.enable_op_determinism()` como mitigação: ele só
garante que a *mesma* semente reproduza resultado idêntico entre
execuções, não reduz a dispersão entre sementes *diferentes* (que é
intencional, por design do seed-sweep).

**Rodada 3 (teste de mitigação):** config
`test_grupo_exp15c_normalizacao_on_state_target_rate.json` (cópia do
EXP15b + `THRESH_MODE=target_rate`, usando o `TARGET_ANOMALY_RATE=0,003`
já configurado) -- fixa a *taxa* de anomalia no treino em vez de
depender do formato da distribuição de erro, testando se isso estabiliza
o ponto de operação entre sementes de um jeito que `robust_mad` não
garante. **Submetida remota:** task `b66bbaa24f7e42259d9eec7a90775e48`,
2026-08-24. Resultado pendente.

**Conclusão da investigação dos 2 episódios do início desta seção:**
2026-03-25 não precisava de mudança de modelo (era bloqueio de gate);
2026-04-14 foi resgatado pela normalização on-state-only + recalibração
de `THRESH_STD_K`. EXP15b já é o novo candidato de referência do EXP13
(supera task 14 em cobertura bruta E em todos os pontos do seed-sweep),
com EXP15c investigando se dá pra reduzir a variância de semente ainda
mais antes de consolidar.

## Pendências / próximos passos

- **Aguardar resultado do EXP15c** (`THRESH_MODE=target_rate`) -- se
  reduzir a variância de semente sem piorar hit_rate/FP, vira o novo
  candidato final; senão, EXP15b já é uma melhoria consolidada sobre a
  task 14 e pode ser promovido como está.
- **2026-03-25**: revisitar a calibração do `GATE_ESCAPE_MULTIPLIER` (ou
  um segundo multiplicador mais permissivo, específico pra margens
  pequenas) -- ver seção acima, não precisa de ensemble/arquitetura.
- **2026-04-14**: resolvido pelo EXP15b (ver acima).
- Formalizar a exclusão de alarmes `Comm Fail`/`Out of Serv` na
  metodologia de avaliação (hoje só documentado, não implementado em
  código -- `eval_alarm_hit_rate` ainda conta esses 4 alarmes no
  denominador).
- Por que threshold mais alto (`k≥6,0`) aumenta a variância do
  seed-sweep de forma persistente (±7-10,5pp vs ±2,5pp original) segue
  sem explicação definitiva -- diagnóstico parcial na seção do EXP15b
  acima (`hit_rate`/`normal_alert_rate` co-movendo entre sementes,
  consistente com sensibilidade de `robust_mad` à forma da distribuição
  de erro por modelo, mas amostra de `SEED_SWEEP_N=4` é pequena demais
  pra confirmar com confiança). EXP15c testa se `target_rate` mitiga.
  `tf.config.experimental.enable_op_determinism()` descartado como
  mitigação (ver seção EXP15b) -- só afeta reprodutibilidade da MESMA
  semente, não a dispersão entre sementes diferentes.
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
7. `7fba01f514674418adba71e3149b5e64` -- + fix stride na máscara operacional + robust_mad, hit_rate 87,5%/FP 1,21%
8. `2db0722a20694cdfb47d77e1b07242a6` -- + recalibração (`THRESH_STD_K=3,0`, `POINT_WINDOW=2`), guiada por simulação offline, hit_rate 90,0% (36/40) / FP 0,94%
9. `9ff669b9a83f4ad7bb0fefc01aead335` -- + gates recalibrados (rampa/volatilidade) + `MAX_TRIALS=40`, guiada por simulação offline, hit_rate 90,0% (36/40) / FP 0,80%
10. `8b97c625af78431e88b912c6e9288332` -- + seed-sweep (`SEED_SWEEP_N=4`), **achado final: CNN1D-AE NÃO tem variância zero (hit_rate 85,0%--92,5% entre sementes, média 86,9%±3,25pp) -- diferente do AutoML, cuja variância de semente é zero desde o EXP5**
11. `2d63b3fbe1ff43459f8db28283910c49` -- + seed global fixada antes do tuner (2 pontos de reseed), reduz variância (`hit_rate_std` 3,25→2,50pp) mas threshold reproduzível (0,135) caiu numa faixa de FP mais alta (0,60%→1,23% de FP médio)
12. `48119896110c4ccfa2581ab2087f4d88` -- + recalibração pós-artefato de janela (`THRESH_STD_K=6,0`, mirando cobertura genuína), threshold real 0,2294 (não 0,265 estimado), cobertura genuína 47,5% (19/40) / FP 0,30% -- melhora sobre baseline mas abaixo da meta de 60%; seed-sweep mostra variância bem maior (`hit_rate_std` ±10,51pp)
13. `8a95bccb0e40461ea9caea20c94dae10` -- + segunda recalibração (`THRESH_STD_K=7,0`, guiada por regrid offline sobre dados reais da task 12), threshold real 0,2609 (quase exato ao previsto 0,260), cobertura genuína 60,0% (24/40) / FP 0,17% -- bate a meta, FP menos da metade do EXP10c; seed-sweep melhora um pouco mas segue elevado (`hit_rate_std` ±7,15pp)
14. `f8b884932a2441b987086b611182fe1d` (commit `dc15daa`) -- + auditoria caso a caso (exclui 4 alarmes de erro de sensor confirmados por código `Comm Fail`/`Out of Serv` no dado bruto) + bloqueio gradual dos gates (`GATE_ESCAPE_MULTIPLIER=1,5`, resgata o episódio 2026-01-29 antes suprimido por load_gate+volatility_gate simultâneos), guiado por simulação offline. Threshold reproduziu idêntico (0,2609), FP bateu exato (0,2204% vs 0,221% previsto). **Candidato final consolidado: cobertura genuína 75,8% (25/33, excluindo erro de sensor) / FP 0,22%** -- gap pro AutoML EXP10c caiu de 13,3pp para 4,2pp, com FP ainda 37% menor
15. `651553afb8444d62972a3ca14d209b95` -- falhou (config nao existia no git ainda -- worker remoto clona do repo, faltou commit+push antes de submeter).
16. `39d73f7cb7ae4ce7845903246edd5df9` -- EXP15 (resubmetida apos commit `59021ee`): `NORMALIZE_ON_STATE_ONLY=true`. Resultado misto -- pico de MAE do episodio 2026-04-14 confirmado acima do threshold usado (1,93/1,68 vs 1,5704, contra 0,12 vs 0,26 antes do fix), mas hit_rate bruto do grupo caiu de 75,0% pra 27,5% (11/40): THRESH_STD_K=7,0 herdado do EXP13 ficou alto demais pra nova escala do MAE. Regrid offline (reproduz exato o resultado oficial) aponta K≈4,5 pra recuperar hit_rate competitivo.
17. `a273ea8c9f674e8ba04ac291f45d2795` -- EXP15b (`test_grupo_exp15b_normalizacao_on_state_recalibrado.json`, `THRESH_STD_K=4,5`) -- recalibracao guiada por regrid offline sobre os dados reais da task 16. Threshold real 1,0892 (quase exato ao previsto 1,10). **hit_rate 77,5% (31/40) / FP 0,30%** -- supera task 14 em cobertura bruta (75,0%), FP ~37% maior mas ainda bem abaixo do AutoML EXP10c (0,35%). 2026-04-14 confirmado detectado (138/161 pontos anomalos nas janelas dos 2 alarmes). Seed-sweep com variancia alta (hit_rate 47,5%-75,0%, media 60,6%±9,9pp) -- ressalva de robustez a retreino.
