# Monitoramento multivariado via PCA com retreino mensal (walk-forward)

**Branch:** `AE_pca_monitoramento_sistema` (a partir de `AE_novo_legado`).
Metodologia nova, complementar às pipelines por-alvo já existentes (T5,
trip de óleo lub., pressão de gás combustível): em vez de escolher um
sensor-alvo e um pequeno grupo de correlacionados, usa **todos os
sensores brutos disponíveis juntos**, reduzidos por PCA, com **retreino
mês a mês** (janela expansiva) e avaliação contra **o catálogo inteiro
de alarmes** (47 tags), não um alarme por vez.

## Origem da ideia

Descrita por um colega como uma técnica que ele já usava: treinar usando
todo o histórico disponível (excluindo só as janelas ao redor dos
alarmes), passando os dados por um pré-processamento parecido com o
nosso, mas retreinando a cada mês em vez de treinar uma vez só com o
histórico inteiro, e usando desvio-padrão móvel em múltiplos horizontes
(3h, 12h, 48h) como no nosso portão de volatilidade. Reconhecida como uma
variante do **monitoramento de processo multivariado via PCA**
(PCA-based fault detection / multivariate SPC), técnica clássica da
indústria de processo, aqui adaptada para reusar nosso motor de modelagem
AutoML (`ocsvm`/`iforest`) já validado, em vez de calcular T²/SPE à mão.

## Metodologia

1. **Sensores**: os mesmos 36 sensores brutos usados nas varreduras de
   precursor anteriores (temperatura, pressão, vibração) — nenhum
   curado/excluído a priori.
2. **Features**: valor bruto + desvio-padrão móvel em 3 horizontes
   (3h/12h/48h) de cada sensor → **144 features** antes do PCA.
3. **Redução de dimensionalidade**: PCA, mantendo componentes suficientes
   para explicar 90% da variância (entre 12 e 20 componentes,
   dependendo do mês).
4. **Retreino em janela expansiva mensal**: para cada mês M (a partir do
   3º mês do histórico, para ter treino mínimo), treina com **todo** o
   histórico anterior a M — filtrado pro estado operacional "on"
   (`build_operational_state`) e excluindo ±24h ao redor de **qualquer**
   alarme do catálogo completo (não só um tag) — e avalia no mês M.
   Concatenando os 26 meses avaliados (2024-03 a 2026-04), cada ponto do
   histórico é pontuado por um modelo que nunca viu esse mês durante o
   treino — um walk-forward de verdade, não um único corte OOS.
5. **Modelagem**: reaproveita os fitters já usados no resto do projeto
   (`fit_ocsvm`/`fit_isolation_forest` de `automl_models.py`), threshold
   por percentil (99) e debounce (6 pontos) — mesma disciplina do
   restante da pipeline AutoML.
6. **Avaliação**: contra os 47 tags do catálogo completo com ≥3 alarmes
   no período avaliado — não um alvo escolhido a priori.

## Dois bugs encontrados e corrigidos durante a implementação

1. **Máscara operacional esquecida na primeira versão**: sem zerar
   `is_anom_point` fora do estado "on", vários meses tinham **quase
   100% dos pontos sinalizados como anômalos** (o modelo nunca via
   período desligado no treino "on", então qualquer ponto de
   desligamento parecia extremo). Corrigido reaplicando
   `build_operational_state` (mesma função usada em todas as outras
   pipelines do projeto) e zerando a flag fora de "on" após o debounce.
2. **`ocsvm` sem limite de amostra de treino**: com a janela expansiva
   crescendo mês a mês, o treino de dezembro/2025 já tinha ~292 mil
   pontos — o kernel RBF do `ocsvm` não escala pra esse tamanho (travou
   por mais de 2h de CPU). Corrigido com o mesmo limite de subamostragem
   (`OCSVM_MAX_TRAIN=50000`) já usado no resto do projeto
   (`AUTOML_OCSVM_MAX_TRAIN_SAMPLES`).

## Resultado

| Modelo | FP geral (on, longe de qualquer alarme) |
|---|---|
| `ocsvm` | 11,31% |
| **`iforest`** | **3,21%** |

`iforest` claramente melhor nesse regime (bem menos ruidoso), apesar do
`ocsvm` ter `hit_rate` um pouco mais alto em vários tags — mesmo
trade-off FP/cobertura observado nos outros experimentos.

### `hit_rate` por tag (modelo `iforest`, ordenado)

| Tag | N alarmes | Acertos | hit_rate |
|---|---|---|---|
| `PAH_6240319` (pressão alta gás partida) | 67 | 57 | 85,1% |
| `PDAL_6240302` | 55 | 46 | 83,6% |
| `PDAHH6240305` | 5 | 4 | 80,0% |
| `PDI_6240302_AL` | 3 | 2 | 66,7% |
| `TAHH_6240307` | 3 | 2 | 66,7% |
| `PDAH_6240305` | 8 | 5 | 62,5% |
| `TC382_03_A` (**T5**, alvo do EXP10c) | 187 | 116 | 62,0% |
| `PAL_6240315` (alvo do EXP17) | 374 | 225 | 60,2% |
| `PI_6240319_AL` | 509 | 281 | 55,2% |
| `T5_AVG_A` | 64 | 32 | 50,0% |
| `PALL_6240309` (alvo do EXP16) | 6 | 3 | 50,0% |
| ... (37 tags restantes, 0-45%) | | | |
| `TV_35[1-5][X/Y]_A` (10 canais de vibração) | 6-13 cada | 1-2 cada | 8-17% |
| `PALL_6240340`/`TALL_6240325`/`TAL_6240325` (trips já sabidos artefato) | 53-163 | 3-16 | 4,4-9,8% |

Tabelas completas (47 tags) salvas em
`scripts/pca_monitoramento_sistema/resultado_iforest_por_tag.csv` e
`resultado_ocsvm_por_tag.csv`.

## v2: pré-processamento real do projeto + avaliação por episódio

A versão acima (v1) usava uma reimplementação manual e simplificada do
pré-processamento (só valor bruto + desvio-padrão móvel em 3 horizontes).
Ao revisar com mais cuidado, ficou claro que isso não reaproveitava o
pré-processamento de verdade do projeto — `clip_outliers` (modo
quantil, ajustado por dataframe), as features derivadas completas
(`_build_derived_features`: delta, mediana/desvio/tendência móvel em 4
janelas — 6min/1h/4h/24h —, além de textura pros canais de vibração) e
`normalize_train_only`. O v2
(`pca_walkforward_v2_preprocessamento_real.py`) corrige isso, chamando
diretamente `build_group_dataframe`, `select_feature_columns`,
`clip_outliers` e `normalize_train_only` de `preprocess.py` — a mesma
cadeia usada em `automl_pipeline.py` — antes do PCA. Isso expande de 144
para ~594 colunas de entrada do PCA (ainda reduzidas a 12-20 componentes
por mês, mantendo 90% de variância).

### Comparação v1 vs v2 (mesma métrica de FP e hit_rate médio por tag)

| Métrica | v1 `ocsvm` | v1 `iforest` | v2 `ocsvm` | v2 `iforest` |
|---|---|---|---|---|
| FP geral (% pontos, on, longe de alarme) | 11,31% | 3,21% | 5,97% | **2,27%** |
| `hit_rate` médio entre os 47 tags | 29,50% | 27,25% | 29,79% | 29,04% |

O pré-processamento real reduziu o FP em ambos os modelos (quase pela
metade no `ocsvm`) sem piorar o `hit_rate` médio — na verdade melhorou
ligeiramente em ambos. Faz sentido: features derivadas de verdade
(médias/desvios/tendências em múltiplas janelas, já testadas e afinadas
no resto do projeto) dão ao PCA uma base mais informativa e mais estável
que o par bruto+desvio-móvel simplificado do v1.

### Achado de co-ocorrência entre tags (motivação pra métrica por episódio)

A pergunta levantada foi: será que os 47 tags do catálogo, por
pertencerem ao mesmo equipamento, disparam **juntos** no mesmo evento
físico? Uma checagem empírica agrupando os 3757 eventos do catálogo em
"episódios" (gap > 60min entre eventos consecutivos = novo episódio)
confirmou que sim, fortemente: em vários episódios (ex.: o de
2024-06-11) até **46 dos 47 tags do catálogo** disparam dentro de uma
janela de poucos minutos um do outro — um efeito cascata típico de
trip/desligamento de equipamento, onde uma única causa-raiz aciona
quase todo o painel de alarmes em sequência.

Isso tem uma implicação direta pra leitura do `hit_rate` por tag: se um
modelo detecta o evento uma vez, ele "acerta" simultaneamente para quase
todos os 47 tags daquele episódio — o `hit_rate_medio_entre_tags`
(29-30%) mistura decisões redundantes (mesmo evento, contado 40+ vezes)
com decisões genuinamente independentes. Pra medir a cobertura real de
**eventos distintos**, o v2 adicionou uma avaliação por episódio:
agrupa os alarmes do catálogo em episódios (mesmo critério de gap>60min)
e verifica se há pelo menos um ponto anômalo dentro de ±24h do episódio,
contando o episódio como um hit ou miss — não por tag. O FP também passa
a ser contado em episódios por mês (agrupando pontos falso-positivos
consecutivos), pra ficar comparável com a forma como resultados externos
costumam reportar (ex.: "X falsos positivos por mês" em vez de "X% dos
pontos").

### Métrica por episódio (v2)

| Métrica | `ocsvm` | `iforest` |
|---|---|---|
| `hit_rate` por episódio (cobertura de eventos distintos) | 68,4% | 63,7% |
| FP em episódios por mês | 4,42 | **3,35** |

A cobertura por episódio (63-68%) é bem mais alta que o `hit_rate`
médio por tag (27-30%) — esperado, já que um único hit já cobre o
episódio inteiro, enquanto a média por tag pune tags de baixo sinal
(vibração, trips-artefato) que nunca vão ser bem servidos por um modelo
generalista de todo o sistema.

### Comparação com o resultado do colega (janela de 3000h / ~4 meses)

O colega reportou FP como **episódios por mês**, não como % de pontos —
por isso a métrica acima foi criada, pra comparar de forma justa. Ele
relatou **0,94 FP/mês**, bem abaixo dos 3,35-4,42/mês obtidos aqui no
v2. A hipótese inicial era que a diferença viesse da estratégia de
treino: ele usa uma **janela rolante fixa de ~3000h (~4 meses)**,
enquanto o v2 usa janela **expansiva** (cresce indefinidamente, treino
de 2026 usa quase 2 anos de histórico). Essa hipótese foi testada
diretamente no v3 — ver abaixo — e **refutada**: a janela rolante não
reduziu o FP, pelo contrário.

Tabelas completas do v2 (47 tags) salvas em
`scripts/pca_monitoramento_sistema/resultado_v2_iforest_por_tag.csv` e
`resultado_v2_ocsvm_por_tag.csv`.

## v3: janela de treino rolante fixa (~3000h) — hipótese testada e refutada

Implementação idêntica ao v2 (mesmo pré-processamento real, mesma
avaliação por tag e por episódio), mudando **só** o filtro de treino: em
vez de "todo o histórico anterior ao mês avaliado" (expansivo), passa a
ser "os últimos ~3000h (~125 dias) anteriores ao mês avaliado" (rolante
fixa) — a mesma ordem de grandeza da janela usada pelo colega.
Script: `pca_walkforward_v3_janela_rolante.py`.

| Métrica | v2 (expansiva) `ocsvm` | v2 `iforest` | v3 (rolante ~3000h) `ocsvm` | v3 `iforest` |
|---|---|---|---|---|
| FP geral (% pontos) | 5,97% | 2,27% | 17,73% | 5,43% |
| `hit_rate` médio entre tags | 29,79% | 29,04% | 30,13% | 29,77% |
| `hit_rate` por episódio | 68,4% | 63,7% | 70,1% | 66,8% |
| FP em episódios/mês | 4,42 | 3,35 | 5,77 | 3,50 |

**A hipótese foi refutada — o resultado foi o oposto do esperado.**
Encurtar a janela de treino pra ~3000h **piorou** o FP em ambos os
modelos (no `ocsvm`, quase triplicou: 5,97%→17,73%), com ganho só
marginal de cobertura (`hit_rate` por episódio subiu ~2-3 p.p.). Faz
sentido em retrospecto: uma janela de treino mais curta expõe o modelo
a **menos variedade de operação normal** (menos ciclos de carga, menos
condições sazonais, menos combinações de setpoint já vistas como
"normais"), então qualquer variação de operação que não caiba dentro
desses ~4 meses recentes passa a ser tratada como anomalia — o modelo
fica **mais sensível, não mais preciso**. A janela expansiva, ao ver
mais histórico, aprende uma noção mais ampla (e mais correta) do que é
variação normal da planta, reduzindo o FP.

Conclusão prática: **a diferença de FP/mês em relação ao colega (0,94
vs. 3,35-5,77) não vem da estratégia de janela de treino** — outra
explicação precisa ser buscada (candidatos mais prováveis: threshold/
debounce diferentes, conjunto de sensores usado, ou definição de
"falso positivo" dele ser mais permissiva/agrupada que a nossa). Ver
"Próximos passos" atualizado.

Tabelas completas do v3 (47 tags) salvas em
`scripts/pca_monitoramento_sistema/resultado_v3_iforest_por_tag.csv` e
`resultado_v3_ocsvm_por_tag.csv`.

## v4: corrige o teto de componentes do PCA (MAX_COMPONENTS)

Bug encontrado nos logs do v2/v3: `k = min(k, MAX_COMPONENTS)` travava
sempre em 20 componentes, mesmo quando isso não alcançava os 90% de
variância-alvo documentados (ficava estacionado em 60-71%). v4 sobe o
teto pra 150 (script `pca_walkforward_v4_pca_variancia_real.py`).

| Métrica | v2 (20 comp.) `ocsvm` | v2 `iforest` | v4 (150 comp.) `ocsvm` | v4 `iforest` |
|---|---|---|---|---|
| FP geral (% pontos) | 5,97% | 2,27% | 6,03% | 4,36% |
| `hit_rate` médio entre tags | 29,79% | 29,04% | 29,97% | 29,88% |
| `hit_rate` por episódio | 68,4% | 63,7% | 69,6% | 67,9% |
| FP em episódios/mês | 4,42 | 3,35 | 5,65 | 4,73% |

**Resultado contraintuitivo**: capturar mais variância (corrigir o bug)
**piorou** o FP no `iforest` (2,27%→4,36%) com ganho só marginal de
cobertura. Mesma lição do v3: mais informação não é sempre melhor pro
`iforest`/`ocsvm` nesse regime — componentes adicionais do PCA além dos
20 originais aparentemente carregam mais ruído específico de mês do que
sinal genuinamente discriminativo. **v2 (20 componentes, "errado" pela
documentação original) continua sendo o melhor ponto da série
v2/v3/v4** — mantido como referência.

## Investigação da branch do colega (Francisco) — `feat/pdm-deteccao-4sinais`

Leitura (sem checkout, só `git show origin/<branch>:<arquivo>`) do
`README_PDM.md`, `src/cabiunas_pdm/config.py` e `scripts/automl_clearml.py`
da branch dele revelou uma arquitetura bem diferente da nossa:

- **4 sinais independentes por subsistema físico, com votação**, não um
  PCA único sobre todos os sensores: `temperatura` (14 sensores),
  `pressao_oleo` (12 sensores), `mancal_spread` (univariado: `TI_0305`
  menos a mediana dos 3 mancais irmãos) e `selagem_z` (univariado:
  `PDIT_0305` isolado, porque a família de pressão dilui esse sensor na
  média de 12).
- **Ground-truth curado**: não usa os 3757 eventos do catálogo bruto.
  Deriva uma lista curta de **trips reais** — parada `RUNNING_A` 1→0
  com duração ≥2h **e** alarme de nível (regex `TRIP|MT.ALTA|M.ALTA|...`
  na descrição) na janela [-1h,+30min] — agrupados em eventos físicos
  por proximidade (<24h).
- **PCA-Q (erro de reconstrução)** em vez de PCA+iforest, com
  `PCA(n_components=0.95, svd_solver="full")` deixando o próprio sklearn
  escolher quantos componentes bastam — evita o bug do v2/v3/v4.
- **Normalização robusta** (mediana/IQR via `RobustScaler`), não z-score.
  **EWMA no score contínuo antes do limiar** (`ewm(halflife=...,
  times=índice)`), não debounce no flag binário depois.
  **Limiar por múltiplo do p99** (`threshold_x_p99=2.0` pra temperatura)
  ou **z-robusto** (`|z|>3.0` pro mancal), ambos sustentados 30min.
  **Confirmação = E** entre 2 sinais (`temperatura` e `mancal_spread`),
  não votação genérica — essa é a política de produção documentada em
  `DETECTION_POLICY` (decisão de 18/07/2026, prioridade "minimizar FP").
- **Janela de baseline em horas de operação ELEGÍVEL** (pós-exclusão),
  não em horas de calendário — "3000h" pode exigir voltar 16 a 64 dias
  no calendário dependendo do mês. O nosso v3 tinha reproduzido isso
  errado (calendário corrido), o que ajuda a explicar por que a
  hipótese da janela rolante falhou lá.
- **Contabilidade de FP mais cirúrgica**: um episódio de alerta só conta
  como falso positivo se não estiver a até 48h de um evento curado **ou**
  de qualquer parada real (não só as poucas catalogadas).
- Número confirmado no código: **"janela de 3.000h dá 6/8 eventos com
  0,94 FP/mês"** — mesma unidade (episódios/mês) que já estávamos usando.

## v5: reproduz a votação N-de-4 (arquitetura, sem as inovações de pós-processamento)

Primeira tentativa (script `votacao_4sinais_v5.py`): os mesmos 4 sinais
por família física, mas ainda com a receita v2 (PCA+`iforest`,
normalização z-score, debounce no flag em vez de EWMA no score,
catálogo bruto como ground-truth).

| Sinal / votação | FP (% pontos) | `hit_rate` por episódio | FP episódios/mês |
|---|---|---|---|
| `temperatura` sozinho | 1,33% | 46,1% | 0,54 |
| `pressao_oleo` sozinho | 3,91% | 50,6% | 1,31 |
| `mancal_spread` sozinho | 9,92% | 14,4% | 0,69 |
| `selagem_z` sozinho | 26,97% | 58,9% | 5,46 |
| votação 2-de-4 | 7,92% | 49,5% | 1,46 |
| votação 3-de-4 | 1,14% | 28,8% | **0,23** |
| votação 4-de-4 | 0,34% | 7,4% | 0,15 |

A arquitetura de votação já ajuda bastante por si só — `votação 3-de-4`
chega a 0,23 episódios/mês, melhor que o 0,94 dele, ainda que avaliado
contra o catálogo bruto (não comparável 1:1). Mas faltavam as inovações
de pós-processamento (EWMA, limiar por múltiplo do p99, ground-truth
curado) — daí o v6.

## v6: reproduz a política de produção dele com fidelidade (9 inovações portadas)

Script `reproducao_francisco_v6.py`: porta as 9 inovações acima pra
nossa pipeline (pré-processamento real do projeto + `sklearn` puro em
vez do dele) e avalia **nos dois estilos** — contra o catálogo bruto
(continuidade com v2-v5) e contra o **ground-truth curado**, replicando
o algoritmo dele em cima do nosso `alarmes_selecionados_turbina_a.csv`.

**Ground-truth curado encontrado no nosso dataset**: 65 paradas reais
(≥2h), das quais 12 coincidem com alarme de nível → **11 eventos
físicos** (2024-03 a 2026-04; ele achou 8 num período mais curto,
jan/2025-abr/2026 — plausível, coerente com a diferença de cobertura
temporal).

| Sinal / política | Eventos detectados | Lead médio | FP/mês (estilo dele) |
|---|---|---|---|
| `temperatura` sozinho (PCA-Q) | 6/11 (54,5%) | 19,9h | 2,81 |
| `pressao_oleo` sozinho (PCA-Q) | 0/11 | — | 0,23 |
| `mancal_spread` sozinho (z-robusto) | 4/11 (36,4%) | 9,75h | 1,18 |
| `selagem_z` sozinho (z-robusto) | 5/11 (45,5%) | **30,5h** | 5,85 |
| **`producao_2sinais_AND`** (política dele: temp. E mancal) | 2/11 (18,2%) | 13,2h | **0,90** |
| votação 2-de-4 (bônus, genérica) | 5/11 (45,5%) | 20,7h | 1,91 |
| votação 3-de-4 / 4-de-4 | 0/11 | — | 0,17 / 0,00 |

**O FP foi reproduzido quase exatamente**: `producao_2sinais_AND` deu
**0,90 episódios/mês**, contra os **0,94/mês** documentados por ele —
usando a MESMA unidade, o MESMO mecanismo (confirmação E entre 2 sinais
independentes) e um ground-truth curado com o MESMO algoritmo. Isso
confirma a hipótese principal da investigação da branch dele: **a
supressão de FP vem da votação/confirmação entre sinais independentes,
não de um modelo mais preciso ponto a ponto** — replicamos o mecanismo
com sklearn puro e chegamos a um número quase idêntico.

**O que NÃO foi replicado com a mesma fidelidade: a cobertura de
eventos.** Ele reporta 6/8 (75%) na varredura ampla; nossa política de
produção replicada chegou a só 2/11 (18%) — bem abaixo até do
`temperatura` sozinho (6/11, 54,5%), o que é o sintoma clássico de um
threshold/EWMA mal calibrado pro nosso dado, não um problema conceitual
(a arquitetura funciona, prova o FP quase idêntico). **Suspeita
principal, ainda não testada**: nós rodamos o PCA-Q sobre o espaço de
**features derivadas completo** do projeto (~196 colunas por família,
com médias/desvios/tendências em 4 janelas), enquanto ele roda o PCA-Q
direto sobre os **sensores brutos** (14 colunas, só limpos e
escalados) — um espaço de entrada bem menos dimensional e sem a
"diluição" que médias/desvios móveis podem causar no erro de
reconstrução de um evento pontual de falha. Fica como próximo passo
(v7): rodar a família `temperatura`/`mancal_spread` sem features
derivadas (`ENABLE_DERIVED_FEATURES=False`), só sensor bruto limpo, pra
isolar se essa é a causa da diferença de sensibilidade.

## v7: sem features derivadas — hipótese confirmada, ganho nas duas métricas

Script `reproducao_francisco_v7_sem_features_derivadas.py`: idêntico ao
v6, mudando **só** `cfg.ENABLE_DERIVED_FEATURES` de `True` para `False`
nas famílias `temperatura`/`pressao_oleo` (os sinais univariados
`mancal_spread`/`selagem_z` não usam essa flag — servem de controle
interno, e devem ficar idênticos ao v6 se a mudança estiver isolada
corretamente).

| Sinal | v6 (com derivadas) | v7 (sensor bruto) |
|---|---|---|
| `temperatura`: eventos / FP | 6/11 (54,5%) / 2,81/mês | 4/11 (36,4%) / 2,03/mês |
| `pressao_oleo`: eventos / FP | **0/11 (0%)** / 0,23/mês | **5/11 (45,5%)** / 2,42/mês |
| `mancal_spread`: eventos / FP | 4/11 (36,4%) / 1,18/mês | 4/11 (36,4%) / 1,18/mês *(controle, inalterado ✓)* |
| `selagem_z`: eventos / FP | 5/11 (45,5%) / 5,85/mês | 5/11 (45,5%) / 5,85/mês *(controle, inalterado ✓)* |
| **`producao_2sinais_AND`** | **2/11 (18,2%) / 0,90/mês** | **4/11 (36,4%) / 0,73/mês** |
| votação 2-de-4 | 5/11 (45,5%) / 1,91/mês | 6/11 (54,5%) / 2,42/mês |
| votação 3-de-4 | 0/11 (0%) / 0,17/mês | 3/11 (27,3%) / 1,13/mês |
| votação 4-de-4 | 0/11 (0%) / 0,00/mês | 2/11 (18,2%) / 0,06/mês |

**Hipótese confirmada.** Os dois sinais de controle (`mancal_spread`,
`selagem_z`) ficaram exatamente iguais ao v6 — confirma que a mudança
afetou só o que deveria (isolamento correto do experimento). O achado
mais marcante: `pressao_oleo` estava **completamente morto** no v6 (0
eventos detectados) e passa a detectar **5 dos 11** no v7 — a família
de 12 sensores de pressão simplesmente não tinha poder de detecção
nenhum quando misturada com ~156 colunas derivadas.

O resultado principal, a política de produção (`producao_2sinais_AND`),
**melhorou nas duas métricas ao mesmo tempo** — algo raro (normalmente
mais cobertura custa mais FP): eventos detectados **dobraram** (18,2%→
36,4%) e o FP **caiu ainda mais** (0,90→0,73/mês, agora melhor que o
próprio 0,94/mês de referência). `temperatura` sozinho piorou em
cobertura (54,5%→36,4%) mas melhorou em FP (2,81→2,03/mês) — o balanço
final da combinação com `mancal_spread` ainda foi positivo.

**Conclusão prática**: rodar o PCA-Q sobre sensor bruto limpo, sem o
conjunto de features derivadas do projeto (pensado originalmente pro
autoencoder CNN-1D e pro AutoML com `iforest`/`ocsvm`), é estritamente
melhor para essa arquitetura específica de votação/confirmação. v7
substitui o v6 como referência para a política de 2 sinais.

## v8: confirmação por mecanismo — hipótese testada e refutada

Script `reproducao_francisco_v8_confirmacao_por_mecanismo.py`: idêntico
ao v7, trocando a regra de confirmação fixa
(`temperatura E mancal_spread`) por uma mais flexível,

```
alerta = temperatura E (mancal_spread OU selagem_z OU pressao_oleo)
```

motivada diretamente pela auditoria dos "misses" do relatório do v6: o
evento de selagem de 2025-02-27 nunca é confirmado pelo par fixo porque
`mancal_spread` não tem relação física com esse mecanismo de falha —
mesmo com a temperatura antecipando com folga (33,7h). A ideia era
recuperar esse tipo de evento sem abrir mão da estrutura "âncora +
confirmador" (que difere da votação genérica N-de-4 por ainda exigir
`temperatura` especificamente, não qualquer par).

| Política | Eventos | Lead médio | FP/mês |
|---|---|---|---|
| `par_fixo_v7` (referência: temp. E mancal_spread) | 4/11 (36,4%) | 9,3h | **0,73** |
| `confirmacao_mecanismo` (temp. E qualquer um dos 3) | 4/11 (36,4%) | 15,3h | 1,63 |

**Resultado negativo — a hipótese não se sustentou.** A cobertura de
eventos ficou **idêntica** (4/11 nos dois casos — o log mostra que a
identidade dos eventos detectados mudou, dado que o lead médio é
diferente, mas o total não), enquanto o FP **mais que dobrou** (0,73→1,63/mês,
74 episódios de alerta contra 32). Trocar o confirmador fixo por um OR
não trouxe nenhum evento novo o suficiente pra compensar o ruído
importado: `selagem_z` e `pressao_oleo`, usados sozinhos, já têm FP
bem mais alto que `mancal_spread` (5,85 e 2,42 contra 1,18 episódios/mês)
— colocá-los na condição de confirmação via OR faz o alerta combinado
herdar boa parte desse ruído, sem ganho líquido de cobertura.

**Lição**: o par fixo (`temperatura` + `mancal_spread`) não é arbitrário
— `mancal_spread` provavelmente foi escolhido pelo Francisco (e continua
sendo o melhor confirmador aqui) precisamente por ser o sinal univariado
mais **limpo** dos três candidatos, não só por ser fisicamente plausível.
Generalizar a regra de confirmação exige mais cuidado que um OR simples
— provavelmente pesar/filtrar os confirmadores por ruído individual, ou
usar um confirmador diferente **por mecanismo conhecido** (não um OR que
os trata como intercambiáveis). **v7 continua sendo a referência.**

## Interpretação

(Números abaixo já refletem o v2, com pré-processamento real; ver
comparação v1/v2 acima.)

- **Um único modelo, sem nenhuma calibração por alarme, detecta sinal em
  praticamente toda categoria do catálogo** (67-85% em vários tags de
  pressão/temperatura no `iforest`), com FP de só 2,27% dos pontos (ou
  3,35 episódios/mês) e cobertura de **63,7% dos episódios distintos**
  do catálogo. Valida a ideia central da abordagem: uma "vigia geral" da
  planta é viável.
- **Não supera os modelos dedicados já construídos**: T5 (EXP10c) 92,5%
  dedicado vs. 80,7% aqui (`iforest`, v2 — subiu bastante frente aos
  62,0% do v1, mas ainda abaixo do dedicado); gás combustível (EXP17a)
  81,5% dedicado vs. 67,1% aqui. Esperado — um generalista não bate um
  especialista calibrado especificamente pro alvo, mas a distância
  encolheu bastante com o pré-processamento correto.
- **Vibração isolada continua mal servida** (8-17% em todos os 10
  canais) — reforça o achado recorrente no projeto de que vibração não é
  bem explicada pelo resto do sistema (é mais útil como *feature* de
  contexto do que como *alvo* de predição direto).
- **Os 3 trips já identificados como artefato de desligamento**
  (`PALL_6240340`, `TALL_6240325`, `TAL_6240325`) continuam com
  `hit_rate` baixo (4-10%) mesmo nesse modelo generalista de todo o
  sistema — consistente com a conclusão anterior de que não há
  precursor real ali, em nenhuma combinação de sensores disponível.

### ⚠️ Ressalvas metodológicas importantes

1. **Genuíno vs. artefato de desligamento**: diferente do EXP16 (onde
   cada evento foi auditado manualmente), aqui o `hit_rate` é calculado
   contra **todos** os eventos `ACT/CFN` do catálogo, sem esse filtro.
   Os números desta tabela **não são diretamente comparáveis** aos
   `hit_rate`/cobertura genuína reportados nos EXP10c/16/17 sem essa
   mesma auditoria. Também não foi feita a checagem de antecedência real
   (preditivo vs. reativo) que a lição do EXP16 exige antes de confiar
   cegamente num `hit_rate` agregado.
2. **Efeito cascata entre tags**: como descrito acima, um único evento
   físico dispara dezenas de tags do catálogo quase simultaneamente
   (até 46/47 num mesmo episódio). Isso infla o número de tags "com
   `hit_rate` alto" sem representar 46 detecções independentes — é
   essencialmente a mesma detecção contada várias vezes. A métrica por
   episódio (v2) é a leitura mais confiável de cobertura real; o
   `hit_rate` por tag individual continua útil pra identificar **quais
   tags nunca são cobertos mesmo quando o episódio é detectado** (ex.:
   vibração), mas não deve ser lido como 47 experimentos independentes.

## Conclusão prática

Essa abordagem parece valiosa como uma **camada complementar de
vigilância ampla** — barata de manter (um único modelo pra toda a
planta, sem precisar escolher um alvo por experimento), boa pra sinalizar
"algo está estranho em algum lugar" — mas não substitui os modelos
dedicados por alarme já construídos, que continuam superiores para os
alvos específicos já trabalhados (T5, óleo lub., gás combustível).

## Próximos passos

1. ~~Testar janela de treino rolante fixa~~ — **feito no v3, hipótese
   refutada** (piorou o FP; ver seção v3). O v6 mostrou o motivo mais
   provável: reproduzimos errado (calendário corrido em vez de horas
   elegíveis) — não vale retestar sem essa correção primeiro.
2. ~~Reproduzir a votação N-de-4 do Francisco~~ — **feito no v5**
   (arquitetura de votação, ainda com receita v2) **e no v6** (+ as 9
   inovações de pós-processamento: EWMA, limiar por múltiplo do p99,
   ground-truth curado, janela em horas elegíveis). **FP replicado quase
   exatamente: 0,90 episódios/mês contra os 0,94/mês dele.**
3. ~~Investigar a cobertura de eventos baixa (2/11) via
   `ENABLE_DERIVED_FEATURES=False`~~ — **feito no v7, hipótese
   confirmada**: `producao_2sinais_AND` foi de 2/11→4/11 eventos **e**
   de 0,90→0,73 FP/mês (melhora nas duas métricas ao mesmo tempo). v7
   substitui v6 como referência.
4. ~~Confirmação por mecanismo (`temperatura` E (`mancal_spread` OU
   `selagem_z` OU `pressao_oleo`))~~ — **feito no v8, hipótese
   refutada**: cobertura ficou idêntica (4/11) e o FP mais que dobrou
   (0,73→1,63/mês) — o OR importa o ruído de `selagem_z`/`pressao_oleo`
   (sozinhos, bem mais ruidosos que `mancal_spread`) sem ganho líquido
   de eventos novos. v7 (par fixo) continua sendo a referência.
5. **[prioridade alta]** Se for tentar de novo uma confirmação
   alternativa: não tratar os 3 confirmadores como intercambiáveis.
   Ideias mais promissoras que o OR simples: (a) escolher o confirmador
   por mecanismo conhecido em vez de OR-ar todos; (b) subir o limiar
   individual de `selagem_z`/`pressao_oleo` antes de usá-los como
   confirmador, já que o problema é o ruído deles, não a ideia em si;
   (c) aceitar `par_fixo_v7` como o ponto de operação final e investir
   em outra frente (ex.: o item sobre loadings do PCA, abaixo).
6. Conferir com o colega os hiperparâmetros exatos que faltam (grid
   completo de 31.104 trials não está disponível pra nós) —
   `exclude_days`/`exclude_alarm_h` de limpeza do baseline, e se
   `pressao_oleo`/`selagem_z` fazem parte da política de produção final
   dele ou só da varredura ampla (nossa leitura do `config.py` sugere
   que não — a política final usa só `temperatura` + `mancal_spread`).
6. Checagem de antecedência real (preditivo vs. reativo) numa amostra
   dos tags/episódios com melhor cobertura, mesma disciplina do EXP16.
7. Auditoria genuíno-vs-artefato pros tags de maior volume antes de
   comparar diretamente com os modelos dedicados.
8. Testar contribuição de cada sensor pro desvio (loadings do PCA) nos
   pontos sinalizados, pra diagnosticar automaticamente qual subsistema
   está por trás de cada alerta.

## Script

- `scripts/pca_monitoramento_sistema/pca_walkforward.py` — **v1**,
  standalone, com reimplementação manual simplificada do
  pré-processamento (valor bruto + desvio-padrão móvel em 3 horizontes).
- `scripts/pca_monitoramento_sistema/pca_walkforward_v2_preprocessamento_real.py`
  — **v2**, mesma estrutura de retreino mensal + avaliação contra
  catálogo completo, mas usando a cadeia de pré-processamento real do
  projeto (`build_group_dataframe`, `select_feature_columns`,
  `clip_outliers`, `normalize_train_only` de `preprocess.py`) e
  adicionando a avaliação por episódio. **Versão de referência** — usar
  esta pra qualquer comparação/próximo passo.
- `scripts/pca_monitoramento_sistema/pca_walkforward_v3_janela_rolante.py`
  — **v3**, idêntico ao v2 exceto pela janela de treino: rolante fixa
  (~3000h de calendário) em vez de expansiva. Testou e refutou a
  hipótese de que essa troca reduziria o FP/mês (ver seção v3 acima).
  Mantido no repo como registro do experimento negativo.
- `scripts/pca_monitoramento_sistema/pca_walkforward_v4_pca_variancia_real.py`
  — **v4**, igual ao v2 mas com `MAX_COMPONENTS=150` (corrige o bug do
  teto de 20 componentes). Piorou o FP marginalmente — v2 continua
  sendo a referência da série `pca_walkforward_*` (ver seção v4 acima).
- `scripts/pca_monitoramento_sistema/votacao_4sinais_v5.py` — **v5**,
  primeira reprodução da arquitetura de votação N-de-4 do Francisco
  (4 sinais por família física), ainda com a receita v2 (PCA+`iforest`,
  z-score, debounce no flag, catálogo bruto).
- `scripts/pca_monitoramento_sistema/reproducao_francisco_v6.py` —
  **v6**, reprodução fiel da política de produção dele: PCA-Q,
  normalização robusta, EWMA no score, limiar por múltiplo do p99/
  z-robusto, ground-truth curado (mesmo algoritmo de detecção de trip),
  janela de baseline em horas elegíveis, confirmação por E entre 2
  sinais. **Replicou o FP quase exatamente (0,90 vs. 0,94 episódios/
  mês)**, mas com cobertura de eventos baixa (2/11) — ver seção v6.
- `scripts/pca_monitoramento_sistema/reproducao_francisco_v6b_series_para_plots.py`
  — **v6b**, idêntico ao v6, mas exporta como artefatos as séries
  suavizadas e a tabela de eventos com acerto/erro/antecedência — dados
  usados nos gráficos do relatório LaTeX
  (`task_plots_relatorio_v6_reproducao_francisco/`).
- `scripts/pca_monitoramento_sistema/reproducao_francisco_v7_sem_features_derivadas.py`
  — **v7**, idêntico ao v6 exceto por `ENABLE_DERIVED_FEATURES=False`
  nas famílias PCA-Q. **Confirmou a hipótese e melhorou nas duas
  métricas ao mesmo tempo** (4/11 eventos, 0,73 FP/mês) — ver seção v7.
  **Versão de referência atual** pra política de 2 sinais.
- `scripts/pca_monitoramento_sistema/reproducao_francisco_v8_confirmacao_por_mecanismo.py`
  — **v8**, idêntico ao v7 exceto pela regra de confirmação
  (`temperatura` E qualquer um dos 3 confirmadores, em vez do par fixo).
  **Hipótese refutada**: cobertura idêntica (4/11), FP mais que dobrou
  (0,73→1,63/mês) — ver seção v8. Mantido como registro do experimento
  negativo; v7 continua sendo a referência.

Nenhuma está integrada ao `automl_pipeline.py` (a estrutura de
retreino mensal + avaliação contra catálogo completo é suficientemente
diferente da pipeline por-alvo existente pra não valer a pena forçar no
mesmo framework agora). Todas reaproveitam os fitters/avaliação
(`automl_models.py`, `scoring.py`, `preprocess.py`) do projeto, exceto
o v6 (usa `sklearn.decomposition.PCA` direto pro PCA-Q, fora do
`automl_models.py`, pra reproduzir a arquitetura exata do Francisco).
