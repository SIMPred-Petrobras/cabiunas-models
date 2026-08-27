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
   refutada** (piorou o FP em vez de melhorar; ver seção v3 acima).
   Manter a janela **expansiva** (v2) como referência daqui pra frente.
2. Grid de threshold/debounce (aqui fixado em percentil 99/debounce 6
   sem otimização) — agora é o candidato mais provável pra explicar o
   gap de FP/mês com o colega (0,94 vs. 3,35 no `iforest` expansivo);
   vale testar percentis mais altos (99,5/99,9) antes de mexer em
   qualquer outra coisa.
3. Conferir com o colega a definição exata de "falso positivo por mês"
   dele (mesmo critério de agrupamento em episódios? mesma janela de
   exclusão ao redor de alarme? mesmo conjunto de sensores?) — a
   comparação só é justa se a métrica for calculada do mesmo jeito dos
   dois lados.
4. Checagem de antecedência real (preditivo vs. reativo) numa amostra
   dos tags/episódios com melhor cobertura, mesma disciplina do EXP16.
5. Auditoria genuíno-vs-artefato pros tags de maior volume antes de
   comparar diretamente com os modelos dedicados.
6. Testar contribuição de cada sensor pro desvio (loadings do PCA) nos
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
  (~3000h) em vez de expansiva. Testou e refutou a hipótese de que essa
  troca reduziria o FP/mês (ver seção v3 acima). Mantido no repo como
  registro do experimento negativo, não como versão de referência.

Nenhuma está integrada ao `automl_pipeline.py` (a estrutura de
retreino mensal + avaliação contra catálogo completo é suficientemente
diferente da pipeline por-alvo existente pra não valer a pena forçar no
mesmo framework agora). Ambas reaproveitam os fitters/avaliação
(`automl_models.py`, `scoring.py`, `preprocess.py`) do projeto.
