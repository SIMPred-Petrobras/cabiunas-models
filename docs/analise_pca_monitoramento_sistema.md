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

## Interpretação

- **Um único modelo, sem nenhuma calibração por alarme, detecta sinal em
  praticamente toda categoria do catálogo** (60-85% em vários tags de
  pressão/temperatura), com FP de só 3,2%. Valida a ideia central da
  abordagem: uma "vigia geral" da planta é viável.
- **Não supera os modelos dedicados já construídos**: T5 (EXP10c)
  92,5% dedicado vs. 62,0% aqui; gás combustível (EXP17a) 81,5%
  dedicado vs. 60,2% aqui. Esperado — um generalista não bate um
  especialista calibrado especificamente pro alvo.
- **Vibração isolada continua mal servida** (8-17% em todos os 10
  canais) — reforça o achado recorrente no projeto de que vibração não é
  bem explicada pelo resto do sistema (é mais útil como *feature* de
  contexto do que como *alvo* de predição direto).
- **Os 3 trips já identificados como artefato de desligamento**
  (`PALL_6240340`, `TALL_6240325`, `TAL_6240325`) continuam com
  `hit_rate` baixo (4-10%) mesmo nesse modelo generalista de todo o
  sistema — consistente com a conclusão anterior de que não há
  precursor real ali, em nenhuma combinação de sensores disponível.

### ⚠️ Ressalva metodológica importante

Diferente do EXP16 (onde cada evento foi auditado manualmente pra
separar genuíno de artefato de desligamento), aqui o `hit_rate` é
calculado contra **todos** os eventos `ACT/CFN` do catálogo, sem esse
filtro. Os números desta tabela **não são diretamente comparáveis** aos
`hit_rate`/cobertura genuína reportados nos EXP10c/16/17 sem essa
mesma auditoria. Também não foi feita a checagem de antecedência real
(preditivo vs. reativo) que a lição do EXP16 exige antes de confiar
cegamente num `hit_rate` agregado.

## Conclusão prática

Essa abordagem parece valiosa como uma **camada complementar de
vigilância ampla** — barata de manter (um único modelo pra toda a
planta, sem precisar escolher um alvo por experimento), boa pra sinalizar
"algo está estranho em algum lugar" — mas não substitui os modelos
dedicados por alarme já construídos, que continuam superiores para os
alvos específicos já trabalhados (T5, óleo lub., gás combustível).

## Próximos passos

1. Checagem de antecedência real (preditivo vs. reativo) numa amostra
   dos tags com melhor `hit_rate`, mesma disciplina do EXP16.
2. Auditoria genuíno-vs-artefato pros tags de maior volume antes de
   comparar diretamente com os modelos dedicados.
3. Grid de threshold/debounce (aqui fixado em percentil 99/debounce 6
   sem otimização) — pode haver ganho fácil.
4. Considerar reduzir a janela de treino de expansiva pra rolante (ex:
   últimos 6-12 meses) em vez de todo o histórico, pra medir se isso
   muda o resultado (a ideia original mencionava "todo mundo", mas vale
   testar a alternativa).
5. Testar contribuição de cada sensor pro desvio (loadings do PCA) nos
   pontos sinalizados, pra diagnosticar automaticamente qual subsistema
   está por trás de cada alerta.

## Script

`scripts/pca_monitoramento_sistema/pca_walkforward.py` — standalone,
não integrado ao `automl_pipeline.py` (a estrutura de retreino mensal +
avaliação contra catálogo completo é suficientemente diferente da
pipeline por-alvo existente pra não valer a pena forçar no mesmo
framework agora). Reaproveita os fitters/avaliação (`automl_models.py`,
`scoring.py`, `preprocess.py`) do projeto.
