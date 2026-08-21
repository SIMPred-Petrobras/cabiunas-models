# Detector de anomalias do TC-330.03A — 4 sinais (`src/cabiunas_pdm`)

Contribuição paralela ao pipeline `cnn1d_ae` deste repositório. Mesmo equipamento e
mesmo ClearML, abordagem diferente: em vez de um autoencoder convolucional único,
**quatro sinais em paralelo com votação**, busca em grade sobre a configuração e
avaliação walk-forward estritamente causal.

O objetivo é **antecipar paradas não programadas com o mínimo de falso positivo**.
São 8 falhas em 16 meses, de 3 mecanismos — não cabe classificador supervisionado,
então o método é modelar o comportamento normal e medir o afastamento.

---

## Como rodar (leia antes de digitar `uv run`)

> **Cuidado.** `uv run` sem `--no-sync` sincroniza o ambiente com o
> `pyproject.toml` da raiz, que é do `cnn1d_ae`. Medido em 21/08/2026:
> `uv sync --dry-run` remove **57 pacotes** e instala 32 — troca scikit-learn por
> TensorFlow e o detector para de importar. Use uma das duas formas abaixo.

```bash
# forma 1 — direta (recomendada)
.venv/bin/python scripts/automl_clearml.py --mode policy_sweep --remote --queue default

# forma 2 — via uv, sem sincronizar
uv run --no-sync python scripts/automl_clearml.py --mode policy_sweep --remote --queue default
```

Ambiente novo, do zero:

```bash
uv venv
uv pip install --python .venv/bin/python \
    "pandas>=2.2" "pyarrow>=16" "scikit-learn>=1.5" "scipy>=1.13" \
    "matplotlib>=3.9" "joblib>=1.4" "clearml>=1.16" pytest
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
```

Python 3.12+. Não usa TensorFlow — os modelos são todos scikit-learn.

Variáveis de ambiente opcionais: `CABIUNAS_PDM_ROOT` (raiz do checkout, se não for a
deduzida do próprio arquivo) e `CABIUNAS_RAW` (pasta dos brutos do PI, só para
ingestão local).

---

## Qual comando roda qual pipeline

| Pipeline | Comando | O que faz | Custo |
|---|---|---|---|
| **Varredura de política** (estágio 1) | `--mode policy_sweep --remote --queue default` | 15 políticas de baseline × 2 modelos, grade de 2 min, fatores baratos em poucos centros. Ordena as políticas. | 4.320 trials, 60 ajustes, **~1,3 h** |
| **Varredura de arquitetura** (estágio 1b) | `--mode arch_sweep --remote --queue default` | As 9 arquiteturas × 3 políticas de baseline. Cruza com três e não com uma porque a arquitetura vencedora depende do tamanho da janela. | 3.888 trials, 54 ajustes, **~1,0 h** |
| **Refino** (estágio 2) | `--mode policy_fine --policy-hours 2000 3000 --remote --queue default` | Grade barata cheia em 30 s + 2 min, em torno das políticas vencedoras. | 31.104 trials, **~7 h** |
| Busca histórica | `--mode full --remote --queue default` | A grade original (task `61b6b109…`). Mantida para reprodução. | 31.104 trials |
| Sanidade local | `--mode quick --local --local-csv <csv> --limit 6` | 6 configs sem rede nem ClearML. Use para validar uma alteração antes de gastar worker. | segundos |
| Controle de modelo único | `--mode fp_first --single-model --remote` | 1 sinal em vez de 4, para medir o ganho da votação. | 2.304 trials |

`--local` sem `--local-csv` ainda busca o dataset no ClearML e fica em backoff se a
rede não responder — sempre passe o CSV em cache nos testes locais.

Artefatos publicados na task: `automl_results` (a grade inteira) e `best_trial`.

---

## Dados

Tudo do ClearML, nenhum arquivo local: dataset `68b25f9db0b8471a90b8100800d26e9a`
("Cabiunas brutos 2025-2026 alarmes mapeados", projeto `TesteMLCab`) — série de 30 s,
01/01/2025 a 30/04/2026, 1.396.800 amostras, 39 colunas, mais a planilha de alarmes.
O primeiro uso grava um cache Parquet em `data/interim/`, fora do Git.

**Rótulo de falha derivado dos dados**, porque não existe ordem de serviço na base:
transições `RUNNING_A` 1→0 (135) → parada durou ≥ 2 h (36; as outras 97 são quedas de
telemetria de ~90 s) → havia alarme de nível em [−1 h, +30 min] (9 trips) →
agrupamento em 24 h = **8 eventos**. Os alarmes que disparam com a máquina **já
parada** (94% dos de nível) são descartados: são eco da parada, não precursores.

---

## Mapa do código

```
scripts/automl_clearml.py    a busca. AUTOCONTIDO de propósito: o worker do
                             ClearML recebe só este arquivo, sem o pacote
                             instalado. Toda a lógica (limpeza, operabilidade,
                             scorers, política de baseline, avaliação) está aqui.

src/cabiunas_pdm/
  replay.py                  reexecuta UMA configuração devolvendo as SÉRIES
                             (score, limiar do mês, alerta, episódios), não só
                             as métricas. Importa o script da busca por caminho
                             para o desenho refletir exatamente o mesmo cálculo,
                             e confere o resultado contra o avaliador.
  viz.py                     46 funções de plot na gramática visual do Transpetro
                             (série azul, anomalias em pontos vermelhos, falha em
                             laranja tracejado).
  config.py                  tags, faixas físicas, limiar de operabilidade.
  cleaning.py, scoring.py    limpeza e scorers reutilizáveis.
  sources.py, dataset.py,    fase de ingestão local, anterior à migração para o
  detector.py, clearml_io.py ClearML. Não são usados pelo resultado atual.
  operability.py

tests/test_baseline_policy.py  33 testes da política de baseline e do limiar.
```

### Os 4 sinais

| Sinal | Tipo | Entrada |
|---|---|---|
| `temperatura` | autoencoder multivariado (MLP) | 14 sensores |
| `pressao_oleo` | autoencoder multivariado | 12 sensores |
| `mancal_spread` | z robusto univariado | `TI_0305` contra os 3 mancais irmãos |
| `selagem_z` | z robusto univariado | `PDIT_0305` |

Os dois univariados existem por uma medição: o `PDIT_0305` respondia por 59% a 95%
do erro antes da falha de selagem e mesmo assim não alarmava, porque o score da
família é a média sobre 12 sensores. Isolado, virou o sinal mais específico dos
quatro. O alarme exige **2 a 4 sinais simultaneamente** acima do limiar — sem isso a
temperatura sozinha alarmaria 47% do tempo.

Vibração ficou fora: o normal dela deriva de ~10 para ~25 ao longo de 2025/26 sem
falha associada.

### As 9 arquiteturas

Cada uma das duas famílias multivariadas pode ser modelada por qualquer destas. O
critério de inclusão foi **escalar**: o baseline vai de 9 mil a 460 mil amostras e o
ajuste é a etapa cara da busca, então OneClassSVM exato, LocalOutlierFactor e
KernelPCA ficaram fora de propósito — são quadráticos em amostras e não terminam.

| Arquitetura | O que mede | Ajuste (45k×14) |
|---|---|---|
| `pca` | erro de reconstrução (estatística Q/SPE) | 1,09 s |
| `pca_t2` | T² de Hotelling nos escores latentes | 0,10 s |
| `mahal` | Mahalanobis com covariância encolhida (Ledoit-Wolf) | **0,06 s** |
| `gmm` | mistura de 3 gaussianas, log-verossimilhança negativa | 1,26 s |
| `iforest` | isolamento por particionamento aleatório | 0,83 s |
| `ocsvm_sgd` | fronteira de uma classe, SGD + kernel de Nyström | 0,40 s |
| `ae` | autoencoder denso, gargalo em n/4 | 2,11 s |
| `ae_deep` | 5 camadas, gargalo em n/6 | 2,53 s |
| `ae_wide` | camada larga (2n), gargalo em n/3 | 2,17 s |

`pca` e `pca_t2` são complementares e é por isso que os dois existem: Q vê o que sai
do subespaço normal, T² vê o que anda longe dentro dele — o par clássico do
monitoramento de processo. `mahal` é a referência honesta contra a qual um
autoencoder precisa provar que vale a complexidade: custa 35× menos.

Medido nos 8 eventos reais com janela móvel de 3.000 h e fatores baratos fixos
(portanto **preview, não veredito** — cada arquitetura tem seu limiar ótimo):
`pca` e `ae` empatam em 6/8, o `gmm` antecipa 27 dos 132 episódios de alarme contra
12 do `pca`, e o `pca_t2` detecta 1 de 8 mas com 46,7 h de antecedência e 0,9 h/mês
de alarme falso.

`gmm`, `ocsvm_sgd` e `iforest` ajustam em subamostra determinística quando o baseline
passa de 60–120 mil pontos (`FamilyScorer.MAX_FIT_SAMPLES`). A subamostra é sempre do
baseline, nunca do teste, então não há vazamento.

---

## Política de baseline

Decide qual trecho de histórico treina o modelo de cada mês. Substituiu o par
`{movel, acumulativo}`, que era um caso degenerado dela:

```python
BaselinePolicy(
    window_hours = 3000,    # horas de operação ELEGÍVEL (pós-exclusões)
    window_days  = None,    # dias de calendário (para comparar com "N meses")
    max_age_days = 180,     # teto de idade do dado
    min_hours    = 100,     # piso; abaixo dele o retreino é inválido
)
```

O mais restritivo vence. Acumulativo é `BaselinePolicy()` — tudo `None`.

**Por que horas de operação e não meses de calendário.** Um mês entrega de 133,8 h a
691,1 h de operação elegível (5,17×), e a escassez é **endógena às falhas**:
`corr(falhas nos 45 dias anteriores, horas elegíveis nos 30 dias) = −0,61`. Contar em
meses encolhe o baseline exatamente nos retreinos que vêm depois de um reparo — o
momento em que o re-baseline mais importa.

**Por que existe teto de idade.** 400 h elegíveis exigem recuar de 16,8 a 64,0 dias
no calendário dependendo do retreino (3,8×). Sazonalidade e reparo agem em tempo de
calendário, não em hora de máquina.

**Cuidado ao interpretar janelas grandes nesta série:** com 3.000 h, 7 dos 15
retreinos ficam **truncados** (em fev/2025 a janela tem 10% do pedido), então no
primeiro semestre o braço "janela longa" roda acumulativo disfarçado. A coluna
`retreinos_truncados` reporta isso em toda configuração.

### Protocolo temporal

Retreino no **dia 1º de cada mês**, sempre só com dado anterior àquele instante
(`indices_elegiveis <= start`). 15 retreinos, fev/2025 a abr/2026. Falha não dispara
retreino — é cadência de calendário. Saem do treino: máquina parada, 2 h pós-partida,
±1 h de cada ativação de alarme (3.757 delas) e, se `exclude_days > 0`, os N dias
antes de cada falha conhecida.

Do score ao alarme: EWMA → limiar → persistência mínima por sinal → votação →
episódios agrupados em 2 h → duração mínima do alerta.

### Limiar

`threshold_kind="percentil"` é o histórico. `threshold_kind="k_maiores"` usa o
k-ésimo maior score do baseline, e existe por uma medição: em janela curta os
percentis de 99,9 a 99,995 são o **mesmo limiar** (o máximo fica só 1,79% acima do
p99,9), então 4 dos 6 níveis da grade antiga eram um só. O k mantém o orçamento de
excedências fixo em qualquer janela, o que torna políticas de tamanhos diferentes
comparáveis. Os dois entram como fator da busca.

---

## Métricas

Além de detecção, antecedência e falso positivo por **mês de operação**:

| Métrica | Para quê |
|---|---|
| `fp_horas_por_mes` | um episódio de 10 min e um de uma semana contam igual na contagem; não são a mesma coisa no plantão |
| `fp_por_mes_liquido` | desconta episódios que coincidem com alarme ativo da instrumentação (±3 h) |
| `alarmes_antecipados` | contra os **132 episódios de alarme em operação** — a única métrica do projeto com poder estatístico razoável |
| `maior_silencio_h` | maior intervalo, em horas de operação, sem emitir nenhum alerta |
| `trimestres_com_alerta` | cobertura temporal |
| `det_1a_metade` / `det_2a_metade` | detecção por metade da série |
| `retreinos_truncados` / `baseline_horas_medio` | diagnóstico da política |

**Por que a liveness não usa rótulo de evento.** Dentro do teto de 1 FP/mês, a
máscara "detectou algo na 2ª metade" é bit-a-bit idêntica a "detectou 26/02/2026":
zero das 11.395 configs aprovadas pegam 04/11/2025 ou 09/12/2025. Removendo aquele
único evento, os sobreviventes de um critério de distribuição temporal caem de 80
para zero. Qualquer critério baseado nos 8 rótulos é um teste de um evento — por isso
o silêncio é medido na própria série de alertas.

`aprovado` = respeita o teto de FP. `vivo` = não está cego. O `select_best` escolhe
entre os dois, mas **toda** configuração vai para o artefato com as métricas ao lado:
nada é descartado, só a escolha automática deixa de premiar o detector que emudece.

---

## Notebooks

| Notebook | Pergunta |
|---|---|
| `01_serie_e_falhas` | como é a série e de onde saem as 8 falhas |
| `02_resultados_automl` | qual ponto de operação escolher na grade |
| `03_anomalias_na_serie` | o detector em funcionamento, com zoom em cada falha |
| `04_comparacao` | 4 sinais em votação × modelo único |
| `06_serie_completa_anomalias` | visão dos 16 meses inteiros em duas figuras |

Os resultados estão nas saídas das células — dá para ler sem executar.

---

## Testes

```bash
.venv/bin/python -m pytest tests/test_baseline_policy.py -q
```

33 testes: janela maior que o histórico, corte por calendário, teto de idade vencendo
a janela em horas, piso de validade, ausência de vazamento de futuro, e a
equivalência **exata** do limiar cacheado com `np.nanpercentile`.

Detalhe que custou duas iterações e está registrado no código: reproduzir o numpy bit
a bit exige `(p/100)*(n-1)` e não `(n-1)*p/100`, além da troca de expressão que o
numpy faz em `frac = 0,5`. Sem isso o limiar diverge na 12ª casa decimal e os
resultados das buscas anteriores deixam de reproduzir.

---

## O que não está no Git

- **dados** (`data/`) — tudo via ClearML
- **grade de resultados** (`automl_out/`) — está nos artefatos da task
- **relatórios e figuras** (`reports/`) — o resultado se lê nos notebooks

---

## Pendências

1. Notebook 07: comparação das políticas de baseline, **fronteira contra fronteira**.
   Comparar em ponto fixo é inválido — uma janela maior desloca a distribuição do
   score, então o mesmo percentil é outro ponto de operação.
2. Warm start com 2024, necessário para testar janelas ≥ 1.500 h sem truncamento. O
   dataset `58a4c230…` ("Cabiunas consolidado 2022-2026", 39 colunas) serve, com três
   obstáculos medidos: **fuso misto** (2024 em UTC, 2022/2025/2026 em UTC-3), nomes de
   coluna **sem o prefixo** `954005_624_`, e **2023 não existe**.
3. O veto de sensor congelado age na pontuação, não na elegibilidade do baseline:
   5,41% do baseline da família de pressão/óleo está sob veto, variando de 0% a 11,4%
   entre retreinos.
4. A janela de detecção de 48 h é convenção não validada, e é medida em calendário
   enquanto a de treino passou a ser medida em horas de operação.
5. `src/cabiunas_pdm/` ainda carrega os módulos da ingestão local anteriores à
   migração para ClearML — revisar o que sobrevive.
