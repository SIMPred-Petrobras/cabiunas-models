# EXP18: EWMA no score antes do limiar (portado da pipeline do Francisco)

**Branch:** `AE_pca_monitoramento_sistema`. Ponto de partida: EXP10c
(`test_grupo_exp10c_portao_volatilidade.json`), o melhor resultado do
projeto até agora — `ocsvm`, grupo `TC382_T5_vibracao_mancais_multiescala`,
percentil 99,9, debounce=1 (ou seja, sem sustentação temporal nenhuma no
flag binário — os portões de rampa/volatilidade fazem o trabalho de
suprimir falso positivo hoje), 92,5% hit_rate / 0,35% FP.

## Motivação

Ao investigar a pipeline do colega Francisco (branch
`feat/pdm-deteccao-4sinais`, ver `docs/analise_pca_monitoramento_sistema.md`
e `ALARMES_POR_SENSOR_EFEITO_CASCATA.md`), identificamos uma diferença de
mecânica que nunca testamos no AutoML deste projeto: ele suaviza o score
contínuo do modelo com uma média móvel exponencial (EWMA, baseada em
**tempo real**, não contagem de amostras) **antes** de comparar com o
limiar — em vez de comparar o score bruto com o limiar e só suavizar
**depois**, no flag binário (que é o que `AUTOML_DEBOUNCE_GRID` faz aqui).

A diferença prática: um pico isolado de 1-2 pontos, que cruzaria o limiar
bruto instantaneamente, pode nunca cruzar o limiar depois de suavizado —
porque o peso desse pico na média é pequeno frente ao histórico recente.
Isso reduz falso positivo por ruído pontual sem depender só do debounce
pós-flag (que no EXP10c está desligado, `debounce=1`).

## O que mudou no código

- `src/cnn1d_ae/config.py`: dois campos novos,
  `ENABLE_SCORE_EWMA: bool = False` e `SCORE_EWMA_HALFLIFE: str = "30min"`.
  Default `False` — nenhum config existente é afetado.
- `src/cnn1d_ae/automl_pipeline.py` (`run_automl_group`): quando
  `ENABLE_SCORE_EWMA=true`, logo após o fit do modelo (antes do laço de
  percentis), `train_err`/`all_err` passam por
  `pd.Series(...).ewm(halflife=SCORE_EWMA_HALFLIFE, times=índice).mean()`
  — o índice de tempo real de `df_normal_fit`/`all_index`, não a
  contagem de amostras (importante porque `df_normal` tem buracos:
  janelas de alarme e período desligado são excluídas do treino).
  Suaviza uma vez só por modelo ajustado, antes do laço de
  percentil/debounce reaproveitar o mesmo score suavizado — mesmo
  princípio de cache que a pipeline dele usa.

Testes existentes (`tests/`) continuam passando sem alteração — a
mudança é 100% opt-in.

## Config do experimento

`configs/calibracao_v4_eq/test_grupo_exp18_ewma_score.json` — idêntica
ao EXP10c, exceto:
- `ENABLE_SCORE_EWMA: true`, `SCORE_EWMA_HALFLIFE: "30min"`.
- `AUTOML_THRESHOLD_PERCENTILES` ampliado para `[99.0, 99.5, 99.9, 99.97]`
  (era só `[99.9]`) — suavizar o score muda a escala/distribuição dele,
  então o percentil antigo não é necessariamente o melhor ponto novo; o
  próprio AutoML re-seleciona via `composite_score`.

Tudo mais (modelo `ocsvm`, grupo de sensores, portões de rampa/
volatilidade, máscara operacional, split OOS em 2025-07-01) idêntico ao
EXP10c, para isolar o efeito da suavização.

## Como reproduzir

```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp18_ewma_score.json
```

## Resultado

**Task ClearML:** `2d772c324e29416da11fba9827ce1876`.

Grade completa (4 percentis testados, `AUTOML_DEBOUNCE_GRID=[1]` fixo):

| percentil | hit_rate | normal_alert_rate | composite_score |
|---|---|---|---|
| 99,00 | 90,0% | 3,59% | 0,9658 (melhor) |
| 99,50 | 85,0% | 1,91% | 0,9498 |
| **99,90 (mesmo ponto do EXP10c)** | **45,0%** | **0,38%** | 0,8167 |
| 99,97 | 40,0% | 0,19% | 0,8000 |

Seed-sweep do trial vencedor (percentil 99,0, 5 seeds): hit_rate
90,0–92,5% (média 92,0%), normal_alert_rate 1,66–3,59% (média 2,07%).

**Comparação direta com a referência EXP10c** (percentil 99,9, sem EWMA):
92,5% hit_rate (37/40) / 0,35% FP.

| | EXP10c (referência) | EXP18 (melhor trial) | EXP18 (mesmo percentil 99,9) |
|---|---|---|---|
| hit_rate | 92,5% | 90,0% (média seeds 92,0%) | **45,0%** |
| normal_alert_rate | 0,35% | 3,59% (média seeds 2,07%) | 0,38% |

**Conclusão: piora, não ajuste favorável.** Dois efeitos, e ambos ruins:

1. **No mesmo percentil (99,9) que o EXP10c usa**, suavizar o score antes
   do limiar derruba o hit_rate de 92,5% para 45,0% — a EWMA de 30min
   borra os picos curtos de anomalia genuína (o próprio evento que
   queremos detectar) tanto quanto borra ruído pontual, porque a escala
   temporal do halflife é maior que a duração típica do pico real neste
   sinal de vibração. O score suavizado simplesmente não sobe o
   suficiente para cruzar um limiar calibrado para o score bruto.
2. **No percentil que o AutoML re-escolhe como melhor (99,0, mais baixo,
   compensando a compressão de escala)**, o hit_rate volta a ficar perto
   do de referência (90–92,5%), mas o normal_alert_rate piora de 6x a 10x
   (0,35% → 2,07–3,59%) — abaixar o limiar para recuperar sensibilidade
   também deixa passar muito mais ruído de fundo suavizado.

Ou seja: não existe, nesta grade, um ponto de operação da EWMA que
supere simultaneamente o hit_rate **e** o FP da referência — é sempre
um dos dois pior, nunca uma melhoria líquida. Isso é consistente com o
diagnóstico já registrado em `ALARMES_POR_SENSOR_EFEITO_CASCATA.md` e em
`docs/analise_pca_monitoramento_sistema.md`: no EXP10c, a supressão de
ruído já é feita pelos portões (máscara operacional + portão de rampa +
portão de volatilidade) atuando sobre o sinal **bruto**, e não depende de
suavização temporal do score. Adicionar suavização do score em cima
disso não soma — compete com o próprio sinal que o modelo está tentando
capturar, porque aqui a "anomalia real" e o "ruído" têm escalas de tempo
parecidas (minutos), diferente do cenário do Francisco onde a EWMA ajuda
(lá o sinal de interesse evolui mais lentamente que o ruído pontual).

**Decisão: EWMA-antes-do-limiar (`ENABLE_SCORE_EWMA`) não é recomendada
para o grupo `TC382_T5_vibracao_mancais_multiescala`/EXP10c.** O código
fica no pipeline como opção opt-in (default `False`, não afeta nenhum
config existente) documentada e testada, para o caso de ser útil em
outro sensor/grupo com dinâmica mais lenta, mas o EXP10c permanece a
referência do projeto sem essa mudança.
