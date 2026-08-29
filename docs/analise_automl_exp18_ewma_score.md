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

_(preenchido após a task terminar)_
