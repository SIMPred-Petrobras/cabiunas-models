# EXP14 — Threshold conformal (branch `exp14_conformal_threshold`)

## Motivação

Ao longo do EXP13 (ver `docs/analise_cnn1dae_exp13.md`), a recalibração
de threshold via `THRESH_MODE="robust_mad"` (mediana + `k`×1,4826×MAD)
se mostrou frágil: `k` é um multiplicador sem garantia estatística, e a
tradução "quero este threshold" → "uso este `k`" não é estável entre
retreinos, porque a mediana/MAD do erro de treino mudam de modelo pra
modelo. Isso custou 2 rodadas remotas de tentativa-e-erro só na
recalibração final do EXP13 (`k=6,0` mirou threshold 0,265 e saiu
0,2294; precisou de mais uma rodada com `k=7,0` pra acertar 0,260).

Levantamento de literatura (pedido do usuário) sobre redução de falso
positivo em detecção de anomalia industrial apontou **predição
conformal** como a técnica mais direta pra essa fragilidade específica:
em vez de multiplicar por `k` e torcer pra cair no percentil certo, o
threshold é calculado como um quantil empírico (com correção de amostra
finita) sobre um conjunto de calibração dedicado, dando uma garantia
formal de FPR marginal sob a suposição de exchangeability — sem
assumir forma de distribuição.

**Decisão de escopo (usuário):** implementar numa branch nova
(`exp14_conformal_threshold`, criada a partir de `AE_novo_legado`) para
não arriscar mudança brusca no pipeline já validado do EXP13. Todas as
mudanças são retrocompatíveis por padrão (`CALIBRATION_FRAC=0.0`
reproduz o comportamento anterior byte-a-byte — ver testes).

## Como funciona (split conformal)

1. **Score de não-conformidade** = o mesmo MAE de reconstrução por
   sequência já usado hoje (`mae_seq`/`train_mae_thresh`). Nenhuma
   mudança aqui.
2. **Split de calibração** — uma 3ª fatia temporal, além de
   treino/val, nunca vista pelo fit nem pelo early stopping. Fica com
   o trecho **mais recente** (mais próximo do corte OOS), por ser o
   que mais precisa se parecer com o período avaliado.
3. **Threshold = quantil empírico com correção de amostra finita:**
   com `n` pontos de calibração e FPR alvo `α`:
   ```
   threshold = calibration_scores_ordenados[ ceil((n+1) × (1-α)) ]
   ```
4. **Garantia:** para qualquer ponto novo "trocável" com o calibration
   set (mesmo regime operacional), `P(score > threshold) <= α`,
   independente da forma da distribuição do erro.
5. **Ressalva:** a garantia só vale sob exchangeability — se o regime
   operacional mudar, ela quebra. Por isso os gates (`operational_state`,
   load gate, volatility gate) continuam necessários como complemento,
   não substituição.

## O que foi implementado

- **`src/cnn1d_ae/sequences.py`** — nova função `train_val_calib_split`
  (3-vias: `train | val | calib`, `calib` por último/mais recente).
  Quando `calib_frac=0.0`, reproduz `train_val_split` byte-a-byte
  (testado — `test_train_val_calib_split_zero_calib_matches_train_val_split`).
  `train_val_split` original não foi tocada (ainda usada por
  `run_one_sensor`, que não recebeu a mudança nesta rodada — mesmo
  padrão do seed-sweep no EXP13, que também ficou restrito a
  `run_one_group`).
- **`src/cnn1d_ae/config.py`** — novo campo `CALIBRATION_FRAC: float
  = 0.0` (default preserva comportamento de todo config existente,
  inclusive os do EXP13 em `AE_novo_legado`).
- **`src/cnn1d_ae/scoring.py`** — novo `THRESH_MODE="conformal"` em
  `compute_threshold`, reaproveitando `TARGET_ANOMALY_RATE` já
  existente como `α` (sem precisar de campo novo). Levanta `ValueError`
  se o conjunto de calibração vier vazio.
- **`src/cnn1d_ae/pipeline.py` (`run_one_group` e
  `_refit_cnn1dae_with_seed`, incluindo o seed-sweep)** — como o MAE do
  `target_sensor` já é calculado sobre `x_train_full` inteiro
  (train+val+calib juntos), a fatia de calibração é só um recorte por
  índice (`train_mae_thresh[-n_calib:]`) quando `THRESH_MODE=="conformal"`
  — **sem inferência extra**, sem custo de memória adicional. Para
  qualquer outro `THRESH_MODE`, o comportamento é idêntico ao de antes
  (usa `train_mae_thresh` inteiro, ignora `CALIBRATION_FRAC` mesmo que
  >0) — confirmado por smoke test dedicado (`smoke_test_conformal.py`,
  3 cenários: baseline sem calibração, conformal com calibração, e
  modo antigo com `CALIBRATION_FRAC>0` que deve ignorá-la).
- **Testes novos** em `tests/test_split_and_threshold.py`: split 3-vias
  (ordem temporal, tamanhos, equivalência com `calib_frac=0`) e
  threshold conformal (quantil manual vs. função, e erro em calibração
  vazia). Suite completa: 19/19 passam.
- **Config novo:** `configs/calibracao_v4_eq/test_grupo_exp14_conformal.json`
  — cópia do candidato final do EXP13 (`k=7,0`, gates recalibrados,
  `MAX_TRIALS=40`, `SEED_SWEEP_N=4`), trocando `THRESH_MODE` para
  `"conformal"` e adicionando `CALIBRATION_FRAC=0,1` (reserva 10% do
  período normal pré-OOS, além do `VAL_FRAC=0,1` já existente).
  `TARGET_ANOMALY_RATE=0,003` reaproveitado como `α`.

## Validação offline feita (antes de qualquer submissão remota)

- `python3 -m compileall` limpo.
- `pytest tests/` — 19/19 (15 pré-existentes + 4 novos).
- Smoke test sintético (`smoke_test_conformal.py`, dados fake,
  `MODE=local`/`RUN_REMOTE=false`) rodou `run_one_group` de ponta a
  ponta em 3 configurações: (1) `CALIBRATION_FRAC=0,0` +
  `robust_mad` (baseline, precisa ficar idêntico ao comportamento de
  sempre), (2) `CALIBRATION_FRAC=0,15` + `conformal` (threshold
  calculado só sobre a fatia de calibração, `α=0,05` → `normal_alert_rate`
  do seed-sweep saiu em 4,25%-4,87%, **bem próximo do alvo de 5%** —
  primeiro sinal de que a garantia de FPR se comporta como esperado
  mesmo em dado sintético pequeno), (3) `CALIBRATION_FRAC=0,15` +
  `robust_mad` (modo antigo deve ignorar a fatia de calibração e usar a
  distribuição inteira — threshold saiu diferente do cenário 2, como
  esperado).

## Achado crítico (primeira submissão remota): calibração contaminada por parada longa

Primeira submissão remota (task `130ed7b5cf9447889443800f8d624b1a`,
commit `f7d81f3`, `α=0,003`) deu um resultado claramente errado:
threshold saiu em **0,6318** — bem mais alto que qualquer candidato
anterior (0,135-0,261 no EXP13) — e o modelo principal teve **hit_rate
0% (0/40)**, enquanto os modelos do seed-sweep (retreinados) detectaram
normalmente (22,5%-50%). O histograma de MAE de treino (`train_mae_hist.png`)
mostrou a linha do threshold muito além do que era visualmente
perceptível na distribuição — sinal de outlier extremo raro na cauda.

**Investigação:** baixando `sequence_scores_all.csv`/`point_anomalies_all.csv`
da task real, localizamos os valores mais altos de MAE do canal-alvo
dentro da janela de calibração (últimos ~10% do período normal
pré-OOS) concentrados em **2025-06-05, madrugada** (MAE até 0,643).
Checando `operational_state` nesse intervalo: **100% `off_longo`** —
uma parada longa da turbina. O autoencoder, treinado majoritariamente
com a turbina ligada, reconstrói mal esse regime (não é anomalia real).

**Causa raiz:** a fatia de calibração é definida por corte temporal
(últimos `CALIBRATION_FRAC` do período normal pré-OOS), sem filtrar por
`operational_state`. A máscara operacional só zera `is_anom_point` na
hora da *avaliação* — nunca filtrou o que entra no treino ou na
calibração. Com `α=0,003` muito baixo, o quantil conformal é literalmente
uma estatística de ordem extrema (~21 pontos entre 7.008 no calibration
set desta task); um único episódio "off" com dezenas de janelas
sobrepostas (mesma parada, `STRIDE=15`/`TIME_STEPS=60`) foi suficiente
pra dominar esses ~21 pontos e inflar o threshold muito além do que a
operação normal jamais atinge. O `robust_mad` não sofre tanto com isso
porque mediana+k×MAD é dominado pelo grosso da distribuição, não pela
cauda extrema.

**Fix (mesma branch, sem nova branch):** filtrar a fatia de calibração
por `operational_state=="on"` antes de calcular o quantil conformal —
alinha o que o threshold "vê" com o que de fato é avaliado (pontos
off/transiente nunca contam como FP).
- `sequences.py`: nova `sequence_all_true(mask_1d, time_steps, stride)`
  — vetorizada (cumsum), retorna por sequência se TODOS os pontos da
  janela são `True`, com indexação idêntica a `make_sequences` (garante
  alinhamento 1:1).
- `pipeline.py` (`run_one_group`): captura `normal_index` (índice de
  `df_normal_z`, só quando `THRESH_MODE=="conformal"` e
  `CALIBRATION_FRAC>0`, pra não pagar custo em nenhum outro caso) antes
  de descartar o dataframe; computa `normal_on_mask` alinhando
  `operational_state` a esse índice; usa `sequence_all_true` pra saber
  quais sequências da fatia de calibração são 100% `"on"`, filtrando
  antes de `compute_threshold`. Mesmo tratamento em
  `_refit_cnn1dae_with_seed` (seed-sweep), passando a máscara já
  calculada (não recalcula por seed).
- **Escopo do fix:** só filtra a *calibração* (o cálculo do threshold).
  `x_train`/`x_val` continuam incluindo períodos off/transiente sem
  mudança — não é o escopo desta correção (evita mudança brusca maior).

**Validação do fix:** smoke test dedicado injetando uma parada
sintética (`RUNNING_A=0`, sensor-alvo cai pra "ambiente") bem dentro da
fatia de calibração. Threshold resultante **com contaminação + fix**
ficou **idêntico** (`1,2654956579208374`) ao cenário limpo sem nenhuma
parada injetada — confirma que o filtro isola corretamente o período
off da calibração, sem afetar o resultado quando não há contaminação.

## Vantagem prática esperada

Como o threshold deixa de ser um hiperparâmetro (`k`) a ser adivinhado
por tentativa-e-erro remota, a expectativa é que **uma única submissão
remota** já baste pra validar o candidato — ao contrário do EXP13, que
precisou de 2 rodadas só na recalibração final.

## Pendências / próximos passos

- Resubmeter `test_grupo_exp14_conformal.json` (com o fix de filtragem
  por `operational_state`) e comparar hit_rate/cobertura genuína/FP
  contra o candidato final do EXP13 (`k=7,0`: cobertura genuína
  60,0%/FP 0,17%) e contra o AutoML EXP10c (80,0%/0,35%).
- Repetir a investigação caso a caso (genuíno vs. suspeito/artefato de
  janela) sobre o resultado do conformal, mesmo critério do EXP13.
- Decidir se vale portar `THRESH_MODE="conformal"` também para
  `run_one_sensor` (não feito nesta rodada, mesmo escopo restrito do
  seed-sweep no EXP13).
- Considerar se o `x_train`/`x_val` também deveriam excluir
  off/transiente (não só a calibração) -- fora de escopo por ora, mas
  o mesmo mecanismo de contaminação de regime pode, em tese, afetar o
  fit do modelo (não só o threshold).
- Se o resultado remoto confirmar a garantia de FPR do `α` escolhido,
  considerar isso a métrica preferencial de calibração daqui pra
  frente (substituindo o ciclo de `THRESH_STD_K` por tentativa-e-erro).

## Tasks ClearML (ordem cronológica)

1. `130ed7b5cf9447889443800f8d624b1a` (commit `f7d81f3`) -- primeira
   submissão, `α=0,003`, **sem** filtro de operational_state na
   calibração: threshold 0,6318 (contaminado por parada longa
   2025-06-05), hit_rate 0% -- descartada, levou ao fix acima.

## Branch e commits

- `exp14_conformal_threshold`, criada a partir de `AE_novo_legado`
  (topo: commit `cc0861d`, candidato final do EXP13 confirmado).
