# Análise de Experimentos — AutoML (grupo T5_temperatura)

Log de resultados da nossa própria pipeline AutoML (`src/cnn1d_ae/automl_pipeline.py`),
que roda dense/ocsvm/iforest ponto-a-ponto (sem janelamento temporal) sobre um
grupo de sensores. Ver `analise_automl_lara.md` para o estudo da pipeline original
da Lara que motivou essa reimplementação.

**Dados:** `serie_consolidada_2025_interpolated_antigo.csv` (`TRAIN_SOURCE: raw`),
2025-01-01 a 2025-10-31, resolução 30s (~875 mil linhas).
**Alarmes T5_AVG_A + TC382_03_A:** 120 no total, **fortemente concentrados em
jan–jun (110/120 = 92%)**:

| Mês | T5_AVG_A | TC382_03_A |
|---|---|---|
| Jan | 4 | 16 |
| Fev | 4 | 4 |
| Mar | 0 | 4 |
| Abr | 6 | 12 |
| Mai | 18 | 28 |
| Jun | 8 | 8 |
| Jul | 2 | 2 |
| Ago | 2 | 2 |

---

## EXP5 — baseline com split OOS em 2025-05-01

**Config:** `test_grupo_T5_automl_exp5_oos.json` (commit `19ebb5d`)
**Task ClearML:** `90866f2bae714ed1acd247bd4849fef7`

Grupo único `T5_temperatura` (T5_AVG_A + TC382_03_A combinados). Fit em
jan–abr (286.090 pontos normais), avaliação OOS em mai–out (70 alarmes).

| Métrica | Valor |
|---|---|
| Melhor modelo | iforest (p95, debounce=1) |
| Hit rate | 57,1% (40/70) |
| composite_score | 0,857 |
| normal_alert_rate | 1,84% |
| anomalias/dia | ~100/dia |

Segundo/terceiro melhores: `dense` e `ocsvm` empatados em ~0,838 de
composite_score, com bem menos falso alerta (~1,0–1,3% vs 1,84% do iforest).

---

## EXP5-v2 — split empurrado para 2025-07-01 (achado negativo)

**Task ClearML:** `e7964380db074cfebd241ace38299d75`

**Motivação:** dar mais dado de treino ao modelo — só 4 dos 10 meses
disponíveis (jan–abr) estavam sendo usados para fit.

**O que quebrou:** como 92% dos alarmes estão concentrados em jan–jun, empurrar
o corte para depois de junho reduziu drasticamente a amostra de avaliação:

| Split | Alarmes na avaliação (T5+TC382) |
|---|---|
| 2025-05-01 (EXP5) | 70 |
| 2025-07-01 (EXP5-v2) | **8** (T5_temperatura) / **4** (TC382 univariado) |

Com `n_alarms` nessa faixa, `hit_rate` vira uma fração quase discreta (4/8 e
2/4 deram exatamente 0,5 na maioria dos trials do ranking) — sem poder de
discriminação real entre modelos/hiperparâmetros. Os `composite_score` dessa
run **não são comparáveis** com os do EXP5 (bases de avaliação diferentes).

**Conclusão:** mover o corte OOS para frente sem olhar a distribuição de
alarmes destrói o poder estatístico da avaliação. Qualquer split temporal
futuro precisa ser escolhido *depois* de checar quantos alarmes sobram de
cada lado.

**Resultado sugestivo (não conclusivo, n pequeno):** o grupo univariado
TC382_03_A teve `normal_alert_rate` quase zero (0,0006%) e apenas 0,02
anomalias/dia, contra ~1,4% do grupo combinado — indício de que isolar
TC382_03_A pode produzir um detector mais específico, mas precisa ser
reavaliado com amostra de alarmes maior antes de tirar qualquer conclusão.

---

## Decisão para a próxima rodada (2026-08-13)

1. **Sem split OOS por enquanto** (`AUTOML_OOS_SPLIT_DATE: null`) — usar toda
   a série (jan–out) tanto para fit quanto para avaliação. Isso é
   intencionalmente *in-sample* nesta fase de busca de modelo/hiperparâmetros
   (mais alarmes disponíveis = comparação mais estável entre trials). Uma vez
   que um candidato bom for escolhido, ele precisa voltar a ser validado com
   split OOS de verdade antes de ir para produção.
2. **Três grupos em paralelo** para comparar multivariado vs. univariado:
   `T5_temperatura` (combinado), `T5_AVG_A_univariado`, `TC382_03_A_univariado`.
3. **DENSE com capacidade maior**: `AUTOML_DENSE_LAYERS` alterado de `[64, 32]`
   para `[256, 128]` — mesma arquitetura do melhor trial da Lara
   (`analise_automl_lara.md`, seção 1.2). Nossa pipeline é ponto-a-ponto (sem
   `seq_len`/janelamento como a dela), então não é uma reprodução exata, mas
   testa se a capacidade extra ajuda no nosso regime também.

---

## EXP5-v3 — sem split OOS, 3 grupos, dense [256,128] (in-sample)

**Task ClearML:** `789ab54015e34bca9f2c1739bb16bec3`

| Grupo | Melhor modelo | Hit rate | composite_score | normal_alert_rate |
|---|---|---|---|---|
| TC382_03_A univariado | **iforest** | 81,6% (62/76) | 0,938 | 2,42% |
| T5_temperatura (combinado) | dense | 78,3% (94/120) | 0,927 | 3,28% |
| T5_AVG_A univariado | dense | 59,1% (26/44) | 0,864 | 0,45% |

Salto grande sobre o EXP5 (57,1%), mas com uma ressalva séria: **o `dense`
venceu com threshold na ordem de 1e-9** (praticamente zero) em dois dos três
grupos. Com 256→128 neurônios reconstruindo um vetor de 1-2 features
contínuas sobre ~720 mil pontos de treino, o autoencoder tem capacidade de
sobra para memorizar os dados normais quase perfeitamente, em vez de
aprender um padrão generalizável — é o mesmo padrão de "resultado sedutor
mas otimizado no mesmo ponto" que a seção 3.4 de `analise_automl_lara.md`
identificou no notebook da própria Lara.

O resultado do **TC382_03_A univariado é o mais confiável dos três**: quem
venceu ali foi o `iforest` (threshold em escala normal, 0,64), não o dense —
consistente com o indício já visto no EXP5-v2 (agora com amostra completa de
76 alarmes em vez de 4).

**Decisão:** nenhum desses candidatos pode ser promovido sem validação fora
da amostra. Próximo passo: restaurar `AUTOML_OOS_SPLIT_DATE: "2025-05-01"`
(preserva ~70+ alarmes na avaliação, ver tabela de distribuição no topo deste
documento) e rodar os mesmos 3 grupos de novo para confirmar se os resultados
se sustentam.
