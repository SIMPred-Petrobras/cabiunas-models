# Análise de Experimentos — Grupo T5_temperatura
**Sensores:** T5_AVG_A (alvo) + TC382_03_A (auxiliar)
**Referência operacional:** NGP_A
**Dados:** `serie_consolidada_2025_interpolated_antigo.csv` (Jan–Nov 2025)
**Alarmes disponíveis:** 120 (8 de T5_AVG_A, 20 de TC382_03_A, 92 de outros)

---

## Arquitetura adotada

Autoencoder CNN-1D **multivariado** com `target_sensor`:
- **Entrada:** janela `(time_steps=180, 2 canais)` — T5_AVG_A + TC382_03_A juntos
- **Saída:** reconstrução dos dois canais (o encoder aprende correlação física entre eles)
- **Detecção:** threshold e anomalia calculados **apenas sobre o MAE do canal alvo (T5_AVG_A)**

O TC382_03_A entra como contexto físico — se T5 sair do padrão esperado *dado o comportamento do TC382*, o erro de reconstrução do T5 sobe. Isso torna a detecção sensível a quebras de correlação entre sensores vizinhos, e não apenas a desvios isolados.

---

## EXP1 — Linha de base

**Config:** `test_grupo_T5_NGP.json`

| Parâmetro | Valor |
|---|---|
| MAX_TRIALS | 5 |
| TRANSIENT_PADDING_MINUTES | 30 min |
| TARGET_ANOMALY_RATE | 0.003 (p99.7) |
| POINT_WINDOW / POINT_MIN_COUNT | 180 / 15 |

### Resultados
| Métrica | Valor |
|---|---|
| Hit rate | **30,0%** (36/120 alarmes) |
| Anomalias/dia | 30,79 pontos |
| Melhor val_loss | 0,001005 |
| Threshold | 0,08427 |

### O que funcionou
- O modelo **detectou os alarmes reais de janeiro** corretamente: os blocos de anomalia coincidem com os alarmes das linhas vermelhas/laranja no período 17–20/Jan/2025
- A máscara operacional NGP_A funcionou bem: períodos `off_longo` e `off_curto` foram excluídos corretamente (22% do dataset)

### O que falhou

**1. Histograma bimodal — dois regimes no treino**
O MAE de T5_AVG_A apresentou dois picos: um em ~0.02 (operação estável) e outro em ~0.08 (colado no threshold). Esse segundo pico corresponde ao período de **aquecimento do forno** (rampa de 20°C → 650°C), que durou horas e não foi completamente excluído pelo padding de apenas 30 minutos. O modelo viu esse regime de aquecimento como "normal", mas com alta incerteza, gerando um threshold alto que mascarou anomalias reais menores.

**2. Modelo subótimo (5 trials)**
Com apenas 5 combinações de hiperparâmetros testadas, o KerasTuner não explorou o espaço adequadamente. O `val_loss=0.001` indica que o modelo ainda tem capacidade de melhorar significativamente.

**3. Concentração de anomalias em janeiro**
7.234 dos 9.359 pontos anômalos estavam em janeiro — um sinal claro de que o modelo estava reagindo ao regime de aquecimento, não a anomalias reais.

---

## EXP2 — Mais trials + padding maior

**Config:** `test_grupo_T5_NGP_exp2.json`

| Parâmetro | Valor |
|---|---|
| MAX_TRIALS | 20 |
| TRANSIENT_PADDING_MINUTES | 60 min |
| TARGET_ANOMALY_RATE | 0.003 (p99.7) |
| POINT_WINDOW / POINT_MIN_COUNT | 180 / 15 |

### Resultados
| Métrica | Valor |
|---|---|
| Hit rate | **38,3%** (46/120 alarmes) |
| Anomalias/dia | 19,59 pontos |
| Melhor val_loss | 0,000019 (**53× melhor que EXP1**) |
| Threshold | 0,00442 |

### O que funcionou

**1. Modelo muito mais preciso**
O `val_loss=1.9e-5` indica que o modelo aprendeu o comportamento normal com alta fidelidade. O histograma do EXP2 é **quase unimodal** — a massa está concentrada próxima de zero, com uma cauda longa separada. O segundo pico de EXP1 desapareceu.

**2. Anomalias distribuídas realisticamente ao longo do ano**
Ao contrário do EXP1 (explosão em janeiro), o EXP2 detectou eventos espalhados por todos os meses — comportamento esperado de equipamento industrial real.

**3. Padding de 60 min excluiu 3.889 pontos a mais de transitório**
O período de aquecimento foi melhor isolado do treino, limpando o segundo pico do histograma.

**4. Janeiro ficou mais preciso**
Os poucos alertas em janeiro agora coincidem diretamente com os alarmes reais — menos ruído, mais sinal.

### O que ainda falhou

**1. Threshold muito restritivo (TARGET_ANOMALY_RATE=0.003)**
Análise dos percentis do MAE nos períodos ON:
```
p99.0  → 0.002676   (1.0% das sequências)
p99.5  → 0.003373   (0.5%)
p99.7  → 0.00442    ← threshold atual
p99.9  → 0.009557   ← salto brusco: cauda de anomalias reais
```
O threshold atual (p99.7) está antes do salto em p99.9. Isso significa que anomalias que geram MAE entre 0.003 e 0.009 (a maioria dos eventos reais) são **classificadas como normais**. Abaixar o threshold para p99.0 (0.0027) permitiria capturar esses eventos.

**2. 62% dos alarmes ainda não detectados**
Os 74 alarmes perdidos têm três prováveis origens:
- **Alarmes de TC382_03_A avaliados pelo MAE de T5_AVG_A:** com `target_sensor=T5_AVG_A`, eventos que afetam mais o TC382 do que o T5 passam despercebidos
- **Alarmes durante períodos de mudança de regime** (partida/parada) que mesmo com 60 min de padding ainda têm contaminação residual
- **Alarmes de outros sensores** (92 de 120 são de NPT_A, NGP_A, TC382_04_A) — esses nunca serão detectados por este grupo que monitora apenas T5+TC382

**3. POINT_MIN_COUNT pode estar perdendo eventos curtos**
Com `POINT_MIN_COUNT=15` em `POINT_WINDOW=180`, exige-se 8,3% da janela como anômala. Eventos de curta duração (picos isolados reais) que não sustentam 15 sequências consecutivas são descartados.

---

## Discussão: os 62% restantes

### Hipótese 1 — Threshold muito alto
**Evidência:** salto brusco em p99.9 (de 0.0034 para 0.0096) indica que há eventos reais na cauda que estão abaixo do threshold atual.
**Solução:** `TARGET_ANOMALY_RATE=0.01` → threshold ≈ p99.0 → mais sensível, mais alarmes capturados.

### Hipótese 2 — Alarmes de TC382 não capturados pelo target T5
**Evidência:** 20 dos 120 alarmes são de TC382_03_A. Com `target_sensor=T5_AVG_A`, o threshold é calculado sobre o canal T5. Se um evento afeta só TC382, não dispara alerta.
**Solução:** testar `target_sensor=null` (MAE global) ou criar dois grupos — um com target T5, outro com target TC382.

### Hipótese 3 — Alarmes fora do escopo deste grupo
**Evidência:** 92 alarmes são de NPT_A, NGP_A, TC382_04_A — sensores que **não estão no grupo**. Esses jamais serão detectados pelo grupo T5_temperatura.
**Impacto real:** se excluirmos esses 92 alarmes, o denominador cai para 28 alarmes relevantes. Dos 46 detectados, certamente muitos são desses 28. O hit rate real sobre alarmes do grupo pode ser bem mais alto que 38%.

### Hipótese 4 — Eventos curtos descartados pela votação
**Solução:** reduzir `POINT_WINDOW` de 180 para 90 e manter `POINT_MIN_COUNT=15` → exige 16% da janela (mais exigente por sequência, mas janela menor = mais reativo).

---

## EXP3 — Ajuste de threshold e votação

**Objetivo:** aumentar sensibilidade via `TARGET_ANOMALY_RATE=0.01` compensado com `POINT_MIN_COUNT=20` para não explodir em falsos positivos.

| Parâmetro | EXP2 | EXP3 |
|---|---|---|
| MAX_TRIALS | 20 | 20 |
| TRANSIENT_PADDING_MINUTES | 60 min | 60 min |
| **TARGET_ANOMALY_RATE** | 0.003 | **0.01** |
| POINT_WINDOW | 180 | 180 |
| **POINT_MIN_COUNT** | 15 | **20** |

**Hipótese:** com threshold mais baixo (p99.0 vs p99.7), mais sequências serão flagadas. O POINT_MIN_COUNT mais alto (20 vs 15) filtra eventos de ruído curto mas mantém anomalias sustentadas. Esperamos hit rate ≥ 45% com anomalias/dia controladas (< 25).

---

## Roadmap de melhorias futuras

| Prioridade | Ação | Esperado |
|---|---|---|
| Alta | Filtrar alarmes por Tag antes de calcular hit rate | Hit rate real do grupo visível |
| Alta | Adicionar TC382_03_A como segundo `target_sensor` ou grupo separado | +20 alarmes no escopo |
| Média | Aumentar `TIME_STEPS` de 180 para 360 (3h de contexto) | Melhor captura de tendências lentas |
| Média | `SPLIT_MODE: random` para validação mais representativa | Melhor generalização |
| Baixa | Adicionar TC382_04_A ao grupo (sensor vizinho) | Mais contexto físico |
