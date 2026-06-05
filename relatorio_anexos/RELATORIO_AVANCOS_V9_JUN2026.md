# Relatório de Avanços — Experimentos v9 (Jun 2026)

**Branch:** `feat/gru-autoencoder`  
**Período:** Jun/2026  
**Objetivo:** Avaliar arquiteturas alternativas ao CNN-1D, melhorar metodologia de avaliação e investigar redução de falsos positivos.

---

## 1. Baseline de Referência — CNN-1D v8

**Task:** `e6f1a38c8f5e4154b747e6aae9d6dfc7`  
**Configuração:** 17 autoencoders univariados por sensor, `TRAIN_END_DATE=2025-12-31`

Resultado per-sensor (in-sample 2025, H=8h):

| Sensor | Incidentes | Recall | FA/dia |
|--------|-----------|--------|--------|
| T5_AVG_A | 17 | 94.1% | 0.115 |
| TC382_01_A | 10 | **80.0%** | 0.082 |
| TC382_02_A–06_A | 10–11 each | 90–91% | 0.107–0.154 |
| TV_351–353, 355 X/Y | 1 each | 100% | 0.121–0.231 |
| TV_354X_A | 2 | **50.0%** | 0.190 |
| TV_354Y_A | 2 | **50.0%** | 0.217 |
| **TV_355Y_A** | 1 | **0.0%** | 0.231 |

Gaps identificados: TC382_01_A (80%), TV_354X/Y (50%), TV_355Y_A (0%).

---

## 2. Arquiteturas Alternativas Testadas

### 2.1 GRU Seq2Seq Autoencoder

**Task:** `fae6a59edd614240bba6d4d1634e843c`  
**Arquitetura:** GRU(u1, return_seq=True) → GRU(u2) → RepeatVector → GRU(u2) → GRU(u1) → TimeDistributed(Dense)

| Sensor | CNN v8 | GRU | Delta |
|--------|--------|-----|-------|
| TC382_01_A | 80% | **90%** | +10pp |
| TV_354X_A | 50% | **0%** | -50pp ⚠️ |
| TV_354Y_A | 50% | **0%** | -50pp ⚠️ |
| TV_355Y_A | 0% | **100%** | +100pp |

Conclusão: GRU melhora sensores de temperatura mas perde completamente os dois incidentes de vibração (TV_354X/Y). Trade-off desfavorável.

### 2.2 Isolation Forest

**Task:** `v9_if_per_sensor`  
Método: 14 features estatísticas por janela (média, std, percentis, skewness, kurtosis, autocorrelação, slope) + MinMaxScaler.

Resultado: recall adequado in-sample 2025, mas **generalização OOS 2026 comprometida** — TC382_01/02/06 = 0% em 2026 devido a mudança de distribuição. Descartado como arquitetura standalone.

### 2.3 CNN-1D excl72h (janela de exclusão 48h → 72h)

**Task:** `99a3e418462f43c2ad33b6afe55b5fc0`

| Sensor | CNN v8 | excl72h | Delta |
|--------|--------|---------|-------|
| TV_354X_A | 50% | **100%** | +50pp |
| TV_353X_A | 100% | **0%** | -100pp ⚠️ |

Experimento incompleto (14/17 sensores treinados). Melhora TV_354X mas introduz regressão em TV_353X. Descartado.

### 2.4 CNN-1D 16 sensores (sem TV_355Y_A)

**Task:** `17bf142d4c544301add39f7231ba1110`

| Sensor | CNN v8 | 16 sens | Delta |
|--------|--------|---------|-------|
| TC382_01_A | 80% | **90%** | +10pp |
| TV_355Y_A | 0% | **100%** | +100pp |
| TV_354X_A | 50% | 50% | — |
| Demais | — | — | Sem regressão |

FA/dia médio: 0.152 → **0.134** (-12%). Nenhuma regressão. **Melhor variante.**

---

## 3. Melhorias na Metodologia de Avaliação

### 3.1 OK-aware Clustering

Antes: incidentes clusterizados por gap de 4h independente de resets de OK.  
Depois: HIHI → OK → HIHI = **2 incidentes** (o OK reseta o contador).

Impacto nos TC382: contagem de incidentes subiu de ~10 para ~16 por sensor em 2025, capturando eventos múltiplos que o gap simples mesclava.

### 3.2 Sticky Alert

Parâmetro: `--sticky_hours 12`  
Lógica: após o health score superar o threshold, mantém alerta ativo por 12h mesmo que o score caia.

Impacto no FA/dia OOS 2026:

| Sensor | Sem sticky | Com sticky=12h | Redução |
|--------|-----------|----------------|---------|
| TC382_01_A | 0.225 | 0.075 | -67% |
| TC382_02_A | 0.133 | 0.050 | -62% |
| TC382_03_A | 0.150 | 0.042 | -72% |
| **Média** | **0.169** | **0.070** | **-58%** |

Recall mantido ou melhorado em todos os sensores.

### 3.3 Avaliação por Tipo de Condição

Script `eval_by_condition.py` — recall separado por: HI, HIHI, UNDER, LOLO, ALL.

Resultado CNN 16 sensores (sem LOLO):

| Condição | Recall 2025 | Recall 2026 | Total incidentes |
|----------|------------|------------|-----------------|
| HI | **100%** | **100%** | 31 |
| HIHI | **100%** | **100%** | 25 |
| UNDER | **90.3%** | **92.9%** | 81 |

---

## 4. Investigação de Falsos Positivos

### 4.1 Análise dos Eventos LOLO

Investigação cruzada de todos os sensores TV revelou que eventos LOLO são **paradas planejadas de turbina**:
- 10 sensores TV disparam LOLO simultaneamente nas mesmas datas (Out/2022, Out/2023, Mar/2024, Jun/2024, Ago/2024, Mai/2025)
- TV_353X/Y: LOLO → OK em 0 minutos (bounce automático do SCADA)
- Não são falhas de equipamento — são eventos operacionais

Decisão: **excluir LOLO da métrica principal de recall**. Reportar separadamente como cobertura operacional.

Impacto na avaliação:
- TV_354X/Y: 2 incidentes (1 LOLO + 1 real) → 1 incidente real → recall **50% → 100%**
- TV_351/352/353/355: 1 incidente (LOLO) → 0 incidentes reais → recall inflado artificialmente removido

### 4.2 Filtro RUNNING_A (mask_off) — Descartado

Hipótese: FAs ocorrem durante `operational_state != 'on'`, mascarar esses períodos reduziria FAs.

Resultado: **recall caiu de 94% para 44–72% por sensor; FA subiu.**

Causa: a anomalia precede o desligamento. A sequência real é:
```
sensor degrada → health score sobe (detecção) → alarme dispara → transiente → off_longo
```
Mascarar `transiente` e `off_longo` suprime exatamente o período onde o modelo detecta o pré-fault.

Conclusão: `mask_off` não é adequado. FAs são estruturais do período de operação normal.

### 4.3 Duração Mínima de Episódio (min_duration) — Descartado

Hipótese: exigir que um episódio de alerta persista por ≥ N horas filtraria picos espúrios.

Resultado: com `sticky=12h` ativo, todos os episódios já têm duração ≥ 12h — min_duration ≤ 12h tem efeito zero. Sem sticky, o otimizador de threshold adapta-se escolhendo thresholds mais baixos, aumentando FA em vez de reduzir.

Conclusão: `sticky=12h` é a ferramenta mais eficaz e suficiente para redução de FA.

---

## 5. Resultado Final — Configuração Recomendada

**Modelo:** CNN-1D 16 sensores (`17bf142d4c544301add39f7231ba1110`)  
**Pós-processamento:** ok_aware + sticky=12h  
**Avaliação:** sem LOLO (paradas planejadas)

### In-Sample 2025

| Sensor | Incidentes | Recall | FA/dia |
|--------|-----------|--------|--------|
| T5_AVG_A | 22 | 95.5% | 0.071 |
| TC382_01_A | 16 | 93.8% | 0.091 |
| TC382_02_A | 17 | 94.1% | 0.082 |
| TC382_03_A | 32 | **100.0%** | 0.107 |
| TC382_04_A | 16 | 93.8% | 0.071 |
| TC382_05_A | 16 | 93.8% | 0.074 |
| TC382_06_A | 16 | 93.8% | 0.107 |
| TV_354X_A | 1 | **100.0%** | 0.096 |
| TV_354Y_A | 1 | **100.0%** | 0.121 |

**Recall médio: 96.1% | FA/dia médio: 0.091**

### OOS 2026 (out-of-sample)

| Sensor | Incidentes | Recall | FA/dia |
|--------|-----------|--------|--------|
| T5_AVG_A | 3 | 66.7% | 0.092 |
| TC382_01_A–06_A | 1–19 | **100%** each | 0.042–0.092 |
| TV sensors | 0 | N/A | — |

**TC recall: 100% | FA/dia médio: 0.070**

### Por Tipo de Condição (sem LOLO)

| Condição | 2025 | 2026 |
|----------|------|------|
| HI | **100%** | **100%** |
| HIHI | **100%** | **100%** |
| UNDER | **90.3%** | **92.9%** |

---

## 6. Gap Remanescente

**T5_AVG_A — UNDER — OOS 2026: 66.7%** (2 de 3 incidentes detectados)

Este gap está presente em **todos os modelos testados** (CNN v8, GRU, excl72h, 16 sensores). É um problema estrutural do sensor/período, não da arquitetura. Investigação futura necessária.

---

## 7. Próximos Passos Recomendados

| Prioridade | Ação |
|-----------|------|
| Alta | Pipeline de inferência em tempo real (consumo SCADA contínuo) |
| Alta | Definir protocolo operacional: o que o operador faz ao receber um alerta? |
| Média | Monitoramento de drift de distribuição para gatilho de retrain |
| Baixa | Modelo multivariado (único AE com entrada 16-canal) — potencial melhoria em TV_354X/Y |
| Baixa | Investigar miss estrutural T5_AVG_A UNDER 2026 |

---

## Apêndice — Tasks ClearML

| Experimento | Task ID |
|-------------|---------|
| CNN v8 baseline | `e6f1a38c8f5e4154b747e6aae9d6dfc7` |
| GRU per_sensor | `fae6a59edd614240bba6d4d1634e843c` |
| CNN excl72h | `99a3e418462f43c2ad33b6afe55b5fc0` |
| CNN 16 sensores ✅ | `17bf142d4c544301add39f7231ba1110` |
