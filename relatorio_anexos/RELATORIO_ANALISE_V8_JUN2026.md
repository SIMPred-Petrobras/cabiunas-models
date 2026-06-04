# Análise v8 — Experimentos e Resultados (Jun 2026)

## Configuração testada

**Task ClearML:** `e6f1a38c8f5e4154b747e6aae9d6dfc7`  
**Config:** `configs/calibracao_v8_prod/v8_prod_oos_per_sensor.json`

Parâmetros principais:
- `MODEL_MODE: per_sensor` — 17 autoencoders CNN-1D univariados
- `TIME_STEPS: 96` (48 min de janela a 30s)
- `STRIDE: 10`
- `TRAIN_END_DATE: 2025-12-31` — treino restrito a 2025
- `PREDICTIVE_EWMA_HALF_LIFE_HOURS: 4.0`
- `PREDICTIVE_ALERT_DEBOUNCE_HOURS: 8.0`
- `MAX_TRIALS: 20`, `PER_SENSOR_EPOCHS: 50`
- Dataset: `sensores_brutos_2025_2026_30s.csv` (Jan/2025 – Abr/2026)

---

## Resultados — Nível Sistema (OR combinado)

Metodologia: OR do health index EWMA por sensor → quantile normalization → sweep de threshold → recall × FA/dia.

| Período | Horizonte | Incidentes | Recall | FA/dia |
|---|---|---|---|---|
| In-sample 2025 | H=8h | 254 | **100.0%** | **0.000** |
| In-sample 2025 | H=24h | 254 | 100.0% | 0.000 |
| In-sample 2025 | H=72h | 254 | 100.0% | 0.000 |
| OOS 2026 | H=8h | 63 | **100.0%** | **0.008** |
| OOS 2026 | H=24h | 63 | 100.0% | 0.008 |
| OOS 2026 | H=72h | 63 | 100.0% | 0.008 |

**Baseline TS=60 (referência):** rec=0.685 FA=0.008 @ H=8h

---

## Resultados — Nível Sensor (avaliação individual)

Cada sensor avaliado contra seus próprios alarmes (`Tag Alarme == sensor`), H=8h.

| Sensor | Inc. 2025 | Recall 2025 | FA/dia 2025 | Inc. 2026 | Recall 2026 | FA/dia 2026 |
|---|---|---|---|---|---|---|
| T5_AVG_A | 17 | 94.1% | 0.115 | 3 | 66.7% | 0.193 |
| TC382_01_A | 10 | 80.0% | 0.082 | 1 | 100.0% | 0.143 |
| TC382_02_A | 11 | 90.9% | 0.115 | 1 | 100.0% | 0.126 |
| TC382_03_A | 23 | 95.7% | 0.179 | 17 | 100.0% | 0.143 |
| TC382_04_A | 10 | 90.0% | 0.121 | 1 | 100.0% | 0.202 |
| TC382_05_A | 10 | 90.0% | 0.107 | 1 | 100.0% | 0.244 |
| TC382_06_A | 10 | 90.0% | 0.154 | 1 | 100.0% | 0.261 |
| TV_351X/Y_A | 1 cada | 100.0% | ~0.15 | 0 | N/A | ~0.13 |
| TV_352X/Y_A | 1 cada | 100.0% | ~0.14 | 0 | N/A | ~0.18 |
| TV_353X/Y_A | 1 cada | 100.0% | ~0.17 | 0 | N/A | ~0.10 |
| TV_354X_A | 2 | 50.0% | 0.190 | 0 | N/A | 0.151 |
| TV_354Y_A | 2 | 50.0% | 0.217 | 0 | N/A | 0.067 |
| TV_355X_A | 1 | 100.0% | 0.154 | 0 | N/A | 0.185 |
| TV_355Y_A | 1 | **0.0%** | 0.231 | 0 | N/A | 0.067 |

**Nota:** FA alto no nível sensor não se traduz em FA no sistema porque o OR + gap-debounce filtra episódios que cobrem incidentes reais.

---

## Diagnóstico de Incidentes Perdidos

### TC382_01_A — 2025-06-17 16:47 UTC (miss tipo A — detecção precoce)

- Anomalia detectada em Jun/14-15 (MAE saturado, health=0.95+)
- Alarme disparado em Jun/17 — 2 dias depois
- EWMA decaiu abaixo do threshold entre a detecção e o alarme
- **Causa:** EWMA com hl=4h não sustenta alerta por 2+ dias
- **Cobertura sistema:** coberto por outros sensores TC382

### TC382_01_A — 2025-08-08 15:31 UTC (miss tipo B — sinal ausente)

- MAE do TC382_01_A flat o tempo todo (0.02–0.04)
- **Causa:** evento de vibração, não de temperatura — o AE de temperatura corretamente não viu
- **Cobertura sistema:** capturado por TV_353Y_A (health=0.511) e TV_354X_A (health=0.700)
- **Conclusão:** não é um miss real — mecanismo de falha diferente, coberto pelos sensores corretos

### T5_AVG_A — 2026-01-17 00:59 UTC OOS (miss tipo A — detecção precoce)

- Spike de MAE=2.9 em Jan/14, health chegou a 0.75
- Alarme do operador só em Jan/17 — 3 dias depois
- EWMA decaiu a ~0.05 até o momento do alarme
- **Com H=72h seria coberto** (antecipação de 3 dias)
- **Cobertura sistema:** coberto via outros sensores TC382

---

## Sweep de Half-life EWMA

Testado com `scripts/sweep_halflife.py` para hl ∈ {2, 4, 6, 8, 12, 16, 24, 36, 48}h.

### TC382_01_A (in-sample 2025):
- hl=2h → recall=90%, FA=0.126 (melhor recall, maior FA)
- hl=4h (atual) → recall=80%, FA=0.082
- hl=24h+ → recall=80%, FA=0.025 (baixo FA mas sem melhora de recall)

### T5_AVG_A (in-sample 2025):
- hl=2–8h → recall=94.1% (estável)
- hl=12h+ → recall cai para 76–82%

### T5_AVG_A (OOS 2026):
- hl=8h → recall=100%, FA=0.076 (**sweet spot**)
- hl=12h → recall=100%, FA=0.034 (melhor FA)
- hl=4h (atual) → recall=67%, FA=0.193

**Decisão:** Testar hl=8h no nível sistema.

### Teste hl=8h — impacto no sistema:

| Período | hl=4h (atual) | hl=8h |
|---|---|---|
| 2025 H=8h | rec=1.000 FA=0.000 | rec=0.996 FA=0.000 |
| 2026 H=8h | rec=1.000 FA=0.008 | rec=1.000 FA=0.008 |

**Conclusão:** hl=8h melhora T5_AVG_A isolado mas o incidente já era coberto pelo OR. Regressão de 1/254 no in-sample. **Manter hl=4h.**

---

## Diagnóstico Cross-Sensor

### TV_354X/Y_A — miss em 2025-05-08 11:08 UTC

- Ambos os eixos do mancal 4 perderam o **mesmo incidente**
- Companheiro não cobriu: TV_354Y max=0.090, TV_354X max=0.185
- Health sobe para 0.95 **depois** do alarme — onset abrupto
- **Conclusão:** falha de onset abrupto, sem precursor detectável. OR cobriu via outros sensores.

### TV_355Y_A — miss em 2025-05-08 11:08 UTC

- TV_355Y_A flat em 0.08 — sensor genuinamente não detectou
- TV_355X_A em 0.980 durante dias — cobriu o incidente
- **Conclusão:** anomalia direcional (só eixo X). TV_355Y_A é deadweight no OR.

### TV_355Y_A exclusion test:

| Período | Com TV_355Y_A | Sem TV_355Y_A |
|---|---|---|
| 2025 H=8h | rec=1.000 FA=0.000 | rec=1.000 FA=0.000 |
| 2026 H=8h | rec=1.000 FA=0.008 | rec=1.000 FA=0.008 |

**Conclusão: TV_355Y_A pode ser excluída com segurança.**

### TC382_01_A Ago/08 — cross-sensor overview

- 15 de 17 sensores abaixo do threshold na janela H=8h
- Sensores que alertaram: **TV_353Y_A (0.511)** e **TV_354X_A (0.700)**
- Todos os TC382 (temperatura) abaixo — evento era de vibração
- **Conclusão:** OR entre mecanismos de falha diferentes é essencial para cobertura total

---

## Scripts desenvolvidos nesta análise

| Script | Função |
|---|---|
| `scripts/eval_per_sensor_level.py` | Avaliação individual por sensor com filtro de período |
| `scripts/diag_missed_per_sensor.py` | Zoom plots dos incidentes perdidos por sensor |
| `scripts/sweep_halflife.py` | Sweep de half-life EWMA por sensor |
| `scripts/diag_cross_sensor.py` | Diagnóstico cross-sensor para TV_354, TV_355Y e TC382_01_A |

---

## Decisões e Próximos Passos

### Aplicar imediatamente
- **Excluir TV_355Y_A** do SENSOR_LIST no config de produção (validado seguro)

### Manter como está
- **hl=4h**: impacto neutro no sistema com hl=8h/12h; 4h é o melhor trade-off global
- **Arquitetura CNN-1D AE**: sistema no teto mensurável com os dados disponíveis

### Investigações futuras (requerem mais dados OOS)
- **GRU** como alternativa ao CNN-1D (dependência temporal mais longa)
- **Isolation Forest** como camada complementar para onset abrupto
- **Sticky alert** para mitigar misses tipo A (detecção precoce + decaimento EWMA)
- **Transformers** apenas com 2+ anos de dados OOS para validação robusta
