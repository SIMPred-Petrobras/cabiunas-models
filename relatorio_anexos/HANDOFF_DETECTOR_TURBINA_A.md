# Handoff — Detector de Anomalias Turbina A

Detector de anomalias por **autoencoder CNN-1D por sensor** (erro de reconstrução do sinal), treinado em 2025, **validado out-of-sample no ano 2024 inteiro**. Entrega **duas saídas complementares**, ambas prontas e sem necessidade de retreino:

1. **Alarme de evento** → sala de controle (detecta anomalia antes do alarme do DCS).
2. **Índice de saúde/condição** → manutenção (degradação lenta de tendência).

---

## Saída 1 — Alarme de evento (ponto de operação por sensor)

Cada sensor tem seu próprio nível de corte (quantil `q`), escolhido pelo menor tempo-em-alerta que mantém recall ≥ 85%. Validação **OOS 2024** (modelo de 2025, threshold congelado, dado nunca visto):

| sensor | q | recall | falsos alarmes/dia | tempo-em-alerta | incidentes |
|---|---|---|---|---|---|
| TC382_06_A | 0,98 | 100% | 0,137 | 3% | 1 |
| TC382_05_A | 0,96 | 86% | 0,121 | 8% | 7 |
| T5_AVG_A | 0,92 | 89% | 0,178 | 8% | 9 |
| TC382_04_A | 0,94 | 100% | 0,134 | 8% | 1 |
| TC382_01_A | 0,92 | 100% | 0,211 | 14% | 4 |
| TC382_02_A | 0,92 | 100% | 0,143 | 23% | 2 |
| TC382_03_A | 0,88 | 92% | 0,079 | 53% | 49 |
| **macro** | | **95,2%** | **< 0,22** | **16,7%** | **73** |

Detalhe e figuras: `relatorio_anexos/RELATORIO_VALIDACAO_OOS_2024.md`, `eval_predictive_out/fig_oos_2024_*.png`.

## Saída 2 — Índice de saúde/condição

Nível suavizado do erro (EWMA 24h) → percentil 0–100 por sensor. **Heatmap sensor × mês**: `eval_predictive_out/health_index_2025_heatmap.png` (revela degradação plant-wide em jan/jun 2025 e recuperação de jul em diante). Resumo/tendência: `eval_predictive_out/health_index_2025_summary.csv`. Script: `scripts/health_index.py`.

É aqui que mora o **TC382_03**: ele fica anômalo boa parte do ano (condição física real), então entra como mapa de condição para planejamento, **não** como alarme de evento.

---

## Como rodar inferência (produção)

- **Modelo**: task ClearML `58bc393c1d7a4e42815236e8897abc88` (artefatos `{sensor}_model_keras`).
- **Bundles**: `production_bundles/{sensor}_inference_bundle.json` (17) — carregam scaler (center/scale), `feature_columns`, `time_steps`/`stride`, `half_life_hours`, **`ewma_abs_threshold`** (ponto de operação por sensor) e `running_col=NGP_A`.
- **Função**: `src/cnn1d_ae/inference.py::score_production(model, bundle, df)`.
- **Pipeline de decisão** (streaming-safe, já embutido): erro → EWMA(half-life) → compara com `ewma_abs_threshold` → **exclui OFF** (NGP ≤ 50) → debounce.
- **Cadência**: 30 s (igual ao treino).

---

## Limites conhecidos (honesto)

- **TC382_03**: baseline cronicamente elevado → nenhum threshold dá alarme limpo (92% recall custa 53% de tempo-em-alerta). Tratar via índice de saúde, não alarme.
- **Cobertura**: só **7 de 17 sensores** têm incidentes avaliáveis em operação (T5 + 6 termopares). Os 10 sensores de vibração TV têm ~1 alarme/ano e em período desligado → não avaliáveis.
- **Becos testados e refutados** (não repetir): mais pré-processamento; trocar arquitetura (GRU/LSTM/Dense/Transformer); ensemble; grupo multivariado + target; threshold absoluto mais alto; detecção por desvio relativo ao baseline. Nenhum moveu o ponteiro — o modelo não é o gargalo.

## Pedidos de dado à Petrobras (onde está o próximo ganho)

1. **Sinais contínuos (historiador)** dos instrumentos analógicos — temp. ar de exaustão (TI_6240315/317) e temps de mancal (TAH/TAHH). Hoje só há o alarme, sem a curva → não modeláveis.
2. **Dados de sensor de 2024-H1 (com NGP) e anos anteriores** — amplia validação e cobre a escassez de incidentes.

## Inventário (scripts-chave)

| script | o que faz |
|---|---|
| `scripts/eval_per_sensor_level.py` | métrica de produção (recall/FA/duty gap-based, EWMA, OFF excl); `--max_duty_cycle` |
| `scripts/validate_deployed_2024.py` | validação OOS 2024 dos bundles deployados |
| `scripts/eval_oos_2024.py` | inferência + recall OOS no ano completo |
| `scripts/regen_bundles_hl.py` | (re)gera `production_alerting` dos bundles (q por sensor) |
| `scripts/health_index.py` | índice de saúde + heatmap |
| `scripts/analyze_duty_cycle.py` | sweep recall×duty por sensor (escolha do q) |

## Backlog (não implementado — ideias para depois)

- Operacionalizar: scoring em streaming + 2 painéis (alarme / saúde) — decisão de produto/infra.
- Detector de **divergência multivariada** (um termopar fugindo dos irmãos = falha local de sensor) — reusa os 6 TC382, sem dado novo.
- Monitor de **drift do baseline** (o duty muda entre anos; avisar quando recalibrar).
