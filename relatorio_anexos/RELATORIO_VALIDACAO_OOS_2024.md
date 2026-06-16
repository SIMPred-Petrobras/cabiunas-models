# Validação Out-of-Sample (2024) — Detector de Anomalias Turbina A

## Resultado principal

O detector (autoencoder CNN-1D por sensor) foi **treinado e calibrado apenas com dados de 2025** e validado **out-of-sample no ano de 2024 inteiro** — dado que o modelo nunca viu, com o ponto de operação **congelado** (nada re-ajustado em 2024).

- **Recall macro: 95,2%** (73 incidentes reais, todos os 7 sensores com ground-truth)
- **Alarme ligado em ~17% do tempo de operação** (duty-cycle médio) — alarme operável, não permanente
- **Falsos alarmes < 0,22/dia** por sensor
- TC382_03 (o termopar mais crítico) sozinho: **49 incidentes em 2024**

É a evidência mais forte de que o detector **generaliza** — não é ajuste fino de um único ano.

## Desempenho por sensor (OOS 2024, ponto deployável — threshold por sensor)

| sensor | q | recall | falsos alarmes/dia | tempo-em-alerta | incidentes |
|---|---|---|---|---|---|
| TC382_06_A | 0,98 | 100% | 0,137 | 3% | 1 |
| T5_AVG_A | 0,92 | 89% | 0,178 | 8% | 9 |
| TC382_05_A | 0,96 | 86% | 0,121 | 8% | 7 |
| TC382_04_A | 0,94 | 100% | 0,134 | 8% | 1 |
| TC382_01_A | 0,92 | 100% | 0,211 | 14% | 4 |
| TC382_02_A | 0,92 | 100% | 0,143 | 23% | 2 |
| TC382_03_A | 0,88 | 92% | 0,079 | **53%** | 49 |
| **macro** | | **95,2%** | **< 0,22** | **17%** | **73** |

Figuras: `eval_predictive_out/fig_oos_2024_*.png`, `fig_series_2025_*.png`.

## Como foi medido (transparência)

- **Métrica de produção**: erro de reconstrução → EWMA (half-life por sensor) → comparação com threshold absoluto calibrado em 2025 → exclusão de períodos com equipamento desligado (NGP ≤ 50) → debounce. É exatamente o que rodaria em produção (streaming).
- **Threshold POR SENSOR**: cada sensor tem uma curva recall×duty própria, então o nível de corte (quantil q) é escolhido individualmente — o menor tempo-em-alerta que mantém recall ≥ 85%. Um q único para todos seria pior (q=0,85 → recall 99% mas 43% de alarme; q=0,90 → 95%/24%). Com q por sensor: **95%/17%**, com 6 dos 7 sensores entre 3–23% de alarme.
- **Caso TC382_03 (limite conhecido)**: é o termopar com **erro de fundo cronicamente elevado** (não picos isolados) — fica anômalo boa parte do ano. Para ele, **nenhum threshold absoluto** quebra o trade-off (92% recall custa 53% de alarme; baixar o alarme para ~20% derruba o recall para ~57%). A solução não é ajustar o nível de corte, e sim **detecção de desvio relativo ao baseline recente** (resíduo do EWMA vs. mediana móvel / CUSUM) — próximo passo recomendado.

## Achados sobre os dados (importantes)

1. **Todos os 17 sensores têm histórico de alarme** no registro completo (`alarmes_selecionados_turbina_a.csv`, 2022–2026) — a ideia anterior de "só 3 com rótulo" era artefato de um arquivo filtrado. Em operação, **7 são avaliáveis** (T5 + 6 termopares); os 10 sensores de vibração têm ~1 alarme/ano (insuficiente).
2. **Os 6 termopares compartilham os mesmos eventos UNDER** (quedas de temperatura coordenadas em toda a turbina) — não são falhas independentes.
3. **30 instrumentos auxiliares** (pressão de gás/óleo, temperatura de mancais, ar de exaustão) têm **apenas alarme registrado, sem sinal contínuo** no dataset → não modeláveis hoje.

## Pedidos à Petrobras (destravam o próximo salto)

1. **Sinais contínuos (séries do historiador)** dos instrumentos analógicos — em especial **temperatura de ar de exaustão (TI_6240315/317)** e **temperaturas de mancal (TAH/TAHH)**. Hoje só temos o alarme; com a curva, o modelo cobre esses pontos.
2. **Dados de sensor de 2024-H1 com NGP** (e anos anteriores) — amplia a validação e resolve a escassez de incidentes que limita os termopares com poucos eventos/ano.
