# Validação Out-of-Sample (2024) — Detector de Anomalias Turbina A

## Resultado principal

O detector (autoencoder CNN-1D por sensor) foi **treinado e calibrado apenas com dados de 2025** e validado **out-of-sample no ano de 2024 inteiro** — dado que o modelo nunca viu, com o ponto de operação **congelado** (nada re-ajustado em 2024).

- **Recall macro: 94,9%** (73 incidentes reais, todos os 7 sensores com ground-truth)
- **Alarme ligado em ~24% do tempo de operação** (duty-cycle médio) — alarme operável, não permanente
- **Falsos alarmes < 0,2/dia** por sensor
- TC382_03 (o termopar mais crítico) sozinho: **49 incidentes em 2024**

É a evidência mais forte de que o detector **generaliza** — não é ajuste fino de um único ano.

## Desempenho por sensor (OOS 2024, ponto deployável)

| sensor | recall | falsos alarmes/dia | tempo-em-alerta | incidentes |
|---|---|---|---|---|
| TC382_03_A | 76% | 0,071 | 33% | 49 |
| T5_AVG_A | 89% | 0,173 | 15% | 9 |
| TC382_05_A | 100% | 0,129 | 40% | 7 |
| TC382_01_A | 100% | 0,184 | 21% | 4 |
| TC382_02_A | 100% | 0,143 | 31% | 2 |
| TC382_04_A | 100% | 0,164 | 12% | 1 |
| TC382_06_A | 100% | 0,192 | 19% | 1 |
| **macro** | **94,9%** | **< 0,2** | **24%** | **73** |

Figuras: `eval_predictive_out/fig_oos_2024_TC382_03_A.png`, `_T5_AVG_A.png`, `_resumo.png`.

## Como foi medido (transparência)

- **Métrica de produção**: erro de reconstrução → EWMA (half-life por sensor) → comparação com threshold absoluto calibrado em 2025 → exclusão de períodos com equipamento desligado (NGP ≤ 50) → debounce. É exatamente o que rodaria em produção (streaming).
- **Ponto de operação deployável**: q = 0,90 (TC382_03 em q = 0,92). Evitamos deliberadamente o ponto que dava "100% de recall" — ele só atingia isso mantendo o **alarme ligado 72–93% do tempo** (inútil na sala de controle). O número honesto e operável é o desta tabela.
- **Decisão de negócio pendente — TC382_03**: é um termopar genuinamente anômalo boa parte do ano. Há um trade-off a definir com a operação:
  - **q = 0,92** → 76% recall, alarme 33% do tempo (atual)
  - **q = 0,88** → 92% recall, alarme 53% do tempo
  - Qual ponto adotar depende de quanto alarme a equipe tolera vs. quanto evento pode deixar passar.

## Achados sobre os dados (importantes)

1. **Todos os 17 sensores têm histórico de alarme** no registro completo (`alarmes_selecionados_turbina_a.csv`, 2022–2026) — a ideia anterior de "só 3 com rótulo" era artefato de um arquivo filtrado. Em operação, **7 são avaliáveis** (T5 + 6 termopares); os 10 sensores de vibração têm ~1 alarme/ano (insuficiente).
2. **Os 6 termopares compartilham os mesmos eventos UNDER** (quedas de temperatura coordenadas em toda a turbina) — não são falhas independentes.
3. **30 instrumentos auxiliares** (pressão de gás/óleo, temperatura de mancais, ar de exaustão) têm **apenas alarme registrado, sem sinal contínuo** no dataset → não modeláveis hoje.

## Pedidos à Petrobras (destravam o próximo salto)

1. **Sinais contínuos (séries do historiador)** dos instrumentos analógicos — em especial **temperatura de ar de exaustão (TI_6240315/317)** e **temperaturas de mancal (TAH/TAHH)**. Hoje só temos o alarme; com a curva, o modelo cobre esses pontos.
2. **Dados de sensor de 2024-H1 com NGP** (e anos anteriores) — amplia a validação e resolve a escassez de incidentes que limita os termopares com poucos eventos/ano.
