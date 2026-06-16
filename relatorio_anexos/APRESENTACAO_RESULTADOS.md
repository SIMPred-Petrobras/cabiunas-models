# Detecção de Anomalias — Turbina A (Cabiúnas)
### Relatório de resultados para apresentação · jun/2026

---

## 1. O problema

Detectar **anomalias nos sensores da Turbina A** (termopares de temperatura T5/TC382 e vibração) **antes** de virarem alarme/falha, a partir das séries temporais. Objetivo: dar à operação um aviso antecipado e à manutenção um indicador de condição.

## 2. O que construímos

Um **autoencoder CNN-1D por sensor**: o modelo aprende o padrão "normal" de cada sensor e mede o **erro de reconstrução** — quando o sinal foge do aprendido, o erro sobe. Sobre esse erro montamos uma **camada de decisão de produção** (suavização EWMA + threshold por sensor + exclusão de equipamento desligado + debounce).

O sistema entrega **duas saídas complementares**:
- **Alarme de evento** → sala de controle.
- **Índice de saúde/condição** → planejamento de manutenção.

## 3. Como validamos — o diferencial

> O modelo foi **treinado e calibrado só com dados de 2025** e testado **no ano de 2024 inteiro, que nunca tinha visto**, com o ponto de operação congelado.

Isso é validação **out-of-sample** de verdade — a evidência mais forte de que o detector **generaliza**, e não é ajuste fino de um único ano. Métrica = a de produção (não número de papel): **recall** (quantos eventos pegou) e **duty-cycle** (quanto tempo o alarme fica ligado).

## 4. Resultados (out-of-sample 2024 — 73 incidentes reais)

| sensor | recall | tempo-em-alerta | falsos alarmes/dia |
|---|---|---|---|
| TC382_06 | 100% | 3% | 0,14 |
| TC382_05 | 86% | 8% | 0,12 |
| T5 (média) | 89% | 8% | 0,18 |
| TC382_04 | 100% | 8% | 0,13 |
| TC382_01 | 100% | 14% | 0,21 |
| TC382_02 | 100% | 23% | 0,14 |
| TC382_03 | 92% | 53% | 0,08 |
| **Conjunto** | **95,2%** | **16,7%** | **< 0,22** |

**Mensagem-chave:** 6 dos 7 sensores avaliáveis pegam **86–100% dos eventos** com o alarme ligado só **3–23% do tempo** — alarme confiável, não permanente.

**Figuras para os slides:**
- Barra recall×duty por sensor: `eval_predictive_out/fig_oos_2024_resumo.png`
- Série temporal TC382_03 (49 eventos): `eval_predictive_out/fig_oos_2024_TC382_03_A.png`
- Mapa de saúde sensor×mês: `eval_predictive_out/health_index_2025_heatmap.png`

## 5. Descobertas que destravaram o resultado

Boa parte do ganho **não veio do modelo, e sim de medir e decidir certo**:
1. **A métrica certa**: a "taxa de falso alarme bruta" era enganosa; a métrica de produção (eventos clusterizados + duty) mostrou o desempenho real.
2. **Excluir equipamento desligado**: ~30% dos alarmes ocorriam com a turbina parada — fora do escopo; contá-los inflava o número.
3. **Ajuste por sensor**: cada sensor tem comportamento próprio → **half-life e threshold individuais** (não um valor único). Isso sozinho levou o conjunto de ~80% para ~95% de recall com baixo alarme.
4. **Ground-truth completo**: descobrimos que os 17 sensores têm histórico de alarme (2022–2026) — uma base de validação muito maior do que se imaginava.

## 6. Limites conhecidos (transparência)

- **TC382_03** fica anômalo boa parte do ano (condição física real, não falso alarme). Não cabe num alarme limpo → entra no **índice de saúde** para a manutenção.
- **Cobertura**: 7 dos 17 sensores têm eventos suficientes para avaliar; os de vibração têm pouquíssimo histórico útil.

## 7. O que testamos e NÃO funcionou (rigor)

Para chegar ao melhor ponto honesto, descartamos por evidência: mais pré-processamento, trocar arquitetura (GRU/LSTM/Dense/Transformer), ensemble, agrupamento multivariado, threshold mais agressivo, e detecção por desvio de baseline. **Conclusão: o modelo não é o gargalo.**

## 8. Próximos passos

**Não dependem de mais modelagem** — dependem de dado e de produto:
1. **Operacionalizar** o que já funciona (scoring em streaming + painéis de alarme e de saúde).
2. **Pedir à Petrobras**: (a) **sinais contínuos** dos instrumentos analógicos hoje sem curva (temperatura de exaustão, temperaturas de mancal); (b) **mais histórico de sensor** (2024-H1 e anos anteriores). É o que amplia cobertura e robustez.

## Resumo executivo (1 frase)

**Detector validado out-of-sample com 95% de recall e alarme ligado só 17% do tempo; pronto para operacionalizar; o próximo salto depende de dado, não de modelo.**

---
*Detalhe técnico e reprodutibilidade: `relatorio_anexos/RELATORIO_VALIDACAO_OOS_2024.md` e `relatorio_anexos/HANDOFF_DETECTOR_TURBINA_A.md`.*
