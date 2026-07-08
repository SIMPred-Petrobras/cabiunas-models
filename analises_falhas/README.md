# Resultados — Detecção de falhas via CNN1D-AE (pipeline Transpetro)

Análise completa dos 12 equipamentos com falha documentada, usando a pipeline multivariada `src/transpetro` (ver `CLAUDE.md` na raiz do repo para detalhes da arquitetura).

## Como navegar

| Pasta | Conteúdo |
|---|---|
| `analise_falha_<EQUIPAMENTO>/` | Análise individual de cada um dos 12 equipamentos: qual falha, sensores usados, resultado (threshold/hit-rate), gráficos de visão geral e zoom na falha. Ponto de partida para qualquer equipamento específico. |
| `modelos_sucesso/` | Os 4 equipamentos que detectaram a falha já na primeira rodada (B-8802B, B-4064A, B-6511502A, B-90001A), com análise comparativa de falsos positivos e recomendações. |
| `modelos_nao_detectados/` | Diagnóstico crítico dos 8 equipamentos que **não** detectaram a falha na primeira rodada — causas-raiz identificadas (máscara operacional suprimindo o sinal, contaminação de outlier, correlação fraca, regra de persistência rígida) com evidência quantitativa nos dados brutos. |
| `modelos_melhorados/` | Resultado do retreino (v2) dos 8 casos acima com correções direcionadas por causa-raiz. 3 confirmaram melhora (B-0302C, B-3403C, B-5401A) — hit_rate foi de 0 para 1, com leitura crítica de lead time real e falsos positivos. |

## Resumo executivo

- **12 equipamentos** avaliados, cada um com falha real documentada (data + descrição).
- **Rodada 1:** 4/12 detectaram a falha dentro da janela de ±48h.
- **Diagnóstico:** causa dominante foi a máscara de estado operacional suprimindo o sinal real em 7 dos 8 casos sem detecção (confirmado nos dados brutos, não suposição) — ver `modelos_nao_detectados/README.md`.
- **Rodada 2 (retreino direcionado):** 3/8 confirmaram melhora até o momento (B-0302C, B-3403C, B-5401A → hit_rate 0 → 1); B-24001B, B-5501B e B-4703.24001B não melhoraram com as correções aplicadas; B-402E e B-8801C ainda em reprocessamento (bug de infraestrutura do ClearML identificado e corrigido — ver commits `76d0154` e `8710b73`).
- **Nenhum modelo "bem-sucedido" está livre de falsos positivos** — mesmo os 7 confirmados com hit_rate=1.0 têm entre 2 e 95 episódios de falso positivo ao longo do período de dados. Ver `modelos_sucesso/README.md` para o comparativo completo.

## Metodologia (resumo)

Pipeline CNN-1D Autoencoder multivariado (`src/transpetro/`), treino real (10 trials × 15 épocas via KerasTuner), execução remota via ClearML Agent com GPU. Cada equipamento usa um grupo de sensores correlacionados (selecionados estatisticamente, ver `analysis/<EQUIPAMENTO>/`) para reconstruir um sensor-alvo; o erro de reconstrução (MAE) acima de um threshold calibrado (`target_rate=1%`) marca pontos anômalos, com uma regra de persistência (`k_of_window`) para filtrar ruído isolado. Detalhes completos de configuração em `configs/transpetro/`.
