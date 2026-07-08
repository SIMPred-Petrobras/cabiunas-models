# Análise de falha — Equipamento B-24001B

## Qual falha estamos tentando predizer

- **Equipamento:** B-24001B
- **Falha(s) registrada(s):** *"Vibração elevada mancal LNA da bomba"*
- **Data(s) da(s) falha(s):** 2025-01-06
- **Task ClearML (produção, real):** `5642c558c9f44fd1b81dcafdbc1386e3`
- **Nº de eventos de falha documentados:** 1

## Modelo configurado

- **Sensor-alvo (o que o autoencoder tenta reconstruir):** VIBRAÇÃO DO MANCAL BOMBA LNA
- **Sensores de entrada (contexto multivariado, além do alvo):** 11
  - VIBRAÇÃO DO MANCAL BOMBA LA
  - VIBRAÇÃO DO MANCAL MOTOR LA (003)
  - VIBRAÇÃO DO MANCAL MOTOR LA (004)
  - TEMPERATURA DO MANCAL MOTOR LA
  - VIBRAÇÃO DO MANCAL MOTOR LNA (005)
  - TEMPERATURA DO MANCAL BOMBA LNA
  - TEMPERATURA DO MANCAL BOMBA LA
  - TEMPERATURA DO MANCAL MOTOR LNA
  - VIBRAÇÃO DO MANCAL MOTOR LNA (006)
  - PRESSÃO NA SUCÇÃO DA BOMBA
  - PRESSÃO NA DESCARGA DA BOMBA

- **Total de sensores no grupo (entrada multivariada):** 12
- Além deste grupo multivariado, a pipeline roda automaticamente modelos **univariados** extras para qualquer outro sensor do feather que não esteja neste grupo (subproduto da execução, não é o foco desta análise).

## Hiperparâmetros / configuração de treino

| Parâmetro | Valor |
|---|---|
| Janela de sequência (`TIME_STEPS`) | 60 passos |
| Stride | 1 |
| Split treino/val | temporal, 90%/10% |
| Normalização | zscore (estatística só do treino) |
| Outliers | quantile (clip 0.001–0.999) |
| Janela de exclusão de alarme (treino/avaliação) | ±2880 min (48h) |
| Busca de hiperparâmetros (KerasTuner) | 10 trials × 15 épocas (patience 5) |
| Batch size | 512 |
| Modo de threshold | target_rate (taxa-alvo 1%) |
| Regra ponto-a-ponto | k_of_window (janela 60, mínimo 5) |
| Máscara operacional | ativada |

## Resultados

| Métrica | Valor |
|---|---|
| Threshold calibrado (MAE) | 0.1630 |
| Taxa de anomalia (pontos/dia) | 20.2 |
| Alarmes documentados | 1 |
| Alarmes com anomalia detectada na janela (±48h) | 0 / 1 (hit_rate = 0.00) |

> ⚠️ **O modelo NÃO detectou nenhuma anomalia dentro da janela de ±48h dos eventos de falha.** Isso não significa necessariamente que o modelo é ruim — pode indicar que a degradação não é visível nos sensores escolhidos, que o threshold calibrado (taxa-alvo 1%) ficou conservador demais para este equipamento, ou que a falha teve caráter mais abrupto/sem precursores mensuráveis nesses canais.

### Antecedência de detecção (lead time)

| Evento | Data da falha | Descrição | Primeiro ponto anômalo (10 dias antes) | Lead time |
|---|---|---|---|---|
| 1 | 2025-01-06 | Vibração elevada mancal LNA da bomba | 2024-12-27 12:03:00 | 227.9h (~9.5 dias) antes |

> **Atenção:** essa coluna mostra o *primeiro* ponto marcado como anômalo nos 10 dias antes da falha — não implica necessariamente um sinal sustentado, nem que caiu dentro da janela de avaliação (±48h). Ver os gráficos de zoom para a forma real do sinal (ramp-up sustentado vs. blip isolado).

## Gráficos

### Visão geral (todo o período de dados)

![Visão geral](01_visao_geral.png)

### Zoom ±10 dias ao redor da falha (2025-01-06)

![Zoom na falha](02_zoom_falha.png)

## Observação sobre granularidade da data da falha

Quando a falha só tem precisão de **dia** (sem horário nos registros originais), a hora usada como âncora nos gráficos (00:00) é apenas convenção. Isso não compromete a janela de exclusão/avaliação, pois ±48h é bem mais larga que a incerteza de 24h do dia.

## Arquivos

- `01_visao_geral.png` — MAE horário no período completo, com threshold e janela(s) de exclusão.
- `02_zoom_falha*.png` — MAE (10 min) e contagem de anomalias/ponto, zoom ±10 dias ao redor de cada evento de falha.
