# Análise de falha — Equipamento B-402E

## Qual falha estamos tentando predizer

- **Equipamento:** B-402E
- **Falha(s) registrada(s):** *"Quebra de barra do rotor do motor com colisão no enrolamento estatórico — TRIP catastrófico"*
- **Data(s) da(s) falha(s):** 2019-10-30 11:06
- **Task ClearML (produção, real):** `1a801a18ede944a18d4fea53356f8286`
- **Nº de eventos de falha documentados:** 1

## Modelo configurado

- **Sensor-alvo (o que o autoencoder tenta reconstruir):** Corrente
- **Sensores de entrada (contexto multivariado, além do alvo):** 5
  - Pressão Descarga
  - Temperatura Estator U
  - Temperatura Estator V
  - Temperatura Estator Wa
  - Temperatura Estator Wb

- **Total de sensores no grupo (entrada multivariada):** 6
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
| Threshold calibrado (MAE) | 0.0180 |
| Taxa de anomalia (pontos/dia) | 7.9 |
| Alarmes documentados | 1 |
| Alarmes com anomalia detectada na janela (±48h) | 0 / 1 (hit_rate = 0.00) |

> ⚠️ **O modelo NÃO detectou nenhuma anomalia dentro da janela de ±48h dos eventos de falha.** Isso não significa necessariamente que o modelo é ruim — pode indicar que a degradação não é visível nos sensores escolhidos, que o threshold calibrado (taxa-alvo 1%) ficou conservador demais para este equipamento, ou que a falha teve caráter mais abrupto/sem precursores mensuráveis nesses canais.

### Antecedência de detecção (lead time)

| Evento | Data da falha | Descrição | Primeiro ponto anômalo (10 dias antes) | Lead time |
|---|---|---|---|---|
| 1 | 2019-10-30 11:06 | Quebra de barra do rotor do motor com colisão no enrolamento estatórico — TRIP catastrófico | — | — (nenhum ponto anômalo nos 10 dias anteriores) |

> **Atenção:** essa coluna mostra o *primeiro* ponto marcado como anômalo nos 10 dias antes da falha — não implica necessariamente um sinal sustentado, nem que caiu dentro da janela de avaliação (±48h). Ver os gráficos de zoom para a forma real do sinal (ramp-up sustentado vs. blip isolado).

## Gráficos

### Visão geral (todo o período de dados)

![Visão geral](01_visao_geral.png)

### Zoom ±10 dias ao redor da falha (2019-10-30 11:06)

![Zoom na falha](02_zoom_falha.png)

## Observação sobre granularidade da data da falha

Quando a falha só tem precisão de **dia** (sem horário nos registros originais), a hora usada como âncora nos gráficos (00:00) é apenas convenção. Isso não compromete a janela de exclusão/avaliação, pois ±48h é bem mais larga que a incerteza de 24h do dia.

## Arquivos

- `01_visao_geral.png` — MAE horário no período completo, com threshold e janela(s) de exclusão.
- `02_zoom_falha*.png` — MAE (10 min) e contagem de anomalias/ponto, zoom ±10 dias ao redor de cada evento de falha.
