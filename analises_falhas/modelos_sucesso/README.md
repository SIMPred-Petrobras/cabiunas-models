# Modelos bem-sucedidos — análise comparativa e próximos passos

Dos 12 equipamentos rodados na pipeline CNN1D-AE (produção real, 10 trials × 15 épocas), **4 conseguiram detectar a falha documentada dentro da janela de avaliação (±48h)**: B-8802B, B-4064A, B-6511502A e B-90001A. Esta pasta reúne a análise de cada um (copiada de `../analise_falha_<EQUIPAMENTO>/`) e um raio-x comparativo de qualidade de detecção — em especial, quantos **falsos positivos** cada modelo gera fora da janela da falha, já que "detectar a falha" sozinho não diz nada sobre confiabilidade operacional do alarme.

## Comparativo quantitativo

| Equipamento | Threshold | Anom./dia (média) | Dias de dados | Tempo anômalo fora da janela | Episódios de falso positivo | Lead time (falha) |
|---|---|---|---|---|---|---|
| **B-4064A** | 0.0389 | 11,3 | 124 | **0,11%** | **2** | ~51h |
| **B-6511502A** | 0.0731 | 3,3 | 365 | 1,00% | 12 | ~9,8 dias* |
| **B-8802B** | 0.1126 | 51,5 | 68 | 1,66% | 9 | ~2 dias |
| **B-90001A** | 0.0552 | 31,4 | 274 | **2,20%** | **95** | ~7 dias* |

\* lead time = primeiro ponto anômalo isolado nos 10 dias antes da falha, não necessariamente o início do sinal sustentado — ver README de cada equipamento.

**Leitura:** "episódios de falso positivo" = blocos contínuos de anomalia de ponto (gap > 1h separa blocos) fora da janela de ±48h da falha, ao longo de todo o período de dados do equipamento.

## Ranking de confiabilidade

1. **B-4064A — melhor caso.** Só 2 episódios de falso positivo em 124 dias (0,11% do tempo). A regra `k_of_window` filtrou quase todo o ruído do MAE bruto (que cruza o threshold com frequência), concentrando o alarme quase exclusivamente na janela da falha real.
2. **B-6511502A — bom, com ressalva.** 12 episódios em 365 dias — mais tempo de operação observado, então a taxa proporcional (1,0%) é comparável à do B-8802B. Falsos positivos são esparsos e curtos.
3. **B-8802B — moderado.** 9 episódios em só 68 dias de dados (janela de observação bem mais curta que os outros) — proporcionalmente mais denso que B-6511502A. Prováveis transientes operacionais, já que `ENABLE_OPERATIONAL_MASK` está desativado nesta config.
4. **B-90001A — pior caso entre os "bem-sucedidos".** 95 episódios de falso positivo em 274 dias — em média mais de 1 episódio a cada 3 dias. Tecnicamente "detectou" a falha (hit_rate=1.0), mas um operador que recebesse esse alarme na prática teria fadiga de alarme quase imediata. Esse resultado não deveria ser tratado como sucesso operacional sem mitigação adicional.

## Recomendações para melhorias futuras

1. **Ativar `ENABLE_OPERATIONAL_MASK` nos equipamentos ruidosos (B-90001A, B-8802B).** Hoje está desativado em todas as configs de produção. Se parte dos falsos positivos vier de partidas/paradas/transientes de operação (não degradação real), a máscara operacional deveria suprimir boa parte deles — vale testar antes de qualquer mudança de modelo.

2. **Calibração de threshold por equipamento, não taxa-alvo genérica de 1%.** `TARGET_ANOMALY_RATE=1%` foi aplicado igual para todos, mas o resultado real variou de 0,2 a 51,5 anomalias/dia entre equipamentos (ver tabela do sweep completo). Para B-90001A em particular, uma taxa-alvo mais conservadora (ex.: 0,1–0,3%) ou um modo de threshold diferente (`MAD`/desvio robusto em vez de `target_rate`) pode reduzir os 95 episódios sem perder a detecção real.

3. **Exigir duração mínima de episódio antes de alarmar (hysteresis).** A regra atual (`k_of_window`, min_count=5 em janela de 60) já filtra ruído ponto-a-ponto, mas ainda deixa passar blocos curtos isolados. Uma camada adicional — só soar alarme se o estado anômalo persistir por, por exemplo, >2h contínuas — provavelmente eliminaria boa parte dos 95 episódios do B-90001A sem afetar a janela real da falha (que teve sinal sustentado por dias).

4. **Validar com mais de 1 evento de falha por equipamento quando possível.** Hoje o hit_rate é sobre N=1 (exceto B-5401A e B-5501B, que têm 2-3 eventos, mas justamente não detectaram nenhum). Um único acerto não é validação estatística — os 4 "sucessos" aqui podem incluir coincidência. Priorizar equipamentos com múltiplos eventos documentados para uma validação mais robusta assim que houver dados suficientes.

5. **Cruzar os episódios de falso positivo com logs de manutenção/operação, se existirem.** Alguns "falsos positivos" podem ser eventos reais não documentados como falha formal (pequenos ajustes, alarmes de campo, manutenção preventiva) — vale checar antes de descartá-los como ruído puro.

6. **Testar features derivadas (`ENABLE_DERIVED_FEATURES`).** Está desativado em todas as configs atuais. Features de rolling window (tendência, variância móvel) podem ajudar o autoencoder a distinguir transiente rápido (ruído) de degradação progressiva (sinal real), especialmente no caso do B-90001A.

7. **Repetir o experimento do B-90001A com a máscara operacional ativada como teste A/B isolado**, comparando hit_rate e nº de episódios de falso positivo antes/depois — é o candidato mais claro para uma melhoria mensurável de curto prazo.

## Conteúdo desta pasta

- `B-8802B/`, `B-4064A/`, `B-6511502A/`, `B-90001A/` — cópia das análises individuais (README + gráficos de visão geral e zoom).
