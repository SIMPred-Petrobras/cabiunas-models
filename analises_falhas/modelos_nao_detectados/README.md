# Diagnóstico crítico — por que 8 dos 12 modelos não detectaram a falha

Este documento investiga, com evidência quantitativa direta nos dados brutos (não inspeção visual), **por que** os 8 equipamentos abaixo tiveram `hit_rate = 0` (nenhuma anomalia de ponto detectada na janela de ±48h da falha documentada), apesar de todos usarem a mesma arquitetura, mesmos hiperparâmetros de treino e mesma metodologia de calibração dos 4 casos bem-sucedidos (ver `../modelos_sucesso/`).

## Conclusão principal (atualizada após investigação completa)

**A máscara de estado operacional (`ENABLE_OPERATIONAL_MASK`) é a causa dominante — não uma hipótese isolada de 1-2 casos.** Reconstruímos o estado operacional (`on` / `off_curto` / `off_longo` / `transiente`, função `build_operational_state` em `src/cnn1d_ae/scoring.py:89`) a partir do sensor de referência real de **cada um dos 8 equipamentos**, para **todos os 11 eventos de falha documentados** (alguns equipamentos têm mais de 1 falha). Resultado:

| Equipamento | Evento (data) | % tempo em estado "on" nas 48h antes da falha |
|---|---|---|
| B-4703.24001B | 2022-10-31 | **0,0%** |
| B-0302C | 2024-08-30 | **0,0%** |
| B-5401A | 2024-12-09 (evento 2) | **0,0%** |
| B-5501B | 2023-10-31 (evento 2) | **0,0%** |
| B-5501B | 2022-10-31 (evento 1) | 0,2% |
| B-5501B | 2024-01-31 (evento 3) | 0,3% |
| B-3403C | 2023-09-12 | 11,1% |
| B-8801C | 2024-07-05 | 24,6% |
| B-402E | 2019-10-30 | 28,3% |
| B-24001B | 2025-01-06 | 90,4% (não é o problema aqui) |
| B-5401A | 2024-08-10 (evento 1) | 98,6% (não é o problema aqui) |

**Em 9 dos 11 eventos de falha avaliados (7 dos 8 equipamentos), o equipamento passou a maior parte (ou a totalidade) das 48h antes da falha em estado diferente de "on".** Como `mask_anomaly_seq_by_operational_state` (`scoring.py:141`) zera qualquer anomalia de sequência fora do estado `on`, isso significa que — independente de o MAE ter subido ou não — **a lógica da própria pipeline impediu o alarme de aparecer** na maioria destes casos, muitas vezes de forma **total** (0,0% de janela "elegível" para alarme).

Isso muda a hierarquia de causas em relação à nossa primeira leitura: o que antes parecia "sinal fraco ou ausente" em B-4703.24001B e B-0302C era, na verdade, **supressão de 100% da janela pela máscara** — o modelo nunca teve chance de soar o alarme, com sinal forte ou fraco.

## Por que a máscara suprime justamente o período da falha?

A hipótese mais provável, coerente em todos os 7 casos: **falhas mecânicas em bombas/motores tendem a ser precedidas por ciclos de liga/desliga** — seja porque o equipamento entra em proteção/trip intermitente à medida que degrada, seja porque operadores já reagem a um problema perceptível cortando a operação repetidamente. O sensor de referência operacional (`Corrente`, na maioria dos casos) cai a valores próximos de zero durante esses ciclos, e a lógica `build_operational_state` — desenhada para evitar falsos alarmes durante paradas/partidas *normais* — classifica esse período como `off_curto`/`off_longo`/`transiente` e suprime a análise, exatamente quando ela seria mais valiosa.

Nos casos com padrão `off_curto`/`transiente` dominante (B-5501B, todos os 3 eventos) em vez de `off_longo`, o padrão é de **ciclos curtos e repetidos**, não uma parada única — reforçando a hipótese de comportamento errático pré-falha, não uma parada de manutenção programada.

## Os 2 casos onde a máscara NÃO é a explicação

### B-24001B (90,4% em "on") — sinal breve, não suprimido

Aqui o equipamento operou quase normalmente até a falha. Investigação separada (ver histórico da análise) mostrou que o MAE de fato sobe perto do topo da distribuição do equipamento (percentil ~100%), mas a excedência acima do threshold dura só ~64 minutos contínuos — insuficiente para a regra `k_of_window` (exige ≥5 de 60 sequências). **Causa aqui: sinal real, mas breve demais para a regra de persistência — não é problema de máscara.**

### B-5401A, evento 1 de 2 (10/08/2024, 98,6% em "on") — threshold degenerado

Neste evento específico a máscara também não é o problema. Encontramos algo mais grave: o sensor-alvo "Corrente" tem **contaminação por outliers extremos** — 0,999 dos dados sobe a ~2,3×10¹⁴ (valor de leitura absurdo, provavelmente falha de instrumentação/SCADA não filtrada), quando o 99º percentil normal é ~285. Como o clip de outliers do pipeline é por quantil (`OUTLIER_MODE=quantile`, corta em 99,9%), e a contaminação afeta mais que 0,1% dos dados, **o próprio corte de outlier calcula um limite contaminado** e não elimina o problema. Resultado: o threshold calibrado (`target_rate=1%`) colapsa para ~0,000115 — um valor que coincide com os quartis 25%, 50% e 75% da distribuição inteira (ou seja, o "MAE" vira uma métrica praticamente constante, sem poder discriminativo). Verificamos que **outros 7 sensores-alvo não têm essa contaminação** (razão p99,9/p99 entre 1,0x–2,1x, normal) — é um problema isolado do sensor "Corrente" neste equipamento específico.

O evento 2 de B-5401A (09/12/2024) tem o problema duplo: além do threshold degenerado, também caiu num período de 0,0% em "on".

## Correlação sensor↔alvo (achado secundário, ainda relevante)

Independente da máscara, **B-3403C** continua tendo o pior conjunto de correlações de todos os 12 equipamentos (`|r| máx = 0,18` entre "Vibração" e os 10 sensores do grupo, contra `>0,6` em todos os 4 casos bem-sucedidos) — mesmo destravando a máscara, é possível que o modelo não tenha estrutura multivariada suficiente para reagir à degradação. **B-5501B** tem problema parecido (só 1 de 2 sensores de entrada com correlação razoável, 0,56) somado a `TIME_STEPS=12` (a menor janela de sequência entre todas as configs — as demais usam 60) e resolução mais esparsa (~5 min por leitura, contra ~1 min nos outros).

## Tabela-resumo de causas-raiz (atualizada)

| Equipamento | Causa-raiz dominante | Confiança |
|---|---|---|
| B-4703.24001B | Máscara operacional — supressão TOTAL (0% on) | 🔴 confirmado |
| B-0302C | Máscara operacional — supressão TOTAL (0% on) | 🔴 confirmado |
| B-5501B | Máscara operacional — supressão quase total (0-0,3% on, nos 3 eventos) + correlação fraca + `TIME_STEPS` curto | 🔴 confirmado (máscara) + 🟠 (correlação) |
| B-8801C | Máscara operacional — supressão parcial (24,6% on) sobre sinal forte e sustentado (859 min contínuos acima do threshold) | 🔴 confirmado |
| B-402E | Máscara operacional (28,3% on) + sinal já fraco (falha tipo "trip catastrófico") | 🔴 confirmado |
| B-3403C | Máscara operacional (11,1% on) + correlação péssima com o alvo | 🔴 confirmado (máscara) + 🟠 (correlação) |
| B-5401A | Threshold degenerado por contaminação de outlier (evento 1) + máscara operacional (evento 2, 0% on) | 🔴 confirmado (ambos, por evento) |
| B-24001B | Sinal real, mas breve (~64 min contínuos) — não sustenta `POINT_MIN_COUNT=5` | 🟡 confirmado que não é máscara; causa é a regra de persistência |

## Ação recomendada por causa

| Causa | Equipamentos | Ação |
|---|---|---|
| Máscara suprimindo sinal | B-4703.24001B, B-0302C, B-5501B, B-8801C, B-402E, B-3403C, B-5401A (evento 2) | Re-treinar com `ENABLE_OPERATIONAL_MASK=false` — é a correção de maior impacto potencial, afeta 7 dos 8 equipamentos |
| Threshold degenerado | B-5401A | Trocar `OUTLIER_MODE: quantile → mad` (mediana é robusta à contaminação; quantil não é quando a contaminação excede a cauda de corte) |
| Correlação fraca | B-3403C, B-5501B | Sem sensor alternativo disponível — testar mesmo assim com máscara desligada; se não melhorar, considerar univariado ou outro sensor-alvo |
| `TIME_STEPS` curto | B-5501B | Igualar a `TIME_STEPS=60` como os demais, para dar mais contexto temporal por sequência |
| Regra de persistência rígida | B-24001B | Reduzir `POINT_MIN_COUNT` (de 5 para 2-3) para captar excedências mais breves |

## Limitações desta análise

- Reconstruímos o estado operacional fora da pipeline (script auxiliar, mesmos parâmetros da config), não a partir do artefato real gerado durante o treino — o resultado deveria ser idêntico, mas não foi extraído diretamente do artefato do ClearML.
- Não cruzamos os períodos de "off"/"transiente" pré-falha com logs de manutenção ou operação (se existirem) — não sabemos se os ciclos de liga/desliga foram reação de operadores a um problema já perceptível (o que mudaria a interpretação de "falha de detecção" para "detecção tardia, já visível para humanos") ou comportamento intrínseco do modo de falha.
- `n_alarms=1` na maioria dos equipamentos — qualquer conclusão aqui é sobre um único evento por equipamento (exceto B-5401A e B-5501B, com 2 e 3 eventos, o que de fato ajudou a robustecer o diagnóstico ao mostrar o mesmo padrão se repetindo).
