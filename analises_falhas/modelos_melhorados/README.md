# Modelos corrigidos com sucesso (v2) — de hit_rate=0 para detecção real

Dos 8 equipamentos com `hit_rate=0` na primeira rodada (ver `../modelos_nao_detectados/`), retreinamos todos com correções específicas por causa-raiz. **3 confirmaram melhora real** até agora: B-0302C, B-3403C e B-5401A. Esta pasta reúne a análise detalhada de cada um.

## Resumo comparativo

| Equipamento | Causa-raiz | Correção aplicada | Hit rate v1 → v2 | Lead time real | Episódios de falso positivo |
|---|---|---|---|---|---|
| **B-0302C** | Máscara operacional (0% "on" pré-falha) | `ENABLE_OPERATIONAL_MASK: false` | 0,0 → **1,0** | 25,1h | 65 (2,19% do tempo) |
| **B-3403C** | Máscara operacional (11% "on") + correlação péssima | `ENABLE_OPERATIONAL_MASK: false` | 0,0 → **1,0** | 6,7h | 84 (2,37% do tempo) |
| **B-5401A** | Outlier extremo (threshold degenerado) + máscara (evento 2) | `OUTLIER_MODE: mad` + `ENABLE_OPERATIONAL_MASK: false` | 0,0 → **1,0*** | Evento 1: 26,7h · Evento 2: **nenhuma** (detecção foi *pós*-falha) | 65 (1,68% do tempo) |

\* Ver ressalva importante no README do B-5401A — o hit_rate de 1,0 (2/2) conta um evento que foi detectado **depois** da falha, não antes. Só o evento 1 é uma antecipação real.

## Leitura crítica dos resultados

**A correção de máscara operacional funcionou como previsto nos 3 casos** — confirma que a supressão da máscara era, de fato, a causa dominante de não-detecção nesses equipamentos, não um problema de arquitetura ou hiperparâmetros do modelo.

**Mas o preço foi ruído.** Todos os 3 modelos corrigidos são significativamente mais barulhentos que os "sucessos limpos" da primeira rodada (comparar com `../modelos_sucesso/`, onde B-4064A tinha só 2 episódios de falso positivo em 124 dias). Aqui, os 3 giram entre 65-84 episódios em ~1,7-2,4% do tempo — a máscara operacional, ao ser desligada, parou de suprimir tanto o sinal real da falha quanto o ruído de transientes normais que ela também filtrava. Isso é um trade-off esperado, não um bug: a correção resolveu o falso-negativo (não detectar a falha) às custas de mais falsos-positivos.

**Nem todo "hit_rate=1.0" é uma vitória de antecipação.** O caso do B-5401A evento 2 mostra a importância de olhar o lead time individualmente, não só a métrica agregada — uma detecção depois da falha não tem valor preditivo, mesmo contando como "hit" na métrica de janela ±48h.

## Próximos passos sugeridos

- Testar uma versão intermediária (ex.: máscara operacional com parâmetros mais permissivos — `OFF_LONG_MIN_HOURS` maior, para não classificar ciclos curtos como "off" — em vez de desligar totalmente) para tentar reduzir os falsos positivos sem reintroduzir a supressão do sinal real.
- Cruzar os episódios de falso positivo com dados de operação/manutenção, se existirem, para confirmar se são transientes benignos ou sinais menores não documentados como falha formal.

## Conteúdo desta pasta

- `B-0302C/`, `B-3403C/`, `B-5401A/` — análise individual (README + gráficos de visão geral e zoom por evento).
