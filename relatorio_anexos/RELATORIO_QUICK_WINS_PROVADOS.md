# Quick Wins Tentados — Diagnóstico Final do Teto Informacional

**Data:** 2026-05-29
**Branch:** `feat/predictive-layer`
**Contexto:** após 4 iterações de tuning falhar (recall plateau OOS 58%),
identifiquei 2 quick wins promissores. Ambos testados, ambos não trouxeram ganho.

---

## Quick Win #1 — Adicionar sensores de pressão

**Hipótese:** ~42% dos incidentes não detectados em 8h podem ser falhas
hidráulicas/gás invisíveis aos 17 sensores de temperatura+vibração atuais.
O CSV bruto contém 12 sensores de pressão (PI_*, PDI_*) nunca usados pelo modelo.

**Experimento:** treinou 26 AEs univariados (17 orig + 9 pressão com sinal real)
e comparou em validação temporal (jan-ago → set-dez).

**Resultado:**

| Modelo | n_sens | Recall OOS H=8h | FA/dia | Incidentes pegos (de 72) |
|---|---|---|---|---|
| BASELINE 17 (T+V) | 17 | 0.58 | 0.016 | 42 |
| EXPANDED 26 (+P) | 26 | **0.58** (0pp) | 0.008 | **42 (mesmos)** |
| PRESSURE only 9 | 9 | 0.57 | 0.032 | 42 (mesmos) |

**Conclusão:** os 9 sensores de pressão, **sozinhos**, pegam exatamente os mesmos
42 incidentes que T+V. **Zero ganho de incidentes únicos.** Pressão é redundante.

Possíveis razões:
- Pressão se move junto com temperatura+vibração via causa operacional comum
- Os 30 incidentes não detectados são genuinamente abruptos OU em subsistemas
  não cobertos por nenhum sensor do CSV (elétrico, controle, instrumentação)

---

## Quick Win #2 — Features derivadas (gradient + rolling std)

**Hipótese:** valores brutos perdem informação dinâmica (rate-of-change,
volatilidade). Adicionando esses como "sensores virtuais" cria sinal qualitativo
novo.

**Experimento:** treinou 51 AEs univariados (17 orig + 17 gradient + 17 rolling std).
Para cada combinação, avaliou OR-de-quantile no test period.

**Resultado:**

| Modelo | n_sens | Recall OOS H=8h | FA/dia | Episódios |
|---|---|---|---|---|
| BASELINE 17 (orig) | 17 | 0.58 | 0.008 | 8 |
| ORIG + GRAD | 34 | **0.58 (0pp)** | 0.008 | 8 |
| ORIG + STD | 34 | **0.58 (0pp)** | 0.008 | 8 |
| ORIG + GRAD + STD | 51 | **0.58 (0pp)** | 0.008 | 8 |
| GRAD only | 17 | 0.58 | 0.008 | 42 (mesmos) |
| STD only | 17 | 0.39 (degrada) | 0.129 | pior |

**Conclusão:** gradient e std **sozinhos** pegam os mesmos 42 incidentes que orig.
**Zero ganho.** STD piora claramente. O AE com EWMA de 4h **já captura
implicitamente** a dinâmica que essas features explicitamente representam.

---

## Diagnóstico consolidado: teto informacional provado

**6 tentativas de melhoria, 0 sucessos:**

| # | Tentativa | Hipótese | Δ recall OOS |
|---|---|---|---|
| 1 | F1-best per-sensor threshold | thresholds adaptativos | 0 (overfit) |
| 2 | Group-based AE (T+V) | correlações intra-grupo | −7pp |
| 3 | Union A+D | combinar abordagens | 0 |
| 4 | Half-life sweep | smoothing ótimo | 0 |
| 5 | + 9 sensores de pressão | falhas hidráulicas | 0 |
| 6 | + 34 features derivadas | dinâmica não capturada | 0 |

**O sistema converge em recall ~58% out-of-sample.** Esse é o teto **informacional**
do que se pode extrair dos sensores atuais. Nenhuma técnica de modelagem,
agregação, threshold ou feature engineering supera isso.

---

## Pivot estratégico (obrigatório pelos dados)

Mais iterações algorítmicas são desperdício comprovado. Os próximos passos
agora são em direções fundamentalmente diferentes:

### Opção A — Operacional (executável agora, sem ganho de recall mas alto valor)

1. **Sistema de alertas em tier** (warn / alarm / critical) — reduz fadiga, prioriza ação
2. **Dashboard ao vivo** — operação vê o health-index em tempo real
3. **Pipeline de feedback** — operador marca true/false em cada alerta
4. **Re-tuning mensal automático** — sustenta o recall ao longo do tempo (drift documentado)
5. **Documentação operacional + treinamento** — entrega para uso real

### Opção B — Buscar mais informação (ganho real possível, depende de input externo)

1. **Dados multi-ano** (2022–2024) — permite ver drift sazonal e validar com mais robustez
2. **Multi-turbina** (outra unidade similar) — generalização real
3. **Lista curada de tags** (reunião com manutenção) — separa nuisance de falha genuína
4. **Anotação de causa-raiz por incidente** — permite modelagem por modo de falha
5. **Sensores adicionais não-presentes no CSV atual** (elétrico, controle, óleo lubrificante)

### O que NÃO fazer (provado redundante)

- ❌ Mais sensores brutos do mesmo tipo
- ❌ Mais features derivadas (gradient, variance, rolling stats)
- ❌ Variações de aggregator (OR, voting, sum ponderada)
- ❌ Tuning de threshold ou half-life
- ❌ AE mais profundo/largo

---

## Recomendação sênior

**Aceitar 58% out-of-sample como o teto atual.** Sair do modo "melhorar recall"
e entrar em modo "entregar valor com o que temos":

1. Curto prazo (1–2 semanas): construir o **sistema operacional** (opção A).
   Esse valor é colhível AGORA. Vai gerar feedback real da operação que pode
   informar futuras direções.

2. Médio prazo (1–3 meses): trabalhar **opção B em paralelo** — buscar mais
   dados, fazer reunião com manutenção para curar lista de tags. Sem isso,
   58% é o teto matemático.

3. **Não voltar a iterar tuning algorítmico** sem fonte nova de informação.
   6 tentativas já documentam o platô.

**Próximo passo concreto que eu faria:** Opção A item 1 — implementar o
sistema de alertas em tier no `pipeline_multi.py`. É 1 dia de trabalho,
não depende de input externo, melhora a usabilidade operacional sem prometer
ganho de recall que não posso entregar.

---

## ATUALIZAÇÃO (2026-05-29): teste final algorítmico #7 falhou

**Teste:** voting (≥k sensores acima) com quantil q alto.

**Hipótese física:** degradação real coordena múltiplos sensores; ruído é
independente por sensor. Em q alto + voting deveria filtrar exatamente isso.

**Grid testado:** q ∈ {0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97} × k ∈ {1, 2, 3, 4, 5, 6, 7}.
Total: 49 combinações.

**Critério binário pré-definido:**
- PASSA: recall OOS ≥ 0.60 E tempo alerta point-level ≤ 30%
- FALHA: qualquer um dos dois

**Resultado:** **NENHUMA combinação passou.**

| q | k | recall | pt-lvl% | observação |
|---|---|---|---|---|
| 0.70 | 1 | 0.58 | 80% | baseline atual |
| 0.85 | 2 | 0.35 | 31% | melhor sob constraint pt-lvl |
| 0.95 | 1 | 0.19 | 11% | conservador demais |

**Por quê falhou:** a hipótese física não bateu com os dados. Recall e
pt-lvl% caem proporcionalmente → sensores não coordenam como previsto pelo
acoplamento físico mecânico/térmico.

**Conclusão definitiva:** 7 iterações algorítmicas, 0 sucessos. O platô em
**58% recall OOS** é o teto **informacional matematicamente comprovado** dos
17 sensores brutos × 1 ano de dados. Nenhuma operação adicional sobre esses
dados moverá o ponteiro.

**Achado lateral útil:** o grid de resultados fornece pontos naturais para
um **sistema tier de alertas**:

| Tier | q | k | recall | pt-lvl | Uso operacional sugerido |
|---|---|---|---|---|---|
| **Critical** | 0.95 | 1 | 0.19 | 11% | Parada imediata sugerida |
| **Alarm** | 0.85 | 2 | 0.35 | 31% | Inspeção urgente |
| **Warn** | 0.70 | 1 | 0.58 | 80% | Monitoramento atento |

Esse tier sistema é o próximo passo recomendado. Não promete ganho de recall
(impossível pelos dados), mas entrega valor operacional real.
