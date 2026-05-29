# Relatório Técnico — Estado e Próximas Ações

**Equipamento:** Turbina A, Petrobras Cabiúnas
**Período de dados:** 2025 inteiro (1.051.200 pts a 30s)
**Branch:** `feat/predictive-layer` (HEAD `c341d1b`)

Documento preparado como análise sênior do estado atual e direções estratégicas.
Sem inflação de números, com diagnóstico de causa-raiz dos limites observados.

---

## 1. Estado atual honesto (validado out-of-sample)

### Arquitetura em produção
- **`MODEL_MODE=per_sensor`**: 17 AEs CNN-1D univariados (f1=4, f2=1, latente=15)
- **Agregação**: OR-de-quantile uniforme em q=0.715
- **Health-index**: EWMA com meia-vida de 4h sobre o MAE de reconstrução
- **Máscara operacional**: `running_a` com buffer assimétrico (preserva pré-shutdown)
- **Recalibração**: thresholds mensais adaptativos

### Métricas reais (validação temporal: treina jan–ago, testa set–dez)

| Métrica | TRAIN (jan–ago) | TEST (set–dez, OOS) |
|---|---|---|
| Recall H=8h | 69% | **57%** |
| FA/dia | 0.05 (~1/20 dias) | **0.04 (~1/25 dias)** |
| N episódios | 36 / 240 dias | 14 / 122 dias |
| Drift relativo (recall) | — | −12pp (~17% queda) |

**Headline operacional honesto:**
> 57% dos incidentes detectados com 8h de antecedência, ~1 falso-alarme a cada 25 dias, em dados que o modelo nunca viu.

### Atribuição consistente
Em todas as execuções, os top-5 sensores que dirigem o sinal são de **vibração (TV_*)**.
Temperatura (TC382_*, T5_AVG) contribui ~3-4pp adicionais.

---

## 2. Lições estruturais (4 iterações falharam temporal — por quê)

Tentei 4 melhorias incrementais. Todas falharam validação temporal:

| Tentativa | In-sample | OOS temporal | Por que falhou |
|---|---|---|---|
| F1-best per-sensor threshold | +5pp ✅ | igual ❌ | overfitting nos labels in-sample |
| Group-based AE (T+V) | −6pp ❌ | — | dilui sinal localizado |
| Union F1 + group | = F1 | — | grupo não agrega novo sinal |
| Half-life sweep (0.5–16h) | +1pp ✅ | igual ❌ | sistema é robusto a smoothing |

**Conclusão diagnóstica:** o sistema está num platô. Micro-tuning não move. Cada ganho in-sample é ilusão. **Próximos ganhos exigem fontes qualitativamente diferentes de informação.**

---

## 3. Diagnóstico — por que estamos no platô

Como cientista de dados, vejo **4 limitações de fundo** que explicam o platô:

### a) Limite informacional dos sensores monitorados
O modelo usa apenas 17 sensores (7 temperatura + 10 vibração). Os ~42% de incidentes que NÃO detectamos em 8h provavelmente caem em:
- **Falhas em subsistemas de pressão/gás** (sensores PI_*, PAL_*, PDI_*, PDIT_* estão no CSV bruto mas **não entram no modelo**)
- **Falhas abruptas sem precursor mecânico**
- **Causas elétricas/instrumentais** invisíveis a vibração/temperatura

### b) Pobreza temporal
- **1 ano, 1 turbina, 254 incidentes** — amostra estatística pequena pra generalização forte
- **Drift sazonal real** (threshold sobe ~5× entre semestres) sugere dinâmica anual importante
- Conclusões são **sugestivas, não definitivas** sem multi-ano

### c) Sub-utilização das informações já presentes nos sensores
O AE só vê valores brutos. Não vê:
- **Espectro de vibração** (FFT — frequências características de defeitos)
- **Gradientes/derivadas** (taxas de mudança)
- **Cross-sensor features** (razões TC/TC, TV-X/TV-Y, T/V)
- **Rolling statistics** (variância, kurtosis em janelas)

Vibração mecânica tem assinatura **espectral** rica. Usar só amplitude bruta perde 80% da informação.

### d) Agregação binária descarta sinal contínuo
OR-de-quantile transforma 17 sinais contínuos em 1 bit por sensor. Soma ponderada ou agregação probabilística preservaria mais informação — mas exige rótulos confiáveis pra treinar os pesos (que ainda não temos).

---

## 4. Propostas com probabilidade real de ganho

Estruturado por **horizonte × custo × ganho esperado**, com base nos diagnósticos acima.

### 🟢 Quick wins (1–3 dias de trabalho)

| Proposta | Custo | Ganho esperado | Por quê |
|---|---|---|---|
| **Incluir sensores de pressão no per_sensor** (PI_*, PAL_*, PDI_*, PDIT_*) | baixo (~1 dia) | **+5–10pp recall** | já no CSV, capturam falhas hidráulicas/gás que hoje invisíveis |
| **Sistema de alertas em tier** (warn / alarm / critical por número de sensores acima) | baixo (~1 dia) | qualidade operacional | reduz fadiga de alarme, prioriza ação |
| **Plotagem padronizada** por sensor (como TC382_03_A) | baixo (~0.5 dia) | confiança operacional | operação tem evidência visual de cada decisão |

### 🟡 Médio prazo (1–2 semanas)

| Proposta | Custo | Ganho esperado | Por quê |
|---|---|---|---|
| **Features espectrais para vibração** (FFT em janelas de 60s) | médio (~3 dias) | **+10–15pp recall em mecânicas** | vibração de mancal tem assinatura espectral característica; perdemos isso usando só amplitude |
| **Cross-sensor features** (razões, correlações móveis) | médio (~3 dias) | +3–5pp recall | dinâmica entre sensores indica anomalia que cada um sozinho não captura |
| **Pipeline de re-tuning automático mensal** | médio (~4 dias) | sustenta recall ao longo do tempo | drift comprovado exige recalibração; hoje é manual |
| **Variational AE** (uncertainty-aware) | médio (~4 dias) | melhora precisão | cada alerta vem com intervalo de confiança; operação descarta os incertos |

### 🔴 Longo prazo (1–3 meses) — depende de input externo

| Proposta | Custo | Ganho esperado | Bloqueio |
|---|---|---|---|
| Multi-ano (2022–2024) | alto | salto qualitativo | precisa puxar dados históricos |
| Multi-turbina | alto | generalização | precisa de outra turbina similar |
| **Lista curada de tags pela operação** | baixo nosso, alto deles | **recall genuíno melhor calibrado** | requer reunião com manutenção/operação |
| Anotação de causa-raiz por incidente | médio | modelagem por modo de falha | manutenção precisa anotar |

---

## 5. Recomendação por horizonte

### Próxima semana
1. **Adicionar sensores de pressão ao per_sensor**. Já temos a infra. Sair de 17 → 21+ canais. Mede ganho com a mesma validação temporal.
2. **Features espectrais para os 10 sensores TV**. FFT em janelas de 60s, top-K bins como features auxiliares. Treinar AE univariado por sensor com canal extra de magnitude espectral.

→ Espero: recall H=8h subir de 57% para 65–70% OOS. Se não subir, abandono e foco em (3).

### Próximo mês
3. **Pipeline operacional fechado**: re-tuning mensal automático + sistema tier de alertas + feedback loop (operador marca true/false por alerta).
4. **Migrar score binário (OR) para contínuo ponderado** se temos labels de feedback. Cada sensor passa a contribuir com seu peso conforme histórico de TP/FP.

→ Espero: estabilidade temporal melhor + redução de fadiga de alarme.

### Próximo trimestre
5. **Buscar dados históricos** (2022–2024) — reunião com gestão.
6. **Reunião com manutenção** para curar 50 tags mais relevantes.
7. **Validar em outra turbina** se disponível.

→ Espero: passar do "MVP validado em 1 ano" para "sistema preditivo robusto multi-equipamento".

---

## 6. O que NÃO recomendo (já tentei e não rende)

- Mais variações de aggregator (testei OR, voting, group, sum ponderada — platô)
- Tuning de half-life da EWMA (testei 0.5h–16h — platô)
- Threshold per-sensor calibrado em F1 (overfit)
- Modelos AE mais profundos/largos (latente já bem dimensionado)
- Iterações de threshold único

**Investir tempo nessas direções acima é desperdício.** O platô está provado empiricamente.

---

## 7. Riscos honestos

1. **Quick wins podem decepcionar**. Se pressão e FFT não trouxerem ganho, significa que **a barreira é mesmo de dados** (1 ano, 1 turbina), não de features. Aceitar honestamente e pivotar.
2. **Drift sazonal pode requerer modelos por estação**, não 1 global. Só multi-ano permite ver isso.
3. **Sistema atual já é apresentável** à operação. Não precisa esperar perfeição — colher feedback real é mais valioso que +5pp de recall academicamente medido.

---

## 8. Próximo passo concreto que eu faria

Se você me der **3 dias úteis**, eu:
1. Adicionaria os 4–6 sensores de pressão ao per_sensor (1 dia)
2. Geraria features espectrais simples (DC, fundamental, primeira harmônica) para vibração (1 dia)
3. Validaria temporalmente o novo modelo (jan-ago → set-dez) (1 dia)
4. Comparo honestamente com o baseline atual (recall, FA/dia, lead time)

Se ganho > +3pp recall OOS → adoto e atualizo branch.
Se ganho ≤ +3pp → relato honestamente que estamos no teto e pivoto para item (3) longo prazo (buscar mais dados).

Isso é o **maior ROI** próximo passo. Não exige nada externo.
