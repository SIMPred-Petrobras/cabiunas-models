# Roadmap Preditivo Validado — Detecção de Anomalias Turbina A

**Equipamento:** Turbina A, Cabiúnas
**Dados:** Ano 2025 completo (1.051.200 pontos a 30s, 17 sensores: 7 temperatura + 10 vibração)
**Modelo:** CNN-1D Autoencoder multivariado, sinal preditivo via EWMA do erro de reconstrução
**Branch:** `feat/predictive-layer`

---

## 1. Estado validado (o que pode ir pra apresentação)

### Headline operacional honesto (corrigido 2026-05-28)

> **Per-sensor + OR-de-quantile no ponto operacional real (q=0.715):**
> **67% de recall a 8h, ~1 falso-alarme a cada 35 dias** em produção in-sample.
> Vence o multivariate em ponto operacional matched: +4pp recall, FA/d 2.6× menor.

> ⚠️ **Armadilha metodológica descoberta:** os 69% recall a q=0.50 reportados antes
> eram **artefato de saturação** (sistema em alerta contínuo, só 24 mega-episódios/ano).
> Pico fica em q≈0.715 onde n_episódios/ano (44-55) é próximo de n_incidentes (254/H)
> — relação 1-pra-1 é a faixa operacional real. **Não usar q<0.7 como ponto operacional.**

| Métrica | Multivariate | **Per-sensor (default)** |
|---|---|---|
| Recall H=8h (produção, in-sample) | 0.66 / FA/d 0.05 | **0.69 / FA/d 0.01** |
| Recall H=8h (out-of-sample, temporal local) | 0.51 / FA/d 0.05 | **0.58 / FA/d 0.01** |
| FA/dia em H=24h (produção) | 0.04 | **0.00** |
| Reprodutibilidade local↔produção | sim | sim |
| Atribuição direta por sensor | via MAE per-sensor | **nativa (1 modelo = 1 sensor)** |

Sobreviveu à **validação temporal** (treina jan→ago, testa set→dez sem retuning). Per-sensor mantém o ganho fora-da-amostra. Esse é o número operacional honesto.

### Resultados por horizonte (validação temporal)

| Horizonte | TRAIN (jan-ago, in-sample) | **TEST (set-dez, out-of-sample)** |
|---|---|---|
| 8h | recall 0,67 / FA/d 0,09 / lead 8h | **recall 0,51 / FA/d 0,05** |
| 24h | recall 0,75 / FA/d 0,06 / lead 24h | **recall 0,57 / FA/d 0,03** |
| 72h | recall 0,83 / FA/d 0,03 / lead 72h | **recall 0,79 / FA/d 0,00** |

> ⚠️ Os números a H=24h e H=72h são parcialmente afetados por **saturação de densidade** (incidentes a cada ~1,2 dias). A 8h o número é o mais honesto.

### Arquitetura final: per_sensor + OR-de-quantile

**Decisão arquitetural (2026-05-28):** `MODEL_MODE=per_sensor` é o default. 17 AEs univariados minúsculos (f1=4, f2=1, latente=15-30) treinados independentemente; alerta combinado via **OR-de-quantile** (cada sensor com seu próprio threshold no seu próprio quantil; alerta dispara se qualquer um cruzar).

Por que venceu o multivariado de 17 canais:
- **Não dilui sinal localizado:** 1 sensor degradando faz seu próprio modelo disparar; no multi, o erro médio dos 17 esconde 1 sensor anormal entre 16 normais.
- **Calibração por sensor:** cada sensor tem sua própria distribuição de health-index e seu threshold ao quantil correto. No multi, threshold único precisa compensar escalas heterogêneas.
- **Atribuição nativa:** qual modelo disparou = qual sensor degradou. Não precisa de heurística de atribuição.

Caveat: per-sensor não captura anomalias por decorrelação cruzada entre sensores. Mas os números mostram que, nesse equipamento, esse efeito é menos importante que evitar diluição.

### O que mais foi provado

- **Vibração carrega ~90% do sinal preditivo.** Modelo só de vibração (10 sensores) perde só 4pp de recall vs 17 sensores. Per-sensor mantém todos os 17 porque o custo de adicionar TC e T5_AVG é trivial (1 modelo univariado a mais cada) e contribuem ~3-4pp.
- **Atribuição consistente entre runs:** TV_355X, TV_354Y, TV_353Y, TV_352X dominam. O modelo está vendo a mesma "história" em todas as execuções.
- **Máscara operacional + buffer assimétrico funciona:** 23% das detecções (de desligamento) eliminadas mantendo o sinal pré-trip.
- **Plot por variável funciona:** alarmes do `T5_AVG_A` cortam de 1093 para 224 — visualização honesta.
- **Recalibração mensal do threshold é necessária:** drift entre períodos (threshold ótimo sobe ~5×). Pipeline já faz isso via `apply_adaptive_monthly_threshold`.

---

## 2. O que NÃO está provado

1. **Multi-ano / multi-turbina.** Tudo em 1 equipamento, 1 ano. Conclusões são **sugestivas, não definitivas**. Precisamos de 2023/2024 ou outra turbina para confirmar.
2. **34% dos alarmes sem precursor de 8h.** Pode ser:
   - falha abrupta sem precursor (limite estrutural),
   - precursor está em **pressão** (sensores que o modelo não vê),
   - evento operacional não-mecânico.
   Sem mais dados/sensores, não dá pra distinguir.
3. **Drift do health-index ao longo do ano:** subiu ~5× em 4 meses. Pode ser **degradação real** (equipamento envelhecendo) ou **covariate shift** (mudança operacional/sazonalidade). Não dá pra separar com 1 ano.

---

## 3. Componentes em produção (branch `feat/predictive-layer`)

| Componente | Arquivo | Estado |
|---|---|---|
| **Backend per_sensor (default)** | `src/cnn1d_ae/per_sensor.py` | ✅ **produção** (`MODEL_MODE=per_sensor`) |
| Backend multivariate (legado, ainda disponível) | `pipeline_multi.py` branch multi | ✅ produção (`MODEL_MODE=multivariate`) |
| Curva preditiva OR-de-quantile (per_sensor) | `compute_predictive_curve_per_sensor` em `predictive.py` | ✅ produção |
| Health-index EWMA + curva preditiva escalar | `src/cnn1d_ae/predictive.py` | ✅ produção |
| Integração no pipeline multivariado | `pipeline_multi.py` bloco 8c | ✅ produção |
| Máscara operacional (running_a) + buffer assimétrico | `scoring.py` + config | ✅ produção |
| Plot por variável de alarme | `pipeline_multi.py` | ✅ produção |
| Recalibração mensal do threshold | `apply_adaptive_monthly_threshold` (já existia) | ✅ produção |
| Validação temporal | `scripts/validate_temporal.py` | ✅ harness disponível |
| Experimento de separação | `scripts/ae_separation_experiment.py` | ✅ harness disponível |

---

## 4. Próximos passos por ordem de prioridade

### ALTA — desbloqueia valor sem desenvolvimento adicional

1. **Documentar o ponto operacional para a operação:**
   - Definir alvo de FA/dia aceitável (sugestão: ≤ 0,05).
   - Documentar resposta operacional do alerta (qual sensor + onde olhar).
   - Estabelecer protocolo de feedback (alarmes "úteis" vs "ruído" para retreino).

2. **Refit periódico do modelo:** dado o drift observado, refazer treino a cada 6 meses (ou quando recall mensal cair abaixo de patamar).

3. **Pedir histórico adicional** (anos anteriores ou outras turbinas) para validação cruzada multi-ano.

### MÉDIA — investigações que decidem arquitetura

4. **Investigar os 34% não-predizíveis:** análise por tag de alarme, se possível classificar em "abrupto" vs "ausência de sinal". Pode justificar pedido de sensores de pressão pro modelo.

5. **Estabilizar infra ClearML** (pedido para engenharia): workers 1/4 com problemas recorrentes (GPU OOM / fileserver upload broken). Estamos rodando com sorteio de worker — não-ideal.

### BAIXA — qualidade de vida

6. **Migrar `hit_rate@threshold_único` para legacy:** a métrica oficial agora é a curva preditiva. O hit_rate continua sendo computado por compatibilidade, mas não deve guiar decisões.

7. **Adicionar plot de cada sensor TV individual** vs alarmes próprios (segue o padrão de `plot_alarms_per_variable`).

---

## 5. Bloqueios externos

| Bloqueio | Impacto | Resolução |
|---|---|---|
| `NGP_A` não disponível | Estamos usando `running_a` como proxy (funciona, mas binário). | Operação |
| Lista curada de tags genuínas | Estamos predizendo todos os onsets (filtro `Condição != "OK"`). Funciona — vibração e temperatura têm sinal. | Operação |
| Infra ClearML instável | ~1/3 dos runs falha. Causa não-determinismo no fluxo. | Engenharia |

---

## 6. Caveats metodológicos (transparência para discussão)

- **Densidade de incidentes:** 254 onsets em 362 dias = ~1,2 dias entre eventos. Em horizontes longos, janelas pré-incidente cobrem boa parte do timeline — saturação inflacional de recall. **A 8h o efeito é mínimo; usar essa figura como referência.**
- **Tuner re-busca a cada run remoto:** variância entre runs (~3-5pp em hit_rate) é maior que muitos ganhos micro. A camada preditiva é **robusta a essa variância** (números reproduziram local↔produção).
- **Avaliação contra TODOS os alarmes de 2025** (conforme a operação confirmou que todos são reais). Aplicado filtro `Condição != "OK"` para descartar clears (retorno ao normal não é onset).

---

## 7. Artefatos anexos

- `resultado_validacao_temporal.json` — números da validação temporal
- `curva_temporal_test_H{8,24,72}h.csv` — curvas preditivas fora-da-amostra
- `fig_curva_preditiva.png`, `fig_health_index_ewma.png` — gráficos do experimento local (versão produção pendente de recuperação de artefato)
- `RELATORIO_SEPARACAO_AE.md` — fase 1 (separação normal vs anomalia)
- `RELATORIO_FILTRO_OPERACAO.md` — fase 2a (máscara operacional)

Scripts reproduzíveis em `scripts/`:
- `ae_separation_experiment.py` — separação normal/anomalia
- `ae_predictive_layer.py` — camada preditiva (validação inicial)
- `ae_running_filter.py` — análise da máscara operacional
- `validate_temporal.py` — validação out-of-sample
