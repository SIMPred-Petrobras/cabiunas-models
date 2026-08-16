# Análise de Experimentos — AutoML EXP8 (reformulação supervisionada + sobrevivência exploratória)

Fecha o plano de 5 itens iniciado no EXP7 (`docs/analise_automl_exp7_planejamento.md`).
Itens 1--3 (multi-escala, textura, mudança de regime) estão documentados em
`docs/analise_automl_exp7.md`, com candidato de referência: `ocsvm`
(p99.9/debounce=1) sobre `TC382_T5_vibracao_mancais` com features
multi-escala + textura + mudança de regime — 29 preditivos, 8 reativos, 3
sem detecção, FP de 1,93--1,94%, mediana de antecedência de 14,7h.

---

## Item 4 — Classificador supervisionado de alerta precoce

**Mudança de paradigma:** em vez de aprender só "normal" e esperar que o
erro de reconstrução suba perto de uma falha (toda a linha EXP5--EXP7), o
item 4 treina um `RandomForestClassifier` para prever diretamente "existe
um alarme real nas próximas `PREDICTION_HORIZON_HOURS` (24h)?", usando os
próprios timestamps de alarme como rótulo. Reusa exatamente a mesma
engenharia de features do EXP7 (`select_feature_columns`, extraída do
`automl_pipeline.py` para `preprocess.py` para as duas abordagens
compartilharem a mesma definição) e a mesma disciplina de split OOS.
Implementação em `src/cnn1d_ae/supervised_pipeline.py`, config
`test_grupo_exp8_supervisionado.json`.

**Task ClearML:** `cb79c953b2ba48e98f2af15cb0552503`

### Resultado: dominado pelo AutoML em todo o grid

| proba\_threshold | hit\_rate | FP | composite |
|---|---|---|---|
| 0,2 (vencedor) | 77,5% (31/40) | **26,98%** | 0,876 |
| 0,3 | 60,0% (24/40) | 8,19% | 0,862 |
| 0,4 | 30,0% (12/40) | 3,18% | 0,766 |
| 0,5+ | 0,0% | ~0% | 0,467 |

As probabilidades do classificador **colapsam** entre 0,4 e 0,5 -- ele
essencialmente nunca atribui probabilidade $\geq$ 0,5 a nenhum ponto do
período OOS, nem aos verdadeiros positivos. É sinal de que o modelo não
generaliza bem para fora do período de treino (ajusta padrões específicos
do período de fit que não se repetem da mesma forma no período de
avaliação).

**Breakdown do trial vencedor (threshold=0,2, debounce=24):** 28
preditivos, 3 reativos, 9 sem detecção -- número de preditivos parecido com
o candidato do AutoML (29), mas:
- FP de 26,98% (**14x pior** que o candidato de referência, 1,94%);
- mediana de antecedência de **23,99h** -- bate quase exatamente no teto de
  24h da janela de avaliação, sinal de que boa parte das detecções
  "preditivas" é artefato da própria largura da janela de rótulo
  (`PREDICTION_HORIZON_HOURS=24h`) em vez de um precursor genuinamente
  bem localizado no tempo, diferente da distribuição mais espalhada e
  crível do candidato do AutoML (0,002h--23,3h, mediana real de 14,7h).

**Em nenhum ponto do grid** (threshold 0,2 a 0,9 $\times$ debounce 1 a 24)
o classificador supervisionado supera o candidato do AutoML nos dois eixos
(detecção e falso alerta) ao mesmo tempo -- ou tem FP muito pior para
detecção parecida, ou detecção muito pior para FP parecido.

### Por que não funcionou (hipóteses)

- **Poucos eventos independentes de verdade:** embora existam ~72 mil
  pontos de treino rotulados como positivo (janelas de 24h antes de cada
  alarme do período de fit), esses pontos vêm de só ~150 eventos
  distintos -- muita redundância/correlação dentro de cada janela, pouca
  informação nova por evento.
- **RandomForest é flexível demais para a quantidade de sinal real:** com
  300 árvores sem limite de profundidade, é fácil memorizar padrões
  específicos do período de fit que não se repetem no período de
  avaliação -- consistente com o colapso de probabilidade observado.
- **Class balancing pode ter distorcido a calibração:** `class_weight="balanced"`
  corrige o desbalanceamento na função de perda, mas não garante que as
  probabilidades resultantes sejam bem calibradas para um grid de corte
  fixo.

### Conclusão

**O candidato de referência continua sendo o do AutoML (EXP7 item 1+2+3).**
A reformulação supervisionada, do jeito que foi tentada aqui, não trouxe
ganho -- é um resultado negativo real, não um bug (o smoke test sintético
validou a implementação antes de rodar contra os dados reais). Documentado
para não reinvestir nessa direção específica sem antes resolver o problema
de generalização (ex: mais eventos rotulados, modelo mais regularizado,
validação cruzada temporal em vez de um único split).

---

## Item 5 — Sobrevivência exploratória (RUL "light")

**Escopo deliberadamente reduzido:** não é um modelo de RUL para uso
operacional. Usa só o histórico de timestamps dos 363 alarmes reais de
`TC382_03_A`/`T5_AVG_A` (sem covariáveis de sensor) para caracterizar o
padrão de recorrência -- decisão registrada em
`docs/analise_automl_exp7_planejamento.md`: com poucos eventos rotulados
no período OOS, um modelo de sobrevivência com covariáveis (Cox, etc.)
teria risco real de overfitting. Script:
`scripts/exploratory_survival_analysis.py`.

### Achado preliminar: decluster é necessário

Os 363 alarmes brutos incluem muitos pares quase-simultâneos (`T5_AVG_A` e
`TC382_03_A` disparando com segundos de diferença para o mesmo evento
físico). Sem tratar isso, o ajuste de sobrevivência fica dominado por uma
massa artificial de intervalos quase-zero e o teste estatístico rejeita
qualquer ajuste (KS test vs. Weibull: D=0,848, p<0,001). Agrupando eventos
com gap $\leq$ 30 min, os 363 alarmes brutos viram **179 eventos
distintos**, e o ajuste Weibull passa a ser estatisticamente razoável (KS
test: D=0,079, **p=0,203**, não rejeitado).

### Resultado

| Métrica | Valor |
|---|---|
| Eventos distintos (declustered) | 179, no período 2022-01 a 2026-04 |
| Intervalo mediano entre eventos | 34,6h (~1,4 dias) |
| Intervalo médio | 209,6h (desvio alto -- distribuição bem assimétrica) |
| Weibull shape (k) | **0,507** |
| Weibull scale ($\lambda$) | 94,8h |

**Interpretação: hazard decrescente (k<1), não o padrão clássico de
desgaste.** Alarmes tendem a vir em rajadas logo após um evento anterior, e
depois ficam mais raros -- mais parecido com um processo autoexcitante
(cada evento aumenta temporariamente a chance de outro) do que com
degradação mecânica acumulando de forma monotônica ao longo do tempo.

**Risco condicional** (dado que já se passaram X horas sem alarme, qual a
chance do próximo vir na próxima janela de Y horas):

| Tempo desde o último alarme | +1h | +6h | +24h |
|---|---|---|---|
| 6h | 2,0% | 9,9% | 26,7% |
| 12h | 1,4% | 7,7% | 23,0% |
| 24h | 1,0% | 5,8% | 18,9% |
| 48h | 0,7% | 4,3% | 14,9% |

O risco condicional cai conforme mais tempo passa sem alarme -- reforça o
achado de hazard decrescente, e sugere uma heurística operacional simples:
vigilância aumentada por ~48--72h após qualquer alarme real, já que a
chance de recorrência próxima é maior logo depois de um evento do que
depois de um período longo de estabilidade.

### Limitações explícitas

- Não usa nenhuma informação de sensor -- é só estatística de recorrência
  temporal dos alarmes em si, não um preditor por si só.
- Não diferencia tipo/severidade de alarme (`Condição do Alarme`) nem
  causa raiz -- trata todo alarme de `TC382_03_A`/`T5_AVG_A` como um evento
  equivalente.
- Janela de decluster (30 min) é uma escolha razoável mas arbitrária; não
  foi sensibilizada.
- **Não deve ser usado como modelo operacional de RUL** -- é uma
  caracterização exploratória do padrão de recorrência, ponto final.

---

## Plano de 5 itens -- fechado

1. ✅ Features multi-escala -- ganho real (mais cobertura, mais FP)
2. ✅ Features de textura -- ganho limpo (melhora tudo ao mesmo tempo) --
   **entra no candidato de referência**
3. ✅ Mudança de regime -- sem ganho adicional (mesmos 3 casos residuais)
4. ✅ Reformulação supervisionada -- **resultado negativo**, dominado pelo
   AutoML em todo o grid testado
5. ✅ Sobrevivência exploratória -- feito no escopo reduzido combinado;
   achado real (hazard decrescente, rajadas de alarme) mas não é modelo
   operacional

**Candidato final da série EXP5--EXP8:** `ocsvm` (p99.9, debounce=1) sobre
`TC382_T5_vibracao_mancais` com features multi-escala + textura (item 2),
sem mudança de regime nem reformulação supervisionada (nenhum dos dois
trouxe ganho sobre o item 2).

## Checagem de variância de semente (2026-08-16)

A rotina de seed-sweep (`AUTOML_SEED_SWEEP_N`, EXP5) era específica pra
`iforest`. Generalizada (`src/cnn1d_ae/automl_pipeline.py`) para cobrir
também `ocsvm` -- que tem uma fonte real de aleatoriedade nesta pipeline:
quando `x_normal` excede `AUTOML_OCSVM_MAX_TRAIN_SAMPLES` (386.492 > 50.000
neste caso), uma subamostra aleatória é usada pra ajustar o SVM (o
algoritmo em si é determinístico, mas *quais* pontos entram no ajuste
muda com a seed).

**Task ClearML:** `58a68b604f1e4207982e151252708a83` (candidato final
isolado: `ocsvm`, p99.9/debounce=1, 5 seeds).

| Seed | hit\_rate | normal\_alert\_rate |
|---|---|---|
| 42 (original) | 92,5% | 1,94% |
| 43 | 92,5% | 1,90% |
| 44 | 92,5% | 1,91% |
| 45 | 92,5% | 1,90% |
| 46 | 92,5% | 1,93% |
| **média / desvio** | **92,5% / std=0** | 1,91% / std=0,015pp |

**Resultado: variância nula em `hit_rate`, variação desprezível em FP.** O
candidato final está confirmado estável -- a subamostragem do `ocsvm` não
afeta a classificação final, provavelmente porque 50 mil pontos já é
amostra grande o suficiente para representar bem a distribuição
independente de qual subconjunto específico é sorteado.

**Pendência fechada.** Com isso, o candidato `ocsvm` (multi-escala +
textura, p99.9/debounce=1) está validado em todos os eixos considerados
nesta série: OOS, breakdown preditivo/reativo/sem-detecção, e agora
variância de semente.
