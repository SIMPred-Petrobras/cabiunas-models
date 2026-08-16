# Ataque ao Falso Positivo — Threshold, `μ + y·σ` e Filtragem

**Data:** 2026-08-15
**Branch:** `backup/thallys`
**Sensor:** TC382_03_A · **Modelo:** braço v13 `b2024` (dense, per_sensor, treino 2024-06-01 → 2025-07-01)
**Origem:** sugestões do professor sobre como reduzir falso positivo
**Método:** pós-processamento offline sobre o MAE em cache (`~/.clearml/cache`), sem retreinar

---

## Contexto

O falso positivo é o gargalo reconhecido do detector. O professor levantou duas
sugestões, testadas aqui na íntegra:

1. **Subir o threshold** para reduzir FP, aceitando perder alguns acertos.
2. **Parametrizar o threshold como `média + y·desvio`**, varrendo valores de `y`.

Testei as duas, mais duas variantes que surgiram da conversa (aplicar a regra ao
MAE bruto; encadear média móvel após a EWMA). Todos os números abaixo usam o
protocolo da auditoria: horizonte 8h (salvo indicação), sticky 12h, incidentes
HI/HIHI com máquina ON e filtro de fantasma (<500 °C), restrição de duty
pós-sticky ≤ 0,25. FP é contado **por episódio**.

**Resumo:** nenhuma das duas sugestões reduz FP. Mas o teste de controle da
segunda revelou uma alavanca que ninguém tinha testado — a **meia-vida da EWMA** —
e uma reinterpretação da fraqueza conhecida do modelo em 2024.

---

## Sugestão #1 — Subir o threshold

**Hipótese:** o alarme dispara em excursões marginais; um limiar mais alto corta
as marginais e preserva os eventos reais.

**Experimento:** varredura de `q` (rank-percentil da EWMA, hl=2h) de 0,80 a 0,998
na janela FULL, 58 incidentes, ~686 dias.

| q | recall | FP | episódios | FA/dia | duty |
|--:|--:|--:|--:|--:|--:|
| 0,800 | 96,6% (56/58) | 87 | 111 | 0,127 | 0,343 |
| **0,884** ← auditoria | **86,2% (50/58)** | **72** | 102 | 0,105 | 0,249 |
| 0,920 | 70,7% (41/58) | 68 | 96 | 0,099 | 0,207 |
| 0,940 | 53,4% (31/58) | 72 | 96 | 0,105 | 0,177 |
| 0,960 | 36,2% (21/58) | 69 | 92 | 0,100 | 0,140 |
| 0,980 | 19,0% (11/58) | 59 | 76 | 0,086 | 0,101 |
| 0,990 | 10,3% (6/58) | 45 | 54 | 0,066 | 0,067 |

**Taxa de câmbio (FP removidos por incidente perdido):** 0,0 a 0,6.
Para comparar, o portão de rampa+nível entrega **6,8**.

**Por que não funciona.** A distribuição do **pico do episódio** é praticamente a
mesma nas duas populações:

| | p10 | p50 | p90 | pico ≥ 0,99 |
|---|--:|--:|--:|--:|
| TP (30 episódios) | 0,960 | **0,9919** | 0,9993 | 18/30 |
| FP (72 episódios) | 0,918 | **0,9894** | 0,9987 | 34/72 |

Cortando em pico ≥ 0,99 sobrevivem 34 dos 72 FP e morrem 12 dos 30 TP. **Nenhuma
altura de limiar separa duas distribuições coincidentes.** O FP não é um erro
pequeno que passou raspando — é um pico tão alto quanto o do evento real, porque
durante manobra de carga o termopar realmente se comporta de forma incomum
(rampa mediana no FP: 172 °C/h; no evento real: 21 °C/h; AUC de amplitude 0,44).

**Ressalva a favor:** subir o threshold **reduz muito o tempo em alarme** (duty
0,343 → 0,101), mesmo sem reduzir o número de alarmes. Se a reclamação da
operação é *"o alarme vive ligado"*, resolve; se é *"aparece alarme demais"*, não.

---

## Sugestão #2 — `threshold = μ + y·σ`

**Descoberta preliminar:** a regra **já está implementada e em uso**. A camada
preditiva varre exatamente `μ + y·σ` sobre a EWMA — `predictive.py:131`,
`y_vals = np.linspace(sigma_y_min, sigma_y_max, n_steps)`, com
`PREDICTIVE_SIGMA_Y_MIN=-2.0` / `MAX=5.0`, comentada no `config.py:310` como
*"SPC / Shewhart"*. O `y` ótimo que ela encontra equivale a **y ≈ 0,9**.

**Experimento:** três parametrizações, μ/σ calibrados só no treino (sem vazamento).

### A — `μ + y·σ` sobre a EWMA (forma deployável), FULL

| y | ≈ quantil | recall | FP |
|--:|--:|--:|--:|
| 0,2 | 0,772 | 98,3% | 88 |
| **1,0** | 0,900 | **79,3%** | **73** |
| 2,0 | 0,978 | 20,7% | 58 |
| 3,0 | 0,992 | 8,6% | 40 |

**`y=2` e `y=3` do livro-texto estão muito acima de tudo** — em y=3 o detector vê
5 dos 58 incidentes. Causa: cauda direita pesada do erro de reconstrução, que
infla σ. Já estava anotado em `scoring.py:50-54`.

### C — `mediana + y·MAD` sobre a EWMA (escala robusta), FULL

| y | recall | FP |
|--:|--:|--:|
| 2,0 | 96,6% (56/58) | 85 |
| 2,5 | 89,7% (52/58) | 79 |
| **3,0** | **81,0% (47/58)** | **73** |

Com MAD, **y=2–3 volta a ser a faixa útil** — que é onde o operador espera que
esteja. É a melhoria concreta a fazer na sugestão, mantendo a forma proposta.

**Mas: é a mesma curva.** Comparando no mesmo recall, quantil e `μ+y·MAD` dão
FP idêntico (89,7% → 78 vs 79; 81% → 73 vs 73). Tem de ser: `μ+y·σ` é
transformação monótona do mesmo escalar, então seleciona os mesmos pontos de
operação. **A parametrização é questão de interface, não de desempenho.**

---

## Variante #1 — Aplicar a regra ao MAE bruto (sem EWMA)

Relevante porque é **o que `scoring.py:55` faz hoje**: `THRESH_MODE="mean_std"`
calcula μ/σ sobre `train_mae_seq`, o MAE bruto, não sobre a EWMA.

| Cenário | | melhor recall (duty ≤ 0,25) | FP | `y` | lead mediano |
|---|---|--:|--:|--:|--:|
| FULL (58) | EWMA | **86,2%** | **72** | 0,90 | **7,4 h** |
| | MAE bruto | 56,9% | 122 | 1,80 | 2,1 h |
| OOS (17) | EWMA | **100%** | **27** | 0,90 | **7,9 h** |
| | MAE bruto | 70,6% | 47 | 2,15 | 0,7 h |

Com o MAE bruto o detector **não alcança 60% de recall no FULL** sob a restrição
de duty, e faz ~120 FP contra 72.

**Achado contra-intuitivo:** a antecedência **despenca** (7,4 h → 2,1 h). Eu
esperava o contrário — sem o atraso do filtro, deveria detectar mais cedo. O
mecanismo é o oposto: o MAE bruto é ruidoso, então para segurar o duty é preciso
um `y` alto, e um limiar alto **só é cruzado no pico do evento**. A EWMA acumula
evidência e cruza um limiar mais baixo ainda na subida. É o argumento clássico
EWMA vs Shewhart em CEP: carta de Shewhart é insensível a desvio pequeno e
sustentado, que é exatamente o nosso caso (rampa real de 21 °C/h).

**Ação:** se `THRESH_MODE="mean_std"` for adotado, `scoring.py` precisa operar
sobre a EWMA. Hoje quem ligar esse botão pega a curva ruim.

---

## Variante #2 — EWMA seguida de média móvel

**Hipótese:** cascata de dois passa-baixa dá roll-off mais íngreme com menos
atraso de fase que um filtro único de banda equivalente.

**Controle obrigatório:** comparar contra **apenas aumentar a meia-vida da EWMA**.
Se `EWMA(2h)+MA(4h)` só empata com `EWMA(6h)`, a cascata não trouxe informação —
só trocou a constante de tempo.

**Grade:** EWMA só (hl 1–24 h) · EWMA2h+MA (1–48 h) · EWMA4h+MA (2–24 h) ·
EWMA2h+mediana móvel (4–24 h). Recall casado, duty ≤ 0,25, FULL.

| recall | melhor filtro | FP | 2º lugar |
|--:|---|--:|---|
| 81,0% | **EWMA hl=4h** | **59** | EWMA2h+MA 1h (67) |
| 77,6% | **EWMA hl=6h** | **54** | EWMA hl=4h (57) |
| 74,1% | **EWMA hl=8h** | **39** | EWMA hl=6h (52) |
| 70,7% | EWMA2h+MED 12h | 36 | EWMA hl=12h (38) |
| 67,2% | **EWMA hl=24h** | **25** | EWMA2h+MED 12h (35) |

**A cascata perde.** No OOS, EWMA sozinha vence em todos os níveis. A mediana
móvel de 12 h aparece uma vez em primeiro (36 contra 38) — diferença de ruído.
A hipótese de que a mediana mataria o transiente de partida **não se confirmou**.

---

## O que emergiu do controle: a meia-vida é a alavanca

Taxa de câmbio a partir de hl=2h (FULL, 50/58, 72 FP):

| mudança | perde | remove | **FP por incidente perdido** |
|---|--:|--:|--:|
| hl 2h → 4h | 3 | 13 | **4,3** |
| hl 2h → 6h | 5 | 18 | **3,6** |
| hl 2h → 8h | 7 | 33 | **4,7** |
| hl 2h → 24h | 11 | 47 | **4,3** |

Contra as demais alavancas medidas:

| alavanca | FP por incidente perdido |
|---|--:|
| subir o threshold (`y`) com hl fixo | 0,0 – 0,6 |
| **alongar a meia-vida da EWMA** | **3,6 – 4,7** |
| portão de rampa+nível | 6,8 |

**A meia-vida é 7 a 10× mais eficiente que o threshold** e joga na mesma liga do
portão operacional. No OOS é quase de graça: hl=4h e hl=6h mantêm 17/17 e ainda
baixam de 27 para 26 FP.

**Nota importante:** os bundles em produção já usam `half_life_hours: 4.0`.
Parte desse ganho **já está entregue** — é validação, não melhoria pendente.

**Detalhe técnico:** σ cai monotonicamente com a meia-vida (0,1155 em hl=0,5h →
0,0550 em hl=24h). Um `y` fixo significa coisas diferentes em cada hl — **mudar a
meia-vida obriga a recalibrar `y`**.

---

## Reinterpretação do BACKCAST 2024

A fraqueza conhecida do modelo (21% de recall em 2024 contra 100% no OOS) muda de
natureza quando o horizonte de avaliação abre:

| horizonte | hl=1h | hl=2h | FP (hl=1h / 2h) |
|---|--:|--:|--:|
| 8 h | 17,9% (5/28) | 25,0% (7/28) | 27 / 21 |
| 24 h | 39,3% (11/28) | 42,9% (12/28) | 25 / 19 |
| **72 h** | **85,7% (24/28)** | 71,4% (20/28) | **16** / 14 |

**O erro de reconstrução sobe antes dos incidentes de 2024 — 20 a 70 horas antes,
não 8.** O detector estava vendo; a janela de avaliação é que era curta demais.

**Cuidado de leitura (crítico):** recall com horizonte maior é **trivialmente ≥**
recall com horizonte menor, e FP cai mecanicamente porque o horizonte entra na
definição de FP (episódio sem incidente em `[s0, s1+H]`). **85,7% em 72 h é uma
afirmação mais fraca que 86,2% em 8 h** — não são comparáveis. O que a tabela
autoriza é dizer que o sinal existe e é antecipado, não que o desempenho melhorou.
O que sustenta a leitura é a FA/dia continuar baixa (0,076/dia ≈ 1 FP a cada 13
dias) — não é o alarme ficando permanentemente ligado.

**Antecedência, agora sem censura.** Com horizonte de 8 h toda meia-vida longa
saturava em ~7,9 h. Abrindo:

| hl | lead @ 24 h | lead @ 72 h |
|--:|--:|--:|
| 2 h | 17,5 h | 48,1 h |
| 8 h | 20,6 h | 44,9 h |
| 24 h | **23,6 h** | **65,9 h** |

Meia-vida longa **antecipa, não atrasa** — o oposto do que eu previa.
*Ressalva:* lead e duty são acoplados; meia-vida longa produz platô de alerta mais
largo, e a métrica "primeiro cruzamento na janela" premia alarme cronicamente
ligado. `duty ≤ 0,25` limita, mas parte da vantagem é da métrica.

---

## Armadilhas metodológicas encontradas

Registradas porque custaram tempo e voltarão a aparecer:

1. **Contar FP por episódio sem restringir duty degenera.** A primeira fronteira
   que montei escolheu `y=−0,5` com duty 0,999 — alarme permanentemente ligado
   colapsa tudo em pouquíssimos episódios e produz FP artificialmente baixo.
   Toda fronteira precisa do `duty ≤ 0,25`.
2. **Lead é censurado pelo horizonte.** Medir antecedência dentro de uma janela de
   8 h faz todo filtro lento saturar em 7,9 h. Conclusões sobre atraso exigem
   horizonte maior.
3. **`THRESH_STD_MULT=3.0` está no config e é inerte.** `compute_threshold`
   (`scoring.py:45`) só lê esse campo quando `THRESH_MODE == "mean_std"`; hoje o
   threshold vem de `target_rate=0.005`. Mexer nele sem trocar o modo não faz nada.
4. **`HL_GRID` estava travada em `[0.5, 1, 2, 4]`** (`sweep_regime_band_offline.py:64`).
   Os pontos de 8/12/24 h — os melhores no eixo de FP — nunca passaram pelo
   protocolo oficial.
5. **Comparar métricas entre horizontes diferentes é inválido** (ver acima).

---

## Configs que afetam este ajuste

**Definem o threshold:** `THRESH_MODE`, `THRESH_STD_MULT`, `TARGET_ANOMALY_RATE`,
`ADAPTIVE_THRESHOLD_MODE`, `PREDICTIVE_SIGMA_Y_MIN/MAX`.

**Definem μ e σ sem parecer** — todo knob de exclusão de treino, porque μ/σ são
calculados sobre `train_mae_seq` = MAE em `df_normal = df_use[~exclude]`
(`pipeline.py:296`): `EXCLUDE_MINUTES_BEFORE/AFTER_ALARM`, `EXCLUDE_STARTUP_MINUTES`
(hoje `0` — as partidas estão dentro do treino, inflando σ), `EXCLUDE_CONSTANT_RUNS`,
`ENABLE_GRADIENT_SPIKE_MASK`, `RUNNING_COL`, `TRAIN_START/END_DATE`, `OUTLIER_MAD_K`,
`SENTINEL_LOW/HIGH`.

**Decidem se um cruzamento vira alarme** (onde o FP nasce): `PREDICTIVE_EWMA_HALF_LIFE_HOURS`
(+ `_PER_SENSOR`, hoje vazio), `PREDICTIVE_ALERT_DEBOUNCE_HOURS`,
`PREDICTIVE_FA_BUDGET_PER_DAY`, `POINT_RULE/WINDOW/MIN_COUNT`, `MIN_ANOMALY_RUN_STEPS`,
**`GRADIENT_SPIKE_SUPPRESS_SCORING`** (hoje `False` — é essencialmente o portão de
transiente recomendado, já implementado; ver a ressalva do `config.py:144-146` de
que em termopar o spike *é* a assinatura de falha).

**Em produção** nada disso vale: o ponto está congelado no bloco `production_alerting`
do bundle. Ajuste sem retreino via
`scripts/set_bundle_threshold.py <bundle.json> --std_mult Y`.

---

## Recomendações

1. **Adotar `mediana + y·MAD` sobre a EWMA** como parametrização oficial. Mantém a
   forma interpretável que o professor propôs, com y=2–3 numa faixa utilizável.
   Custo zero — é reparametrização.
2. **Corrigir `scoring.py::mean_std` para operar sobre a EWMA**, não sobre o MAE
   bruto. Ganho real: 122 → 73 FP no mesmo recall.
3. **Não esperar redução de FP vinda do `y`.** O ganho vem do eixo ortogonal à
   amplitude: portão de rampa+nível (6,8) e máscara de pós-partida (40% dos FP são
   pós-partida, 27 dos 29 disparando na primeira meia hora após o arranque).
4. **Estender `HL_GRID` no protocolo oficial** para `[0.5, 1, 2, 4, 8, 12, 24]`.
5. **Avaliar a mudança de enquadramento do produto:** no horizonte de 24 h,
   `hl=12h` dá 79,3% de recall, **36 FP** (metade dos 68 de hl=2h), FA 0,052/dia e
   20,6 h de antecedência. Um *aviso antecipado de 24 h* com 1 falso alarme a cada
   20 dias pode ser operacionalmente mais útil que um *alarme de 8 h* com o dobro
   de FP.

---

## Limitações

- **Um sensor** (TC382_03_A), **um braço** (v13 `b2024`), **um retreino**. GPU/cuDNN
  + KerasTuner são não-determinísticos e o recall por sensor oscila ±10 pontos
  entre retreinos da mesma config — nenhuma diferença de um retreino único é
  evidência.
- Toda a análise é **pós-processamento do mesmo MAE**. Não testa se outro modelo
  produziria fronteira diferente.
- Os braços restantes do v15 (`rerun_t50e80`, `b2024_t20e20`, `rerun_t20e20`) ainda
  não rodaram; sem eles não dá para separar efeito de orçamento de ruído de retreino.
- A vantagem de antecedência das meias-vidas longas é parcialmente artefato da
  métrica (ver ressalva acima).

---

## Artefatos

| Arquivo | Conteúdo |
|---|---|
| `scripts/sweep_hl_grid_offline.py` | Varredura de meia-vida × horizonte × `y`, offline |
| `eval_predictive_out/hl_grid_sweep_TC382_03_A.csv` | 8.883 linhas — grade completa |
| `scripts/sweep_threshold_mean_std.py` | Script pré-existente para a mesma sugestão |
| `scripts/plot_fp_mensal_on_off.py` | Classificação dos FP em pós-partida / parada / estável |

Reprodução: `PYTHONPATH=. python scripts/sweep_hl_grid_offline.py`
(offline, lê `~/.clearml/cache`, não precisa do servidor ClearML nem de GPU).
