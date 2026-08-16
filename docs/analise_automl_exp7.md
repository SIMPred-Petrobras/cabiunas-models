# Análise de Experimentos — AutoML EXP7 (features multi-escala + textura)

Continuação do EXP6 (`docs/analise_automl_exp6.md`) e do planejamento
registrado em `docs/analise_automl_exp7_planejamento.md`. Este documento
consolida os **ganhos e perdas** de cada mudança testada.

**Ponto de partida (EXP6):** grupo `TC382_03_A` + `T5_AVG_A` + 10 canais de
vibração, features derivadas numa única janela (6min), `iforest`
(p99.9/debounce=6) — 65% de antecedência real (26/40), FP de 0,06%, mediana
de 12,4h. Dois pontos fracos identificados: **4 alarmes só detectados depois**
(reação, não previsão) e **10 sem detecção nenhuma**.

---

## Item 1 — Features multi-escala (6min / 1h / 4h / 24h)

**Mudança:** `DERIVED_ROLLING_WINDOWS: [12, 120, 480, 2880]` (amostras a 30s)
em vez de uma janela única de 12. Para cada sensor e cada janela: média
móvel, desvio móvel e `trend_w` (valor atual − valor de w passos atrás).
48 → 168 features de entrada.

**Task ClearML:** `aad8bb00fdc24e2d86e35fec83cc2027`

| | EXP6 (janela única) | EXP7 multi-escala (vencedor: `ocsvm`) |
|---|---|---|
| Melhor modelo | iforest p99.9/db6 | ocsvm p99.5/db6 |
| hit_rate | 75,0% (30/40) | 87,5% (35/40) |
| normal_alert_rate (FP) | 0,06% | 2,21% |
| **Preditivo (antecedência real)** | 26 | **28** |
| Reativo (só depois) | 4 | 7 |
| **Sem detecção** | **10** | **5** |
| Mediana de antecedência | 12,4h | **15,0h** |

**Ganho:** os "sem detecção" caíram pela metade (10→5) e a antecedência
mediana melhorou (12,4h→15,0h) — a hipótese se confirmou, features de
janela longa pegam precursores que a janela de 6min não via.
**Perda:** falso alerta ~37x maior (0,06%→2,21%). Ainda baixo em termos
absolutos, mas é uma troca real, não um upgrade de graça.

### Desvio: o "candidato de baixo FP" que parecia bom mas não era

No mesmo grid multi-escala, o melhor trial de `iforest` com FP mínimo e
hit_rate≥70% (p99.0/debounce=12: 70% hit / 0,38% FP) parecia, pelos números
agregados, uma alternativa mais conservadora ao `ocsvm`. Isolamos esse
trial numa run dedicada (task `85bd0918ba6042c780d96c16f66055bb`,
`test_grupo_exp7_iforest_lowfp.json`) para conferir o breakdown
preditivo/reativo:

| | EXP7 ocsvm (multi-escala) | EXP7 iforest "baixo FP" (multi-escala) |
|---|---|---|
| hit_rate | 87,5% | 70,0% |
| FP | 2,21% | 0,38% |
| **Preditivo** | **28** | **13** ⚠️ |
| Reativo | 7 | **15** ⚠️ |
| Sem detecção | 5 | 12 |

Apesar do FP baixo, esse candidato é o **pior dos três em antecedência
real** — só 13 das 28 detecções são preditivas, o resto é reação tardia
(debounce=12, o dobro do EXP6, provavelmente atrasa o primeiro ponto que
"conta" como detecção o suficiente para empurrar previsões para dentro da
janela reativa). **Lição:** `hit_rate`/`normal_alert_rate` agregados enganam
— é preciso sempre abrir o breakdown preditivo/reativo/sem-detecção antes
de escolher um candidato pelo score agregado.

**Candidato de referência após o item 1:** `ocsvm` multi-escala (p99.5/db6).

---

## Item 2 — Features de textura do sinal (kurtosis, skewness, crest factor)

**Mudança:** para cada janela ≥ 60 amostras (1h+; a janela de 6min não tem
amostra suficiente para esses estimadores serem estáveis), adicionadas 3
novas features por sensor: kurtosis móvel, skewness móvel e crest factor
móvel (pico/RMS). Capturam mudança de *forma* do sinal (ficando mais
"impulsivo"/assimétrico), não só nível ou tendência — features padrão em
condition monitoring de vibração. 168 → ~276 features de entrada.

**Task ClearML:** `3eb2f1f166114118a5c2dbbf759da5bb`

| | EXP7 item 1 (multi-escala) | EXP7 item 1+2 (+ textura, vencedor: `ocsvm` p99.9/db1) |
|---|---|---|
| hit_rate | 87,5% (35/40) | **92,5% (37/40)** |
| normal_alert_rate (FP) | 2,21% | **1,94%** |
| **Preditivo** | 28 | **29** |
| Reativo | 7 | 8 |
| **Sem detecção** | 5 | **3** |
| Mediana de antecedência | 15,0h | 14,7h |

**Ganho limpo:** ao contrário do item 1 (que trocou FP por menos casos
perdidos), o item 2 melhorou hit_rate, FP *e* a divisão preditivo/sem-detecção
ao mesmo tempo — as features de textura (kurtosis/skewness/crest factor)
parecem estar capturando sinal genuíno de vibração ficando mais "irregular"
antes do evento, não só reclassificando detecções reativas como preditivas.
Sem perda identificada nesta etapa (a única ressalva é a amplitude da
antecedência: de 0,002h a 23,3h — ainda há casos detectados quase em cima
da hora, junto com os de quase 24h).

---

## Resumo acumulado

| Etapa | Preditivo | Reativo | Sem detecção | FP | Mediana antecedência |
|---|---|---|---|---|---|
| EXP6 (baseline, janela única) | 26 | 4 | 10 | 0,06% | 12,4h |
| EXP7 item 1 (multi-escala) | 28 | 7 | 5 | 2,21% | 15,0h |
| **EXP7 item 1+2 (+ textura)** | **29** | 8 | **3** | 1,94% | 14,7h |

**Candidato de referência atual:** `ocsvm` (p99.9, debounce=1) sobre o grupo
`TC382_T5_vibracao_mancais_multiescala` com `ENABLE_DERIVED_FEATURES=true`,
`DERIVED_ROLLING_WINDOWS=[12, 120, 480, 2880]` e features de textura
(kurtosis/skewness/crest factor) nas janelas ≥1h.

## Próximos passos (do plano em 5 itens)

1. ✅ Features multi-escala — feito, ganho real (com custo de FP)
2. ✅ Features de textura — feito, ganho limpo (sem custo adicional)
3. ⬜ Detecção de mudança de regime (PELT/CUSUM) — ainda restam 3 alarmes
   sem detecção nenhuma; candidato natural para atacá-los especificamente
4. ⬜ Reformulação supervisionada ("vai alarmar em N horas?") — considerar
   depois do item 3, com cuidado redobrado de validação temporal dado o
   tamanho pequeno da amostra (40 eventos)
5. ⬜ RUL/sobrevivência — não priorizado, requer mais eventos rotulados

**Pendência de validação:** nenhum dos candidatos do EXP7 teve checagem de
variância de semente ainda (feita no EXP5 via `AUTOML_SEED_SWEEP_N`) —
recomendado antes de considerar o candidato atual definitivo.
