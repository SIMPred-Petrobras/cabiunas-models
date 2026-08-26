# EXP16 — Predição de alarmes TRIP de óleo lubrificante

**Branch:** `AE_trip_oleo_lub` (a partir de `AE_novo_legado`, onde está o
candidato final consolidado do EXP13/EXP15b para T5). Objetivo: abrir uma
frente nova de detecção, independente da T5, focada nos alarmes TRIP do
circuito de óleo lubrificante. Os demais modelos (T5, e futuros) continuam
sendo ajustados em paralelo, em outras branches.

## Os 3 alarmes TRIP candidatos

Levantados na planilha `Alarmes Selecionados Turbina A.xlsx` — os únicos 3
tags com "TRIP" na descrição, todos do circuito de óleo lubrificante da
multiplicadora/compressor (não têm relação com o alvo T5/TC382):

| Tag alarme | Descrição | Tag PI bruto | Prioridade |
|---|---|---|---|
| `PALL_6240340` | Pressão baixa óleo header (proteção turbina) | `954005_624_PI_0340` | HIGH |
| `PALL_6240309` | Pressão baixa óleo header compressor | `954005_624_PI_0308` | HIGH |
| `TALL_6240325` | Temperatura baixa tanque óleo | `954005_624_TI_0325` | HIGH |

## Achado crítico: 2 dos 3 são artefato de desligamento, não falha incipiente

Puxamos a série bruta dos 3 tags do ClearML (dataset
`a97ba56ba14840fbb1125c2a82f883c9`, `sensores_full_2024_2026_30s.csv`,
2024-01-01 a 2026-04-30) e cruzamos com os timestamps reais de cada evento
TRIP (`ACT/UNACK` + `CFN`) e com `RUNNING_A`:

| Tag | Valor mediano em operação normal | Valor mediano no instante do trip | % de trips com `RUNNING_A=0` |
|---|---|---|---|
| `PALL_6240340` | 3,25 kgf/cm² | 0,01 (~zero) | **99,4%** (168/169) |
| `TALL_6240325` | 60,8 °C | 25,5 (caindo pro piso -19) | **100%** (53/53) |
| `PALL_6240309` | 1,38 kgf/cm² | 0,95 (mediana geral) | 57,1% (4/7) |

`PALL_6240340` e `TALL_6240325` disparam quase exclusivamente com a turbina
já parada — o sensor já colapsou pra zero/piso antes do alarme, é
consequência do desligamento, não um precursor. **Não há sinal aprendível
aí com os dados atuais** (mesma conclusão já documentada para
`PALL_6240340` em `docs/analise_automl_exp11.md`, agora confirmada também
para `TALL_6240325` e generalizada com evidência direta na série bruta).

`PALL_6240309` é diferente: dos 7 eventos totais, **3 dispararam com a
turbina rodando**, com o valor bem colado no teto da faixa normal
(p95=1,421 / p99=1,434):

| Data do trip | Valor no disparo | `RUNNING_A` |
|---|---|---|
| 2024-03-07 13:55 | 1,363 | 1 (rodando) |
| 2025-11-04 06:22 | 1,415 | 1 (rodando) |
| 2026-02-26 15:34 | 1,348 | 0,17 (transição) |

## Alvo recomendado: `954005_624_PI_0308` (alarme `PALL_6240309`)

É o único dos 3 com eventos genuínos de trip em operação — os outros dois
não têm o que aprender (seriam 100% "artefato de desligamento" como alvo,
inviabilizando qualquer avaliação supervisionada). `PI_0340` e `TI_0325`
entram como **features do grupo multivariado** (mesmo circuito de óleo,
fisicamente relevantes), não como alvo.

### ⚠️ Ressalva estatística importante

**Só há 3 eventos genuínos em ~2,3 anos de dados, e apenas 2 caem no
período OOS (pós `2025-07-01`)**. Isso é ordens de grandeza menor que os
40 alarmes de T5 usados no EXP13/EXP15b. Consequências:
- `hit_rate`/FP calculados sobre n=2-3 têm variância enorme — qualquer
  conclusão de "funcionou" ou "não funcionou" precisa ser tratada como
  anedótica, não estatisticamente robusta.
- Vale considerar ampliar a janela de dados (puxar 2023 também, já que o
  catálogo de alarmes cobre 2022–2026, ver pendência aberta em
  `docs/analise_cnn1dae_exp13.md`) antes de rodar remoto, se houver série
  bruta de sensores disponível para esse período.
- Alternativa a avaliar: já que a assinatura física do trip parece ser
  "aproximar-se do teto da faixa normal enquanto opera" (não um evento
  raro e distinto), pode fazer mais sentido tratar isso como um problema
  de **regressão/previsão de tendência** (prever o valor de `PI_0308`
  N minutos à frente e alarmar quando a previsão cruza o teto) em vez de
  detecção de anomalia por reconstrução — não implementado ainda, é só
  uma direção alternativa a considerar se o AE não performar bem com
  tão poucos exemplos positivos.

## Estratégia: repetir o arco AutoML EXP5→EXP10c antes de ir para CNN1D-AE

Igual ao que foi feito pra T5 (EXP5: grid amplo dense/ocsvm/iforest →
EXP7: multi-escala/textura vence com ocsvm p99,9/db1 → EXP10/10b/10c:
refina com máscara operacional + portão de rampa + portão de
volatilidade, chegando a 92,5% hit_rate / 0,35% FP), vamos primeiro rodar
um **grid amplo de AutoML** pra descobrir qual modelo/threshold/debounce
se sai melhor com os dados de `PI_0308`, antes de partir pro CNN1D-AE
(que fica pra depois, como segunda rodada — ver `test_grupo_exp16_trip_oleo_lub.json`,
guardado como referência CNN1D-AE mas não é o próximo passo agora).

**Config do grid:** `configs/calibracao_v4_eq/test_grupo_exp16a_trip_oleo_lub_automl_grid.json`
— clone do template que gerou o resultado do EXP7 (`test_grupo_exp7_multiescala.json`):
3 modelos (`dense`, `ocsvm`, `iforest`) × 7 percentis (90–99,9) × 6
debounces (1–24), sem seed-sweep ainda (roda depois de escolher o
vencedor, como no EXP7→EXP10 original).

## Grupo multivariado (usado tanto no grid AutoML quanto no CNN1D-AE de referência)

Escolhido por correlação com o alvo (`RUNNING_A=1`, n≈1,65M pontos) +
relevância física (mesmo circuito de lubrificação/mancais):

| Sensor | Correlação com `PI_0308` | Motivo de inclusão |
|---|---|---|
| `954005_624_PI_0340` | 0,898 | mesma linha de óleo (header turbina) |
| `954005_624_PI_0339` | 0,895 | terceira pressão de óleo lub. correlacionada |
| `954005_624_TI_0325` | 0,531 | temperatura do tanque, mesmo circuito |
| `954005_624_TI_0301/0303/0305/0307` | 0,44–0,65 | temperatura dos mancais (dependem de lubrificação) |
| `TV_35X_A` (10 canais) | 0,12–0,51 | vibração dos mancais, textura apenas (como nos outros grupos) |

Config segue o formato do candidato final T5 (EXP15b): CNN1D-AE,
`NORMALIZE_ON_STATE_ONLY=true`, gates de rampa/volatilidade, seed-sweep
n=4. **Valores de threshold/gate (`THRESH_STD_K=4,5`,
`VOLATILITY_GATE_THRESHOLD=0,1745`, `OFF_TARGET_ABS_THRESHOLD=0,7`) são
placeholders herdados do EXP15b — não foram recalibrados para a escala
de `PI_0308`.** Antes de submeter remoto, seguir a mesma metodologia do
EXP13 (regrid/simulação offline sobre um resultado piloto) para calibrar
esses valores à nova escala, em vez de gastar rodadas remotas às cegas.

## Task ClearML

1. `1009af2fce754951baa90d137ad058e0` — EXP16a (`test_grupo_exp16a_trip_oleo_lub_automl_grid.json`), submetida em 2026-08-25. Grid ampliado em relação ao template original do EXP7: 2 grupos (`controle_alvo_vibracao` univariado+vibração vs. `PI0308_trip_oleo_lub_multivariado` completo) × 3 modelos (`dense`, `ocsvm` com grade `AUTOML_OCSVM_NU_GRID=[0.01,0.03,0.05,0.1]`/`AUTOML_OCSVM_GAMMA_GRID=["scale","auto"]` = 8 variantes, `iforest`) × 7 percentis × 6 debounces — **420 trials/grupo, 840 no total** (20 treinos reais: 10 por grupo). Objetivo duplo: (a) achar o modelo/threshold/debounce vencedor, igual ao papel do EXP7 pra T5; (b) comparar univariado-com-vibração vs. multivariado-completo pra medir se os sensores extra do circuito de óleo realmente ajudam a explicar o comportamento de `PI_0308` antes do trip.

## Resultado do EXP16a (grid AutoML, 840 trials)

**Vencedor: `iforest`, percentil 99,5, debounce 24 pontos** — idêntico nos dois
grupos (`controle_alvo_vibracao` e `PI0308_trip_oleo_lub_multivariado`):

| Grupo | hit\_rate (OOS, 2 alarmes) | FP (`normal_alert_rate`) |
|---|---|---|
| `controle_alvo_vibracao` (alvo + vibração só) | 100% (2/2) | 0,143% |
| `PI0308_trip_oleo_lub_multivariado` (+ pressões/temps do óleo) | 100% (2/2) | 0,144% |

**Os sensores extra do circuito de óleo (`PI_0340`, `PI_0339`, `TI_0325`,
temperaturas de mancal) não agregaram nada** mensurável sobre o grupo
simples alvo+vibração — resultado praticamente idêntico. Escolhido o
grupo **`controle_alvo_vibracao`** (mais simples, mesmo desempenho) como
base daqui pra frente.

### Sanity-check: o resultado é real, não artefato do n=2

Simulado offline um threshold ingênuo (percentil direto sobre o valor
bruto de `PI_0308`, sem nenhum modelo, mesma avaliação/gates):

| Abordagem | Threshold | hit\_rate | FP |
|---|---|---|---|
| `iforest` multivariado | p99,5/db24 | **100%** | **0,143%** |
| Ingênuo (só `PI_0308`) | p99,5/db24 (mesmo corte) | **0%** | 1,28% |
| Ingênuo (só `PI_0308`) | p90/db24 (bem mais frouxo) | 100% | **10,3%** (72x pior) |

No mesmo threshold que o `iforest` usa, um corte direto no valor bruto
erra os 2 alarmes; pra acertar os 2 com um corte ingênuo, o FP explode
72x. **Conclusão: o `iforest` está usando o contexto da vibração pra
separar os 2 eventos genuínos de forma muito mais precisa do que o
valor do sensor sozinho permite — não é artefato do conjunto de teste
pequeno.**

### Caracterização dos falsos positivos residuais (OOS, `controle_alvo_vibracao`)

16 episódios de FP fora da janela dos 2 alarmes genuínos, entre
2025-07-31 e 2026-04-15, variando de 1 a 235 pontos (o maior,
2026-01-29 12h04–14h01, coincide com a partida real documentada em
`docs/analise_cnn1dae_exp13.md` que também gerou falso alerta na T5 —
possivelmente o mesmo evento físico de partida atravessando os dois
alvos, não uma coincidência).

**Investigação caso a caso (cruzando `PI_0308`, vibração e `T5_AVG_A`
ao redor de cada episódio):** não há uma causa única — pelo menos 3
fenômenos misturados:

1. **Em todos os 16 casos, `PI_0308` está no teto da faixa normal mas
   nunca extremo** (1,38–1,44; normal p50=1,391/p95=1,426/p99=1,436) —
   confirma que é o conjunto multivariado que pega, não o valor bruto
   sozinho (consistente com o sanity-check).
2. **~5-6 episódios coincidem com variação real de `T5_AVG_A`** de
   dezenas de graus na hora ao redor (manobra de carga real) — mesmo
   tipo de gatilho que gerou FP na T5 no EXP10. Ex: 2025-08-24,
   2025-12-02, os dois de 2026-01-29, 2026-04-08.
3. **1 episódio (2026-04-15 14h50) é glitch de sensor confirmado**, não
   evento real: `T5_AVG_A` despenca de ~760°C pra 415°C num único ponto
   (14h40) e volta ao normal no ponto seguinte — ruído entrando na
   janela de features derivadas.
4. **Vibração elevada só explica 1 episódio** (2026-01-29, já sabido
   ser real) — baseline justo de volatilidade (amostra on-state, janela
   90min): mediana 0,142; a maioria dos outros 15 episódios fica em
   0,07–0,39, dentro do normal.
5. **~9-10 episódios sem causa identificada** com os 3 sinais checados —
   provavelmente vêm de alguma combinação nas features derivadas
   (`DERIVED_ROLLING_WINDOWS`) que o IsolationForest usa; atribuir a
   causa exata exigiria análise de importância de feature por ponto
   (SHAP ou equivalente), não feita aqui.

**Conclusão:** mistura de manobras de carga reais, 1 glitch de sensor
confirmado, e o resto sem causa clara — não muda a decisão de
consolidar o candidato sem portões (FP de 0,143% já é baixo; um portão
de volatilidade seguiria arriscado dado o papel da vibração como sinal,
não ruído, pra este alvo).

### ⚠️ Cuidado antes de copiar os portões do EXP10c/EXP15b direto

Nos experimentos de T5, o portão de volatilidade (`VOLATILITY_GATE`)
suprime `is_anom_point` quando a vibração fica volátil demais —
funciona lá porque vibração é só uma feature de contexto, a T5
(temperatura) é quem carrega o sinal de falha. **Aqui é diferente: o
sanity-check acima mostrou que é justamente a vibração que faz o
`iforest` separar os 2 eventos genuínos do ruído.** Aplicar o mesmo
portão de volatilidade sem checar primeiro se ele suprime os pontos
que hoje acertam os 2 alarmes genuínos pode reduzir o FP à custa de
also perder a detecção real — o oposto do que aconteceu na T5. Qualquer
portão de rampa/volatilidade aqui precisa ser validado offline
especificamente contra os 2 episódios genuínos antes de entrar num
config remoto.

## Candidato final consolidado do EXP16

**`iforest`, grupo `controle_alvo_vibracao` (alvo `954005_624_PI_0308` +
10 canais de vibração), percentil 99,5, debounce 24 pontos.** hit_rate
100% (2/2 alarmes genuínos OOS) / FP 0,143%, sem nenhum portão de
pós-processamento. Decisão de não adicionar portões de rampa/volatilidade
agora: FP já está bem baixo sem eles (diferente do ponto de partida da T5,
1,94%), e o portão de volatilidade em particular seria arriscado aqui —
ver alerta acima sobre vibração ser sinal, não ruído, neste alvo. Os 16
episódios de FP residual (0,143%) ficam como possível investigação futura,
não bloqueiam a consolidação.

**EXP16b (task `fc787faca377406890fed0309cd95b3c`, config
`test_grupo_exp16b_trip_oleo_lub_iforest_final.json`):** config final
travado (1 grupo, 1 modelo, percentil/debounce fixos) reproduzindo o
vencedor do EXP16a — resultado idêntico (threshold, hit_rate, FP), como
esperado. **Seed-sweep (`AUTOML_SEED_SWEEP_N=4`, sementes 42-46):
`hit_rate_std = 0,0pp`** — os 2 alarmes genuínos são detectados em
**todas as 5 sementes**, FP variando só entre 0,133%-0,143%
(`normal_alert_rate_std` ≈ 0,004pp). Variância de semente nula, bem mais
estável que o CNN1D-AE jamais foi pra T5 (que tinha `hit_rate_std` de
vários pontos percentuais) — reforça a robustez do candidato.

### ⚠️ Correção importante: `hit_rate=100%` mistura previsão real com detecção reativa

`eval_alarm_hit_rate` conta "acerto" se **qualquer** ponto anômalo cai
dentro de ±24h do alarme, **antes ou depois** — a mesma armadilha
metodológica já identificada e corrigida pra T5 no EXP13 (categorias
"preditivo genuíno" vs "reativo"), que ainda não tinha sido aplicada
aqui. Checando a antecedência real de cada um dos 2 alarmes genuínos
(`point_anomalies_all.csv` da task EXP16b):

| Alarme | Pontos anômalos na janela ±26h | Quando | Antecedência |
|---|---|---|---|
| 2025-11-04 06:22:18 | 126, todos antes | -- | **13,3–14,3h antes -- preditivo genuíno** |
| 2026-02-26 15:34:20 | 26, todos depois | -- | **6,7–6,9h depois -- puramente reativo** |

**Cobertura genuinamente preditiva real: 50% (1/2), não 100%.** O
segundo caso só "conta" na métrica porque a janela de avaliação também
olha pra depois do alarme — na prática, um alerta que chega 7h depois
do trip já disparado não tem valor operacional (o sistema já avisou
antes). Isso não muda a decisão de manter `iforest`/`controle_alvo_vibracao`
como candidato (ainda é o melhor resultado disponível, e o sanity-check
de vibração continua válido), mas **a alegação correta é "detecta 1 de
2 alarmes com antecedência real de ~14h", não "acerta 100% dos
alarmes".** Com n=2 isso já era estatisticamente frágil; agora sabemos
que nem os 2 casos genuínos se comportam da mesma forma.

**Esta é a pipeline vencedora consolidada do EXP16 — com essa ressalva
de cobertura preditiva registrada.**

## Prontidão para produção real-time: não está pronto

Pergunta levantada explicitamente e vale registrar a resposta. O pipeline
atual é retrospectivo/batch (`python src/main.py` → ClearML baixa
snapshot CSV → treina → avalia contra corte OOS fixo → relatório), não
um serviço de inferência contínua. Faltam, no mínimo:

1. **Conexão viva com o Historian PI** — hoje só existe leitura de
   snapshot CSV via ClearML Dataset, sem tag subscription em tempo real.
2. **Job de scoring agendado/contínuo** que pontue a janela mais recente
   e dispare alerta (o pipeline hoje só roda sob demanda, uma vez).
3. **Integração com alerta operacional** (notificar operador) — não existe.
4. **Rotina de retreino/recalibração periódica** — o `iforest`
   consolidado aprendeu "normal" só até jun/2025; sem atualização desde
   então, o threshold pode já estar defasado (deriva sazonal, manutenções).
5. **Base de validação pequena demais pra garantia operacional** — n=2
   no teste, com só 1 caso genuinamente preditivo (ver acima). Não dá
   pra afirmar uma taxa de detecção com confiança estatística.

O resultado do EXP16 é uma prova de conceito que confirma que existe
sinal real (o caso de 14h de antecedência), não um sistema pronto pra
prever falha em produção.

## Próximos passos

1. ~~Submeter grid AutoML~~ — feito, EXP16a (task
   `15833383325746c999e8b866dda4b5e9`), candidato final acima.
2. ~~Refinar com portões~~ — decidido não fazer agora (FP já baixo sem
   portões; portão de volatilidade seria arriscado aqui, ver alerta acima).
3. O CNN1D-AE de referência (`test_grupo_exp16_trip_oleo_lub.json`) fica
   engavetado — só entra em jogo se surgir motivo concreto pra revisitar
   (ex: mais dados genuínos tornando a avaliação mais robusta e valendo a
   pena testar uma arquitetura mais pesada).
4. ~~Decidir se amplia a janela de dados pra antes de 2024-01-01~~ —
   investigado, **não compensa**. Existe um dataset ClearML mais amplo
   (`Cabiunas consolidado 2022-2026`, id `58a4c230ff30420aa31f1d83d2da79ee`,
   colunas sem o prefixo `954005_624_`) cobrindo 2022–2026, mas: **2023
   está com 0 linhas de dado de sensor** (ausência total nesse dataset)
   — e é justamente 2023 que teria os 6 eventos adicionais do alarme
   (todos caem em `-1.000`/`NaN`, sem leitura). **2022 tem ótima
   cobertura (99,9%+) mas zero eventos do alarme `PALL_6240309`
   registrados nesse ano.** Ampliar a janela só acrescentaria dado de
   treino "normal" (2022), sem resolver o gargalo de eventos genuínos de
   teste. Se existir uma exportação separada de 2023 em algum lugar não
   catalogado no ClearML, valeria checar com o time — mas não foi
   encontrada nos datasets disponíveis. O gargalo do EXP16 continua
   sendo n=2 alarmes genuínos no período OOS, sem solução de dado à vista.
5. ~~Investigar os 16 episódios de FP residual~~ — feito, ver seção
   acima (mistura de rampas reais, 1 glitch de sensor, resto sem causa
   clara).
6. ~~Investigar por que 2026-02-26 é puramente reativo~~ — feito, ver
   seção abaixo. **Achado: não há precursor, é limitação física/de
   instrumentação, não do modelo.**

### Investigação do caso 2026-02-26 (reativo)

Série minuto-a-minuto ao redor do trip (`sensores_full_2024_2026_30s.csv`):

| Horário | `PI_0308` | `RUNNING_A` | `T5_AVG_A` |
|---|---|---|---|
| 13h30–15h32 (2h antes) | 1,3537 → 1,3478, declínio linear suavíssimo (~0,003/h) | 1,0 | ~713–716°C, estável |
| **15h34:00 (minuto do trip)** | 1,3477 | **0,167 (colapsando)** | **295,9 (despenca)** |
| 15h36+ | -0,26 (valor de off) | 0,0 | ~200–280 |
| 21h20 (partida) | volta a 1,43 | 1,0 | 460→698 |

**Não há precursor.** Nas 2h antes do trip, `PI_0308` está completamente
estável dentro da faixa normal (1,348–1,354 — nem perto do teto,
diferente dos episódios de FP caracterizados acima) e a vibração também
sem nada fora do normal. **O trip e o desligamento acontecem no mesmo
minuto** (15h34) — evento físico abrupto (pressão cruza o setpoint de
proteção), sem degradação gradual visível em nenhum dos sensores
disponíveis. Diferente do caso 2026-01-29 da T5 (EXP13), onde havia
degradação real acontecendo antes do alarme só bloqueada pelos portões
— aqui não há nada bloqueado, o sinal simplesmente não existe nos dados.

**Os 26 pontos "reativos" (22h17–22h30, ~1h após a partida às 21h20)
provavelmente nem são reconhecimento tardio da falha** — coincidem com
a acomodação suave de `PI_0308` pós-partida (1,432→1,424 na hora
seguinte), o mesmo mecanismo que gera vários dos 16 episódios de FP.
Ou seja: este alarme provavelmente não tem **nenhuma** detecção real
(nem preditiva nem reativa de fato) — o "acerto" na métrica é
coincidência de ruído de partida caindo dentro da janela de ±24h.

**Conclusão: limitação de instrumentação/física do processo, não do
modelo ou do pipeline** — mesma categoria de achado negativo documentada
pro EXP13/T5 (`docs/analise_cnn1dae_exp13.md`, seção de achados
negativos). Sem um sensor que capture o precursor real desse tipo de
trip (se é que existe um fisicamente), não há o que ajustar no modelo.

### Checagem do 3º evento genuíno (2024-03-07, fora da avaliação por cair no treino)

Pergunta natural: "mesmo com engenharia de deploy perfeita, esse modelo
prediria a falha de verdade?" Só dava pra responder isso checando
também o 3º evento genuíno conhecido (`2024-03-07 13:55:37`), que nunca
foi avaliado (cai no período de treino, não no OOS). Resultado: **também
sem precursor.** Curiosamente o alarme dispara às 13h55:37 mas a queda
real de pressão/desligamento só acontece ~2h depois (15h56–16h16);
`PI_0308` fica completamente estável (1,360–1,364) tanto nas 2h antes
quanto no próprio horário oficial do alarme — sem nenhuma reação.

**Contagem real com os 3 eventos genuínos conhecidos (não só os 2 do
OOS):**

| Evento | Sinal detectável? |
|---|---|
| 2024-03-07 | ❌ sem precursor |
| 2025-11-04 | ✅ ~14h de antecedência real |
| 2026-02-26 | ❌ sem precursor |

**Cobertura genuinamente preditiva real: 1 em 3 (~33%), não 1 em 2
(50%).** Resposta à pergunta "com deploy perfeito, esse modelo prediria
a falha?": **parcialmente, não sempre.** Não é limitação de engenharia
de deploy — é limite de informação disponível nos sensores atuais. Se
esse tipo de trip tiver duas causas físicas distintas (uma degradação
lenta detectável, outra um evento abrupto/mecânico sem aviso),
aumentar dados ou trocar de modelo não resolve o segundo tipo; exigiria
um sensor que hoje não está disponível (ou o precursor pode
genuinamente não existir fisicamente pra esse modo de falha). Com n=3,
essa taxa de 33% carrega incerteza estatística enorme, mas é o retrato
mais completo disponível hoje.

### Duas hipóteses de precursor testadas e refutadas (EXP16c)

Buscando destravar 2024-03-07 e 2026-02-26, varreu-se ~36 sensores
brutos disponíveis (z-score robusto nas 24h antes de cada evento,
comparado contra o caso preditivo 2025-11-04 como controle). Dois
candidatos surgiram e foram testados a fundo — **os dois refutados:**

1. **Declínio de temperatura do óleo/mancais (`TI_0301/03/05/07/0325`)
   antes de 2024-03-07** — real e coerente entre os 5 sensores (~1,5-2°C
   em ~20h), mas checagem de taxa-base mostrou que esse padrão acontece
   **32,5% do tempo operacional** (387 episódios em 2,3 anos), com só
   0,8% seguidos de trip — é variação normal, não precursor. Confirmado
   também que o modelo multivariado já treinado (que já incluía essas
   5 temperaturas) não sinalizou nada nessa janela (0 pontos anômalos).

2. **Elevação de `PDIT_0305` (pressão vazamento gás selagem) antes de
   2026-02-26** — muito mais específico à primeira vista (só 0,036% do
   tempo, 16 episódios, 12,5% seguidos de trip, 2 desses 16 precedendo
   exatamente esse evento). **EXP16c** (tasks
   `640251258fcc471bb8f25bf8fe3ffac8` e `3f679264f5cb4c2081f658bdb2fa154b`,
   config `test_grupo_exp16c_trip_oleo_lub_pdit0305.json`) testou
   adicionar `PDIT_0305` ao grupo (`candidato_com_pdit0305` vs
   `controle_atual_sem_pdit0305`). **Resultado: não mudou nada** — o
   primeiro ponto anômalo em 2026-02-26 continua exatamente no mesmo
   horário (22h17:30, ~6,7h **depois** do trip) com ou sem `PDIT_0305`;
   só 6 pontos a mais na mesma janela reativa, nenhum antes do alarme.
   O sinal correto na análise de taxa-base não sobreviveu ao score
   conjunto do IsolationForest com os outros 11 sensores — mesmo padrão
   de diluição do achado da temperatura.

**Conclusão desta linha de investigação:** duas hipóteses concretas
testadas com rigor (taxa-base + confirmação no modelo real), ambas
refutadas. Fecha o ciclo de busca por precursores baratos pros 2 casos
sem sinal — mesma natureza dos "achados negativos" documentados no
EXP13 pra T5. Não há evidência de que mais dados ou mais tempo de busca
nos ~36 sensores já disponíveis destravem esses 2 casos; um novo
sensor (não coletado hoje) ou uma mudança de arquitetura seriam os
próximos passos plausíveis, fora do escopo de uma investigação offline.

### Nota técnica: falha de infraestrutura na reprodução local

Durante essa investigação, `point_anomalies_all.csv` do grupo
`candidato_com_pdit0305` falhou o upload/download no ClearML por duas
vezes (404, depois timeout de rede) antes de baixar com sucesso na
5ª tentativa. Reprodução local (bypassando o ClearML) esbarrou em OOM
real do host compartilhado — o `IsolationForest` do scikit-learn
precisa de ~5,6GB de RSS pra pontuar 2,45M pontos mesmo com só 50
árvores (contra 200 do config oficial), confirmado via
`dmesg`/`journalctl` (oom-kill do kernel, não bug de código). Se este
tipo de diagnóstico precisar ser repetido no futuro, considerar
pontuar `x_all` em lotes (batches) em vez de de uma vez, ou rodar em
máquina com mais memória disponível.
