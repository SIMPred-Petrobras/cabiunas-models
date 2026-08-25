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

## Próximos passos

1. Submeter `test_grupo_exp16a_trip_oleo_lub_automl_grid.json` (remoto,
   ClearML) — grid amplo de AutoML, primeiro passo do arco.
2. Com o resultado, escolher o modelo/threshold/debounce vencedor
   (equivalente ao "ocsvm p99,9/db1" que venceu pra T5 no EXP7) e criar
   um EXP16b de refinamento (máscara operacional + portões, igual
   EXP10/10b/10c) mirando reduzir FP sem perder os 3 eventos genuínos.
3. Só depois, se fizer sentido, tentar o CNN1D-AE
   (`test_grupo_exp16_trip_oleo_lub.json`, já preparado como referência).
4. Decidir se amplia a janela de dados pra antes de 2024-01-01 (mais
   eventos genuínos = avaliação menos frágil) — independente da escolha
   de modelo.
5. Se nenhum modelo performar dado o n baixo (3 eventos genuínos), avaliar
   a alternativa de regressão/previsão de tendência descrita acima.
