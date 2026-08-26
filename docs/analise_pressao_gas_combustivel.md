# EXP17 — Predição do alarme de pressão baixa de gás combustível

**Branch:** `AE_pressao_gas_comb` (a partir de `AE_novo_legado`). Nova frente
de detecção, independente da T5 (`AE_novo_legado`) e do trip de óleo
lubrificante (`AE_trip_oleo_lub`). Objetivo: continuar expandindo a
pipeline de modelagem preditiva multivariada pros demais alarmes/sensores
disponíveis no catálogo.

## Por que este alvo

Levantamento do catálogo completo de alarmes (13 tags mais frequentes),
cruzado com o estado operacional (`RUNNING_A`) no instante de cada
disparo, para achar candidatos com volume suficiente de eventos genuínos
(problema que limitou o EXP16 a n=3):

| Tag | Descrição | Eventos no período c/ dado | Genuínos (`RUNNING_A=1`) | % |
|---|---|---|---|---|
| **`PAL_6240315`** | Pressão baixa gás combustível | 402 | **132** | 32,8% |
| `PDAL_6240302` | Diferencial pressão linha balanceamento | 64 | 57 | 89,1% |
| `TAHH_6240305` | Temp. muito alta mancal radial LNA | 57 | 10 | 17,5% |
| `TAH_6240305` | Temp. alta mancal radial LNA | 69 | 9 | 13,0% |
| `PAL_6240339` | Pressão baixa header óleo lub. | 187 | 12 | 6,4% |
| `TAH_6240301/6240303/6240307` | Temp. alta outros mancais | 15-24 | 0 | 0% (puro artefato) |

**`PAL_6240315` escolhido primeiro**: maior amostra de eventos genuínos de
longe (132) — mais de 40x a amostra do EXP16 (n=3) — permitindo uma
avaliação de `hit_rate` estatisticamente muito mais robusta.
`PDAL_6240302` fica como segundo candidato (Seção de próximos passos),
por ter a maior pureza (89,1%) mas amostra menor.

Sensor mapeado: `PAL_6240315` → `954005_624_PI_0315` (via planilha
`Alarmes X Instrumentos`).

## Caracterização do sinal

- **Faixa de operação muito estreita** (on-state): p1=16,74, mediana=16,97,
  p95=17,20, p99=17,29, máx=17,91 — um sinal bem regulado/controlado, com
  pouca variação em condição normal. Colapsa pro piso de desligamento
  (~-11) quando a turbina para.
- **Correlação fraca com todos os outros sensores disponíveis** — a maior
  é com vibração (`TV_353X_A`, r=-0,32), o resto abaixo disso
  (`PDI_0338` r=-0,26, `PDI_0301` r=0,21, `PDI_0317` r=-0,17). Nenhum
  sensor explica bem seu comportamento normal.
- **Interpretação:** por ser um sinal pouco correlacionado com carga/
  temperatura, um desvio genuíno tende a não ficar confundido com manobra
  operacional normal — mesma lógica que fez o EXP16 funcionar bem com
  alvo+vibração simples, sem precisar de features extras.

## Grupos multivariados propostos (`configs/calibracao_v4_eq/test_grupo_exp17a_pressao_gas_comb_automl_grid.json`)

Mesma estrutura de comparação do EXP16a — controle simples vs. grupo com
os poucos correlatos disponíveis:

1. **`controle_alvo_vibracao`**: `PI_0315` + 10 canais de vibração.
2. **`PI0315_gas_comb_multivariado`**: + `PDI_0338`, `PDI_0301`, `PDI_0317`
   (os 3 sensores de pressão diferencial com correlação acima de 0,17).

Grid idêntico ao EXP16a: `dense`/`ocsvm` (grade nu×gamma)/`iforest` × 7
percentis × 6 debounces × 2 grupos = 840 trials.

`OFF_TARGET_ABS_THRESHOLD=10,0`: calibrado pro salto grande entre on
(~17) e off (~-11) deste sensor — placeholder inicial, não recalibrado
por simulação offline ainda (seguir a mesma disciplina do EXP13/16 antes
de confiar cegamente nele).

## Task ClearML

1. `116fc65495984744ad991df9d1320ee8` — EXP17a, grid AutoML (840 trials),
   `test_grupo_exp17a_pressao_gas_comb_automl_grid.json`. **Resultado
   abaixo.**

## Resultado do EXP17a

Amostra de teste muito mais robusta que o EXP16: **108 alarmes genuínos
no período OOS** (contra 2 do trip de óleo).

| | `controle_alvo_vibracao` | `PI0315_gas_comb_multivariado` |
|---|---|---|
| Modelo vencedor | `ocsvm` (não `iforest`) | `ocsvm` |
| Threshold/debounce | percentil 97 / 6 pontos | percentil 99,5 / 3 pontos |
| `hit_rate` | 80,6% (87/108) | **81,5% (88/108)** |
| FP (`normal_alert_rate`) | 4,45% | **2,90%** |
| `anomaly_rate_points_per_day` | 281 | 209 |

**Achados:**
- Ao contrário do EXP16 (trip de óleo), aqui **o grupo multivariado ajudou
  de verdade** — hit_rate maior e FP quase pela metade. As 3 pressões
  diferenciais correlacionadas (`PDI_0338`, `PDI_0301`, `PDI_0317`) agregam
  sinal real, não são redundantes com a vibração.
- **Modelo vencedor mudou** de `iforest` (EXP16) para `ocsvm` — reforça
  não presumir o vencedor anterior e sempre rodar o grid completo.
- **FP bem mais alto** que o trip de óleo (2,9-4,45% vs 0,143%) —
  `anomaly_rate_points_per_day` de ~209-281 é uma taxa de alerta bem mais
  barulhenta. Pode ser inerente a esse alarme (mais frequente/rotineiro)
  ou sinal de que precisaria de portões de refinamento (como na T5).
- **Ainda não verificado**: antecedência real ponto a ponto (preditivo vs.
  reativo) numa amostra dos 87-88 acertos — pendência antes de consolidar
  este candidato, mesma lição do EXP16 (hit_rate agregado pode mascarar
  detecção puramente reativa).

## Status: pausado

Trabalho pausado em 2026-08-26 para retomar o EXP10c/T5 (branch
`AE_novo_legado`). Este branch (`AE_pressao_gas_comb`) fica pronto pra
retomar quando for a vez: próximo passo é a checagem de antecedência
real, seguida (se justificar) de um EXP17b com config travado + portões
de refinamento pra reduzir o FP mais alto observado aqui.

## Próximos passos (ao retomar)

1. Checagem de antecedência real (preditivo vs. reativo) numa amostra dos
   88 acertos do grupo multivariado.
2. Se justificar, EXP17b: config final travado + seed-sweep, e considerar
   portões de refinamento (rampa/volatilidade) dado o FP mais alto que no
   EXP16 -- aqui o risco de suprimir sinal via vibração precisa ser
   reavaliado (correlação de `PI_0315` com vibração é mais fraca, -0,32,
   que no caso do óleo, então o portão de volatilidade pode ser menos
   arriscado aqui).
3. Depois, considerar `PDAL_6240302` como segunda frente (menor amostra,
   mas maior pureza e correlação nativa mais forte com temperatura).
