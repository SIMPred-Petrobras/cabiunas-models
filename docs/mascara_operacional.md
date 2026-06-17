# Máscara Operacional — Documentação

## O Problema

O modelo CNN1D-AE é treinado para aprender o padrão de **operação normal** da turbina.
Quando a turbina está desligada ou em transição (liga/desliga), o comportamento dos
sensores é completamente diferente da operação normal — e qualquer detecção nesses
períodos seria um **falso positivo** sem valor operacional.

A máscara operacional resolve isso classificando cada instante temporal em um de
quatro estados, e suprimindo alertas fora da operação normal.

---

## Sensor de Referência

Para este experimento (grupo NPT_A + NGP_A + TC382_03_A), o sensor de referência é:

```
OPERATIONAL_REF_SENSOR = "NPT_A"   (velocidade da turbina)
```

NPT_A é ideal porque:
- Vai a **zero exato** quando a turbina para
- Sobe de forma estável durante operação normal (60–96 rpm equivalentes)
- Não tem ambiguidade: zero = parado, acima de zero = girando

---

## Threshold OFF

O threshold que separa "ligado" de "desligado" é calculado automaticamente como o
**percentil 5% (p5) do NPT_A** sobre toda a série:

```
OFF_VALUE_QUANTILE = 0.05
off_threshold = NPT_A.quantile(0.05) = 0.00
```

O p5 cai em **zero** porque a turbina ficou parada em ~22.5% do período total
(Jan–Out 2025). Portanto, a regra efetiva é:

```
NPT_A = 0  →  turbina DESLIGADA
NPT_A > 0  →  turbina LIGADA
```

---

## Os Quatro Estados

```
NPT_A > 0  e  fora de borda
│
├── on          ← operação normal  [VERDE nos plots]
│                  detecção de anomalia ATIVA
│
NPT_A = 0
│
├── off_curto   ← parada < 24h    [LARANJA nos plots]
│                  detecção SUPRIMIDA
│
├── off_longo   ← parada ≥ 24h    [LARANJA nos plots]
│                  detecção SUPRIMIDA
│                  também excluído do TREINO
│
Bordas de transição (±20 min ao redor de cada liga/desliga)
│
└── transiente  ← liga ou desliga  [LARANJA nos plots]
                   detecção SUPRIMIDA
```

---

## Distribuição Real dos Estados (Jan–Out 2025)

| Estado | Pontos | Horas | % do tempo |
|--------|--------|-------|-----------|
| `on` (verde) | 681.930 | 5.682,8h | **77,9%** |
| `off_longo` (laranja) | 165.787 | 1.381,6h | 18,9% |
| `off_curto` (laranja) | 27.803 | 231,7h | 3,2% |

---

## Exemplo Real: Transição Desliga → Liga (01/Jan/2025)

A turbina estava parada desde as 00:00. Às 11:37 houve uma tentativa de partida
(NPT_A subiu para 5.2) que não sustentou — voltou a zero. TC382_03_A mostrou
comportamento caótico nesse período:

```
Timestamp          NPT_A    TC382_03_A    Estado
─────────────────────────────────────────────────
2025-01-01 11:27    0.0       32.0        off_longo  ← turbina parada
2025-01-01 11:32    0.0       32.6        off_longo
2025-01-01 11:37    5.2      539.1        transiente ← tentativa de partida
2025-01-01 11:42    0.0      150.7        off_curto  ← voltou a parar
2025-01-01 11:47    0.0      122.5        off_curto  ← temperatura caindo
2025-01-01 11:52    0.0      103.3        off_curto
2025-01-01 11:57    0.0       93.2        off_curto
2025-01-01 12:02    0.0       49.0        off_curto
```

Sem a máscara, o salto de TC382_03_A de 32°C para 539°C em 5 minutos seria
detectado como anomalia — mas é apenas o comportamento esperado de partida a frio.
Com a máscara, **esse período todo é laranja e suprimido**.

---

## Períodos off_longo Reais (paradas ≥ 24h)

| Início | Fim | Duração |
|--------|-----|---------|
| 2025-01-01 12:08 | 2025-01-08 01:33 | 157,4h (~6,6 dias) |
| 2025-01-08 15:00 | 2025-01-14 16:19 | 145,3h (~6,1 dias) |
| 2025-01-21 08:35 | 2025-01-23 12:46 | 52,2h |
| 2025-03-06 17:28 | 2025-03-08 22:52 | 53,4h |
| 2025-04-11 17:05 | 2025-04-13 10:26 | 41,4h |
| 2025-04-19 16:26 | 2025-04-20 17:39 | 25,2h |
| **2025-04-29 14:21** | **2025-05-22 15:15** | **552,9h (~23 dias)** |
| 2025-06-04 23:49 | 2025-06-09 13:59 | 110,2h |
| 2025-06-09 14:04 | 2025-06-15 18:01 | 148,0h |
| 2025-08-19 10:56 | 2025-08-23 10:33 | 95,6h |

O maior período foi de **~23 dias contínuos parada** (abril–maio 2025).
Todos esses períodos são excluídos do treino e marcados como laranja nos plots.

---

## Por Que Isso Importa para o Modelo

O treino do CNN1D-AE usa **apenas os períodos verdes** (`on`).
Isso garante que o modelo aprende exclusivamente o padrão de operação normal,
sem contaminação por comportamentos de partida, desligamento ou parada prolongada.

Na inferência (avaliação de toda a série), anomalias detectadas em períodos laranja
são descartadas — apenas alertas em períodos **verdes** chegam ao operador.

```
Dado bruto completo (Jan–Out 2025)
        │
        ├── VERDE (77,9%)  →  entra no TREINO + gera ALERTAS
        └── LARANJA (22,1%) →  excluído do TREINO, ALERTAS suprimidos
```
