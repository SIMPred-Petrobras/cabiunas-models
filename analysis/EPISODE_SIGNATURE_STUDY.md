# Estudo de assinatura de episódio (magnitude + duração) — near falha vs far

v2 — episódios extraídos de `is_anom_point` (já pós k_of_window, suaviza
quedas breves abaixo do threshold), não do cruzamento bruto de mae_seq>threshold.
`near_48h` = episódio dentro de ±2 dias da falha documentada; `near_10d` = 2–10
dias; `far` = >10 dias (candidato a falso positivo). Não decide regra nova — só
caracteriza o padrão.

### Uni_sensor (univariado) — pooled

Total de episódios: 817 | equipamentos com episódio: 11

| Categoria | n | duração mediana (min) | duração p75 | peak_ratio mediana | peak_ratio p75 |
|---|---|---|---|---|---|
| far | 781 | 20.0 | 50.0 | 1.38 | 1.83 |
| near_10d | 29 | 56.0 | 91.0 | 1.79 | 2.00 |
| near_48h | 7 | 105.0 | 282.5 | 2.18 | 2.65 |

**Perto da falha (near_48h+near_10d), n=36:** duração mediana=57.0min (p25=27.5, p75=106.0) | peak_ratio mediana=1.79
**Longe da falha (far), n=781:** duração mediana=20.0min (p25=3.0, p75=50.0) | peak_ratio mediana=1.38

Se usarmos como corte o p25 dos episódios `near` (duração≥27.5min OU peak_ratio≥1.33): **67.3%** dos episódios `far` seriam eliminados (duração isolada eliminaria 55.2%, peak_ratio isolado 44.6%).

### Uni_sensor (univariado) — por equipamento

| Equip | n_near | dur mediana near | peak_ratio near | n_far | dur mediana far | peak_ratio far |
|---|---|---|---|---|---|---|
| `B-0302C` | 0 | — | — | 16 | 43.5 | 1.25 |
| `B-24001B` | 7 | 114.0 | 1.817 | 38 | 60.0 | 1.57 |
| `B-3403C` | 0 | — | — | 136 | 4.0 | 1.15 |
| `B-402E` | 0 | — | — | 41 | 9.0 | 1.28 |
| `B-4064A` | 3 | 56.0 | 1.168 | 3 | 58.0 | 1.19 |
| `B-4703.24001B` | 0 | — | — | 58 | 23.0 | 1.50 |
| `B-5501B` | 3 | 50.0 | 1.788 | 156 | 40.0 | 1.56 |
| `B-6511502A` | 3 | 105.0 | 2.504 | 19 | 275.0 | 1.55 |
| `B-8801C` | 0 | — | — | 3 | 111.0 | 2.19 |
| `B-8802B` | 8 | 78.5 | 1.591 | 20 | 98.0 | 2.08 |
| `B-90001A` | 12 | 17.5 | 1.782 | 291 | 12.0 | 1.51 |

### Mult_sensor (multivariado, canal-alvo) — pooled

Total de episódios: 1096 | equipamentos com episódio: 11

| Categoria | n | duração mediana (min) | duração p75 | peak_ratio mediana | peak_ratio p75 |
|---|---|---|---|---|---|
| far | 1038 | 76.0 | 99.0 | 1.43 | 1.75 |
| near_10d | 43 | 90.0 | 121.5 | 1.77 | 2.25 |
| near_48h | 15 | 101.0 | 110.0 | 2.11 | 2.45 |

**Perto da falha (near_48h+near_10d), n=58:** duração mediana=97.5min (p25=70.5, p75=119.8) | peak_ratio mediana=1.79
**Longe da falha (far), n=1038:** duração mediana=76.0min (p25=55.0, p75=99.0) | peak_ratio mediana=1.43

Se usarmos como corte o p25 dos episódios `near` (duração≥70.5min OU peak_ratio≥1.39): **62.1%** dos episódios `far` seriam eliminados (duração isolada eliminaria 45.2%, peak_ratio isolado 46.2%).

### Mult_sensor (multivariado, canal-alvo) — por equipamento

| Equip | n_near | dur mediana near | peak_ratio near | n_far | dur mediana far | peak_ratio far |
|---|---|---|---|---|---|---|
| `B-0302C` | 2 | 145.0 | 1.526 | 74 | 88.5 | 1.31 |
| `B-24001B` | 17 | 108.0 | 2.497 | 61 | 103.0 | 1.74 |
| `B-3403C` | 7 | 88.0 | 1.168 | 64 | 87.0 | 1.17 |
| `B-402E` | 3 | 81.0 | 1.628 | 196 | 90.0 | 1.35 |
| `B-4064A` | 6 | 86.5 | 1.7375 | 27 | 74.0 | 1.47 |
| `B-4703.24001B` | 1 | 20.0 | 1.765 | 97 | 64.0 | 1.56 |
| `B-5501B` | 5 | 75.0 | 1.651 | 238 | 75.0 | 1.55 |
| `B-6511502A` | 3 | 780.0 | 1.329 | 14 | 300.0 | 1.30 |
| `B-8801C` | 4 | 105.0 | 2.1704999999999997 | 139 | 59.0 | 1.52 |
| `B-8802B` | 4 | 145.5 | 2.009 | 9 | 82.0 | 1.12 |
| `B-90001A` | 6 | 98.5 | 1.829 | 119 | 77.0 | 1.35 |
