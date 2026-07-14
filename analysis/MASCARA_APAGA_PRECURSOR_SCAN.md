# Varredura: máscara apagando precursor sustentado perto da falha

Critério: bins de 2h dentro de ±10d/+6h da falha com `pct_mae_over>=0.3` (sinal bruto cruzando threshold) E `pct_anom_seq<=0.05` (quase nada sobrevive à máscara) E `pct_on>=0.2` (não é caso de máquina genuinamente desligada).

**Equipamentos afetados: 7 ocorrência(s) (uni+mult) de 24 modelos avaliados.**

| Equip | Modo | Falha | Bins apagados | Horas apagadas | %MAE-over máx | %transiente médio | flicker (transições/h) |
|---|---|---|---|---|---|---|---|
| `B-3403C` | uni | 2023-09-12 | 5 | 30 | 0.93 | 0.47 | 2.89 |
| `B-4064A` | uni | 2024-08-30 | 5 | 30 | 0.53 | 0.47 | 0.48 |
| `B-4064A` | mult | 2024-08-30 | 4 | 24 | 0.66 | 0.0 | 0.12 |
| `B-24001B` | uni | 2025-01-06 | 2 | 12 | 0.47 | 0.63 | 0.11 |
| `B-3403C` | mult | 2023-09-12 | 2 | 12 | 0.81 | 0.0 | 0.07 |
| `B-402E` | uni | 2019-10-30 | 1 | 6 | 0.46 | 0.44 | 0.36 |
| `B-8801C` | mult | 2024-07-05 | 1 | 6 | 0.33 | 0.01 | 0.08 |
