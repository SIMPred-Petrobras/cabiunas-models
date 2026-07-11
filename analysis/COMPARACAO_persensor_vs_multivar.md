# Comparação: modelo por-sensor (univariado) vs multivariado (multisensor)

Métrica central: as **anomalias de ponto** caíram perto da falha documentada?
- **BOM** = detectou na janela ±48h da falha · **PARCIAL** = só precursor nos 10 dias antes · **FRACO** = nada em ±10 dias.

**Placar por-sensor:** BOM=2 · PARCIAL=3 · FRACO=7

**Placar multivariado:** BOM=4 · PARCIAL=2 · FRACO=6

### Por-sensor (univariado)

| Equip | Classe | Anom ±48h | Anom pré-10d | Lead (dias) | Anom/dia | Total anom |
|---|---|---|---|---|---|---|
| `B-0302C` | **FRACO** | 0 | 0 | — | 4.689203857619675 | 1041 |
| `B-24001B` | **PARCIAL** | 0 | 979 | 9.5 | 19.684968959225568 | 7185 |
| `B-3403C` | **FRACO** | 0 | 0 | — | 5.6887313249333245 | 1462 |
| `B-402E` | **FRACO** | 0 | 0 | — | 0.9753810082063306 | 832 |
| `B-4064A` | **PARCIAL** | 0 | 204 | 6.7 | 2.927435749528167 | 363 |
| `B-4703.24001B` | **FRACO** | 0 | 0 | — | 6.280606917500586 | 1746 |
| `B-5401A` | **FRACO** | 0 | 0 | — | 1.1639366346702995 | 426 |
| `B-5501B` | **FRACO** | 0 | 0 | — | 1.5980511571254568 | 1312 |
| `B-6511502A` | **BOM** | 109 | 109 | 1.8 | 3.2630447397711166 | 1191 |
| `B-8801C` | **FRACO** | 0 | 0 | — | 0.27124585374292925 | 233 |
| `B-8802B` | **BOM** | 923 | 979 | 8.8 | 54.01580914642864 | 3673 |
| `B-90001A` | **PARCIAL** | 0 | 386 | 7.1 | 34.56204379562044 | 9470 |

### Multivariado (multisensor)

| Equip | Classe | Anom ±48h | Anom pré-10d | Lead (dias) | Anom/dia | Total anom |
|---|---|---|---|---|---|---|
| `B-0302C` | **FRACO** | 0 | 0 | — | 16.4234747981569 | 3646 |
| `B-24001B` | **PARCIAL** | 0 | 1262 | 9.5 | 20.180860313661174 | 7366 |
| `B-3403C` | **PARCIAL** | 0 | 209 | 7.1 | 12.548672040294099 | 3225 |
| `B-402E` | **FRACO** | 0 | 0 | — | 7.899179366940211 | 6738 |
| `B-4064A` | **BOM** | 1209 | 1406 | 2.1 | 11.338773178613232 | 1406 |
| `B-4703.24001B` | **FRACO** | 0 | 0 | — | 1.6654759466224351 | 463 |
| `B-5401A` | **FRACO** | 0 | 0 | — | 1.1639366346702995 | 426 |
| `B-5501B` | **FRACO** | 0 | 0 | — | 1.2850182704019488 | 1055 |
| `B-6511502A` | **BOM** | 161 | 491 | 9.8 | 3.3014012690379477 | 1205 |
| `B-8801C` | **FRACO** | 0 | 0 | — | 0.19324811897564914 | 166 |
| `B-8802B` | **BOM** | 1972 | 2004 | 2.2 | 51.47163953512122 | 3500 |
| `B-90001A` | **BOM** | 55 | 524 | 7.1 | 31.427007299270073 | 8611 |

### Comparação lado a lado (classe)

| Equip | Por-sensor | Multivariado | Vencedor |
|---|---|---|---|
| `B-0302C` | FRACO | FRACO | empate |
| `B-24001B` | PARCIAL | PARCIAL | empate |
| `B-3403C` | FRACO | PARCIAL | multivariado |
| `B-402E` | FRACO | FRACO | empate |
| `B-4064A` | PARCIAL | BOM | multivariado |
| `B-4703.24001B` | FRACO | FRACO | empate |
| `B-5401A` | FRACO | FRACO | empate |
| `B-5501B` | FRACO | FRACO | empate |
| `B-6511502A` | BOM | BOM | empate |
| `B-8801C` | FRACO | FRACO | empate |
| `B-8802B` | BOM | BOM | empate |
| `B-90001A` | PARCIAL | BOM | multivariado |
