# Comparação: exp2_supressao_transiente vs exp3_mask_transiente

- **exp2_supressao_transiente:** `resultados/experimento_2_supressao_transiente`
- **exp3_mask_transiente:** `resultados/experimento_3_mask_transiente`

Classe: BOM (detectou em ±48h da falha) · PARCIAL (só precursor ≤10d) · FRACO (nada em ±10d). `rate/dia` = anomalias de ponto por dia (ruído).

### Uni_sensor

| Equip | exp2_supressao_transiente | exp3_mask_transiente | rate/dia (exp2_supressao_transiente) | rate/dia (exp3_mask_transiente) | Δrate | suprimidos (exp2) | Veredito |
|---|---|---|---|---|---|---|---|
| `B-0302C` | FRACO | SEM_DADOS | 3.8423543617190994 | — | — | 0 | ⚠️ PIOROU classe |
| `B-24001B` | PARCIAL | SEM_DADOS | 18.04386994648011 | — | — | 0 | ⚠️ PIOROU classe |
| `B-3403C` | FRACO | SEM_DADOS | 2.770435501609116 | — | — | 0 | ⚠️ PIOROU classe |
| `B-402E` | FRACO | SEM_DADOS | 0.7444314185228605 | — | — | 0 | ⚠️ PIOROU classe |
| `B-4064A` | FRACO | SEM_DADOS | 1.7096869942147974 | — | — | 0 | ⚠️ PIOROU classe |
| `B-4703.24001B` | FRACO | SEM_DADOS | 4.780599423458351 | — | — | 0 | ⚠️ PIOROU classe |
| `B-5401A` | FRACO | SEM_DADOS | 1.1639366346702995 | — | — | 0 | ⚠️ PIOROU classe |
| `B-5501B` | FRACO | SEM_DADOS | 1.7490864799025578 | — | — | 0 | ⚠️ PIOROU classe |
| `B-6511502A` | BOM | SEM_DADOS | 3.526060940457957 | — | — | 0 | ⚠️ PIOROU classe |
| `B-8801C` | FRACO | SEM_DADOS | 0.24796294784224862 | — | — | 0 | ⚠️ PIOROU classe |
| `B-8802B` | BOM | SEM_DADOS | 39.42727588390285 | — | — | 0 | ⚠️ PIOROU classe |
| `B-90001A` | PARCIAL | SEM_DADOS | 30.624087591240876 | — | — | 0 | ⚠️ PIOROU classe |

### Mult_sensor

| Equip | exp2_supressao_transiente | exp3_mask_transiente | rate/dia (exp2_supressao_transiente) | rate/dia (exp3_mask_transiente) | Δrate | suprimidos (exp2) | Veredito |
|---|---|---|---|---|---|---|---|
| `B-0302C` | BOM | SEM_DADOS | 34.82443325961355 | — | — | 0 | ⚠️ PIOROU classe |
| `B-24001B` | BOM | SEM_DADOS | 24.041141630786967 | — | — | 0 | ⚠️ PIOROU classe |
| `B-3403C` | BOM | SEM_DADOS | 25.859992055750258 | — | — | 0 | ⚠️ PIOROU classe |
| `B-402E` | BOM | SEM_DADOS | 22.438452520515828 | — | — | 0 | ⚠️ PIOROU classe |
| `B-4064A` | BOM | BOM | 27.306604539675963 | 22.540448815237543 | -4.77 | 0 | ✅ menos ruído, classe igual |
| `B-4703.24001B` | PARCIAL | SEM_DADOS | 26.669197987599855 | — | — | 0 | ⚠️ PIOROU classe |
| `B-5401A` | FRACO | SEM_DADOS | 1.5136640741956477 | — | — | 0 | ⚠️ PIOROU classe |
| `B-5501B` | FRACO | SEM_DADOS | 4.378806333739342 | — | — | 0 | ⚠️ PIOROU classe |
| `B-6511502A` | BOM | SEM_DADOS | 4.060312598103102 | — | — | 0 | ⚠️ PIOROU classe |
| `B-8801C` | BOM | BOM | 11.298030088305271 | 10.600707056579887 | -0.70 | 4 | ✅ menos ruído, classe igual |
| `B-8802B` | BOM | SEM_DADOS | 36.66251353173063 | — | — | 0 | ⚠️ PIOROU classe |
| `B-90001A` | PARCIAL | SEM_DADOS | 33.47810218978102 | — | — | 0 | ⚠️ PIOROU classe |
