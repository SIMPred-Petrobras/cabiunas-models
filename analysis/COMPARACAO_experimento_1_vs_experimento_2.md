# Comparação: exp1_mascara_v3 vs exp2_supressao_transiente

- **exp1_mascara_v3:** `resultados/experimento_1_mascara_v3`
- **exp2_supressao_transiente:** `resultados/experimento_2_supressao_transiente`

Classe: BOM (detectou em ±48h da falha) · PARCIAL (só precursor ≤10d) · FRACO (nada em ±10d). `rate/dia` = anomalias de ponto por dia (ruído).

> ⚠️ **Ressalva metodológica:** cada experimento RE-TREINA o modelo do zero (tuner roda
> de novo). A comparação mistura o efeito da mudança de código com ruído de
> retreinamento (mesmo com `RANDOM_SEED` fixo, TF/tuner não são 100% determinísticos).
> Um único equipamento mudando de classe não é conclusivo por si só — ver caso B-4064A
> abaixo, investigado e confirmado como ruído de retreinamento, não efeito da supressão.
> Confiar mais no padrão agregado (maioria com `rate/dia` menor, classe estável) do que
> em qualquer célula isolada.

## Achado: B-4064A (Uni_sensor) PARCIAL → FRACO — investigado, não é a supressão

Único caso que mudou de classe. Investigação: os 2 episódios suprimidos no exp2
(10/mai e 16/ago/2024) **não têm relação com a janela da falha** (23-26/ago/2024).
Os 204 pontos anômalos que existiam nesse período no exp1 simplesmente não aparecem
no exp2 — o MAE do modelo re-treinado é mais baixo ali (mesmo com threshold do exp2
menor: 0,0108 vs 0,0125 do exp1). Causa: variância de retreinamento (novo resultado
do tuner), não a lógica de supressão de `transiente_curto` (que nem tocou nesse
período). Fica como alerta para o próximo experimento: se quisermos isolar de fato o
efeito de uma mudança de pós-processamento, considerar reaproveitar os pesos já
treinados em vez de re-treinar do zero.

### Uni_sensor

| Equip | exp1_mascara_v3 | exp2_supressao_transiente | rate/dia (exp1_mascara_v3) | rate/dia (exp2_supressao_transiente) | Δrate | suprimidos (exp2) | Veredito |
|---|---|---|---|---|---|---|---|
| `B-0302C` | FRACO | FRACO | 4.689203857619675 | 3.8423543617190994 | -0.85 | 0 | ✅ menos ruído, classe igual |
| `B-24001B` | PARCIAL | PARCIAL | 19.684968959225568 | 18.04386994648011 | -1.64 | 0 | ✅ menos ruído, classe igual |
| `B-3403C` | FRACO | FRACO | 5.6887313249333245 | 2.770435501609116 | -2.92 | 80 | ✅ menos ruído, classe igual |
| `B-402E` | FRACO | FRACO | 0.9753810082063306 | 0.7444314185228605 | -0.23 | 0 | ✅ menos ruído, classe igual |
| `B-4064A` | PARCIAL | FRACO | 2.927435749528167 | 1.7096869942147974 | -1.22 | 2 | ⚠️ PIOROU classe |
| `B-4703.24001B` | FRACO | FRACO | 6.280606917500586 | 4.780599423458351 | -1.50 | 7 | ✅ menos ruído, classe igual |
| `B-5401A` | FRACO | FRACO | 1.1639366346702995 | 1.1639366346702995 | +0.00 | 0 | = igual |
| `B-5501B` | FRACO | FRACO | 1.5980511571254568 | 1.7490864799025578 | +0.15 | 1 | ⚠️ mais ruído, classe igual |
| `B-6511502A` | BOM | BOM | 3.2630447397711166 | 3.526060940457957 | +0.26 | 0 | ⚠️ mais ruído, classe igual |
| `B-8801C` | FRACO | FRACO | 0.27124585374292925 | 0.24796294784224862 | -0.02 | 0 | ✅ menos ruído, classe igual |
| `B-8802B` | BOM | BOM | 54.01580914642864 | 39.42727588390285 | -14.59 | 4 | ✅ menos ruído, classe igual |
| `B-90001A` | PARCIAL | PARCIAL | 34.56204379562044 | 30.624087591240876 | -3.94 | 18 | ✅ menos ruído, classe igual |

### Mult_sensor

| Equip | exp1_mascara_v3 | exp2_supressao_transiente | rate/dia (exp1_mascara_v3) | rate/dia (exp2_supressao_transiente) | Δrate | suprimidos (exp2) | Veredito |
|---|---|---|---|---|---|---|---|
| `B-0302C` | BOM | BOM | 33.83343916866607 | 34.82443325961355 | +0.99 | 0 | ⚠️ mais ruído, classe igual |
| `B-24001B` | BOM | BOM | 23.306893658473474 | 24.041141630786967 | +0.73 | 0 | ⚠️ mais ruído, classe igual |
| `B-3403C` | BOM | BOM | 24.560377649096544 | 25.859992055750258 | +1.30 | 2 | ⚠️ mais ruído, classe igual |
| `B-402E` | BOM | BOM | 25.305978898007034 | 22.438452520515828 | -2.87 | 7 | ✅ menos ruído, classe igual |
| `B-4064A` | BOM | BOM | 19.04849377516675 | 27.306604539675963 | +8.26 | 0 | ⚠️ mais ruído, classe igual |
| `B-4703.24001B` | PARCIAL | PARCIAL | 26.831069299906574 | 26.669197987599855 | -0.16 | 3 | ✅ menos ruído, classe igual |
| `B-5401A` | FRACO | FRACO | 1.5136640741956477 | 1.5136640741956477 | +0.00 | 0 | = igual |
| `B-5501B` | FRACO | FRACO | 4.292326431181486 | 4.378806333739342 | +0.09 | 1 | ⚠️ mais ruído, classe igual |
| `B-6511502A` | BOM | BOM | 3.9014069768548025 | 4.060312598103102 | +0.16 | 0 | ⚠️ mais ruído, classe igual |
| `B-8801C` | BOM | BOM | 10.016306118472803 | 11.298030088305271 | +1.28 | 3 | ⚠️ mais ruído, classe igual |
| `B-8802B` | BOM | BOM | 58.82473089728139 | 36.66251353173063 | -22.16 | 0 | ✅ menos ruído, classe igual |
| `B-90001A` | PARCIAL | PARCIAL | 33.92335766423358 | 33.47810218978102 | -0.45 | 4 | ✅ menos ruído, classe igual |
