# Bundles de produção — detector CNN-1D per-sensor (Cabiunas, base 2025)

Artefatos de inferência por sensor: `{sensor}_inference_bundle.json`. Cada bundle é
auto-contido e carrega tudo para reproduzir o scoring em dados novos **sem refitar**.

- **Modelo-fonte:** ClearML task `58bc393c1d7a4e42815236e8897abc88`
  (config `configs/calibracao_v9_cnn_variants/v9_prod_sensores_2025.json`, dataset 2025
  `424e5b58…`, 17 sensores). Os pesos de cada sensor estão no artefato `{sensor}_model_keras`.
- **Filtro de operação:** NGP_A > 50 (`RUNNING_COL`/`RUNNING_THRESHOLD`).

## Ponto de operação por sensor (métrica de produção, gap-based)

| Sensor | half_life(h) | thr_q | ewma_abs_threshold | recall@op | FA/dia@op |
|---|---|---|---|---|---|
| T5_AVG_A | 4.0 | 0.500 | 0.01266 | 88.2% | 0.086 |
| TC382_03_A | 4.0 | 0.515 | 0.01404 | 90.5% | 0.066 |
| TC382_04_A | 0.5 | 0.510 | 0.02627 | 77.8% | 0.079 |
| TC382_01/02/05/06_A | 4.0 | 0.500 | (ver bundle) | N/A* | 0.09–0.13 |
| TV_351…355 (X/Y) | 4.0 | 0.500 | (ver bundle) | N/A* | 0.07–0.10 |

\* **N/A** = 0 incidentes não-LOLO em 2025 → recall não medível; só a FA é observável
(o `recall_at_op=0.0` no bundle desses sensores significa "sem incidente", não recall zero).
**TC382_04 com half_life curto (0.5h)** porque seus eventos (UNDER) são dips breves que a
EWMA de 4h alisaria. Ver memória `fp-rootcause-exclusao-cargaalta`.

## Inferência em dados novos

```python
from tensorflow import keras
from src.cnn1d_ae.inference import load_bundle, score_production

bundle = load_bundle("production_bundles/TC382_03_A_inference_bundle.json")
model  = keras.models.load_model("<TC382_03_A_model_keras local copy>")
scores = score_production(model, bundle, df_novo)   # df indexado por tempo, com a coluna do sensor + NGP_A
# scores: mae_seq, health_ewma, operational_state, alert (1 = anomalia confirmada)
```

`score_production` aplica: `(x − center)/scale` (estatísticas de treino) → erro de
reconstrução → EWMA(half_life) → `>= ewma_abs_threshold` → máscara NGP → debounce.

## Notas importantes
- **Não clipar dados de scoring/inferência** nos bounds de treino: cortaria anomalias
  fora-de-faixa (UNDER/drift). `transform_features` usa `clip=False` por padrão; o
  `clip_bounds` no bundle só documenta o scaler.
- `ewma_abs_threshold` = `quantil(ewm(mae, half_life), thr_q)` na janela de calibração —
  conversão do `thr_q` (quantil de rank do eval) para um limiar **absoluto** streaming-safe.
  É uma reprodução **aproximada** do ponto de operação do eval (a EWMA online + máscara OFF
  diferem do rank global), suficiente para produção; os incidentes são detectados.
- **Variância de treino:** GPU/cuDNN+KerasTuner são não-determinísticos; o recall per-sensor
  varia ~±10pt entre re-treinos (ex: TC382_04 oscilou 78–89%). Para robustez, considerar
  best-of-N por sensor ou ensemble num passo futuro.
