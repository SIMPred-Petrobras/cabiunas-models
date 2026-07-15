# Índice de experimentos — Transpetro

Cada rodada completa (12 equipamentos, Uni_sensor + Mult_sensor) que muda a
pipeline de forma relevante vira uma pasta `experimento_N_<apelido>/` própria,
preservada intacta — nunca sobrescrita — para permitir comparar sempre com o
resultado anterior e ver o que de fato melhorou.

## experimento_1_mascara_v3 (2026-07-11/12)

- **O que mudou:** máscara operacional calibrada por equipamento (sensor de
  status + limiar), `TRANSIENT_PADDING_MINUTES=0`/`TRANSIENT_DIFF_QUANTILE=0.999`,
  plots `TARGET_`/`contexto/` no multivariado.
- **Conteúdo:** `Uni_sensor/`, `Mult_sensor/` (12 equip cada) + `MODELS_INDEX.md`;
  `baseline_v2_mascara_antiga/` = resultados multivariados da máscara ANTERIOR
  (fixa, `OFF_ABS_THRESHOLD=5.0`), mantidos só para referência comparativa.
- **Análises que usam este experimento:** `analysis/COMPARACAO_persensor_vs_multivar.md`,
  `analysis/ANALISE_MASCARA_OPERACIONAL.md`, `analysis/EPISODE_SIGNATURE_STUDY.md`,
  `analysis/EPISODE_TRIAGE.md`.
- **Conclusão que motivou o próximo experimento:** o balde de falsos positivos
  "far" não é ruído homogêneo — é uma mistura de glitch de sensor, mudança de
  regime operacional, precursor de parada e sustentado sem causa aparente.
  Só o bucket `transiente_curto` (curto, fraco, sem nenhum desses padrões) é
  seguro de suprimir (~1% de sobreposição com episódios perto da falha real).
  Ver `ESTUDO_E_DECISOES_TRANSPETRO.md` §7.

## experimento_2_supressao_transiente (concluído 2026-07-14)

- **O que mudou:** `SUPPRESS_SHORT_TRANSIENT_EPISODES=True` (novo, default),
  implementado em `scoring.py::suppress_short_transient_episodes` — suprime
  episódios `transiente_curto` do `is_anom_point` antes de salvar
  `point_anomalies_all.csv` e dos plots. Mesma máscara/config do experimento_1
  fora isso.
- **Configs:** os mesmos `*_persensor*.json` / `*_mult_v3.json`, com
  `OUTPUT_ROOT` apontando para `resultados/experimento_2_supressao_transiente/`.
- **24 tasks** (12 uni + 12 mult) rodadas no ClearML, coletadas com
  `scripts/collect_persensor_results.py --task-ids analysis/experimento_2_uni_task_ids.json`
  e `scripts/collect_multivar_results.py --task-ids analysis/experimento_2_mult_task_ids.json`.
- **Comparação:** `scripts/compare_experiments.py` → `analysis/COMPARACAO_experimento_1_vs_experimento_2.md`.

### Resultado (24 comparações: 12 Uni_sensor + 12 Mult_sensor)

| Veredito | n |
|---|---|
| ✅ menos ruído, classe de detecção igual | 12 (50%) |
| ⚠️ mais ruído, classe igual | 9 (37,5%) |
| = igual (sem mudança) | 2 |
| ⚠️ classe piorou | 1 — **investigado, não foi a supressão** (ver abaixo) |
| ✅ classe melhorou | 0 |

**Conclusão:** a supressão reduziu ruído (`anomaly_rate_points_per_day`) na metade dos
casos, sem derrubar nenhuma classificação de detecção real por causa da supressão em si.
O único caso que mudou de classe (B-4064A, Uni_sensor: PARCIAL→FRACO) foi investigado a
fundo — os episódios suprimidos não têm relação com a janela da falha; a mudança veio de
**variância do retreinamento** (novo resultado do tuner), não da lógica de supressão.

**Ressalva metodológica importante para os próximos experimentos:** cada experimento
re-treina os modelos do zero, então a comparação mistura o efeito da mudança de código
com ruído de retreinamento. Considerar, se possível, reaproveitar pesos já treinados ao
isolar mudanças futuras de pós-processamento (que não afetam o treino em si).

- **O que NÃO mudou ainda:** threshold por regime (`mudanca_regime`, maior bucket),
  filtro de glitch de sensor no dado bruto, revisão manual de
  `precursor_parada`/`sustentado_sem_causa`.

## experimento_3_mask_transiente (preparado, aguardando decisão de rodar)

- **O que muda:** `MASK_ALLOW_TRANSIENTE=True` — máscara aceita `transiente` como
  válido (não só `on`). Motivado por: gráficos perto da falha mostrando MAE sustentado
  sendo apagado pela máscara em rampas de liga/desliga (ver
  `ESTUDO_E_DECISOES_TRANSPETRO.md` §7.4-7.6).
- **Testado por simulação em 6 equipamentos** (`scripts/simulate_allow_transiente.py`,
  `scripts/simulate_majority_mask.py`, `scripts/simulate_smoothed_mask.py`) — só
  **B-4064A (mult)** e **B-8801C (mult)** recuperam detecção sem explodir ruído; os
  outros 4 (B-3403C, B-4064A uni, B-402E, B-24001B uni) têm sensor de referência
  cronicamente instável e pioram com qualquer afrouxamento — **não ativado neles**.
- **Escopo:** só 2 configs mudaram (`B-4064A_mult_v3.json`, `B-8801C_mult_v3.json`),
  `OUTPUT_ROOT` apontando para `resultados/experimento_3_mask_transiente/Mult_sensor/`.
  Os outros 22 modelos ficam idênticos ao experimento_2 (não precisam re-rodar).
- **Como comparar:** `scripts/compare_experiments.py` entre experimento_2 e
  experimento_3, só nesses 2 equipamentos (mult) — esperado: hit_rate igual ou melhor,
  rate/dia com aumento pequeno (<5%, validado por simulação).

## Convenção para os próximos

1. Nomeie a pasta `experimento_N_<apelido-curto-da-mudança>`.
2. Nunca sobrescreva um experimento anterior — sempre uma pasta nova.
3. Aponte o `OUTPUT_ROOT` dos configs para dentro da pasta do experimento antes
   de enfileirar/rodar.
4. Registre aqui: o que mudou, o que NÃO mudou (para isolar o efeito), e como
   comparar com o experimento anterior.
5. Atualize `ESTUDO_E_DECISOES_TRANSPETRO.md` e `PIPELINE_REVISADA.md` com a
   conclusão assim que o experimento terminar.
