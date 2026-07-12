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

## experimento_2_supressao_transiente (em andamento)

- **O que muda:** `SUPPRESS_SHORT_TRANSIENT_EPISODES=True` (novo, default),
  implementado em `scoring.py::suppress_short_transient_episodes` — suprime
  episódios `transiente_curto` do `is_anom_point` antes de salvar
  `point_anomalies_all.csv` e dos plots. Mesma máscara/config do experimento_1
  fora isso (comparação isolada desse único efeito).
- **Configs:** os mesmos `*_persensor*.json` / `*_mult_v3.json`, com
  `OUTPUT_ROOT` já apontando para `resultados/experimento_2_supressao_transiente/`.
- **O que NÃO mudou ainda (fica para o experimento_3):** threshold por regime
  (`mudanca_regime`, maior bucket), filtro de glitch de sensor no dado bruto,
  revisão manual de `precursor_parada`/`sustentado_sem_causa`.
- **Como comparar com o experimento_1:** mesma métrica de sempre (placar
  BOM/PARCIAL/FRACO de `scripts/analyze_failure_detection.py`) + contagem de
  `suppressed_transient_episodes` no `calibration_report.json` de cada modelo +
  `anomaly_rate_points_per_day` antes/depois (deve cair sem derrubar o hit_rate).

## Convenção para os próximos

1. Nomeie a pasta `experimento_N_<apelido-curto-da-mudança>`.
2. Nunca sobrescreva um experimento anterior — sempre uma pasta nova.
3. Aponte o `OUTPUT_ROOT` dos configs para dentro da pasta do experimento antes
   de enfileirar/rodar.
4. Registre aqui: o que mudou, o que NÃO mudou (para isolar o efeito), e como
   comparar com o experimento anterior.
5. Atualize `ESTUDO_E_DECISOES_TRANSPETRO.md` e `PIPELINE_REVISADA.md` com a
   conclusão assim que o experimento terminar.
