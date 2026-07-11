# Pipeline revisada (v3) — o que mudou e por quê

> Registro das mudanças na pipeline CNN1D-AE motivadas pela análise dos resultados e
> pela documentação do time PdM (`DOC/`). Convenção do projeto: **documentar a pipeline
> a cada mudança**, comparando com a versão anterior. Atualizado em 2026-07-11.
>
> Racional completo do diagnóstico: `ESTUDO_E_DECISOES_TRANSPETRO.md`.

---

## Resumo executivo

A rodada anterior tinha **7/12 equipamentos FRACOS** (nenhuma anomalia perto da falha).
Investigando, descobrimos que **o modelo via a falha** (o MAE subia), mas a **máscara
operacional apagava o sinal** — não pelo filtro ligado/desligado, e sim pela marcação de
**transiente**. A v3 corrige a máscara, melhora a visualização e a documentação por modelo.

---

## Mudanças v2 → v3

### 1. Máscara operacional — configuração por equipamento

| Aspecto | v2 (anterior) | v3 (revisada) |
|---|---|---|
| Sensor de status | `OPERATIONAL_REF_SENSOR` genérico (quase sempre `Corrente`) | **melhor sensor por equipamento** (pressão de descarga / vazão / velocidade), validado por preservação do sinal |
| Limiar de "desligado" | `OFF_ABS_THRESHOLD` **fixo = 5.0** | **limiar operacional por equipamento** (≈15% da mediana operacional) |
| Transiente | `TRANSIENT_PADDING_MINUTES = 60`, `TRANSIENT_DIFF_QUANTILE = 0.99` | **`= 0`** e **`= 0.999`** (para de apagar operação real) |
| B-0302C | máscara ativa (mas sensores zerados) | **máscara desativada** (instrumentação zerada, documentado) |

**Por quê:** validação local no B-8801C — a preservação do sinal de vibração alta como `on`
subiu de **78% → 99,2%** com `padding=0, diffq=0.999`. Os pontos de sinal já passavam o
filtro on/off; era a marcação de transiente (em máquinas que ligam/desligam muito) que os
apagava. Fonte do sensor de status: doc do time PdM (bomba parada não gera pressão de descarga).

Config-fonte por equipamento: `analysis/status_sensor_choice.json`.

### 2. Novo produto visual — plot duplo-eixo (bruto + MAE + anomalia)

- **Novo:** `plots.py::plot_signal_mae_anomaly` — um painel com **eixo Y esquerdo = sinal bruto**
  e **eixo Y direito = erro de reconstrução (MAE)**, com o limiar tracejado, a **área acima do
  limiar realçada**, **anomalias** como pontos sobre o bruto, **falha** (linha vermelha),
  **alarmes** e **estado operacional** (faixas). Aceita janelas → gera **visão geral** e
  **zoom(s) na falha** no mesmo formato.
- Arquivos gerados por modelo: `figs/signal_mae_anomaly.png` e `figs/signal_mae_anomaly_zoom.png`
  (no grupo, um par por sensor, usando o MAE do canal correspondente).
- **Por quê:** ligar visualmente *bruto ↔ MAE ↔ anomalia* num só gráfico torna óbvio se a
  anomalia acompanha a subida do erro perto da falha.

### 3. Model card e índice por experimento (introduzidos nesta linha de trabalho)

- `model_card.py::write_model_card` → `MODEL_CARD.md` por modelo (dados, pré-processamento,
  janela, hiperparâmetros, limiar, máscara, avaliação, figuras).
- `pipeline.py::_write_models_index` → `MODELS_INDEX.md` consolidado por execução.

### 4. Gráfico de zoom na falha e curva de loss apresentável

- `plot_series_failure_zoom` (série ±`FAILURE_ZOOM_DAYS` na falha) e `plot_loss` reformulado
  (melhor época marcada, treino/validação, grid).
- Novo campo `PipelineConfig.FAILURE_ZOOM_DAYS` (default 10).

### 5. Organização dos resultados

- **v2:** `runs_transpetro/<eq>_*` misturados.
- **v3:** `resultados/Uni_sensor/<eq>` e `resultados/Mult_sensor/<eq>` — experimentos
  separados por modo, cada um autocontido (figs, csv, `MODEL_CARD.md`).

---

## O que NÃO mudou (intencionalmente)

- Arquitetura do autoencoder, tuner (KerasTuner RandomSearch) e loop de treino — idênticos.
  As mudanças de máscara agem **só no pós-processamento** (scoring), não no treino.
- `THRESH_MODE=target_rate` (1%) e a regra de ponto `k_of_window` (60/5) — mantidos nesta
  rodada para isolar o efeito da máscara. (Candidatos a ajuste na próxima rodada.)

---

## Backlog (herdado da auditoria do time PdM — `DOC/`)

1. **FP honesto em held-out** (o in-sample é otimista).
2. **Resample preservando picos** (`max`/`rms`), não `.last()`.
3. **CUSUM/debounce** como política de alarme para rampas.
4. **Threshold por regime** de carga (equipamentos multi-regime, ex. B-4064A).
5. **Sem espectro/FFT** (limite físico) → falhas de banda estreita (B-4703, B-0302C) seguem
   difíceis; nenhuma mudança de máscara resolve isso.

---

## Histórico de versões

| Versão | Data | Mudança principal |
|---|---|---|
| v1 | — | pipeline por-sensor + gráficos (loss, geral, zoom) + model cards |
| v2 | 2026-07-10 | rodada por-sensor e multivariada nos 12 equipamentos (ClearML) |
| **v3** | **2026-07-11** | máscara por equipamento (sensor+limiar+transiente=0), plot duplo-eixo, organização `resultados/` |
