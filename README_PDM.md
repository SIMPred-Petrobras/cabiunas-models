# Detector de anomalias do TC-330.03A — 4 sinais (`src/cabiunas_pdm`)

Contribuição paralela ao pipeline `cnn1d_ae` deste repositório. Mesmo equipamento e
mesmo dataset ClearML, abordagem diferente: em vez de um autoencoder convolucional
único, **quatro sinais em paralelo com votação** (dois autoencoders densos
multivariados + dois z-scores robustos univariados), busca em grade sobre 31.104
configurações e avaliação walk-forward causal.

O código ainda **não segue a organização do repositório** (`configs/*.json` +
`src/main.py --config`). Esta branch é o ponto de partida para essa adaptação; ver
"Pendências" no fim.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[pdm]"          # extras do detector: scikit-learn, scipy, pyarrow, joblib
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
```

Requer Python 3.12+. Não usa TensorFlow — os modelos são todos scikit-learn.

Variáveis de ambiente reconhecidas (ambas opcionais):

| Variável | Para quê |
|---|---|
| `CABIUNAS_PDM_ROOT` | raiz do checkout, se não for a deduzida do próprio arquivo |
| `CABIUNAS_RAW` | pasta dos arquivos brutos do PI — só os módulos de ingestão local usam |

## Dados

Vem tudo do ClearML, nada de arquivo local: dataset `68b25f9db0b8471a90b8100800d26e9a`
(série de 30 s, 01/01/2025 a 30/04/2026, 39 colunas). O primeiro uso baixa o CSV pelo
`Dataset.get()` e grava um cache Parquet em `data/interim/` — que fica fora do Git.

## Rodar a busca

```bash
python scripts/automl_clearml.py --mode quick                        # 16 configs, local, sanidade
python scripts/automl_clearml.py --mode full --remote --queue default # 31.104 configs, ~6 h no worker
```

O script é **autocontido de propósito**: o worker do ClearML executa só esse arquivo,
sem depender do pacote instalado. Publica dois artefatos na task — `automl_results`
(a grade inteira) e `best_trial`.

`--mode` aceita `quick` (16), `balanced` (1.536), `fp_first` (2.304) e `full` (31.104).

## Notebooks

`src/cabiunas_pdm/replay.py` reexecuta uma configuração da busca devolvendo as
**séries** (scores, limites, alertas), não só as métricas — é o que os notebooks usam.
Ele importa `scripts/automl_clearml.py` por caminho, para o desenho refletir exatamente
o mesmo cálculo que a busca avaliou.

| Notebook | Pergunta que responde |
|---|---|
| `01_serie_e_falhas` | como é a série e de onde saem as 8 falhas |
| `02_resultados_automl` | qual ponto de operação escolher na grade |
| `03_anomalias_na_serie` | o detector em funcionamento, com zoom em cada falha |
| `04_comparacao` | 4 sinais em votação × modelo único |
| `06_serie_completa_anomalias` | visão dos 16 meses inteiros em duas figuras |

O notebook 02 lê o artefato `automl_results` direto da task; os outros recalculam a
partir do dataset. Os resultados estão salvos nas saídas das células — dá para ler sem
executar nada.

## O que não está no Git

- **dados** — série, planilha de alarmes e caches (`data/`), tudo via ClearML
- **grade de resultados** (`automl_out/`) — está nos artefatos da task
- **relatórios e figuras** (`reports/`) — o resultado se lê nos notebooks

## Pendências para seguir o padrão do repositório

1. mover a configuração do detector para `configs/*.json`, como o `cnn1d_ae`
2. expor a busca por `src/main.py --config` em vez do script solto
3. cobrir com testes em `tests/` o critério de evento e o cálculo de falso positivo
4. `src/cabiunas_pdm/` ainda carrega módulos da fase de ingestão local
   (`sources.py`, `dataset.py`, `detector.py`, `clearml_io.py`, `operability.py`),
   anteriores à migração para ClearML — revisar o que sobrevive
