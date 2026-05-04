# cabiunas-models

Pipeline CNN-1D Autoencoder para deteccao de anomalias em series temporais de Cabiunas, integrado ao ClearML para versionamento de dataset, tracking de tasks e execucao remota via ClearML Agent.

## ClearML

Valores padrao do fluxo:

- Projeto: `TesteMLCab`
- Dataset: `Cabiunas 2025`
- Dataset ID: `e2765c3eef2349cda5f5cbcb0fcd5a40`
- Queue remota: `default`

O arquivo `clearml.conf` deve existir localmente na raiz do projeto, mas fica fora do Git.

## Preparar Ambiente Local

```bash
cd /home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabiunas-models
source ../venv/bin/activate
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
```

## Criar/Atualizar Dataset

O dataset ja foi criado no ClearML. Para criar uma nova versao a partir de um config:

```bash
scripts/create_clearml_dataset_from_config.sh configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_T5_AVG_A.json "TesteMLCab" "Cabiunas 2025"
```

## Execucao Remota

Garanta que um worker existente esteja escutando a fila `default`.

```bash
clearml-agent daemon --queue default
```

Em outro terminal, submeta a task:

```bash
cd /home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabiunas-models
source ../venv/bin/activate
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
PYTHONPATH=. ../venv/bin/python src/main.py --config configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_T5_AVG_A.json
```

Com `RUN_REMOTE=true`, a maquina local apenas enfileira a task. O worker remoto baixa o codigo do Git, baixa o dataset ClearML e executa o treino.

## Execucao Local

Para rodar localmente, use um config com `RUN_REMOTE=false` ou remova essa chave temporariamente:

```bash
PYTHONPATH=. ../venv/bin/python src/main.py --config configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_T5_AVG_A.json
```

## Validacao

```bash
python3 -m compileall src/main.py src/cnn1d_ae
python3 -m json.tool configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_T5_AVG_A.json >/tmp/cabiunas_config_check.json
```
