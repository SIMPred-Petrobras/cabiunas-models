# Análise do detector do Diego

Scripts que rodam **na branch dele**, não nesta. Produziram os achados sobre o
canal 4 documentados em `../CONTEXTO_05-06_SET_2026.md` §4.

## Como rodar

```bash
git worktree add /tmp/wt-diego origin/feat/exp32_alinhamento_equipe
cp *.py /tmp/wt-diego/
cd /tmp/wt-diego
export CLEARML_CONFIG_FILE=<repo>/clearml.conf
export PYTHONPATH="$PWD"

python testa_nossa_decisao.py   # controle + 128 configs da nossa camada nos canais dele
python quem_vota.py             # quais canais votam em cada uma das 8 detecções
python tags_na_janela.py        # o que sustenta o canal 4 em cada detecção
python enriquecimento_tags.py   # controle negativo das 5 tags do canal 4
```

O primeiro baixa ~300 MB de artefatos do ClearML (os três `point_anomalies_all.csv`
das tasks `805fbf34` / `7815d2cf` / `18a61687`) e cacheia em `canais/merged.parquet`.

A lista de falhas aponta para o nosso `falhas.csv` porque
`dataset_francisco_lara/alarmes_francisco_falhas.csv` é dado e não está na branch —
são os mesmos 8 eventos da Tabela 7 do relatório dele, e o script confere a contagem.

## O que mediram

- **controle**: reproduz o publicado dele exato — 8/8, 2,88 FP/mês, 25 inconclusivos
- a nossa camada de decisão **não transfere**: de 128 configurações só a dele faz 8/8
- o canal 4 fica aceso **47,6%** do tempo e é **essencial em 5 das 8 detecções**
- enriquecimento das 5 tags antes dos trips: **1,45× (p = 0,051)**, não significativo
- sem as 3 tags de utilidade/gás: **8/8 · 2,88 → 3/8 · 1,51**
