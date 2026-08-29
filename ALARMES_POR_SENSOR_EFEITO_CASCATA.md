# Alarmes são por sensor, mas falha real dispara vários em cascata

**Pergunta que originou este doc**: quando ocorre um alarme de temperatura,
vibração ou pressão, ele é configurado por sensor individual, ou existe
"um alarme" que cobre vários sensores de uma vez?

**Resposta curta**: é **por sensor**. Cada tag do catálogo
(`alarmes_selecionados_turbina_a.csv`, coluna `Tag Alarme`) tem seu próprio
limite configurado independentemente no sistema de controle — ex.:
`TAHH_6240305` é um limite de temperatura ALTA-ALTA num ponto de medição
específico; `PALL_6240309` é um limite de pressão BAIXA-BAIXA em outro
ponto. Não existe "um alarme" compartilhado entre sensores.

**Mas** quando ocorre uma falha física real (um trip, um desligamento),
ela afeta fisicamente dezenas de grandezas ao mesmo tempo, porque todas
estão medindo o mesmo equipamento no mesmo instante — a turbina para,
temperatura cai em vários pontos, pressão de gás/óleo cai, vibração muda
de padrão, tudo dentro de segundos a minutos um do outro. Cada grandeza
cruza **seu próprio limite individual** de forma independente, mas como
todas reagem à mesma causa raiz, o resultado observado no catálogo é uma
"cascata" de dezenas de tags disparando quase juntos.

## Evidência empírica

Script: `scripts/analise_alarmes/check_coocorrencia_alarmes.py`. Agrupa os
3757 eventos `ACT` do catálogo (47 tags distintos) em "episódios" por
proximidade temporal (gap entre eventos consecutivos > `WINDOW_MIN` = novo
episódio) e conta quantos tags **distintos** aparecem em cada episódio.

| Janela de agrupamento | Episódios com ≥2 tags distintos | Episódios com ≥3 tags | Máximo de tags num episódio |
|---|---|---|---|
| 30 min | 460/1204 (38,2%) | 181 (15,0%) | 46 |
| 60 min | 452/1099 (41,1%) | 186 (16,9%) | 46 |
| 180 min | 413/881 (46,9%) | 199 (22,6%) | 46 |
| 24h | 248/370 (67,0%) | 152 (41,1%) | 46 |

O maior episódio encontrado é **2024-06-11, 14:28–14:31** (3 minutos):
**46 dos 47 tags do catálogo** disparam juntos — praticamente o painel de
alarmes inteiro, cobrindo temperatura (`TAH_*`, `TAHH_*`, `TC382_*`),
pressão (`PAL_*`, `PALL_*`, `PDAH_*`, `PDIT_*`, `PI_*`) e os 10 canais de
vibração (`TV_35*`), tudo num único evento físico.

Outros episódios grandes (2024-03-15: 40 tags; 2024-07-09: 39 tags;
2024-08-21/22: 38-39 tags cada) mostram que esse não é um caso isolado —
é o comportamento normal do sistema quando há um evento físico
significativo.

## Por que isso importa pra este projeto

1. **`hit_rate` por tag infla artificialmente.** Se um modelo detecta um
   evento uma vez, ele "acerta" simultaneamente para os 40+ tags daquele
   episódio — a mesma detecção contada dezenas de vezes, misturando
   decisões redundantes com genuinamente independentes. É por isso que
   as pipelines deste projeto passaram a reportar também uma métrica
   **por episódio** (evento físico distinto), não só por tag — ver
   `docs/analise_pca_monitoramento_sistema.md`.
2. **Justifica fisicamente a arquitetura de votação/confirmação** usada
   pelo colega Francisco (branch `feat/pdm-deteccao-4sinais`): exigir que
   2 *famílias* de sinal independentes (ex. temperatura E spread de
   mancal) concordem não é redundante — é forçar que dois **sintomas
   diferentes** da mesma causa apareçam juntos, o que filtra ruído de um
   sinal isolado sem depender de coincidência entre tags que sabemos que
   disparam em cascata de qualquer jeito.
3. **Ajuda a distinguir "causa" de "eco"**: um alarme que dispara só
   porque a máquina já estava parada por outro motivo entra na cascata
   mas não tem precursor nenhum pra prever — é consequência, não sintoma
   antecipável. Essa é a mesma lógica por trás da auditoria
   genuíno-vs-artefato feita no EXP16 (`docs/analise_trip_oleo_lub.md`) e
   do algoritmo de ground-truth curado do Francisco (parada real +
   alarme de nível dentro de uma janela específica, não qualquer alarme
   isolado).

## Como reproduzir

```bash
PYTHONPATH=. python scripts/analise_alarmes/check_coocorrencia_alarmes.py
```
