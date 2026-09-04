# Sondagem do híbrido — o detector de 4 sinais dentro da máquina do Francisco

**04/09/2026 · 960 configurações em quatro braços · resultado negativo e conclusivo**

A proposta era: colocar o nosso canal de vibração como quinto sinal na varredura
AutoML do Francisco e medir se dá para ter os oito eventos com o custo de alarme
da fronteira dele. Rodamos. **Nenhuma peça do nosso detector transfere isoladamente.**

---

## 1. O controle, antes de qualquer conclusão

Um resultado negativo só vale se a montagem estiver provada. O pipeline dele,
rodando sobre a **nossa** grade exportada, reproduz os números publicados dele:

| métrica | nosso run | publicado por ele |
|---|---|---|
| eventos detectados | **6 de 8** | 6 de 8 |
| antecedência média | **18,3 h** | 18,3 h |
| falso positivo | **0,94/mês** | 0,94/mês |
| horas em alarme falso | 22,3/mês | 22,0/mês |
| dias avaliados | **350,5** | 350,5 |
| episódios totais | **20** | 20 |

Cinco de seis batem casa por casa; a diferença de 1,4% nas horas é atribuível à
nossa grade de 2 min vir de mediana sobre 30 s. Ele derivou os **mesmos 8 eventos
físicos a partir de 9 trips**, os mesmos 132 episódios de alarme e as mesmas 15
exclusões de baseline. Régua alinhada, comparação válida.

Configuração do controle: `pca | 2min | baseline 3000h | ewma 2h | p99,9 |
sustain 30min | confirm 2`.

## 2. Primeira sondagem — o canal, sozinho

| braço | detecção máxima (sem teto) | FP nesse ponto |
|---|---|---|
| controle, 4 sinais dele | 6/8 | **0,94** FP/mês |
| 5 sinais, com o nosso `vb` | 6/8 | **4,02** FP/mês |

Mesma cobertura, quatro vezes o custo. O corte por `confirm` diz por quê:

| limiar de `vb` | confirm | dentro do teto | det. máx. no teto |
|---|---|---|---|
| p60 | **2** | **0 de 48** | — |
| p60 | 3 | 16 de 48 | 4/8 |
| p90 | 2 | 5 de 48 | 4/8 |

Com `vb` em p60 e `confirm=2`, **nenhuma das 48 configurações passa no teto**. O
canal fica aceso ~40% do tempo e qualquer segundo sinal fecha o voto. As únicas que
passam são `confirm=3`, estrito demais — caem para 4/8.

**Diagnóstico:** não falta grade de limiar, falta **estrutura de voto**. Ele tem
`confirm = N de todos`, uma contagem. Nós temos `voto ≥ 2 E (sp OU vb)` — contagem
mais um canal obrigatório qualificado.

## 3. Segunda sondagem — o portão

Quatro braços, para não confundir sinal com estrutura:

| braço | sinais | portão | dentro do teto | **det. máx. no teto** |
|---|---|---|---|---|
| **A** — o detector dele hoje | 4 | — | 28/96 | **6/8** |
| B — primeira sondagem | 5 (+`vb`) | — | 69/384 | 4/8 |
| C — o portão sozinho | 4 | `spread` | 32/96 | 4/8 |
| D — o nosso desenho | 5 (+`vb`) | `spread`+`vb` | 77/384 | 4/8 |

**O portão não transfere.** O braço C, que não usa nada nosso além da ideia
estrutural, piora de 6/8 para 4/8. O braço D, réplica do nosso desenho dentro da
máquina dele, também dá 4/8. **A configuração dele sem portão continua sendo a
melhor de todas as 960.**

## 4. A conclusão medida

> O 8/8 não vem do canal `vb` nem do portão de canal obrigatório. Vem de `vb` em
> **referência rolante** operando em p69, **mais** o portão, **mais** o CUSUM,
> **mais** o refratário de 48 h com duração mínima de 120 min — calibrados juntos.
> Retirar qualquer peça, ou transplantá-la para outra máquina de decisão, não
> reproduz o resultado.

Duas peças nossas que **não** foram para a máquina dele, e que são as candidatas
mais fortes para a diferença:

- **CUSUM** — soma acumulada de `(E/limiar − κ)` acima de `H`. Pega desvio fraco e
  persistente que o limiar sozinho nunca cruzaria, e é o que entrega 04/11/2025
  pelo canal de pressão (69% daquele episódio é CUSUM sozinho). A máquina dele não
  tem equivalente: `sustain` exige N amostras **consecutivas acima**, o que é outra
  coisa.
- **Refratário de 48 h + duração mínima de 120 min** — é onde a nossa seletividade
  real mora. O `min_alert` dele filtra por duração, mas não há refratário.

Isso também explica, de forma direta, por que **cinco famílias de modelo empataram
em 6/8** nos times paralelos (PCA, autoencoder, Mahalanobis, OCSVM, isolation
forest): o teto não está no modelo nem no sinal — está na política de decisão.

## 5. O que isto encerra e o que abre

**Encerra:** a proposta "coloca o seu canal como quinto sinal na nossa varredura"
já está medida, e a resposta é não. Não vale uma semana de trabalho para chegar ao
mesmo lugar.

**Abre:** o inverso. Em vez de transplantar peças nossas para a máquina dele,
**portar a máquina de decisão inteira** — CUSUM e refratário incluídos — e então
varrer. Mas isso não é mais "quinto sinal": é reimplementar o detector dentro da
infraestrutura dele, e só vale se o objetivo for a **auditoria**, não o ganho.

---

## Armadilhas técnicas encontradas

Valem para qualquer nova tentativa nesse caminho.

**`--with-vibration` dele não serve para testar o nosso canal.** Injeta as 10 tags
`TV_*` **cruas** como família de PCA (`self.families["vibracao"] = VIBRATION`). É a
construção que falha por deriva de campanha — a que ele mesmo mediu obtendo 5/8 a
2,48 FP/mês. O nosso canal tem de entrar pré-computado, porque a referência rolante
de 400 h **é parte do sinal**, não pré-processamento.

**Sinal externo não pode passar pela normalização mensal dos derivados.** Ela faz
`abs((x − mediana)/MAD)`, o que **dobra a distribuição na mediana** e mapeia amostra
quieta para escore alto — invertido. Daí o `PRE_NORMALIZADOS` no patch: centro 0,
escala 1.

**Com limiar por sinal, o `DetectionReplay` e o avaliador do AutoML divergem** na
contagem de falso positivo (24 contra 21) e a assertiva interna dele dispara. Usar o
avaliador do AutoML — é o caminho que produziu a fronteira publicada.

**A exportação tem de ser tz-naive.** O loader dele faz `pd.to_datetime` sem
`utc=True` e depois fatia com strings ingênuas; índice tz-aware quebra com
`Cannot compare tz-naive and tz-aware`.

## Como reproduzir

Requer um worktree da branch dele e a nossa grade exportada:

```bash
git worktree add /tmp/wt-chico origin/feat/pdm-deteccao-4sinais
cd /tmp/wt-chico && git apply <este_dir>/patch_automl_clearml.diff
# exportar grade2min.parquet como CSV tz-naive de 2025-01-01 em diante,
# com a coluna VB_ROLANTE_Z, em dados_locais/ junto de alarmes_mapeados_colunas.csv
export PYTHONPATH="$PWD/src:$PWD/scripts"
python controle_reproducao.py     # o controle, primeiro
python valida_sinal_externo.py    # confirma ordem e participação do sinal
python sondagem_vb.py             # primeira sondagem  (480 trials, ~5 min)
python sondagem_portao.py         # segunda sondagem   (960 trials, ~7 min)
```

Um replay leva ~10 s; o ajuste mensal é cacheado por (modelo, grade, política,
exclusões) e reaproveitado por centenas de trials.

## Arquivos

| arquivo | o que é |
|---|---|
| `controle_reproducao.py` | reproduz os números publicados dele sobre a nossa grade |
| `valida_sinal_externo.py` | confirma ordem dos sinais e que o externo participa |
| `sondagem_vb.py` | primeira sondagem — o canal sozinho, com braço de controle |
| `sondagem_portao.py` | segunda sondagem — quatro braços, sinal × portão |
| `patch_automl_clearml.diff` | as duas mudanças no script dele, comentadas e prontas para enviar |
