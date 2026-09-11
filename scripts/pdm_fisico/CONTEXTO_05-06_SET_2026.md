# Contexto — sessões de 04 a 06/09/2026

Documento de passagem. Cobre o que foi medido, o que mudou de entendimento e o que
está pendente. Complementa o `README.md`, que descreve o detector; aqui está o que
**mudou desde que ele foi escrito**, incluindo uma correção importante nele.

---

## 1. O achado principal: as duas réguas não são iguais

**O `README.md` está errado onde diz que a régua é a mesma da equipe paralela.** As
*constantes* batem (janela 48 h, fusão de episódio 2 h, denominador em tempo
monitorado, categoria "inconclusivo" = parada real). A **regra de associação**
evento↔episódio não:

| | pergunta que responde |
|---|---|
| **nossa** | o alarme estava **ativo** em algum instante de [t−48 h, t]? |
| **deles** | o **início** do episódio caiu dentro de [t−48 h, t]? |

No código deles: `inside = [ep for ep in episodes if window_start <= ep <= evento]`.

**Consequência: o nosso 8/8 vira 4/8 pela régua deles.** Divergem os quatro eventos
de lead acima de 48 h — 27/02 (143,9 h), 17/03 (194,9 h), 29/04 (51,8 h) e 26/02
(670 h).

**A régua é inerte para eles e cara só para nós.** O detector do Francisco dá
6/8 · 18,3 h · 0,94 FP/mês · 22,3 h/mês **idêntico nas duas**, e o melhor de 384
configurações também não se move: **0 de 8** episódios dele atravessam a borda das
48 h (mediana de episódio 6,43 h). Os leads do Diego são todos < 48 h, mesma coisa.
Só nós produzimos episódios de semanas.

**Decisão tomada com o usuário: a régua de início é a tecnicamente correta** e foi
adotada como referência. Motivo: uma métrica de detecção mede a *transição* de normal
para alarmante, porque é a transição que dispara ação. EEMUA 191 e ISA-18.2 tratam
alarme permanente (*standing alarm*) como defeito do sistema, não como informação.

**Mas ela tem dois defeitos próprios**, medidos:
- **não tem piso** — uma detecção a 1,3 h do trip conta igual a uma de 43 h;
- **um τ_max único para modos de falha com relógios diferentes** — o 17/03 tem quatro
  canais subindo por 8 dias, o melhor precursor do conjunto, e é marcado como erro.

Ver `_tmp_regra_associacao.py`, `regra_inicio_varredura.py`, `banda_acionabilidade.py`.

---

## 2. A banda de acionabilidade — o diagnóstico que reformula o problema

`banda_acionabilidade.py`. Distribuição dos leads de início:

- **nós** : 1,3 · 2,8 · 5,0 · 8,8 · 20,2 · 23,6 · 51,8 · 194,9 h (mediana 14,5)
- **Diego**: 3,8 · 8,4 · 13,7 · 19,8 · 31,2 · 33,8 · 36,7 · 43,2 h (mediana 25,5)

Os leads dele estão **todos dentro de uma faixa usável**. Os nossos se espalham por
duas ordens de grandeza. **Em toda banda com τ_min ≥ 4 h ele ganha em detecção.**

> **O nosso defeito não é falso positivo — ganhamos 4× nisso. É que o instante do
> alarme é errático.** Erro nas duas pontas, contra erro no meio.

E os dois eventos que perdemos **não são detecções ausentes, são detecções cedo
demais**: o 17/03 tem precursor 194,9 h antes e nada novo nas 48 h finais (o pico de
t a 8,38× acontece no último sample, 2 min antes do trip — é a falha, não precursor);
o 29/04 tem precursor 51,8 h antes e **perde a janela por 3,8 h**.

---

## 3. Estado dos pontos de operação

### Publicado (o que está no `publica_clearml.py`)
`8/8 nossa régua · 4/8 régua de início · banda 3/8 · 0,517 FP/mês · 7,1 h/mês · lead 10,1 h`

### Melhor medido nesta sessão — **in-sample, NÃO adotado**
`8/8 · 6/8 início · banda 5/8 · 0,517 FP/mês · 7,8 h/mês · lead 19,7 h`

Configuração: gatilho de dois níveis com limiar livre por canal no nível sensível.
- **nível A (sensível)**: ≥3 de 4 canais, `k_lo = t:1,1 · p:0,7 · sp:0,9 · vb:1,8`
- **nível B (específico)**: ≥2 de 4 canais + portão `sp|vb`, `k = 1,7 / 2,2`
- alarme = A **ou** B
- religamento `frac = 0,03`, escalada `idade = 96 h · ABS = 20 · piso 60 min`,
  refratário 72 h, duração mínima 120 min

**+2 na banda acionável e quase o dobro do lead, ao mesmo custo do publicado.**
181 células chegam a banda 5/8. `dois_niveis_por_canal.py`.

**PENDENTE**: confirmar a vizinhança do ponto (variar um canal por vez) e a tabela
por evento — o script `confirma_ponto_final.py` foi escrito mas a execução foi
interrompida.

### Ressalvas honestas
- Sete parâmetros ajustados em oito rótulos. Os mecanismos têm justificativa física
  independente; os números estão ajustados.
- **Sem confirmação fora da amostra.** Nas 54 paradas reais que nunca entraram no
  ajuste, o ponto novo e o publicado são **idênticos** (2/54 nascem na janela nos
  dois, lead médio 22,6 h nos dois). Ressalva justa: só 2 das 54 são detectadas por
  qualquer configuração, então o teste tem poder quase nulo. Não refuta, não confirma.

---

## 4. O que descobrimos sobre o detector do Diego

Branch `origin/feat/exp32_alinhamento_equipe` (05/09/2026). Worktree em
`/home/thallys/Documents/projeto-petrobras/wt-diego`. O relatório dele é
`relatorio_pipeline_operacional.pdf` na raiz do repo.

**O resultado dele**: 3 canais OCSVM (temperatura, vibração, óleo) + 1 canal
estatístico (proximidade a alarme de processo), voto ≥2 de 4, filtro de duração
45 min, refratário 48 h → **8/8 · 2,88 FP/mês · lead 23,8 h**.

**Rodamos o pipeline dele sobre os artefatos dele. O controle reproduz exato: 8/8,
2,88 FP/mês, 25 inconclusivos.** (`wt-diego/testa_nossa_decisao.py`)

### 4.1 A nossa camada de decisão NÃO transfere
128 configurações (4 portões × 2 votos × 4 durações × 4 refratários). **Só uma faz
8/8: a dele.** Portão de mancal → 7/8; portão de vibração → 6/8; voto≥3 → **0/8**.

Segunda confirmação independente — a primeira foram 960 configurações na máquina do
Francisco (`sondagem_hibrido/`).

### 4.2 O canal 4 dele carrega o resultado
`quem_vota.py`, `tags_na_janela.py`, `enriquecimento_tags.py`:

- O canal de alarme fica aceso **47,62% do tempo em operação**
- Dois canais **modelados** dele acendem juntos em apenas **1,2% do tempo**
- O canal 4 é **essencial em 5 das 8 detecções**
- Portão "≥2 canais modelados" → **0/8**
- As 5 tags somam 1.535 ativações = **40,9% de todo o catálogo** de 47 tags
- Três das cinco são de **utilidade/gás** (duas do motor de partida, uma delas
  *falha de instrumento*). Removendo essas três: ciclo cai de 47,6% para 8,1% e o
  resultado vai de **8/8 · 2,88 para 3/8 · 1,51**
- **Enriquecimento das tags antes dos trips: 1,45× (p = 0,051)** — não significativo.
  Para comparação, o Passo 3 do relatório dele mede 6,9× e chama de forte,
  corretamente.

> Leitura: o detector dele é, na prática, **três detectores OCSVM de canal único em
> OU, desbastados pela metade por um canal quase aleatório.** Não é consenso de
> quatro canais — e é por isso que o nosso portão não tem onde operar.

**Sendo justo:** nada disso invalida a medida dele sob a régua dele, e a **calibração
por canal dele é genuinamente boa** — é dela que vem o perfil de lead, que continua
melhor que o nosso.

### 4.3 O Passo 2 da auditoria dele não tem controle negativo
`controle_negativo_zscore.py`. O critério "pico |z| ≥ 3 em ≥2 sensores" dá
**95,6–96,5% em janelas aleatórias** da mesma máquina, contra os 90,5% dele
(enriquecimento 0,94×). A mediana de uma janela sorteada já tem 6 de 14 sensores
acima de |z| = 3. A recomendação dele de "não suprimir mais" se apoia nisso.

---

## 5. Ideias testadas e REFUTADAS nesta sessão

Não repetir sem informação nova.

| ideia | por que caiu |
|---|---|
| **CUSUM com esquecimento** (`cusum_vazante.py`) | ataca a causa certa dos episódios longos, mas dá 5/8 a **62,5 h/mês**. O CUSUM é o que pega deriva fraca — 69% do episódio de 04/11 é CUSUM sozinho |
| **Ordem de acendimento dos canais** (`ordem_canais.py`) | sem separação: `p` primeiro sai em 4 TP, 5 NEUTRO e 6 FP. Os atrasos são quase todos +0,0 h — os canais acendem juntos |
| **Portão de temperatura sub-limiar** (`portao_temperatura.py`) | melhor feature isolada (TP 0,98 vs FP 0,31, AUC 0,78) mas **reprovado no LOEO**: corta 2 de 8 FP e custa 1 detecção. O limiar é fixado pelo mínimo dos TP — instabilidade de estatística de mínimo |
| **Troca limiar↔voto isolada** (`troca_limiar_voto.py`) | não passa de 4/8 na banda. Só funciona como união com o nível específico |
| **Limiar livre por canal, sozinho** (`limiar_por_canal.py`) | banda 5/8 a 0,947 FP/mês — pior que o dois níveis. Precisa da estrutura de dois regimes |
| **Teto de permanência sob régua de início** (`teto_sob_regra_inicio.py`) | `det_ini` cravado em 4/8 em 300 configs. O rearme exige a condição **cair**, e episódio de 670 h é condição contínua |
| **Histerese** | direção errada — histerese *prolonga* alarme. FP 6→11, horas 7,1→18,2 |
| **Corte por decaimento SEM religamento** | parecia funcionar mas era **defeito de implementação**: marcava "morto" e nunca ressuscitava. Ver §6 |
| **Nossa camada de decisão nos canais do Diego** | 128 configs, só a dele faz 8/8 |
| **Reanúncio periódico de alarme permanente** | creditaria *todo* alarme de pé e derrotaria o propósito da régua de início. Não testado por ser gaming |

---

## 6. Erros meus nesta sessão, para não se repetirem

**Cheguei a afirmar que a detecção de 26/02/2026 era artefato de "alarme travado" e
que o número honesto seria 7/8. ERRADO, e retratado.** O corte que usei para "provar"
marcava o episódio como morto e nunca religava enquanto o voto seguisse de pé — como
o CUSUM mantinha o voto, a subida real de 24/02 ficava silenciada.

A anatomia real (`anatomia_travado.py`): o episódio de 670 h são **dois eventos
genuínos separados por três semanas** — p a 35× em 29/01, depois 576 h (86%) sem nada
acima de 1,5×, depois p a 2,75× nas 48 h antes do trip. O defeito é **fusão de
episódio**, não detecção falsa. **O 8/8 está de pé.**

**Também afirmei 5/8 no LOEO com o corte aplicado.** Não se sustenta: o número passeia
4/8 → 5/8 → 7/8 conforme o orçamento de FP. Era ponto de uma curva instável.

**Regra geral que sai daí:** qualquer corte por decaimento PRECISA de religamento; e
nunca citar número de LOEO sem declarar a regra de desempate e o orçamento.

---

## 7. Arquivos criados nesta sessão

Todos em `scripts/pdm_fisico/`, exceto onde indicado.

**Sobre a régua e o diagnóstico**
| arquivo | o que faz |
|---|---|
| `_tmp_regra_associacao.py` | mede os 8 alvos sob as duas regras de associação |
| `regra_inicio_varredura.py` | `avalia_inicio()` + varredura de 1.440 pontos sob a régua estrita |
| `banda_acionabilidade.py` | detecção por banda [τ_min, τ_max], nós contra Diego |
| `anatomia_travado.py` | dissecação do episódio de 670 h em fatias de 2 dias |
| `episodios_longos_tendencia.py` | regressão do escore dentro de cada episódio longo |
| `diagnostico_tres_faltantes.py` | o que os 3 eventos que falham fazem nas 48 h finais |

**Mecanismos novos**
| arquivo | o que faz |
|---|---|
| `corte_com_rearme.py` | `corta_rearma()` — fecha episódio no decaimento, **reabre na subida** |
| `escalada_por_idade.py` | `quebra_idade()` + `pos_idade()` — reanúncio por magnitude condicionado à idade |
| `checa_degenerado.py` | `pos_dur_esc()` — piso de duração no episódio escalonado |
| `dois_niveis.py` | gatilho de dois níveis (sensível-corroborado OU específico) |
| `dois_niveis_por_canal.py` | **o ponto atual** — dois níveis com limiar livre por canal |
| `combina_final.py` | `constroi()` e `mede()` reutilizáveis |
| `corte_vetorizado.py` | versão vetorizada do corte, verificada bit a bit |

**Diagnóstico de FP e validação**
| arquivo | o que faz |
|---|---|
| `fp_producao_diagnostico.py` | FP por combinação de canal + corroboração por catálogo |
| `features_episodios.py` | tabela de features por episódio, TP × NEUTRO × FP |
| `valida_fora_amostra.py` | as 54 paradas reais como alvo independente |
| `controle_negativo_zscore.py` | o nulo do Passo 2 do Diego |
| `taxa_base_regua.py` | nulo por permutação da nossa regra de associação |

**No worktree `wt-diego/`**
| arquivo | o que faz |
|---|---|
| `testa_nossa_decisao.py` | controle + 128 configs da nossa camada nos canais dele |
| `quem_vota.py` | quais canais votam em cada uma das 8 detecções dele |
| `tags_na_janela.py` | o que sustenta o canal 4 em cada detecção |
| `enriquecimento_tags.py` | controle negativo das 5 tags |
| `canais/*.csv` | os 3 `point_anomalies_all.csv` baixados do ClearML (~300 MB, não versionar) |

---

## 8. Pendências

1. **Confirmar a vizinhança do ponto novo** — `confirma_ponto_final.py` está escrito,
   a execução foi interrompida. Sem isso, não adotar.
2. **Mandar o pedido ao Diego** — texto pronto na conversa; três testes: ablação do
   canal 4, controle negativo do Passo 2, LOEO no filtro de 45 min.
3. **Decidir se adota o ponto novo** no `publica_clearml.py`. Recomendação atual:
   manter o publicado como validado e apresentar o novo como alternativa
   declaradamente in-sample.
4. **Commitar os scripts** desta sessão — só código, sem CSV/PNG/PDF/npz/parquet
   (restrição permanente do usuário).
5. **Fechar a branch `exp/sondagem-hibrido`** ou abrir uma nova para este trabalho.

## 9. Recomendação de fundo

A comparação entre os times deve usar a **banda de acionabilidade**, não a contagem
crua a 48 h. Nela: **nós 5/8 a 0,517 FP/mês, Diego 7/8 a 2,88.** Ele detecta mais,
nós custamos ~5,6× menos — mas 5 das 8 detecções dele dependem de um canal com
enriquecimento 1,45×.

E o limite real não é mais parâmetro: são **oito rótulos**. Já medimos que provar
1,03 → 0,75 FP/mês exigiria 92 meses de operação.
