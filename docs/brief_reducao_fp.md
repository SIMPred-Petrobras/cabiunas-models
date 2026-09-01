# Detector de anomalias TC-330.03A — estado atual e o problema a resolver

*Brief para revisão externa. Pedido: agir como engenheiro de IA sênior e propor como reduzir falsos positivos sem depender do silêncio pós-partida. As perguntas específicas estão na última seção.*

## Contexto

Turbocompressor a gás em Cabiúnas. Detecção precoce de falha, não supervisionada, com protocolo temporal estrito (walk-forward, retreino mensal). Teto acordado: 1 falso positivo por mês.

Pedido específico: como reduzir falso positivo? A solução que temos hoje (silenciar o alerta nas 24 h após cada partida) funciona nos números mas é fraca operacionalmente, porque cega o detector justamente quando a máquina volta de reparo.

## Dados

- 1.396.800 amostras a 30 s, 01/01/2025 a 30/04/2026 (16 meses), 39 tags do PI.
- Grade de trabalho: 2 min por mediana, ou 30 s cru.
- 14 temperaturas, 12 pressões/diferenciais, 10 tags de vibração (em quarentena), RUNNING_A.
- A máquina fica parada boa parte do tempo: ~350 dias de operação nos 16 meses.
- Planilha de alarmes 2022–2026, 6.851 ativações.

## Definição de falha

trip = queda de RUNNING_A durando ≥ 2 h e coincidente com alarme de nível na janela [−1 h, +30 min]. Trips a menos de 24 h contam como um evento. Resultado: **8 falhas** (1 selagem, 5 mancal, 2 óleo) de 9 trips.

O piso de 2 h existe porque 65 das 135 transições do RUNNING_A duram menos de 3 min.

| # | início da parada (UTC−3) | duração | mecanismo |
|---|--------------------------|---------|-----------|
| 1 | 27/02/2025 08:38 | 4,4 h | selagem |
| 2 | 17/03/2025 18:16 | 6,2 h | mancal (inclui trip de 18/03 11:16, agrupado) |
| 3 | 07/04/2025 21:18 | 18,3 h | mancal |
| 4 | 11/04/2025 17:04 | 41,4 h | mancal |
| 5 | 29/04/2025 03:04 | 564,2 h | mancal |
| 6 | 04/11/2025 06:22 | 144,1 h | óleo lubrificante |
| 7 | 09/12/2025 08:36 | 7,8 h | mancal |
| 8 | 26/02/2026 15:34 | 5,7 h | óleo lubrificante |

## Detector

Quatro sinais, retreino mensal com janela de 3.000 h de operação elegível:

1. **temperatura** — erro de reconstrução de PCA (95% da variância) sobre 14 sensores
2. **pressao_oleo** — idem sobre 12 sensores
3. **mancal_spread** — TI_0305 menos a mediana dos 3 mancais irmãos, z robusto
4. **selagem_z** — PDIT_0305 isolado, z robusto

Decisão: EWMA → limiar = percentil do score do próprio baseline daquele mês → persistência → alerta se ≥ 2 dos 4 sinais concordam. Episódios separados por menos de 2 h são o mesmo.

Limpezas já aplicadas ao baseline: 2 h após cada partida, ±1 h ao redor de cada uma das 3.757 ativações de alarme, veto de instrumento congelado por família.

## Resultados atuais

Cinco varreduras completas no ClearML, 10.368 configurações cada (51.840 no total). Melhor ponto de cada variante em **6 de 8 falhas, dentro do teto**:

| variante | lead | FP/mês | líquido | h/mês em alarme falso | configs aprovadas | dias sob vigilância |
|---|---|---|---|---|---|---|
| sem filtro | 11,8 h | 0,95 | 0,60 | 18,1 | 7 | 348,6 |
| silêncio 12 h | 11,8 h | 0,93 | 0,74 | 13,2 | 6 | 322,6 |
| silêncio 24 h | 18,3 h | 0,87 | 0,48 | 13,8 | 1 | 311,4 |
| filtro de exaustão T5 | 7,3 h | 0,96 | 0,52 | 27,7 | 3 | 343,9 |
| T5 + silêncio | — nenhuma dentro do teto | | | | 0 | |

Aceitando **5 de 8**:

| variante | lead | FP/mês | h/mês |
|---|---|---|---|
| sem filtro | 9,0 h | 0,52 | 8,7 |
| silêncio 12 h | 9,0 h | 0,56 | 3,3 |
| silêncio 24 h | 8,5 h | 0,58 | 2,0 |

Nas 51.840 configurações avaliadas, **nenhuma chega a 8/8**. O máximo é 7/8, e só a 1,13 FP/mês (com T5, todas Mahalanobis) ou 5,51 (sem). A falha de 04/11/2025 não é antecipada por nenhuma.

## Anatomia dos 11 falsos positivos (configuração de produção)

257,2 h no total. Dois episódios respondem por 80% das horas (19/08 com 104,5 h e 27/10 com 101,8 h). Dos 11:

- 4 começam nas primeiras 12 h após uma partida (transiente de religar)
- 4 coincidem com alarme de processo ativo — de 19 a 25/08 há 4 a 5 ativações de suprimento de gás (PI_0319 "Falha PI Gás Motor p/Partida" e PI_0315 "Pressão Baixa Gás Comb.") na janela de ±12 h de cada episódio
- 4 estão dentro do aglomerado de falhas de mancal de março–abril, quando a máquina falhou 3 vezes em 6 semanas
- 1 (o de 27/10) precede a falha de 04/11 e provavelmente não é falso positivo (ver "problemas na régua")

Contraprova de que o detector não segue alarme cegamente: setembro/2025 teve 21 ativações de alarme (todas de gás, nenhuma de nível) e o detector ficou calado o mês inteiro.

## O mecanismo que trava qualquer limpeza

Observado três vezes, sempre igual. O limiar é um percentil do score do próprio baseline. Quando se removem amostras contaminadas do baseline, a cauda se concentra e o percentil desaba: ao tirar 155 h (1,8% do tempo), o limiar de temperatura caiu para 3% do valor anterior em 12/2025 e o de pressão para 0,3% em 04/2025. O detector fica muito mais sensível e, no mesmo percentil, isso vira falso positivo.

Foi o que derrubou:

- filtro de exaustão T5 (piora de 6/8 para 5/8 na vizinhança do ponto atual)
- ampliar o descarte de partida de 2 h para 24 h no baseline (4/8, FP 2,41)
- combinar T5 com silêncio (nenhuma configuração aprovada em 6/8)

Trocar percentil por "k-ésimo maior score do baseline" não recupera.

## Outras coisas testadas que não funcionaram

| tentativa | resultado |
|---|---|
| resíduo de carga (regressão dos sensores em T5, PI_0307, PI_0308) | lead sobe de 13,8 para 22,8 h, mas FP vai de 1,46 para 3,00/mês |
| vibração como nível (10 tags TV_*) | percentil 79–85 de 315 janelas de controle nas 48 h antes das falhas; não separa |
| vibração residual como 5º sinal | perde uma falha, FP 3,17/mês |
| canais lentos (EWMA de 12, 24 e 72 h) | cai a 2/8, até 175 h/mês em alarme |
| autoencoder no lugar do PCA | empata: mesmas 6 falhas, leads diferindo menos de 1,5 h |

Varredura exaustiva de precursor (36 sensores × 3 horizontes × cru e normalizado por carga, contra 182–241 janelas de controle): a falha de 04/11 tem precursor visível só em 7 dias (resíduo de TI_0305 no percentil 98,6), e a de 09/12 não tem nada (máximo percentil 87).

## Problemas conhecidos na própria régua de avaliação

Isto pode ser onde está o maior ganho, porque parte do "falso positivo" pode ser artefato de classificação:

**1.** A janela de crédito é de 48 h de calendário, num equipamento que fica parado metade do tempo. O episódio de 27/10 termina 78 h de calendário antes da falha de 04/11, mas apenas 13,3 h de operação antes. Hoje é falso positivo. Medimos: contando a janela em horas de operação e casando pelo fim do alerta, o detector vai de 6/8 para 7/8 com FP 0,82/mês, sem mudar nada no modelo.

**2.** A janela de perdão (parada real depois do alerta) é ancorada no início do episódio. Para episódios de 100 h a janela termina dentro do próprio alerta. Testadas três ancoragens na configuração de produção:

| ancoragem | FP | FP/mês |
|---|---|---|
| [início, início+48h] (atual) | 11 | 0,90 |
| [fim, fim+48h] | 12 | 0,99 |
| [início, fim+48h] | 10 | 0,82 |

Recalculando tudo com a terceira regra: produção vai de 0,94 para 0,82 FP/mês e de 22,0 para 12,8 h/mês; o ponto de PCA em 30 s sem filtro vai de 0,52 para 0,12 FP/mês e de 8,7 para 2,1 h/mês.

**3.** Duas métricas de FP discordam. Contando episódios, um ponto ganha; contando horas em alarme, outro ganha. Um alarme travado 100 h é um episódio só. Ainda não decidimos qual é a métrica oficial.

**4.** O filtro de "detector vivo" exige silêncio máximo de 2.000 h de operação, mas o mínimo alcançável entre as configurações aprovadas é 1.578 h e o percentil 1 é 2.194 h. Ele reprova a configuração campeã, que alarma em 6 dos 6 trimestres.

## A objeção operacional ao silêncio

O silêncio de 24 h suprime o alerta (não o score) nas 24 h após cada partida, e desconta essas horas do denominador de FP/mês. Ganho medido: 22,0 → 13,8 h/mês mantendo 6/8.

Problemas:

- cega 11% do tempo de operação, e justamente após religar, que é quando a máquina volta de reparo;
- reduz as configurações viáveis em 6/8 de 7 para 1 — o resultado passa a depender de um único ponto do grid;
- em 36 h de silêncio perde-se a detecção de 29/04/2025, que veio 28,7 h após uma partida, o que mostra que o risco é real.

O silêncio de 12 h é menos agressivo (7,5% do tempo cego, 6 configurações viáveis, 13,2 h/mês) e hoje é a nossa preferência, mas continua sendo o mesmo tipo de solução.

## Limitações da amostra

- 8 falhas em 16 meses. A diferença entre configurações costuma ser de 1 evento.
- A janela de 3.000 h trunca e entrega ~2.400 h de média, porque a série começa em jan/2025. Faltam os dados de 2024 (existem localmente, 12 parquets mensais; 2024 tem 210 ativações de alarme de nível em 49 dias contra 104 em 2025).
- Vibração só existe agregada a 30 s, sem conteúdo espectral.
- Uma única máquina, sem frota para comparar.

## O que eu gostaria que você respondesse

1. **Como reduzir falso positivo sem cegar o detector?** O transiente de partida é responsável por 4 dos 11 FP. Existe forma de tratá-lo como condição de operação (feature, regime, normalização) em vez de janela cega de tempo?
2. **Como quebrar o acoplamento entre limpeza de baseline e limiar?** Toda limpeza que tentamos derruba o percentil e piora tudo. Existe formulação de limiar que seja robusta a mudança na distribuição do baseline (limiar absoluto calibrado, conformal prediction, controle de taxa de excedência)?
3. Os 4 FP que coincidem com alarme de suprimento de gás deveriam ser tratados como **entrada do modelo** (contexto de processo) em vez de desconto a posteriori?
4. Qual métrica de falso positivo adotar como oficial — **episódios por mês ou horas em alarme por mês** — dado que elas ordenam as soluções de forma diferente?
5. A correção da régua (janela em horas de operação, perdão ancorado no fim do episódio) tem algum viés que estamos deixando passar? Ela nos dá 7/8 com FP menor sem tocar no modelo, o que parece bom demais.
6. Com 8 eventos, **que protocolo de validação** sustenta escolher entre configurações que diferem por um evento?
