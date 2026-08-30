# Análise de Experimentos — AutoML EXP10 (redução de falso alerta)

Parte do candidato de referência EXP7 item 1+2 (`ocsvm`, p99,9/debounce=1,
multiescala + textura — 92,5% hit_rate, 1,94% de falso alerta) e investiga
a estrutura do falso alerta em vez de tratá-lo como ruído agregado.
Relatório completo com gráficos:
`task_plots_exp10_reducao_fp/relatorio_exp10_reducao_fp.pdf`.

## Método

Fragmentação da série `is_anom_point` em episódios contínuos (gap ≤5 min),
cruzados com dados brutos (temperatura, vibração) e com o catálogo
completo de 47 tags de alarme — não só os 2 sensores avaliados. Achado
estrutural: 72,5% dos 12.782 pontos de falso alerta (OOS) se concentram
em apenas 10 dos 295 episódios.

## Achado 1 — desligamento real mal rotulado (65,3% do FP)

Os 5 maiores episódios (19–24/08/2025) coincidem com `TC382_03_A` caindo
de ~478°C para ~28–32°C por 3,5 dias, enquanto `RUNNING_A` (referência da
máscara operacional) permanece ~0,96–1,0 o tempo todo — a máscara nunca
excluía esse período. Confirmado como desligamento real (não artefato de
dado) via `PI_6240319_AL`/`PAL_6240315` no catálogo completo, disparando
exatamente nas bordas do período.

**Correção:** `build_operational_state` (`scoring.py`) ganhou
`secondary_series`/`secondary_off_abs_threshold` — o próprio sensor-alvo
do grupo caindo abaixo de um piso físico (`OFF_TARGET_ABS_THRESHOLD`,
150°C) também conta como "off", unido por OR ao critério de
`OPERATIONAL_REF_SENSOR`, antes da classificação off_curto/off_longo/
transiente já existente.

**Resultado (EXP10):** hit_rate idêntico (92,5%, 37/40); FP 1,94% → 0,67%.
Seed-sweep: desvio-padrão de 0,019pp.

## Caminho testado e descartado — debounce

Simulação offline da grade de debounce (1 a 60) mostrou ganho de FP
pequeno até debounce=10 (1,94%→1,66%, já custando 3 preditivos) e perda
de alarmes reais inteiros além disso (32→26→20→17 de 32). Causa: a
duração mediana dos episódios de FP residual (6 pontos/2,5min) se
sobrepõe à dos episódios mais curtos que ainda precedem um alarme real
(25º percentil = 6min). Debounce não separa os dois.

## Achado 2 — rampas de carga reais (25,0% do FP)

Dos 4.441 pontos de FP residual (287 episódios), variabilidade local de
`TC382_03_A` é 5–9x maior que em pontos normais (nível idêntico — não é
viés de faixa). Investigação individual dos 10 maiores episódios: 8 de
10 mostram `TC382_03_A`/`T5_AVG_A` variando dezenas de graus em ~1h, com
desvio-padrão de vibração 3–6x mais alto — manobra de carga real, sem
alarme, sem falha.

**Correção:** `ENABLE_LOAD_GATE`/`apply_load_gate` já existia para o
CNN1D-AE (`pipeline.py`) mas nunca fora portado ao AutoML. Portado para
o loop principal de trials e para `_seed_sweep` em `automl_pipeline.py`.
Parâmetros default do CNN1D-AE (halflife=120min/janela=360min) custavam
até 6 casos preditivos — uma rampa de falha real e uma rampa de carga
legítima têm a mesma assinatura de taxa de variação numa janela longa.
Janela curta (halflife=15min/janela=30min) + `ramp_max=100°C/h`,
encontrados por simulação offline, preservam 29/29 preditivos.

**Resultado (EXP10b):** hit_rate idêntico (92,5%, 37/40); FP 0,67% →
0,48%. Seed-sweep: desvio-padrão de 0,016pp.

## Achado 3 — volatilidade que persiste, não que sobe

O portão de rampa reage bem a variação de *nível* em `T5_AVG_A`, mas o
padrão de vibração observado no Achado 2 é diferente: o desvio-padrão
sobe e **permanece** elevado por 1–2h, não é só um pico na subida.

**Tentativa 1 (falhou):** alimentar `apply_load_gate` com um índice de
volatilidade de vibração (desvio-padrão móvel médio dos 10 canais) no
lugar do proxy de temperatura, mantendo o mecanismo de taxa de variação.
O ponto de custo zero (`thr=8`) reduzia o FP de 0,480% para apenas
0,477% — taxa de variação não distingue o início de uma rampa real do
início de uma escalada de falha, o mesmo problema do portão de rampa
original, agora por uma causa relacionada mas distinta (métrica errada,
não janela errada).

**Tentativa 2 (funcionou):** `compute_volatility_index`/
`apply_volatility_gate` (`scoring.py`) — desvio-padrão móvel causal
(`VOLATILITY_GATE_WINDOW_MINUTES=60min`) de cada um dos 10 canais de
vibração, reduzido pela média entre canais, bloqueando quando o **nível**
(não a taxa) ultrapassa `VOLATILITY_GATE_THRESHOLD`. Simulação offline
encontrou `threshold=0,39` como ponto de custo zero: preserva 29/29
preditivos com FP caindo 27% relativo (0,480%→0,348%).

**Resultado (EXP10c):** hit_rate idêntico (92,5%, 37/40); FP 0,48% →
0,35%. Seed-sweep: desvio-padrão de 0,011pp.

## Resultado consolidado

| Etapa | hit_rate | FP |
|---|---|---|
| EXP7 item1+2 (base) | 92,5% (37/40) | 1,94% |
| EXP10 (+ máscara operacional) | 92,5% (37/40) | 0,67% |
| EXP10b (+ portão de rampa) | 92,5% (37/40) | 0,48% |
| **EXP10c (+ portão de volatilidade)** | **92,5% (37/40)** | **0,35%** |

**Candidato de referência atualizado:** `ocsvm` (p99,9/debounce=1) sobre
multiescala + textura + máscara operacional corrigida
(`OFF_TARGET_ABS_THRESHOLD=150`) + portão de rampa
(`ENABLE_LOAD_GATE`, `LOAD_GATE_SENSOR=T5_AVG_A`, `LOAD_GATE_RAMP_MAX=100`,
`LOAD_GATE_RAMP_HALFLIFE_MINUTES=15`, `LOAD_GATE_WINDOW_MINUTES=30`) +
portão de volatilidade (`ENABLE_VOLATILITY_GATE`,
`VOLATILITY_GATE_SENSORS`=10 canais de vibração,
`VOLATILITY_GATE_WINDOW_MINUTES=60`, `VOLATILITY_GATE_THRESHOLD=0.39`).
Nenhuma das três correções toca no modelo de anomalia — todas atuam na
camada de pós-processamento/avaliação (rotulagem operacional e contexto
de manobra), com custo de detecção zero confirmado em produção (task
remota) nas três vezes, batendo a simulação offline com <0,01pp de
diferença. Redução total: 1,94%→0,35% (~82% relativo).

## Pendências (não endereçadas aqui)

- ~9,7% do FP original coincide com outro alarme real do catálogo
  completo (pressão/partida a gás) — fora do escopo de avaliação de 2
  sensores, não é falso alerta genuíno, mas não foi corrigido.
- Resíduo de falha de sensor pontual (~2,1%) que contamina a feature de
  tendência de 24h — nenhuma das três correções mira esse mecanismo.

## Tasks ClearML

- EXP10 (máscara operacional): `5fc24eb564284436912dd189fddf747d`
- EXP10b (+ portão de rampa): `24b3e27a4241412f99beed4e029554b4`
- EXP10c (+ portão de volatilidade): `6ac3b1b52a45433a83568d61fafadda6`

## Validação por evento físico (2026-08-29) — o 37/40 é real ou é cascata?

Motivação: `docs/analise_pca_monitoramento_sistema.md` fez LOEO na
pipeline do Francisco e achou otimismo real (66,7% do grid virou 62,5%
no LOEO honesto); `ALARMES_POR_SENSOR_EFEITO_CASCATA.md` mostrou que um
único evento físico dispara dezenas de tags quase juntos. As duas coisas
levantam a mesma dúvida sobre o 92,5% (37/40) do EXP10c: é 37 eventos
físicos distintos detectados, ou é um número menor de eventos inflado
por múltiplas linhas de alarme do mesmo evento?

**Método:** reconstrução local dos 40 alarmes OOS (`Tag` ∈
{`TC382_03_A`, `T5_AVG_A`}, status `ACT`, `Data da Ocorrencia >=
2025-07-01`) a partir do dataset ClearML, cruzados com
`point_anomalies_all.csv` da task EXP10c (mesma lógica de
`eval_alarm_hit_rate`, janela ±1440min). Os 40 alarmes foram agrupados
em episódios por proximidade temporal (gap > 1440min = novo episódio) —
a mesma janela usada para decidir "detectado", então dois alarmes do
mesmo episódio têm janelas de avaliação quase idênticas.

**Resultado: não é artefato de cascata.** 40 alarmes → 16 episódios
físicos distintos. Dentro de cada episódio, os alarmes são **uniformemente
todos detectados ou todos perdidos** (nenhum episódio tem detecção
parcial) — ou seja, o 37/40 ao nível de linha de alarme corresponde
exatamente a **14/16 (87,5%) ao nível de evento físico**, sem inflação:
os 3 alarmes perdidos (40-37) são exatamente os alarmes dos 2 episódios
perdidos (2 alarmes do episódio de 2025-08-08 + 1 do episódio de
2025-11-29).

**Os 2 episódios perdidos são falha de instrumento, não precursor real.**
Ambos são um único tag disparando alarme `UNDER` com valor fisicamente
impossível para o sinal (T5_AVG_A/TC382_03_A a **-21°C** em
2025-08-08 15:31 e **-18°C** em 2025-11-29 07:52 — mesmo padrão de leitura
negativa fora de faixa já investigado e descartado como precursor em
`docs/analise_pca_monitoramento_sistema.md` v10, "veto+faixa física").
Não é um evento perdido pelo modelo por fraqueza de generalização — é
um tipo de evento estruturalmente diferente do que o modelo foi
desenhado para capturar (glitch pontual de um sensor, provavelmente já
removido pelo clipping de outlier `OUTLIER_Q_LOW/HIGH=0.001/0.999` antes
de chegar no score, e não uma deriva coordenada de múltiplos sensores
precedendo um evento mecânico real).

**O que isso NÃO responde — risco de otimismo ainda em aberto.** Este
teste usa a config já congelada do EXP10c (modelo, threshold p99,9 e os
3 portões com seus parâmetros fixos: `LOAD_GATE_RAMP_MAX=100`,
`VOLATILITY_GATE_THRESHOLD=0.39`, `OFF_TARGET_ABS_THRESHOLD=150`) — ele
mostra que o resultado final generaliza bem entre os 16 eventos, mas
**não** re-deriva esses 3 parâmetros excluindo cada evento por vez. Os
parâmetros dos portões foram escolhidos por "simulação offline" olhando
o efeito em todos os eventos disponíveis de uma vez (achado 2 e achado
3 acima, critério "preserva 29/29 preditivos") — a mesma classe de viés
de seleção que motivou o LOEO no pipeline do Francisco, só que aqui não
há uma grade de trials já computada para reaproveitar (o AutoML deste
grupo roda com grade unitária, `AUTOML_THRESHOLD_PERCENTILES=[99.9]`,
`AUTOML_DEBOUNCE_GRID=[1]`) nem o score bruto pré-portão foi salvo como
artefato (só `point_anomalies_all.csv`, já pós-portão). Fazer o LOEO
completo desse viés exigiria re-simular a escolha dos 3 limiares de
portão excluindo um evento por vez — reprocessamento novo (o modelo
`ocsvm` em si é barato, mas precisa recarregar os dados brutos e
recalcular os índices de rampa/volatilidade por fold), não uma leitura
de artefato existente como foi possível aqui.

**Conclusão prática:** o 92,5%/0,35% do EXP10c resiste ao teste de
inflação por cascata (14/16 eventos físicos genuinamente distintos,
não 37 linhas de um punhado de eventos) e as 2 falhas residuais têm
explicação física específica, não são sinal de fragilidade geral. O
risco de otimismo nos 3 limiares de portão continua sem verificação
formal — fica registrado aqui como pendência explícita, a fazer sob
demanda se o ganho justificar o reprocessamento.

## LOEO completo dos limiares de portão (2026-08-29)

Script: `scripts/loeo_exp10c_portoes.py`. Task ClearML:
`e33a4e804707489dbe791fd437a49332`. Fecha a pendência da seção anterior.

**Escopo:** `LOAD_GATE_RAMP_MAX` (100°C/h) e `VOLATILITY_GATE_THRESHOLD`
(0,39) — os 2 limiares com critério de seleção "custo zero" documentado
(escolhidos varrendo uma grade contra os eventos disponíveis,
preservando os "preditivos" e minimizando FP). `OFF_TARGET_ABS_THRESHOLD`
(150°C) ficou de fora — veio de um piso físico observado num
desligamento real específico, não de uma varredura contra os eventos de
avaliação.

**Método:** o modelo (`ocsvm`) e o pré-processamento foram reproduzidos
localmente **uma única vez** (não usam rótulo de alarme, então não há
necessidade de re-treinar por fold). As séries contínuas dos 2 portões
(rampa, volatilidade) também foram calculadas uma única vez, com as
janelas fixas do config — só o limiar de corte final é re-selecionado
por fold. **Sanity check:** a reprodução local bateu com a task remota
original: hit_rate 0,9250 (37/40), normal_alert_rate 0,00348 (referência:
0,00350) — confirma que a reprodução é fiel antes de confiar no resto.

Dos 16 episódios físicos OOS, 14 são "predizíveis" (o modelo sem nenhum
portão já os alcança) — os 2 não-predizíveis são exatamente os glitches
de instrumento já identificados na seção anterior (-21°C/-18°C), fora do
escopo de qualquer ajuste de portão.

Para cada um dos 14 episódios predizíveis: excluí-lo da visão, buscar
numa grade (22 valores de `ramp_max` x 49 de `vol_threshold`, 1078
combinações) o combo que preserva os outros 13 com o **menor**
`normal_alert_rate` (mesmo critério "custo zero" documentado, sem ver o
episódio em questão) e checar se esse combo ainda detecta o episódio
retido.

**Resultado: existe otimismo real, na mesma direção (e ordem de
grandeza) do que já tínhamos visto na pipeline do Francisco.**

| | valor |
|---|---|
| Fixo, vendo tudo (config atual: ramp=100/vol=0,39) | 14/14 (100%) |
| Melhor combo da grade, vendo tudo (ramp=60/vol=0,21) | 14/14 (100%), FP=0,237% |
| **LOEO honesto (re-seleciona por fold)** | **12/14 (85,7%)** |

Os 2 episódios perdidos no LOEO:

- **2026-03-13 10:33–10:38** (2 alarmes) — combo escolhido sem vê-lo:
  ramp=50/vol=0,19 (mais agressivo que o necessário para os outros 13).
- **2026-03-25 10:45** (1 alarme único) — combo escolhido: ramp=30/vol=0,23
  (ainda mais agressivo).

Os dois têm em comum ser episódios com poucos alarmes (1–2), no limite
inferior da distribuição — exatamente o tipo de evento mais sensível a
qual conjunto de "outros eventos" está definindo a fronteira de
custo-zero em cada fold.

**Achado colateral (fora do LOEO em si):** a grade encontrou um combo
(ramp=60/vol=0,21) que **preserva os 14/14 com FP menor** que o
configurado em produção (0,237% vs 0,348%) — os limiares atuais
(100/0,39) não estão na fronteira ótima da própria grade cheia. Vale
como candidato de ajuste independente do resultado do LOEO.

**Leitura honesta para o EXP10c:** o "92,5%/0,35%" (ou "87,5% por
evento", 14/16, da seção anterior) é o número de melhor-caso, calibrado
vendo todos os eventos de uma vez. A estimativa fora-da-amostra mais
honesta para a parcela dos 3 portões cai para **12/14 (85,7%) dos
episódios predizíveis** — ou **12/16 (75,0%)** se incluirmos os 2
glitches estruturalmente fora de alcance. Não invalida o EXP10c (a queda
é de ~15pp, não um colapso, e concentrada em episódios de 1-2 alarmes só,
o tipo de evento mais marginal da amostra) — mas o número a comunicar daqui
pra frente é essa faixa (75-86%), não o 92,5% otimista, sempre que o
assunto for expectativa de detecção em produção, não o placar do
backtest.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/loeo_exp10c_portoes.py
```

## Diagnóstico de deriva temporal — vale a pena retreino mensal? (2026-08-29)

MOTIVAÇÃO: próximo item do "menu" de melhorias do EXP10c era avaliar
retreino periódico (walk-forward mensal) em vez do corte único atual
(modelo treinado uma vez em dados anteriores a `AUTOML_OOS_SPLIT_DATE`
2025-07-01, aplicado sem re-treino nos 10 meses seguintes). Antes de
construir a infraestrutura (retreinar, re-selecionar limiar/portões a
cada mês, redefinir a régua de avaliação OOS), vale checar se há
evidência de que o modelo congelado está degradando ao longo desses 10
meses — dois diagnósticos baratos, reaproveitando dados já existentes,
sem retreinar nada.

**Diagnóstico 1 — FP mensal.** `normal_alert_rate` do EXP10c (task
`6ac3b1b5...`), quebrado por mês dentro do período OOS:

| Mês | FP | Mês | FP |
|---|---|---|---|
| 2025-07 | 0,086% | 2025-12 | 0,325% |
| 2025-08 | 0,405% | 2026-01 | 0,081% |
| 2025-09 | 0,177% | 2026-02 | 0,357% |
| 2025-10 | 0,861% | 2026-03 | 0,365% |
| 2025-11 | 0,182% | 2026-04 | 0,556% |

Correlação com "meses desde o corte": **0,20** (fraca). Inclinação da
reta: 0,016pp/mês, desprezível frente ao desvio-padrão entre meses
(0,225pp) — variação parece ruído operacional (meses com mais
manobra), não degradação progressiva. Reforça isso: os 2 episódios
perdidos (glitches de instrumento, seção anterior) caem em **agosto e
novembro de 2025** — logo no início da janela OOS, não no fim, o oposto
do que "modelo ficando desatualizado" preveria.

**Diagnóstico 2 — deriva na distribuição das features brutas.** Pontos
"on" e fora de qualquer janela de alarme (±24h, RUNNING_A≥0,9 como
proxy), agregados por mês, para os 12 sensores do grupo
(`TC382_03_A`, `T5_AVG_A`, 10 canais de vibração `TV_35*`) — z-score de
cada mês em relação à média/desvio do próprio período de treino
(tudo antes de 2025-07-01), toda a série 2024-07 a 2026-04:

- |z| médio dos 10 meses OOS: **0,774**. |z| médio dentro do próprio
  período de treino (variabilidade interna, mesma régua): **0,609**
  (desvio 0,318). Ou seja, o período OOS está dentro de ~1 desvio-padrão
  da variabilidade que o próprio treino já continha — não é um regime
  novo.
- Correlação entre |z| mensal e "meses desde o corte", **dentro do OOS**:
  **0,07** — praticamente zero. Sem tendência crescente.
- Os meses com |z| mais alto no OOS (2026-02 e 2026-03, ~1,2-1,3,
  puxados principalmente pelos canais de vibração `TV_354Y_A`/`TV_355X_A`)
  coincidem com o trimestre de maior atividade de alarme (episódios em
  10/02, 24/02, 25/02, 13/03, 16/03, 25/03 — ver seção do LOEO) — leitura
  mais provável é "trimestre mecanicamente mais agitado" (mais eventos
  reais próximos, contaminando o rótulo de "normal" nas bordas da janela
  de exclusão de ±24h), não "sensor perdendo calibração com o tempo".
  Esses picos de |z| tem magnitude comparável a picos que já existem
  DENTRO do próprio período de treino (ex: 2024-08, |z|=1,373) — não é
  um patamar novo.

**Conclusão: sem evidência de deriva nos 10 meses observados.** Os dois
diagnósticos (FP-rate e distribuição bruta das features) concordam: não
há tendência monotônica de degradação no horizonte OOS atual. O caso de
negócio para construir retreino walk-forward mensal *agora*, motivado
por deriva já observada, não se sustenta com os dados disponíveis. Isso
não fecha a pergunta em definitivo (10 meses pode ser curto demais para
deriva lenta aparecer, e o argumento de "robustez preventiva" é
independente de evidência already-observed), mas muda a prioridade: não
é um problema com sintoma já visível hoje.

**Reprodução:** os dois diagnósticos foram feitos como scripts ad-hoc
(não commitados como script reprodutível — reaproveitam artefatos já
existentes via ClearML `Task.artifacts`/`Dataset.get`, sem processamento
novo do modelo). Query 1: agrupa `point_anomalies_all.csv` da task
`6ac3b1b52a45433a83568d61fafadda6` por mês, filtra `operational_state ==
"on"` e fora de ±1440min de qualquer um dos 40 alarmes OOS. Query 2: lê
`RUNNING_A` + os 12 sensores do grupo direto do CSV do dataset
(`a97ba56ba14840fbb1125c2a82f883c9`), mesmo filtro "on"/fora-de-alarme
(aplicado a toda a série 2024-2026, não só OOS), z-score mensal contra
média/desvio do período de treino.

## Retreino walk-forward mensal (2026-08-29)

Script: `scripts/walkforward_exp10c_retrain_mensal.py`. Task ClearML:
`16a98571ab16434cbc8f7429048cd0de`. Mesmo sem evidência de deriva (seção
anterior), o teste empírico direto é barato o suficiente pra vir na
frente de qualquer suposição.

**Escopo:** isola só a cadência de retreino do modelo (`ocsvm` +
normalização + limiar de percentil, recomputados a cada mês, janela
**expansiva** — cada mês usa todo o histórico normal anterior). Máscara
operacional, portões de rampa/volatilidade e limites de clipping de
outlier ficam **fixos** nos valores de produção — não é sobre isso este
teste (já tratado na seção do LOEO).

**Método:** para cada um dos 10 meses do período OOS, treina do zero em
todo o normal disponível antes do início do mês, mede hit/FP só daquele
mês. Roda também o modelo **congelado** atual (o do EXP10c) pela mesma
rotina de avaliação mês a mês, para comparação pareada.
**Sanity check:** o congelado, reavaliado mês a mês, bateu exatamente
com a referência conhecida — hit_rate 0,9250 (37/40), normal_alert_rate
0,00348 (0,35%).

| Mês | Congelado (FP) | Walk-forward (FP) |
|---|---|---|
| 2025-07 | 70/81406 | 70/81406 |
| 2025-08 | 283/69853 | 292/69853 |
| 2025-09 | 152/85665 | 91/85665 |
| 2025-10 | 693/80458 | 650/80458 |
| 2025-11 | 51/28089 | 42/28089 |
| 2025-12 | 280/86108 | 177/86108 |
| 2026-01 | 47/57789 | 41/57789 |
| 2026-02 | 220/61658 | 118/61658 |
| 2026-03 | 160/43871 | 98/43871 |
| 2026-04 | 310/55744 | 247/55744 |

**Resultado agregado:**

| | hit_rate | normal_alert_rate |
|---|---|---|
| Congelado (referência) | 92,50% (37/40) | 0,348% |
| **Walk-forward mensal** | **92,50% (37/40)** | **0,281%** |

**Ganho real, sem custo de detecção.** Mesmos 37/40 alarmes detectados
(hit_rate idêntico, nenhum episódio perdido nem ganho), FP cai ~19%
relativo (0,348%→0,281%). Em 9 dos 10 meses o walk-forward empata ou
melhora o congelado; só agosto/2025 piora ligeiramente (283→292, 1º mês
completo após o corte, janela de treino ainda quase idêntica à do
congelado).

**Sobre o mecanismo — não é "correção de deriva".** O treino do `ocsvm`
já era subamostrado para `AUTOML_OCSVM_MAX_TRAIN_SAMPLES=50000` pontos
mesmo no congelado (386492 normais disponíveis) — cada mês do
walk-forward *também* amostra só 50000, então o ganho não vem de "mais
dados de treino" no sentido bruto. Vem de a amostra de 50000 ser tirada
de um conjunto-candidato cada vez maior e mais recente — mais variedade
de condições operacionais normais representadas no fit, não uma
correção de um alvo que se move (a seção anterior não achou deriva). É
uma leitura mais modesta que "o modelo estava ficando desatualizado",
mas o ganho de FP é real e mensurável.

**Recomendação:** candidato sólido para produção — ganho líquido sem
custo de detecção, mecanismo simples de entender (retreinar mensalmente
com janela expansiva, tudo mais igual). Custo operacional: implica
infraestrutura de retreino periódico (hoje o pipeline treina uma vez por
task); não foi construída aqui, só o experimento que mede o potencial
ganho.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/walkforward_exp10c_retrain_mensal.py
```

## Veto de sensor congelado (2026-08-29)

Script: `scripts/veto_sensor_congelado_exp10c.py`. Task ClearML:
`d28df72a954d46409bafff29207a5e16`.

MOTIVAÇÃO: a ideia já tinha sido tentada (e refutada) na reprodução
manual da pipeline do Francisco (`docs/analise_pca_monitoramento_sistema.md`,
seção v10) — mas lá foi testada **em conjunto** com outra mudança (faixa
física fixa em vez de clip por quantil) e o próprio doc registra que
"pode ainda valer a pena isolado — não foi testado separadamente". Além
disso é uma pipeline diferente (PCA multi-sinal, não o `ocsvm` do
EXP10c). Existe uma coluna pré-computada no dataset (`any_sensor_constant_run`)
mas é `True` ~99,8% do tempo (calculada sobre todo o painel de ~40 tags,
não só os 12 do grupo) — inútil como está.

**Método:** aditivo sobre a config de referência do EXP10c (modelo,
portões, máscara — tudo fixo). Sensor "congelado" = valor bruto
literalmente sem mudar (`diff()==0`) por uma janela sustentada de W
minutos, em qualquer um dos 12 sensores do grupo; veto suprime
`is_anom_point` quando congelado (mesma direção dos outros portões — só
remove, nunca adiciona detecção). Varrida grade W ∈ {5, 10, ..., 120}min.
**Sanity check:** reprodução local bateu com a referência (0,9250/37-40,
FP 0,00348 vs 0,00350).

| W (min) | FP | Δ vs. referência | episódios perdidos (de 14) | % tempo congelado (OOS) |
|---|---|---|---|---|
| 5–45 | 0,323% | **−7,2%** | 0/14 | 11,0% → 4,4% |
| 60 | 0,325% | −6,8% | 0/14 | 3,9% |
| 90–120 | 0,327% | −6,2% | 0/14 | 3,5% → 3,4% |

**Ganho real, modesto, sem custo de detecção.** Melhor ponto: W=5min (o
mais agressivo testado, resultado idêntico até W=45min) — FP cai ~7,2%
(0,348%→0,323%), nenhum dos 14 episódios predizíveis é perdido em
nenhuma janela testada. Menor que o ganho do walk-forward (~19%) ou do
ajuste de limiares de portão (~32%, seção do LOEO), mas é um mecanismo
diferente — **combinável** com os outros (é uma camada adicional de
supressão, não substitui nenhuma existente). Not-LOEO-validado: assim
como os outros limiares de portão, W foi escolhido vendo todos os 14
episódios de uma vez — o mesmo risco de otimismo de seleção da seção do
LOEO se aplica aqui, não verificado formalmente.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/veto_sensor_congelado_exp10c.py
```

## Votação 2-de-2 entre famílias independentes (2026-08-29) — refutado

Script: `scripts/votacao_2de2_exp10c.py`. Task ClearML:
`17ce17b63e00474fb91798051682e064`.

MOTIVAÇÃO: ver `ALARMES_POR_SENSOR_EFEITO_CASCATA.md` — um evento físico
real dispara dezenas de tags porque várias grandezas reagem à mesma
causa raiz. A hipótese: treinar 2 `ocsvm` **independentes** — família
térmica (`TC382_03_A`, `T5_AVG_A`) e família de vibração (10 canais
`TV_35*`) — e exigir que as duas concordem (dentro de uma janela de
tolerância causal) deveria filtrar ruído de uma família isolada, igual à
lógica de votação N-de-4 da pipeline do Francisco.

**Método:** mesma máscara operacional e mesmos alarmes de exclusão
compartilhados entre as 2 famílias; portões de rampa/volatilidade da
produção aplicados sobre o resultado combinado. Família "ativa" em t =
sinalizou em algum ponto de `[t-W, t]` (dilatação causal). Combinado =
ativa(térmica) E ativa(vibração). Varrida grade de janela de votação W
∈ {30min, ..., 24h}. **Sanity check:** o modelo único (mesma rotina,
recalculado neste script) bateu com a referência (0,9250/37-40, FP
0,00348).

| W | FP | FP vs. modelo único | episódios (de 14 do modelo único) |
|---|---|---|---|
| 30min | 0,804% | **2,3x pior** | 13/14 |
| 1h | 1,741% | 5,0x pior | 13/14 |
| 2h | 3,849% | 11,1x pior | 13/14 |
| 4h | 7,446% | 21,4x pior | 13/14 |
| 8h | 13,402% | 38,5x pior | 13/14 |
| 12h | 18,978% | 54,5x pior | 13/14 |
| 24h | 32,372% | 93,0x pior | 13/14 |

**Refutado com clareza — pior nos dois eixos, em toda a grade.** Mesmo
na janela mais curta testada (30min), o FP já é 2,3x maior que o modelo
único; cresce descontroladamente com a janela (93x pior em 24h) e a
detecção **nunca** chega a 14/14 (fica em 13/14 mesmo com um dia inteiro
de tolerância) — ou seja, uma das duas famílias, sozinha, simplesmente
nunca sinaliza nada perto de um dos episódios, independente de quanto
tempo de tolerância se dê à outra.

**Leitura:** separar os 12 sensores em 2 modelos independentes joga fora
a estrutura de correlação conjunta que o modelo único explora — uma
combinação temperatura+vibração pode estar fora do padrão normal
*junto*, mesmo que cada família isolada ainda pareça dentro do seu
próprio envelope individual (a família de vibração, com 10 canais
correlacionados comprimidos num único detector, parece especialmente
ruidosa sozinha). A dilatação causal (rolling-max) amplifica esse ruído
de cada família antes do "E", e o resultado da combinação nunca volta a
ficar melhor que o modelo conjunto original. A intuição por trás da
votação (histórica, da pipeline do Francisco — sinais fisicamente
independentes concordando é evidência mais forte) não se traduziu numa
implementação vantajosa aqui: o EXP10c já captura naturalmente a
correlação entre as famílias por tratá-las como um único espaço de
features, o que a arquitetura de votação joga fora deliberadamente.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/votacao_2de2_exp10c.py
```

## Combinação dos ganhos validados (2026-08-29)

Script: `scripts/combinado_exp10c_final.py`. Tasks ClearML:
`16e4bbab957042b2bae2005e6a154b01` (waterfall A→B→C) e
`c344509b8c034cf68235b5a37970284f` (adiciona a variante D). Pergunta:
os 3 ganhos validados isoladamente (walk-forward mensal, limiares de
portão mais agressivos, veto de sensor congelado) se somam?

**Método:** mesma rotina de retreino mensal do walk-forward. Em cada
mês, o MESMO modelo retreinado daquele mês é pontuado 4 vezes, variando
só a camada de pós-processamento (waterfall aditivo):

- **A** — só walk-forward (portões de produção 100/0,39, sem veto)
- **B** — A + limiares novos (ramp=60/vol=0,21, achado do LOEO)
- **C** — B + veto de sensor congelado (W=5min)
- **D** — walk-forward + veto, **sem** trocar os limiares (100/0,39 mantidos)

| Combinação | hit_rate | FP | vs. produção |
|---|---|---|---|
| Produção hoje (referência) | 92,50% (37/40) | 0,348% | — |
| A. + walk-forward | 92,50% (37/40) | 0,281% | FP −19% |
| B. + limiares novos | **80,00% (32/40)** | 0,206% | **perde 5 detecções reais** |
| C. + veto (em cima de B) | 80,00% (32/40) | 0,181% | perde as mesmas 5 |
| **D. walk-forward + veto (sem trocar limiar)** | **92,50% (37/40)** | **0,256%** | **FP −26%, mesmo hit_rate** |

**Achado central: os ganhos NÃO se somam livremente — há uma
incompatibilidade real entre dois deles.** Os limiares de portão
(ramp=60/vol=0,21) foram validados como "custo zero" no LOEO contra o
modelo **congelado** de produção — mas essa calibração é específica à
distribuição de score daquele modelo. Assim que o modelo passa a ser
retreinado mês a mês (walk-forward), a distribuição de score muda, e os
mesmos limiares fixos passam a bloquear pontos que eram precursores
reais em janeiro (6→3 detectados) e fevereiro (9→7) — a mesma classe de
risco de otimismo já flagrada no LOEO, agora se manifestando como uma
**interação entre dois ganhos**, não um viés isolado de um deles.

**Combinação segura e recomendada: D (walk-forward + veto de sensor
congelado, portões de produção inalterados).** Mesmo hit_rate da
produção (37/40, nenhuma detecção perdida), FP cai 26% relativo
(0,348%→0,256%) — mais que qualquer um dos dois isoladamente (19% e
7,2%), e a composição é aproximadamente multiplicativa
(0,281%×(1−0,072)≈0,261%, próximo do 0,256% observado), confirmando que
walk-forward e veto atacam fontes de FP diferentes e não se sobrepõem.

**Lição para qualquer combinação futura:** um ganho validado contra uma
config específica (mesmo com LOEO) não é automaticamente seguro quando
outra parte do sistema muda. Recalibrar/revalidar limiares dependentes
do modelo (portões, thresholds) sempre que o próprio modelo mudar —
nunca assumir que ganhos independentes se somam sem testar a combinação
de verdade, como feito aqui.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/combinado_exp10c_final.py
```

## Duração do score: TP vs FP residual (2026-08-29)

Script: `scripts/analise_duracao_score_exp10c.py`. Task ClearML:
`51106f4ef05f4bc0a7460b2d774443d6`.

PERGUNTA DO USUÁRIO: os precursores reais (os 37 alarmes batidos) ficam
com o score do modelo continuamente acima do limiar por mais tempo do
que os episódios de falso positivo residual? Se sim, duração mínima
poderia ser mais um filtro. Diferente do debounce por **contagem de
pontos** já testado e descartado no EXP7 (`docs/analise_automl_exp10.md`,
seção "Caminho testado e descartado — debounce") — aquele teste foi
feito **antes** dos portões de máscara/rampa/volatilidade existirem,
sobre um perfil de FP muito maior; valia re-testar no residual de hoje.

**Método:** score bruto (sem nenhum portão, só máscara operacional) >
limiar p99,9 — episódios contínuos medidos por duração real (RLE). Para
cada um dos 37 alarmes batidos, pega o maior episódio que se sobrepõe à
janela ±24h (o "sinal" que gerou a detecção). Para FP: episódios fora de
qualquer janela de alarme, no período OOS, marcando quais sobrevivem aos
portões de produção hoje (FP residual, n=285) vs. os que os portões já
suprimem.

| | n | média | mediana | p10 | p25 | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| TP (precursor batido) | 37 | 127,7min | **49,5min** | 6,6 | 8,0 | 62,0 | 247,0 | 1379,0 |
| FP residual (sobrevive hoje) | 285 | 5,8min | **2,5min** | 0,5 | 1,0 | 5,0 | 7,0 | 240,5 |

**Separação real de ~20x na mediana** (49,5min vs 2,5min) — confirma a
intuição. Mas há sobreposição nas caudas: o TP mais curto é 0,5min (um
único ponto de 30s) e o FP mais longo é 240,5min (4h) — duração sozinha
não separa 100% dos casos, só a maioria.

**Filtro de duração mínima — grade fina em torno do ponto de virada:**

| duração mín. | TP perdidos (de 37) | normal_alert_rate | redução vs. 0,348% |
|---|---|---|---|
| 1min | 1 | 0,342% | 1,8% |
| 3min | 1 | 0,310% | 11,0% |
| **6min** | **1** | **0,235%** | **32,7%** |
| 7min | 4 | 0,218% | 37,5% |
| 9min | 13 | 0,206% | 41,0% |
| 15min | 14 | 0,184% | 47,1% |

(`normal_alert_rate` estimado ponderando cada episódio residual pela sua
duração real em pontos, não por contagem de episódios — 74% dos
episódios têm duração ≤5min mas pesam pouco no total de pontos; o FP
residual é dominado por um número menor de episódios mais longos.)

**Ponto de virada nítido em 6min.** Até 6min, só se perde 1 dos 37 TP
(um único ponto de 30s isolado — o alarme `2026-01-17 00:59`, uma
detecção já no limite de ser ruído, não um precursor sustentado) e o FP
cai 32,7%. Passar de 6 para 7min já custa 4 TP (3 detecções reais a mais
perdidas por 5pp de FP a mais) — claramente do lado ruim da curva.
**6 minutos de duração mínima contínua é o melhor candidato de filtro
encontrado até agora nesta investigação** — maior redução de FP
(32,7%) pelo menor custo (1 detecção marginal) de qualquer mecanismo
testado (walk-forward: 19%; veto de sensor congelado: 7,2%).

**Ainda não testado:** interação deste filtro com walk-forward mensal e
com o veto de sensor congelado — a lição da seção "Combinação dos
ganhos validados" (limiares de portão que pareciam custo-zero quebraram
ao trocar o modelo) se aplica igualmente aqui: um filtro calibrado
contra o score do modelo **congelado** não é garantidamente seguro
contra o score de um modelo retreinado mensalmente. Precisa validação
conjunta antes de qualquer recomendação de combinar os dois.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/analise_duracao_score_exp10c.py
```

## Filtro de duração + walk-forward + veto — refutado com força (2026-08-29)

Script: `scripts/combinado_exp10c_com_duracao.py`. Task ClearML:
`dfda933c81f245c4a02389ccfbe08dae`.

Testando exatamente a pendência da seção anterior: o filtro de duração
mínima (6min, −32,7% de FP isolado) sobrevive à combinação com
walk-forward mensal + veto de sensor congelado (a combinação segura D)?

**Método:** as flags de cada mês do walk-forward (D completo: modelo
retreinado + portões de produção + veto) são concatenadas numa série
contínua do período OOS inteiro **antes** do RLE, evitando que o
chunking mensal corte artificialmente um episódio que atravesse
fronteira de mês. **Sanity check:** D reproduzido aqui bateu exatamente
com o valor já conhecido (92,50%/0,256%). Aplicado o filtro de 6min
sobre essa série:

| | hit_rate | normal_alert_rate |
|---|---|---|
| D (walk-forward + veto, sem filtro) | 92,50% (37/40) | 0,256% |
| **E = D + filtro de duração (6min)** | **40,00% (16/40)** | 0,124% |

**Refutado com força — pior que qualquer combinação testada até agora.**
21 de 37 detecções reais desaparecem (hit_rate despenca pra 40%), muito
mais grave que a quebra do ajuste de limiares (que custou 5 de 37). A
explicação é a mesma raiz, só que mais severa: o filtro de 6min foi
calibrado sobre a duração dos episódios de score do modelo **congelado**
único. Sob walk-forward, cada mês tem um modelo com fronteira de decisão
diferente — o formato da subida do score perto de um precursor real muda
de mês pra mês, e para muitos meses o mesmo precursor que durava >6min
sob o modelo congelado passa a durar menos sob o modelo daquele mês
específico, sendo descartado pelo filtro.

**Conclusão — o filtro de duração fica associado ao modelo congelado,
não ao walk-forward.** Não há evidência, com o que foi testado, de que
essa combinação específica valha a pena sem uma re-calibração completa
do limiar de duração por mês (o que reabriria a mesma questão de viés
de seleção já tratada na seção do LOEO, sem garantia de payoff). A
recomendação prática do EXP10c permanece a combinação **D** (walk-forward
+ veto, sem filtro de duração, sem trocar limiares de portão): 92,50%
hit_rate / 0,256% FP — o filtro de duração de 6min fica registrado como
um ganho válido **apenas em cima do modelo congelado de produção
atual** (92,50%/0,235%, seção anterior), não como parte de uma pilha
com walk-forward.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/combinado_exp10c_com_duracao.py
```

## Filtro de duração adaptativo — pior ainda (2026-08-29)

Script: `scripts/combinado_exp10c_duracao_adaptativa.py`. Task ClearML:
`7ce47c0fc6c44eaabf33632540a6ae78`.

IDEIA (levantada em conversa): em vez de um número fixo de minutos, o
corte de duração deveria se auto-calibrar a cada retreino — a mesma
lógica já usada pro limiar de **score**, que nunca é um valor fixo, é o
percentil 99,9 do erro do próprio treino daquele mês. Aplicando o mesmo
princípio um nível acima: por definição, ~0,1% dos pontos de treino
cruzam esse limiar — são os "falsos positivos" intrínsecos aquele
modelo específico, ocorrendo em dados 100% normais. Hipótese: a duração
desses episódios de ruído do próprio treino seria a régua natural de
"ruído de fundo" daquele mês, ajustando-se sozinha quando o retreino
muda a fronteira de decisão.

**Método:** para cada mês do walk-forward, mede a duração dos episódios
onde o erro do PRÓPRIO treino cruza o limiar (RLE ciente de buracos no
índice de tempo — o treino exclui períodos inteiros, então vizinhos na
tabela podem estar longe no tempo real). Corte adaptativo = percentil 99
dessa distribuição, aplicado só nos pontos daquele mês.

| | hit_rate | normal_alert_rate |
|---|---|---|
| D (walk-forward + veto, sem filtro) | 92,50% (37/40) | 0,256% |
| Filtro fixo (6min, seção anterior) | 40,00% (16/40) | 0,124% |
| **Filtro adaptativo (p99 do ruído de treino)** | **15,00% (6/40)** | 0,102% |

**Piorou ainda mais — a hipótese "duração do ruído de treino = régua de
ruído de fundo" não se sustenta.** Os cortes adaptativos calculados
(8 a 15min por mês) saíram **maiores** que o filtro fixo de 6min que já
tinha sido refutado. Causa: o treino cobre uma janela histórica de mais
de um ano (crescente no walk-forward) — "ruído" nesse conjunto não é
white noise pontual e independente; existem trechos estruturais de erro
sistematicamente mais alto (variação sazonal, pequenos desvios não
capturados pelos 2 sensores de alarme avaliados, etc.) que se
manifestam como episódios de cruzamento-de-limiar bem mais longos do
que um "flicker" de ruído puro. O percentil 99 dessa distribuição
contaminada vira um corte mais agressivo que o fixo, não menos.

**Conclusão — duração não é seguro contra walk-forward de nenhuma forma
testada até agora.** Nem o valor fixo (calibrado contra o modelo
congelado) nem a versão auto-calibrada (calibrada contra o próprio
ruído do treino de cada mês, mas contaminada por estrutura de longo
prazo) sobrevivem à troca de modelo mensal. Um caminho ainda não testado
seria restringir a medição de "ruído" a uma janela **recente** (ex:
últimos 30 dias de treino) em vez do histórico expansivo inteiro, para
evitar a contaminação por efeitos de escala longa — mas dado que já são
3 tentativas de duração falhando em cadeia, a recomendação prática
permanece a mesma: **combinação D (walk-forward + veto, sem nenhum
filtro de duração), 92,50% hit_rate / 0,256% FP.**

**Reprodução:**
```bash
PYTHONPATH=. python scripts/combinado_exp10c_duracao_adaptativa.py
```

## EXP19: combinação D promovida a config de produção (2026-08-29)

A combinação segura (walk-forward mensal + veto de sensor congelado,
portões/mascara inalterados) deixou de ser um script ad-hoc e virou
recurso de primeira classe do pipeline, do mesmo jeito que os outros
portões (EXP10/10b/10c) — dois campos novos em `config.py`, ambos com
default que preserva 100% o comportamento anterior:

- `ENABLE_FROZEN_SENSOR_VETO` / `FROZEN_SENSOR_VETO_WINDOW_MINUTES`
  (default `False`/`5.0`) — `scoring.py:compute_frozen_sensor_mask`/
  `apply_frozen_sensor_veto`, mesma camada dos outros portões (só
  remove detecção, nunca adiciona).
- `ENABLE_WALKFORWARD_RETRAIN` / `WALKFORWARD_RETRAIN_FREQ` (default
  `False`/`"MS"`) — `automl_pipeline.py:_walkforward_fit_periods`:
  re-treina do zero a cada período (janela expansiva), com
  normalização e limiar de percentil recalculados a partir do **próprio**
  `train_err` de cada período — nunca um valor global aplicado depois.
  É essa disciplina (percentil sempre relativo ao modelo do momento,
  nunca um número fixo herdado de outro modelo) que faltou nas duas
  tentativas de filtro de duração — documentado como comentário no
  código, ao lado do campo, para não repetir o erro em cima disso no
  futuro.

**Limitação assumida:** `AUTOML_SEED_SWEEP_N` é ignorado quando o trial
vencedor usou walk-forward (`_seed_sweep`/`_refit_with_seed` só sabem
re-treinar no split único) — mediria a variância de semente do modelo
errado. Rodar walk-forward com `AUTOML_SEED_SWEEP_N=0`.

**Config de produção:**
`configs/calibracao_v4_eq/test_grupo_exp19_walkforward_veto.json` —
idêntica ao EXP10c, exceto `ENABLE_WALKFORWARD_RETRAIN: true` e
`ENABLE_FROZEN_SENSOR_VETO: true` (`window=5min`). `LOAD_GATE_RAMP_MAX`/
`VOLATILITY_GATE_THRESHOLD` **mantidos em 100/0,39** (os valores de
produção) — a seção "Combinação dos ganhos validados" mostrou que
trocá-los para os "custo zero contra o congelado" (60/0,21) perde 5
detecções reais sob walk-forward. Nenhum filtro de duração (fixo ou
adaptativo) — as duas tentativas de compor duração com walk-forward
foram refutadas.

**Resultado esperado (validado via script ad-hoc antes de virar
feature): 92,50% hit_rate (37/40) / 0,256% normal_alert_rate** — mesma
detecção de hoje, ~26% menos falso alerta. Suíte de testes (`pytest
tests/`) passa inalterada (15/15) — confirma que nenhum config existente
é afetado pelos campos novos.

**Reprodução (via entrypoint padrão, não script ad-hoc):**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp19_walkforward_veto.json
```
