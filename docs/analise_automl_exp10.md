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
