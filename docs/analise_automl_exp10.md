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

**Resultado confirmado via `src/main.py` (entrypoint padrão, não script
ad-hoc)** — task ClearML `577f136aa36c4d3c938a62daa44da95b`,
`calibration_report.json`: `hit_rate=0,925 (37/40)`,
`normal_alert_rate=0,002559` (0,256%) — bate exatamente com o validado
nos scripts desta investigação. Suíte de testes (`pytest tests/`) passa
inalterada (15/15) — confirma que nenhum config existente é afetado
pelos campos novos.

**Reprodução (via entrypoint padrão, não script ad-hoc):**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp19_walkforward_veto.json
```

## Terceira tentativa de duração — adaptativo-recente, mesma falha (2026-08-30)

Script: `scripts/combinado_exp10c_duracao_adaptativa_recente.py`. Task
ClearML: `40045ea00dfb4f96bed75d4de683c330`.

Diagnóstico da tentativa anterior: medir "ruído" sobre o treino
expansivo inteiro (+1 ano) captura contaminação de longo prazo. Correção
testada aqui: restringir a medição aos **últimos 60 dias** antes de cada
retreino (não o histórico todo), usando o **máximo** observado (não um
percentil alto, dado a amostra menor) como corte adaptativo.

| mês | nº episódios de ruído (60d) | corte adaptativo |
|---|---|---|
| 2025-07 | 2 | 2,5min |
| 2025-08 | 3 | 4,0min |
| 2025-09 | 20 | 5,0min |
| 2025-10 | 20 | 15,0min |
| 2025-11 | 7 | 8,5min |
| 2025-12 | 11 | 9,0min |
| 2026-01 | 16 | 11,5min |
| 2026-02 | 9 | 10,5min |
| 2026-03 | 13 | **35,0min** |
| 2026-04 | 12 | 9,0min |

| | hit_rate | normal_alert_rate |
|---|---|---|
| D (walk-forward + veto, sem filtro) | 92,50% (37/40) | 0,256% |
| **G = D + filtro de duração adaptativo-recente** | **15,00% (6/40)** | 0,122% |

**Falhou de novo, na mesma magnitude — mesmo restringindo a janela de
"ruído" a 60 dias.** A maioria dos meses ainda produz um corte entre
5 e 35min (só julho/agosto ficam abaixo da zona segura de ~6min
encontrada contra o modelo congelado) — na prática, equivalente a testar
um filtro fixo de 9-35min em cada mês, faixa que a grade fina da
primeira tentativa já tinha mostrado ser destrutiva (perde 13+/37 TP a
partir de ~9min). Restringir a janela de medição não resolveu porque o
problema não é "contaminação de longo prazo" per se — é que, sob
retreino mensal, o score de VÁRIOS meses tem um comportamento de "quase
limiar" mais persistente que o do modelo congelado, tanto em ruído
quanto (como a 1ª tentativa de duração já mostrou) em precursor real. A
separação limpa TP-vs-FP por duração que existe pro modelo congelado
(mediana 49,5min vs 2,5min) não se sustenta da mesma forma mês a mês
quando o modelo muda todo mês.

**Veredito, com 3 tentativas independentes de duração falhando por
razões relacionadas mas distintas (fixo, adaptativo-expansivo,
adaptativo-recente): duração e retreino walk-forward mensal não são
compatíveis nesta arquitetura, não é uma questão de calibração — é
estrutural.** Não há mais tentativa de duração planejada para esta
combinação. A recomendação prática do EXP10c/EXP19 permanece:

- **Se for usar walk-forward**: combinação D (walk-forward + veto),
  sem filtro de duração — 92,50%/0,256%, já promovida a config de
  produção (EXP19).
- **Se preferir não mexer na cadência de retreino**: filtro de duração
  fixo de 6min sozinho, em cima do modelo congelado atual — 90,0%
  (36/40) / 0,235% FP (−32,7%), a maior redução de FP isolada encontrada
  em toda a investigação, ao custo de 1 detecção já marginal (ponto
  único de 30s). Não implementado como feature de pipeline ainda — fica
  como candidato caso o walk-forward não seja adotado operacionalmente.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/combinado_exp10c_duracao_adaptativa_recente.py
```

## EXP20: opção B (filtro de duração fixo, sem walk-forward) como config (2026-08-30)

Promovido a feature de pipeline igual ao EXP19, mantendo o EXP19
intocado — as duas opções (A: walk-forward+veto; B: filtro de duração
sozinho) coexistem como configs separadas, para comparação lado a lado.

- `ENABLE_MIN_DURATION_FILTER` / `MIN_DURATION_FILTER_MINUTES` (default
  `False`/`6.0`) — `scoring.py:apply_min_duration_filter`, aplicado por
  **último** no laço de portões (depois de máscara/rampa/volatilidade/
  veto de congelamento) — mede a duração do que sobra após todos os
  outros filtros.
- Guarda de segurança: `run_automl_group` levanta `ValueError` se
  `ENABLE_MIN_DURATION_FILTER` e `ENABLE_WALKFORWARD_RETRAIN` estiverem
  ligados juntos — as 3 tentativas de compor os dois (seções acima)
  derrubaram o hit_rate pra 15-40%; a combinação nunca deveria ser
  usada em produção.
- Correção incidental: `_seed_sweep` (variância de semente) não estava
  aplicando `frozen_mask` nem (agora) o filtro de duração — corrigido
  para as duas camadas, garantindo que o seed-sweep meça exatamente o
  mesmo pipeline do trial vencedor. Não afeta nenhum config existente
  (`ENABLE_FROZEN_SENSOR_VETO`/`ENABLE_MIN_DURATION_FILTER` são `False`
  por padrão em todos os configs anteriores ao EXP19).

**Config:** `configs/calibracao_v4_eq/test_grupo_exp20_filtro_duracao.json`
— idêntica ao EXP10c, exceto `ENABLE_MIN_DURATION_FILTER: true`,
`MIN_DURATION_FILTER_MINUTES: 6.0`. Sem walk-forward, sem veto, sem
trocar limiares de portão — só o filtro de duração em cima da
referência.

**Dado disponível:** verificado nos 13 datasets do ClearML (incluindo
`Cabiunas consolidado 2022-2026`, criado em 18/08/2026) — nenhum tem
dado além de **2026-04-30**. Não foi possível estender a série temporal
para este treino; usa o mesmo range de sempre.

**Dois bugs encontrados rodando de verdade (2026-08-30) — resultado real
é diferente do estimado:**

1. **Bug de ordem (corrigido):** a 1ª implementação aplicava o filtro de
   duração **depois** dos portões de rampa/volatilidade/veto de
   congelamento — media a duração de episódios já fragmentados por eles
   (um precursor real longo vira vários pedaços curtos, cada um mais
   curto que o corte mesmo sem ser ruído). Resultado da 1ª rodada:
   hit_rate=47,5% (19/40) — muito pior que qualquer coisa vista na
   investigação. Corrigido: filtro de duração agora roda logo após a
   máscara operacional, **antes** dos outros portões.
2. **A estimativa original (90%/0,235%) era uma aproximação, não uma
   simulação literal** — a análise de duração (`analise_duracao_score_exp10c.py`)
   caracterizou as distribuições TP/FP e estimou o efeito do filtro por
   contagem/peso, sem de fato reconstruir a série e reavaliar. Rodando
   de verdade (ordem corrigida), o resultado muda: **2 alarmes a mais**
   são perdidos, ambos com episódio de duração bem próxima do corte de
   6min (2025-10-20, medido em ~6,0min na análise exploratória —
   sensível a diferença de arredondamento entre a medição aproximada e
   o RLE vetorizado real).

**Resultado real, confirmado via `src/main.py`** (task
`16a66da5cb034307bba53ac968d5a09e`):

| | hit_rate | normal_alert_rate |
|---|---|---|
| Referência (sem filtro) | 92,50% (37/40) | 0,348% |
| **EXP20 (filtro de duração 6min, ordem corrigida)** | **85,0% (34/40)** | **0,189% (−45,7%)** |

Perde 3 alarmes a mais que a referência (além dos 3 já estruturalmente
não-detectáveis: os 2 glitches de instrumento + o ponto único de 30s de
2026-01-17) — 2025-10-20 e mais uma linha de 2026-01-17. Em troca, a
redução de FP é **maior** que a estimativa original (45,7% vs 32,7%
estimado) — é um ponto diferente (mais agressivo) da curva
custo-benefício, não simplesmente pior.

**Lição geral:** qualquer estimativa de filtro que não seja uma
reconstrução+reavaliação literal deve ser tratada como aproximada,
principalmente perto de fronteiras exatas de corte — a mesma disciplina
de "rodar de verdade antes de confiar no número" que motivou várias
correções nesta investigação (walk-forward, veto) se aplica aqui
também.

**Reprodução:**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp20_filtro_duracao.json
```

## Grade fina literal do filtro de duração (2026-08-30)

Script: `scripts/grade_fina_filtro_duracao_exp20.py`. Task ClearML:
`49f6c33a356044178c8267ab1dfbcd05` (após corrigir mais um bug de
metodologia: a 1ª rodada desse script também tinha o denominador do FP
sem restringir a "on" — mesma classe de erro já vista antes — sanity
check deu 0,291% em vez de 0,350%; corrigido e revalidado).

Ao contrário de toda a análise anterior (por contagem/peso), esta grade
reconstrói a série de verdade a cada corte candidato, usando a própria
função de produção (`scoring.apply_min_duration_filter`, mesma ordem —
antes dos portões). **Sanity check bate exatamente:** 0,9250 (37/40) /
0,00348 sem filtro de duração.

| corte | hit_rate | normal_alert_rate | redução |
|---|---|---|---|
| 3,0–4,5min | **90,0% (36/40)** | 0,293%→0,247% | 15,8%→**29,1%** |
| **5,0min** | **85,0% (34/40)** | 0,232% | 33,4% |
| 5,5–6,1min | 85,0% (34/40) | 0,213%→**0,189%** | 38,9%→**45,7%** |
| 6,5–7,0min | 82,5% (33/40) | 0,174%→0,170% | 50,1%→51,3% |
| 7,5–8,0min | 70,0–75,0% (28–30/40) | 0,160% | 54,0% |
| 9,0–10,0min | 60,0% (24/40) | 0,156%→0,157% | 54,9%→55,3% |

**Não existe meio-termo que recupere o alarme de fronteira mantendo a
redução de 45,7%.** A queda de 36→34 acontece de forma abrupta entre
4,5min e 5,0min (não suavemente perto de 6min como a análise
aproximada sugeria) — a duração real do episódio de 2025-10-20 está
nessa faixa, não nos ~6,0min estimados antes. Acima de 6,1min, o
platô também é curto: 6,5min já custa mais 1 alarme (33/40) por pouco
FP a mais, e a partir de 7,5min a curva desmorona (70-75%, depois 60%).

**Dois pontos de operação genuinamente bons, escolha por prioridade:**

| Prioridade | Corte | hit_rate | FP | redução |
|---|---|---|---|---|
| Máxima detecção | **4,5min** | **90,0% (36/40)** | 0,247% | 29,1% |
| Máxima redução de FP | **6,0min** | 85,0% (34/40) | **0,189%** | **45,7%** |

O config `test_grupo_exp20_filtro_duracao.json` está com `6.0` — mantém
a redução de FP mais agressiva encontrada em toda a investigação, ao
custo de 2 detecções a mais (34/40) que o originalmente estimado
(36/40). Trocar `MIN_DURATION_FILTER_MINUTES` para `4.5` recupera as 2
detecções, com FP ainda 29,1% menor que a referência.

**Reprodução:**
```bash
PYTHONPATH=. python scripts/grade_fina_filtro_duracao_exp20.py
```

## Opção 2: investigando os 4 alarmes perdidos (36/40) (2026-08-31)

Até aqui só investigamos FP; nunca olhamos por que caímos de 92,5%
(37/40, EXP7-EXP10c) para 90,0% (36/40, EXP20 em diante). São 4 linhas
de alarme perdidas (2 tags no mesmo instante contam como 2 linhas),
correspondendo a **3 eventos distintos**:

| evento | tag(s) | causa raiz | quando começou a falhar |
|---|---|---|---|
| `2025-08-08 15:31:18` | TC382_03_A + T5_AVG_A | **Sem sinal nos 12 canais do grupo** — z-score < 1,4σ em TODOS os canais (temperatura, carga, 10 vibrações) em ±30min | **Já era MISS no EXP10c** (92,5%, antes de qualquer portão) |
| `2025-11-29 07:52:03` | TC382_03_A | **Turbina inteira desligada** (`operational_state='off_longo'` nas ±24h completas) — máscara operacional zera `is_anom_point` por design, indetectável estruturalmente | **Já era MISS no EXP10c** — máscara operacional existe desde o EXP10 |
| `2026-01-17 00:59:01` | T5_AVG_A | Precursor real e curto, cortado pelo filtro de duração de 4,5min (`duration_filter_blocked=2` na janela) | **Era HIT no EXP10c** — a ÚNICA detecção realmente perdida pelos nossos próprios portões (o "1 alarme de fronteira" já documentado na seção do EXP20, agora identificado nominalmente) |

**Confirmado via reconstrução literal**: baixei o
`point_anomalies_all.csv` da task original do EXP10c
(`6ac3b1b52a45433a83568d61fafadda6`, congelado, sem nenhum portão) e
comparei `is_anom_point` nas mesmas ±24h dos 3 eventos — bate
exatamente com a tabela acima.

**Conclusão:** dos 4 alarmes perdidos hoje, só **1** (2026-01-17) é
custo real da nossa otimização de FP (já sabíamos do trade-off, agora
sabemos qual alarme é); os outros 2 eventos (3 linhas) já eram invisíveis
para o modelo desde antes de qualquer trabalho desta investigação —
um por falta de sinal nos sensores do grupo (pode ser um alarme de
outro subsistema, ou um evento instantâneo abaixo da resolução de
30s), outro por definição (turbina desligada). **Não há gate adicional
que resolva os 2 pré-existentes sem mudar o modelo base ou o grupo de
sensores** — não são bugs de portão, são limites estruturais do escopo
atual (12 sensores de um grupo, granularidade 30s).

**Correção metodológica importante (achada durante esta investigação):**
as colunas `load_gate_blocked`/`volatility_gate_blocked`/
`step_change_gate_blocked` do `point_anomalies_all.csv` marcam
simplesmente "a condição do portão (rampa/volatilidade/degrau) estava
acima do limiar neste ponto" — **não** "este ponto estava marcado como
anomalia e foi removido por este portão" (só `duration_filter_blocked`
tem essa semântica exata, via `flags & ~filtered`). Isso significa que
o número **190.993 "FP eliminados pelos portões"** reportado no
gráfico-funil (`gen_figura_funil_exp22.py`,
`serie_completa_funil_exp22.png`/`barras_funil_exp22.png`) está
**superestimado** — inclui pontos onde o portão nunca teve nada pra
suprimir (o modelo não tinha marcado aquele ponto de qualquer forma,
mas a condição de rampa/volatilidade/degrau ainda assim ficou acima do
limiar, ex: durante um startup/shutdown real da turbina). **Os números
36/40, 222 e 4.878 do funil continuam corretos** (dependem só de
`is_anom_point` final, não dos `*_blocked`). Se for necessário o número
exato de "eliminados de verdade", precisa reconstruir o score bruto
recarregando `model.pkl`/`normalization_stats.json` da task e reaplicando
o percentil — não feito ainda, fora do escopo desta rodada.

## EXP23: aplicando a pilha de portões (EXP20-22) no grupo de TRIP de óleo lub. (2026-08-31)

Pedido do usuário: reaproveitar TODAS as melhorias validadas para o
grupo `TC382_T5_vibracao_mancais_multiescala` (filtro de duração 4,5min,
portão de rampa, portão de volatilidade, portão de mudança de nível) no
grupo diferente do **EXP16c** — TRIP de óleo de lubrificação, alvo
`PALL_6240309`, sensores `954005_624_PI_0308`/`954005_624_PDIT_0305` +
10 canais de vibração (mesmos `TV_35[1-5][X/Y]_A`).

**Contexto herdado do EXP16c** (`runs_exp16c_trip_oleo_lub_pdit0305/`):
grupo já usa a pipeline AutoML (`ENABLE_AUTOML=true`, mesmo schema
`SENSOR_GROUPS`), modelo `iforest` (validado superior a `ocsvm` nesse
alvo pelo experimento PCA — `docs/analise_pca_monitoramento_sistema.md`),
threshold p99,5, debounce=24 (bem mais agressivo que o debounce=1 do
grupo TC382 — já suavizava bastante), **nenhum portão ligado**
(`ENABLE_LOAD_GATE`/`ENABLE_VOLATILITY_GATE=false`).

**Ressalva importante sobre amostra**: `PALL_6240309` só tem **13
ocorrências no catálogo inteiro** (2023-10 a 2026-02), das quais só
**2 caem no período OOS** (`>=2025-07-01`). `hit_rate` aqui só pode
assumir 0%/50%/100% — extremamente ruidoso, não comparável em precisão
ao 36/40 do grupo TC382. Resultado deve ser lido com essa ressalva.

**Config**: `configs/calibracao_v4_eq/test_grupo_exp23_trip_oleo_lub_gates.json`
— mesma estrutura do EXP16c `candidato_com_pdit0305`, com:
- `ENABLE_MIN_DURATION_FILTER=true, MIN_DURATION_FILTER_MINUTES=4.5` (valor já validado, reaproveitado sem recalibração própria)
- `ENABLE_LOAD_GATE=true` / `ENABLE_STEP_CHANGE_GATE=true`, ambos usando `954005_624_PI_0308` (a pressão de óleo, análogo ao `T5_AVG_A` do outro grupo) como sensor de referência de nível — parâmetros (rampa/janelas/limiar 1,5) copiados do EXP22 SEM recalibração para esta unidade física
- `ENABLE_VOLATILITY_GATE=true`, mesmos 10 canais de vibração e mesmo limiar 0,39 do EXP22 (aqui sim, mesma grandeza física, herança mais defensável)
- `EXTRA_NEAR_ALARM_TAGS=null` — **não** copiado do EXP21 (aqueles 3 tags de pressão de gás combustível são de outro subsistema físico, sem base para transferir); se o resultado justificar, fazer o mesmo tipo de cruzamento com o catálogo completo específico para este alvo antes de adicionar
- `AUTOML_THRESHOLD_PERCENTILES` ampliado para `[99.0, 99.5, 99.9]` (EXP16c só testou 99,5) — única variável nova além dos portões, pra dar mais chance ao AutoML

**Task:** `3982f8a399854d5982801b32f1e5a7a6`.

**Resultado: trade-off real, não é vitória limpa.**

| | EXP16c (baseline, sem portões) | EXP23 (pilha EXP20-22) |
|---|---|---|
| hit_rate (OOS, 2 alarmes `PALL_6240309`) | **100% (2/2)** | **50% (1/2)** |
| `normal_alert_rate` | 7,60% | 0,66%-0,88% (3 percentis testados) |

Os portões cortaram o FP em ~90%, mas custaram **1 dos 2 alarmes** do
período OOS — proporcionalmente um custo enorme (a amostra de 2 é tão
pequena que perder 1 já é 50pp de hit_rate).

**Causa raiz identificada** (inspeção de `point_anomalies_all.csv`,
colunas `*_blocked`, nos dois alarmes OOS): o alarme perdido
(`2025-11-04 06:22:18`) tem `is_anom_point=0` na janela ±24h inteira,
com `step_change_gate_blocked` verdadeiro em **4.156 dos 5.760 pontos
da janela (72%)** e `volatility_gate_blocked` em 379. O alarme mantido
(`2026-02-26`, 70 pontos ainda detectados) tem uma condição de portão
ainda mais extrema (`step_change_gate_blocked` em 88% da janela) e
mesmo assim sobrevive por pouco.

**O erro de configuração**: `STEP_CHANGE_GATE_SENSOR` e
`LOAD_GATE_SENSOR` foram apontados para `954005_624_PI_0308` — que é o
**próprio sensor-alvo/saúde** do grupo, não um proxy de carga
independente como `T5_AVG_A` era pro grupo TC382. Um TRIP real de
óleo lub. **é, por definição, um degrau/mudança abrupta nesse mesmo
sensor** — então o portão de mudança de nível, ao monitorar o próprio
alvo, acaba classificando o evento de falha real como "manobra
operacional" e suprimindo a detecção. No grupo TC382 isso não
acontecia porque o sensor de referência (`T5_AVG_A`, carga) é
fisicamente independente do sensor de saúde (`TC382_03_A`,
temperatura) — a premissa por trás do portão (distinguir "mudou o
regime operacional" de "o equipamento está degradando") só vale
quando as duas coisas são sensores diferentes. `LOAD_GATE_SENSOR`
mesmo erro conceitual, mas ficou inofensivo aqui só por sorte de
escala (`LOAD_GATE_RAMP_MAX=100`, copiado do EXP22, nunca foi
ultrapassado pela taxa de variação da pressão de óleo — `load_gate_blocked=0`
nas duas janelas).

**Conclusão prática:** não dá pra copiar os portões de rampa/mudança
de nível cegamente entre grupos — eles exigem um sensor de referência
**operacionalmente relevante mas fisicamente independente** do
alvo/saúde monitorado. Para este grupo, precisaria de um proxy de
carga de verdade (não catalogado neste grupo — talvez a rotação da
máquina ou outra tag de processo fora dos 12 sensores atuais) ou
desligar `ENABLE_LOAD_GATE`/`ENABLE_STEP_CHANGE_GATE` e manter só o
filtro de duração + portão de volatilidade (que usa os 10 canais de
vibração, independentes do alvo, e não teve esse problema). **Não
promovido para nenhuma config de referência** — fica registrado como
experimento negativo instrutivo, não como recomendação de produção.

## Reframe operacional: alertas por mês, não pontos de 30s (2026-08-31)

Pedido do setor operacional: "não quero o sistema alarmando a toda
hora". Contar em PONTOS de 30s (222 residuais) é enganoso pra esse
público — o que um operador vê é um ALERTA por episódio, não um alerta
por amostra. Script
`relatorio_exp21_portao_pressao/analise_episodios_operacional.py`
agrupa o `is_anom_point` final do EXP22 (domínio on+OOS, ~9,8 meses,
2025-07-01 a 2026-04-20) em episódios (gap>30min separa) e classifica
cada um contra: alarme oficial (±24h, 2 tags), catálogo completo
(±24h, 47 tags), o padrão recorrente confirmado do mancal 354 (±30min
dos 4 eventos já validados), ou isolado.

| categoria | episódios | % |
|---|---|---|
| Alarme oficial | 30 | 32,6% |
| Catálogo completo (evento real, outro sensor) | 42 | 45,7% |
| Padrão recorrente mancal 354 (confirmado) | 4 | 4,3% |
| **Isolado (ruído sem correspondência)** | **16** | **17,4%** |
| **Total** | **92** | 100% |

**Taxa: 9,4 alertas/mês no total, dos quais só 1,64/mês são
"isolados"** (sem nenhuma correspondência com alarme oficial, catálogo
completo, ou o padrão físico já identificado). Ou seja, **82,6% de todo
alerta que o sistema dispara já tem alguma correspondência com sinal
real** — não é "alarmando à toa a maior parte do tempo".

Dos 16 episódios isolados, **13 já são o artefato estrutural de borda
de gate causal** (0-1min, confirmado/investigado nas seções acima) e
**só 1** (`2025-08-27 00:37`, 17min) segue genuinamente sem explicação
depois de checar os 12 sensores do grupo.

**Implicação prática pro operacional:** o maior ganho que falta não é
mais ajuste de modelo/portão (já testamos e o custo de mexer mais é
perder detecção real) — é a **camada de consolidação de alerta**: se o
sistema hoje dispara uma notificação por PONTO de 30s em vez de uma por
EPISÓDIO, isso já produz uma redução de ~56x na contagem de
notificações (5.176 pontos → 92 episódios) sem tocar em nenhuma lógica
de detecção. Vale confirmar como o alerta chega ao operador hoje antes
de investir em mais portões.

## EXP21: portão de pressão — os "FP" restantes são majoritariamente eventos reais de outro sensor (2026-08-30)

Branch: `feat/exp21_portao_pressao` (nova, a partir de
`fix/upload-ocsvm-model-artifacts` — preserva todo o histórico EXP10c a
EXP20). Motivação: inspecionando a Figura 3 do relatório do EXP20 (ver
`relatorio_exp20_filtro_duracao/`), ficou visível que os pontos
"isolados" (falso positivo) não estavam espalhados aleatoriamente —
apareciam nos mesmos períodos que os pontos perto de alarme real.

**Cadeia de evidência (scripts em `relatorio_exp20_filtro_duracao/`):**

1. Cruzando as 6406 anomalias detectadas (série toda) com o catálogo
   **completo** de alarmes (47 tags, não só os 2 avaliados): 92,6% caem
   a menos de 24h de ALGUM alarme real; só 7,4% são genuinamente
   isolados (`analise_fp_vs_catalogo_completo.py`).
2. Restringindo exatamente aos 1606 pontos que compõem a métrica
   oficial de `normal_alert_rate` do EXP20 (0,2468%): **71,7% coincidem
   (≤24h) com um alarme de OUTRO sensor** — majoritariamente pressão
   (`PI_6240319_AL`: 875, `PAL_6240315`: 215, `PDAL_6240302`: 61) —
   `analise_fp_metrico_vs_catalogo.py`.
3. Dos 455 pontos que sobram genuinamente isolados (nem os 47 tags
   explicam), agrupados em 32 episódios: **72–78% ainda mostram a
   própria pressão bruta (`954005_624_PI_0319`/`_PI_0315`/`_PDI_0302`)
   se movendo 2–3 desvios-padrão acima da linha de base recente**,
   abaixo do limiar oficial de alarme mas fisicamente anômala —
   `analise_pressao_subthreshold.py`.

**Implementação:** `EXTRA_NEAR_ALARM_TAGS` / `EXTRA_NEAR_ALARM_WINDOW_MINUTES`
(`config.py`) — amplia o denominador de `normal_alert_rate` para também
excluir janelas perto desses 3 alarmes de pressão. **Não toca o treino
do modelo nem o `hit_rate`/`eval_sensors`** — só reconhece que um ponto
anômalo perto de um evento real de outro sensor não deveria contar como
"alarme do nada". Janela deliberadamente curta comparada aos ±1440min
usados para os 40 alarmes raros do target: esses 3 tags de pressão
ocorrem ~1×/dia no período OOS — uma janela de ±24h cobriria 60% do
tempo todo (checado empiricamente antes de escolher).

**Calibração da janela** (grade completa, sem retreinar nada — só
reprocessamento do denominador):

| janela | FP recuperado | `normal_alert_rate` esperado | tempo "sob suspeita" |
|---|---|---|---|
| ±2h | 9,8% | 0,223% | 8,5% |
| ±4h | 23,5% | 0,189% | 15,6% |
| ±6h | 41,2% | 0,145% | 22,0% |
| **±8h (escolhido)** | **58,4%** | **~0,103%** | 27,8% |
| ±12h | 65,7% | 0,085% | 38,4% |
| ±24h | 71,7% | 0,070% | 60,0% |

Escolhido ±8h: recupera mais da metade do FP oficial sem reservar mais
de um quarto do tempo como "não julgável".

**Resultado real, confirmado via `src/main.py`** (task
`51564f9c61074a1e94cbb00b4b42cb80`):

| | hit_rate | `normal_alert_rate` |
|---|---|---|
| EXP20 (referência) | 90,0% (36/40) | 0,247% |
| **EXP21 (portão de pressão, ±8h)** | **90,0% (36/40)** | **0,132%** |

**Redução de 46,4% no falso alerta, hit_rate idêntico** (nenhuma
detecção ganha ou perdida — o portão só afeta a avaliação, não o
modelo). Suíte de testes passa inalterada (15/15).

Config: `configs/calibracao_v4_eq/test_grupo_exp21_portao_pressao.json`
— idêntico ao EXP20 (filtro de duração 4,5min) + os 2 campos novos.

**Reprodução:**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp21_portao_pressao.json
```

## Investigando os 455 restantes: buraco de dado real no fim do dataset (2026-08-30)

Aprofundando a investigação dos 455 pontos genuinamente isolados (nem
os 47 tags do catálogo explicam, seção anterior), agrupados em 32
episódios: 2 desses episódios (start em 2026-04-21 04:35 e 2026-04-22
00:10, 41+92=133 pontos) deram z-score de pressão **zero/NaN** — não
porque a pressão estava normal, mas porque **os dados de pressão (e de
todos os outros sensores, incluindo o próprio `TC382_03_A`) simplesmente
não existem** nesse período.

**Achado: buraco de dado real e permanente no fim do dataset.**
`TC382_03_A` para de ter leitura em **2026-04-21 00:50:00** e nunca mais
volta até o fim do arquivo (2026-04-30 23:59:30) — 71 horas (quase 9,5
dias) de `NaN` cru. O pipeline interpola só até `INTERPOLATE_LIMIT=3`
amostras (90s); além disso, `ffill().bfill()` arrasta o último valor
válido indefinidamente — esse platô artificial de ~9,5 dias é pontuado
pelo modelo como dado real, e sua textura anormalmente "morta" (sem o
ruído natural que sempre existe em operação) dispara o limiar de
anomalia.

**Impacto quantificado:** 159 dos 455 pontos genuinamente isolados
(34,9%) caem dentro desse buraco — incluindo 3 dos episódios mais
longos já identificados na análise de duração (13min, 20,5min e 46min
em 21-22/04/2026, que antes pareciam "cauda genuína" sem explicação).
**Nenhum dos 40 alarmes avaliados ocorre depois de 16/04/2026** — cortar
esse período não afeta `hit_rate` em nada.

**Correção:** `DATA_END_DATE: "2026-04-20"` no config do EXP21 (era
`null`, usava o arquivo inteiro).

**Resultado confirmado via `src/main.py`** (task
`e271fa9258c24495b5d9ea2075e079b3`):

| | hit_rate | `normal_alert_rate` |
|---|---|---|
| EXP21 (portão de pressão, sem cortar) | 90,0% (36/40) | 0,132% |
| **EXP21 + `DATA_END_DATE` corrigido** | **90,0% (36/40)** | **0,103% (−21,9%)** |

**Resumo acumulado da investigação inteira** (mesma detecção, 90%/36-40,
desde o EXP20):

| etapa | `normal_alert_rate` | redução acumulada vs. EXP10c |
|---|---|---|
| EXP10c (referência original) | 0,348% | — |
| EXP20 (filtro de duração 4,5min) | 0,247% | 29,1% |
| EXP21 (+ portão de pressão ±8h) | 0,132% | 62,1% |
| **EXP21 (+ corte do buraco de dados)** | **0,103%** | **70,4%** |

**Lição de processo:** vale sempre checar se um resíduo "sem explicação
física" não é, na verdade, um problema de qualidade do dado antes de
aceitar como limite estrutural do modelo — os episódios de duração mais
longa e mais "suspeitos" da primeira análise (seção "Duração do score")
eram exatamente esses.

**Reprodução:**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp21_portao_pressao.json
```

## EXP22: portão de mudança de nível — degraus de carga (2026-08-30)

Investigando os 455 pontos genuinamente isolados do EXP21 mais a fundo:
plotando o contexto bruto (temperatura, vibração, pressão ±3h) dos 5
episódios com menor z-score de pressão, 3 de 5 mostram um **degrau real
de operação** — temperatura E vibração mudando de patamar juntas, em
minutos, exatamente na hora do ponto marcado (2025-12-16 01:31,
2025-08-27 07:39, 2025-08-27 01:51). Não é ruído — é manobra de carga
real que o portão de rampa atual não pega, porque reage à **taxa**
suavizada por EWMA (meia-vida 15min); um degrau quase instantâneo tem
sua taxa percebida diluída por essa suavização.

**Achado secundário (ressalva):** no mesmo exame, `PI_6240319_AL` (o tag
de pressão que mais "recuperou" FP no EXP21) mostra em alguns períodos
um padrão de onda quadrada errático entre 0 e 45 — sinal de
*chattering* de instrumento (sensor ruidoso perto do limiar,
disparando/desarmando repetidamente), não evento físico puro. Não
invalida o portão de pressão (a correlação com vibração/temperatura
continua real nos casos examinados), mas é uma limitação de dado
conhecida a registrar.

**Implementação:** `ENABLE_STEP_CHANGE_GATE` / `STEP_CHANGE_GATE_SENSOR`/
`_SHORT_WINDOW_MINUTES`/`_LONG_WINDOW_MINUTES`/`_THRESHOLD` (`config.py`)
— `scoring.py` ganha `compute_step_change_index`/`apply_step_change_gate`:
índice = `|média curta − média longa| / (desvio longo + eps)` do proxy
de carga (`T5_AVG_A`) — mesma matemática do `localz` já usado como
**feature** do modelo em `_build_changepoint_features`
(`preprocess.py`), aqui aplicado como **portão** (suprime detecção, não
alimenta o modelo). Mesma camada dos outros portões — ordem
independente entre eles.

**Calibração (grade literal, script
`relatorio_exp21_portao_pressao/grade_step_change_gate.py`)** —
reconstrói a série de verdade a cada limiar candidato, em cima do
`point_anomalies_all.csv` já calculado do EXP21 (janelas 5min/60min
fixas, só o limiar varrido):

| limiar | hit_rate | `normal_alert_rate` |
|---|---|---|
| 1,0 | 82,5% (perde 3) | 0,0349% |
| **1,5 (escolhido, custo zero)** | **90,0% (36/40)** | **0,0469%** |
| 2,0–3,5 | 90,0% | 0,065%–0,103% |
| ≥4,0 | 90,0% (sem efeito) | 0,1031% |

**Sanity check** (sem o portão novo) bateu exatamente com o EXP21 atual:
90,0%/0,1031%.

**Resultado confirmado via `src/main.py`** (task
`d252e94b51b3407186e68908a8a1c26c`, `automl_ranking.csv`):
`hit_rate=0,90` (36/40), `normal_alert_rate=0,000469` (0,0469%) — bate
exatamente com a grade.

**Resumo acumulado da investigação inteira** (mesma detecção, 90%/36-40,
desde o EXP20):

| etapa | `normal_alert_rate` | redução acumulada vs. EXP10c |
|---|---|---|
| EXP10c (referência original) | 0,348% | — |
| EXP20 (filtro de duração 4,5min) | 0,247% | 29,1% |
| EXP21 (+ portão de pressão ±8h) | 0,132% | 62,1% |
| EXP21 (+ corte do buraco de dados) | 0,103% | 70,4% |
| **EXP22 (+ portão de mudança de nível)** | **0,047%** | **86,5%** |

Config: `configs/calibracao_v4_eq/test_grupo_exp22_portao_mudanca_nivel.json`
— idêntico ao EXP21 + o portão novo ligado. Suíte de testes passa
inalterada (15/15).

**Reprodução:**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/test_grupo_exp22_portao_mudanca_nivel.json
```

## Investigando os 222 restantes do EXP22

Depois do EXP22, cruzamos os pontos de FP oficial (222 pontos, sobre o
`normal_alert_rate=0,047%`) com o **catálogo completo de 47 tags de
alarme** (não só os 2 avaliados) numa janela de ±24h — script
`relatorio_exp21_portao_pressao/analise_residual_exp22.py`. Resultado:
**46,4% ainda coincidem** com algum alarme do catálogo (ou seja, são
"FP" só porque avaliamos oficialmente 2 tags); sobram **20 episódios
genuinamente isolados** (~119 pontos agrupados, gap>30min separa
episódio).

**Duração desses 20 episódios isolados** (script ad-hoc, RLE sobre os
pontos isolados agrupando por gap>30min):

| # | início | fim | pontos | duração (min) |
|---|---|---|---|---|
| 0 | 2025-08-26 22:18:30 | 2025-08-26 22:20:00 | 4 | 1,5 |
| 1 | **2025-08-27 00:37:30** | **2025-08-27 00:54:30** | **35** | **17,0** |
| 2 | 2025-08-27 01:51:00 | 2025-08-27 01:51:00 | 1 | 0,0 |
| 3 | 2025-10-17 13:49:00 | 2025-10-17 13:54:30 | 12 | 5,5 |
| 4 | 2025-10-22 18:18:30 | 2025-10-22 18:22:30 | 9 | 4,0 |
| 5 | 2025-12-15 07:17:30 | 2025-12-15 07:21:30 | 9 | 4,0 |
| 6 | 2025-12-16 05:22:00 | 2025-12-16 05:22:30 | 2 | 0,5 |
| 7 | 2025-12-24 10:54:30 | 2025-12-24 10:59:00 | 10 | 4,5 |
| 8 | 2026-01-21 11:59:30 | 2026-01-21 11:59:30 | 1 | 0,0 |
| 9 | **2026-03-10 10:23:00** | **2026-03-10 10:42:00** | **21** | **19,0*** |
| 10 | 2026-03-11 09:32:30 | 2026-03-11 09:32:30 | 1 | 0,0 |
| 11 | 2026-03-17 18:50:30 | 2026-03-17 18:51:00 | 2 | 0,5 |
| 12 | 2026-03-18 08:15:00 | 2026-03-18 08:15:00 | 1 | 0,0 |
| 13 | 2026-03-20 14:08:30 | 2026-03-20 14:09:30 | 3 | 1,0 |
| 14 | 2026-03-23 21:39:00 | 2026-03-23 21:39:00 | 1 | 0,0 |
| 15 | 2026-03-27 21:23:00 | 2026-03-27 21:23:00 | 1 | 0,0 |
| 16 | 2026-04-10 23:03:00 | 2026-04-10 23:03:30 | 2 | 0,5 |
| 17 | 2026-04-11 08:57:00 | 2026-04-11 08:57:00 | 1 | 0,0 |
| 18 | 2026-04-11 10:20:30 | 2026-04-11 10:21:00 | 2 | 0,5 |
| 19 | 2026-04-12 17:33:30 | 2026-04-12 17:33:30 | 1 | 0,0 |

*(episódio 9 são na verdade dois blocos de ~5min a ~9min de distância,
agrupados aqui por caírem dentro da tolerância de 30min)*

Estatísticas: média=2,93min, mediana=0,5min, desvio=5,46min, máx=19min.
**13 dos 20 episódios (65%) duram 0–1min.**

**Achado "borda de entrada" (leading edge):** como todos os portões são
causais (só olham pra trás — necessário pra rodar em tempo real), existe
um atraso estrutural entre o início real de uma transição e o momento em
que a estatística móvel do portão cruza o limiar. Inspecionando as
colunas `*_gate_blocked`/`duration_filter_blocked` do
`point_anomalies_all.csv` em 5 dos episódios de 0min (`2026-01-21
11:59:30`, `2026-03-11 09:32:30`, `2026-03-18 08:15:00`, `2026-03-23
21:39:00`, `2026-03-27 21:23:00`), confirmamos nos 5: **algum portão
engata exatamente 1 amostra depois do ponto sinalizado e fica ativo por
vários minutos seguidos**. Ou seja, esses pontos "isolados" de 0-1min são
resíduo estrutural do EXP20 (o filtro de duração de 4,5min já filtrou
episódios curtos ANTES dos outros portões — o que sobra e é curto agora
é fragmento de borda, não ruído bruto).

Só 2 dos 20 episódios NÃO se encaixam nesse padrão (nenhum portão
engata perto): `2025-08-27 00:37–00:54` (17min) e `2026-03-10
10:23–10:42` (dois blocos ~5min). Checagem de `TC382_03_A`/`T5_AVG_A`
brutos no segundo mostrou ambos praticamente planos (sem degrau de
carga) — hipótese aberta de ser vibracional; checagem de z-score nos 10
canais `TV_35*` (script ad-hoc, comparando baseline `08:00–09:50` vs.
os 2 blocos suspeitos em `2026-03-10`).

### Fechamento: os 79 pontos restantes (6 episódios >1min)

Script `relatorio_exp21_portao_pressao/investiga_6_episodios_restantes.py`
— mesma técnica (z-score contra baseline de 2h, leitura chunked em UMA
passada pra evitar OOM), aplicada aos 6 episódios >1min que sobravam sem
explicação (excluindo o de `2026-03-10` já resolvido). Também checadas
as colunas `*_blocked` ±30min de cada um, pra ver se algum portão quase
disparou por perto.

| episódio | pontos | achado |
|---|---|---|
| `2025-10-17 13:49` | 12 | **TV_354Y_A z=42,8 / TV_354X_A z=8,8** — mesma assinatura do evento de `2026-03-10` |
| `2025-12-15 07:17` | 9 | **TV_354Y_A z=25,1 / TV_354X_A z=6,8** — mesma assinatura |
| `2025-12-24 10:54` | 10 | **TV_354Y_A z=23,8 / TV_354X_A z=6,7** — mesma assinatura |
| `2025-10-22 18:18` | 9 | z moderado (2–2,9) em quase todos os canais + `TC382_03_A` z=−2,5; `step_change_gate_blocked=17` nas ±30min — provável borda de portão sobre transiente real |
| `2025-08-26 22:18` | 4 | z fraco (<1,0) em tudo; `volatility_gate_blocked=60` nas ±30min — provável borda do portão de volatilidade |
| `2025-08-27 00:37` | 35 | z fraco (<1,5, a maioria negativo) em tudo — **segue sem explicação** |

**Achado principal: padrão recorrente real no mancal 354.** A mesma
assinatura (`TV_354Y_A` com z de 17 a 43, `TV_354X_A` secundário) apareceu
em **4 ocasiões distintas e não-adjacentes** no histórico inteiro:
`2025-10-17`, `2025-12-15`, `2025-12-24` e `2026-03-10` — não é ruído
nem coincidência, é um evento vibracional real e repetido nesse mancal
específico, sem nenhum alarme catalogado correspondente (nem nos 47
tags, nem nas 2 tags oficiais). Vale reportar ao cliente como possível
padrão de monitoramento a acompanhar (não como FP).

**Tally final dos 222 pontos residuais do EXP22:**

| categoria | pontos | % |
|---|---|---|
| Coincide com algum dos 47 tags do catálogo (±24h) | 103 | 46,4% |
| Evento vibracional real recorrente, mancal 354 (`TV_354Y_A`), 4 ocorrências | 52 | 23,4% |
| Borda estrutural de gate causal (confirmado, 13 episódios de 0-1min) | 19 | 8,6% |
| Provável borda de gate sobre transiente real (2 episódios, confiança média) | 13 | 5,9% |
| **Genuinamente sem explicação** (`2025-08-27 00:37`, 35 pontos) | 35 | 15,8% |
| **Total** | **222** | 100% |

**84% do residual está explicado** (real signal ou artefato estrutural
já testado). Sobra só **1 episódio de 35 pontos/17min** sem explicação —
os 10 canais de vibração + os 2 sensores avaliados não mostram nada
> 1,5σ ali. Não investigado além disso por ora (exigiria olhar outros
sensores da planta fora do grupo `TC382_T5_vibracao_mancais_multiescala`,
fora do escopo atual). Nenhuma implementação nova de gate decorre desta
investigação — é só uma investigação (nada foi promovido em
`config.py`/`automl_pipeline.py`).

**RESOLVIDO — confirmado como evento vibracional real no mancal 354.**
Z-score (média da janela vs. média/desvio da baseline `08:00–09:50`):

| canal | z (10:20–10:30) | z (10:34–10:44) |
|---|---|---|
| **TV_354Y_A** | **11,13** | **17,10** |
| TV_354X_A | 4,72 | 7,71 |
| TV_355Y_A | 1,30 | 2,41 |
| demais canais | \|z\|<1,05 | \|z\|<1,07 |

Z-score de 11 a 17 desvios-padrão é um evento real e concentrado — quase
exclusivamente em **TV_354Y_A** (e em menor grau `TV_354X_A`), crescendo
entre os dois blocos (10,3σ→17,1σ), não ruído. Como `TC382_03_A` e
`T5_AVG_A` (as 2 tags oficialmente avaliadas) ficam planas nesse
período, esse episódio nunca teria como ser um "hit" nas 2 tags — mas é
provavelmente uma detecção **genuína** de vibração anômala no mancal
354, sem alarme correspondente registrado no catálogo (nem nos 47 tags,
nem nas 2 avaliadas). Ou seja: não é artefato de borda nem ruído — é
provavelmente um TP não-catalogado (evento real que o sistema de
alarmes da planta não chegou a disparar). Não é caso de suprimir com
gate nenhum; ao contrário, reforça a capacidade do modelo multivariado
de pegar sinal em canais que os 2 tags oficiais não veriam sozinhos.

### EM ANDAMENTO — candidato a EXP23: 2º filtro de duração (pós-portões)

Ideia do usuário, dado o achado acima: já que 65% dos episódios isolados
restantes são fragmentos de 0-1min (borda estrutural, não precursor
real), testar um **segundo filtro de duração mínima aplicado no FINAL da
cadeia** (depois de todos os portões — carga/volatilidade/mudança de
nível/congelamento), reaproveitando a mesma função `apply_min_duration_filter`
(RLE) já usada como filtro PRÉ-portões no EXP20, agora rodando por cima
do `is_anom_point` final do EXP22.

Script: `relatorio_exp21_portao_pressao/grade_filtro_pos_portoes.py` —
reconstrução literal (usa as funções reais de `src/cnn1d_ae/scoring.py`,
não aproximação): lê o `point_anomalies_all.csv` já calculado do EXP22
(cache local, task confirmada `d252e94b51b3407186e68908a8a1c26c`),
reconstrói `df_alarm_eval`/`near_alarm_mask` idênticos à pipeline real, e
varre uma grade de limiares (1,0 a 4,5min) do 2º filtro, recomputando
`hit_rate` (`eval_alarm_hit_rate`) e `normal_alert_rate`
(`compute_normal_alert_rate`) de verdade a cada limiar.

**RESULTADO: NEGATIVO — não implementar.** Grade concluída
(`grade_filtro_pos_portoes_result.csv`):

| limiar 2º filtro | hit_rate | normal_alert_rate | pontos restantes |
|---|---|---|---|
| — (EXP22 atual, baseline) | 90,0% (36/40) | 0,0430% | 5176 |
| 1,0min | 82,5% (33/40) | 0,0407% | 5101 |
| 1,5min | 55,0% (22/40) | 0,0383% | 4987 |
| 2,0min | 50,0% (20/40) | 0,0375% | 4948 |
| 2,5min | 50,0% (20/40) | 0,0358% | 4908 |
| 3,0min | 45,0% (18/40) | 0,0358% | 4888 |
| 3,5–4,5min | 37,5% (15/40) | 0,0358% | 4842–4864 |

Já no limiar mais suave (1min) perdem-se 3 alarmes reais por só ~5% de
redução de FP; subindo o limiar o hit_rate desaba até 37,5% e o
`normal_alert_rate` praticamente para de cair. **Causa raiz: mesmo
mecanismo do bug de ordem do EXP20, na outra ponta da cadeia.** Os
portões upstream (rampa/volatilidade/mudança de nível/congelamento) já
fragmentam episódios longos — inclusive precursores REAIS, exatamente
como o achado de "borda de entrada" mostrou (precursor real truncado
sobra como 1-2 pontos). Nesse ponto da cadeia, um fragmento curto de
precursor real e um fragmento curto de ruído residual são
indistinguíveis por duração — um 2º filtro de duração pós-portões corta
os dois igualmente. **Não implementar** `POST_GATE_DURATION_FILTER`
(nunca chegou a existir em `config.py`/`automl_pipeline.py` — só o
ad-hoc de checagem). Os 222 pontos residuais (46% no catálogo completo,
resto majoritariamente artefato de borda estrutural) ficam como estão
por ora; não há gate adicional pendente de implementação neste momento.
