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
