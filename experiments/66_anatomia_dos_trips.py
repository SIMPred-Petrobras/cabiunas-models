#!/usr/bin/env python3
"""EXP 66 — ANATOMIA DOS TRIPS, UM A UM: o alarme de nível de trip que disparou cada parada (first-out), a sequência, o precursor físico e o que o detector viu.

DEFINIÇÕES FIXADAS EM 24/10/2026, ANTES DE RODAR. DESCRITIVO (nada se adota). Roda no worker do ClearML (`scripts/worker_sota.py --exp ...`), tag `trips-anatomia`.
O QUE JÁ SE SABE (declarado; anatomia a 30 s de 21/10, física por evento de 20/10, as 7 tags de nível de trip de 24/08): 5 trips de mancal radial LNA, 2 de óleo súbitos,
27/02/2025 provável instrumento (PDIT), 29/04/2025 talvez parada controlada sem HIHI; 2 eventos fora do registro (27/02/2025 17:33 e 18/03/2025 11:16 UTC); as 7 tags de nível
de trip no diário (pressão muito baixa do header de óleo PI-0340, temperatura muito baixa do tanque TI-0325, PALL-6240309, temperatura muito alta dos mancais TI-0305/0303/0307,
ΔP muito alto do vazamento do selo PDIT-0305). Esta análise sistematiza isso numa regra única e acrescenta o que o detector viu em cada caso.
CASOS: os 9 trips do registro (UTC) e os 2 eventos fora dele (27/02/2025 17:33 e 18/03/2025 11:16 UTC).
DEFINIÇÕES:
  PARADA: o instante, na grade de 2 min, em que RUNNING_A > 0,5 vira falso, o mais próximo do horário do caso dentro de ±30 min.
  ALARME DE NÍVEL DE TRIP: ativação (ACT) do diário cuja descrição tem "Mt." (muito alta/muito baixa) ou "TRIP". FIRST-OUT = o primeiro deles em [parada − 15 min, parada + 2 min].
    O diário está em UTC (EXP 38, calibrado em 2025-2026); se não houver nenhum alarme nessa janela, procura-se em ±6 h e o deslocamento é relatado (fuso de 2024 não verificado).
  SEQUÊNCIA: todas as ativações em [parada − 15 min, parada + 5 min], com o minuto relativo à parada (causa antes, consequência depois).
  PRÉ-ALARME: ativações da MESMA tag do first-out em nível de alarme (descrição sem "Mt.") em [parada − 48 h, parada): a primeira e a contagem.
  PRECURSOR FÍSICO: a variável analógica do first-out (mancal -> a própria TI; selagem -> PDIT_0305; óleo -> PI_0339 e PI_0340; tanque -> TI_0325) na grade de 2 min: valor a
    48, 24, 6 e 1 h e 10 min antes, o extremo das 48 h, e o início da EXCURSÃO = primeiro instante a partir do qual a variável fica além de mediana ± 4 MAD (dos 30 d de operação
    estável antes da janela, no lado do trip) até a parada, só em operação.
  CONTEXTO: carga L (percentil do T5 em [parada − 2 h, parada] contra os 30 d estáveis antes; EXP 61); horas desde a última partida; manutenção (modo HSX >= 24 h, regra do
    pacote) terminada nos 7 d antes; horas parada depois (até RUNNING_A voltar).
  DETECTOR: nos ledgers do controle (atual, G168, B800), o primeiro episódio de alarme que intersecta [caso − 48 h, caso]: início, lead (censurado em 48 h), início dentro da
    janela e banda (>= 4 h); na base atual, os canais acesos no nascimento e as 3 tags que mais pesaram (H6, artefato da tarefa 6e11bc03).
  MODO: pelo first-out — mancal, selagem, óleo, tanque de óleo; sem first-out -> "sem alarme de trip" (parada não atribuível a proteção).
EXPECTATIVAS: parada a até 30 min do horário do registro nos 8 do alvo (~90%); first-out em >= 7 dos 8 (~80%); o 29/04 sem first-out de nível de trip (~55%); os 2 eventos fora do
registro com first-out e máquina operando (~75%); a excursão física começa > 1 h antes em <= 4 dos 8 (~65%: 17/03, 09/12 e os de óleo são súbitos).
VERIFICAÇÃO: V1 o diário tem first-out em [parada − 15 min, parada + 2 min] para os trips de 2025-2026 (fuso UTC); V2 os ledgers reproduzem as referências (7/8, 8/8, 8/8).

RESULTADO (24/10/2026, rodada LOCAL: o servidor do ClearML estava inacessível nesta máquina; repetir no worker quando voltar). Saídas em data_local/sota/anatomia_trips/.
DESVIOS DECLARADOS (correções do executor depois da 1ª rodada; as regras de parada, first-out, pré-alarme, precursor e contexto não mudaram):
  (1) FUSO DO DIÁRIO: até 2024 ele está em hora LOCAL; desde 02/2025, em UTC. Medido pelas paradas da grade com alarme a ±5 min: até 15/11/2024 só batem lendo o diário
      como local (15 contra 1), desde 03/02/2025 só como UTC (43 contra 3); entre as duas datas não há parada com alarme (corte adotado: 01/01/2025). Confirmação
      independente: o HI/HH da TI_0305 de 16/01/2024 no diário (08:23) coincide com a subida da TI_0305 no export de 30 s, que é hora local (cruza 120 °C às 08:23:30 e
      125 °C às 08:24:00), com a máquina parada desde 06:11.
  (2) INÍCIO E BANDA: o texto acima pedia o primeiro episódio que intersecta a janela; o protocolo da frente (`sota/protocolo.py`) conta qualquer episódio que nasce nela.
      O executor passou a seguir o protocolo; o lead continua sendo o do primeiro episódio (censurado em 48 h). Muda 27/02/2025 (início) e 26/02/2026 (início e banda).
  (3) CASO NOVO, FORA DO PRÉ-REGISTRO: 17/01/2024 07:56 local (10:56 UTC), achado com o relógio corrigido; entra como "fora do registro".
V1 FALHA só no 29/04/2025, o caso que a expectativa 3 previa sem first-out (os outros 7: first-out de −0,7 a +0,9 min da parada). V2 PASSA (atual 7/8 FP 9; G168 8/8 FP 14;
B800 8/8 FP 12).
PLACAR: parada a até 30 min nos 8 — CUMPRIDA (8/8, todas a 0 min); first-out em >= 7 dos 8 — CUMPRIDA (7/8); 29/04 sem first-out — CUMPRIDA; eventos fora do registro com
first-out e máquina operando — CUMPRIDA (3/3, com o novo); excursão > 1 h em <= 4 dos 8 — CUMPRIDA (2: 17/03 com 8,7 h e 07/04 com 6,6 h).
POR CASO (hora local; minutos relativos à parada; detector na base atual, salvo indicação):
  16/01/2024 06:10 (registro) — NÃO É TRIP. Nenhum alarme na parada; carga reduzida às 05:38 (T5 686 -> 647 °C), parada às 06:10, religada às 14:04. O registro tomou a
    data do HI/HH da TI_0305 lido como UTC; no relógio certo esse alarme vem 2,2 h DEPOIS da parada, com a máquina parada e a TI_0305 lendo 229 °C (artefato de máquina parada).
  17/01/2024 07:56 (fora do registro) — MANCAL LNA, súbito: TI_0305 77,7 °C às 07:36, 97,7 às 07:54, 122,6 na parada; muito alto +0,8 min; carga 0,99; 17,9 h após a partida;
    parada de 25 h. É o trip real de janeiro de 2024 (antes do período avaliado; não há ledger em 2024). O único pré-alarme é o artefato de 16/01.
  27/02/2025 05:38 — SELAGEM: PDIT_0305 muito alto −0,2 min; 2 pré-alarmes, o primeiro 1,6 h antes. Detector de pé desde 21/02 05:42 (140 h) e episódio NOVO às 04:04
    (1,6 h antes; p+vb; PDIT_0305 a 203x): início sim, banda não.
  27/02/2025 14:33 (fora do registro) — SELAGEM outra vez, 4,5 h após religar, em carga 0,01; 6 pré-alarmes desde 10,5 h antes; parada de 0,2 h.
  17/03/2025 15:16 — MANCAL LNA: muito alto −0,7 min; TI_0305 num platô de ~74 °C (base 70) nas 48 h; excursão de 8,7 h e subida final de 36 min. Detector só pelo episódio
    de 196,7 h nascido em 09/03 (p+sp+vb; TI_0305 pelo sp à frente): de pé, sem início.
  18/03/2025 08:16 (fora do registro) — MANCAL LNA: religada 3,1 h antes e já quente (91,6 °C 1 h antes); muito alto +0,2 min; o mesmo episódio de 09/03.
  07/04/2025 18:18 — MANCAL LNA: TI_0305 acima da base nas 48 h inteiras (81 -> 89 -> 97 °C; spread de 20-25 contra 14); excursão de 6,6 h. Detector nasce 06/04 10:10
    (32 h antes; t+sp+vb; vibração no top-3): início e banda. É o precursor mais longo e mais claro.
  11/04/2025 14:02 — MANCAL LNA: normal 48 h antes, 81,7 °C 6 h antes, 114,7 °C aos 10 min; muito alto +0,9 min. Detector nasce 09:56 (4,1 h antes; TV_351X_A a 9,0x): banda
    por 6 min.
  29/04/2025 00:04 — PARADA CONTROLADA, SEM ALARME DE TRIP: HI −22,4 min; carga reduzida às 23:50; parada com a TI_0305 a 124,3 °C (abaixo do HH de 125); HH +12,5 min, de calor
    residual; 564 h de manutenção depois. Subida de ~21 h (79 °C 6 h antes, 102 °C 1 h antes). Detector nasce 27/04 11:16 (37 h antes; t+vb).
  04/11/2025 03:22 — ÓLEO, súbito: PALL_6240309 TRIP +0,3 min, depois PI_0339 baixa e PI_0340 TRIP; pressão do header normal nas 48 h (mínimo a −1,2 MAD). A base atual
    NÃO detecta; G168 e B800 detectam com início e banda.
  09/12/2025 05:36 — MANCAL LNA: episódio nasce 08/12 02:32 numa subida de carga (27 h antes; t+sp+vb; vibração à frente); o spread da TI_0305 sobe de 15 para 22-23 °C a
    partir das 15:00; parada curta 20:04-20:14 sem alarme de trip (só ΔP baixo da linha de balanço, PDI_0302); depois de religar, o spread fica em 22-23 °C e não nasce
    episódio (refratário do episódio que fechou às 20:02); muito alto −0,1 min em carga baixa (0,22).
  26/02/2026 12:34 — ÓLEO, súbito: PI_0339 baixa −0,2 min, PALL_6240309 TRIP +0,3, PI_0340 TRIP +0,7. O header cai devagar de 3,213 para 3,157 em 21 h, mas o mês anterior
    foi mais baixo (menor média de 21 h: 3,096 contra 3,186). Detector de pé desde 29/01 (648 h) e episódio NOVO em 25/02 16:18 (20,3 h antes; t+p+sp), que nasce num pico
    ISOLADO do PDIT_0305 (1,11 contra 0,006-0,03) num degrau de carga (T5 691 -> 760 °C); picos iguais em 10/02 e 27/02. A detecção não tem relação física com o modo de falha.
LEITURA: dos 8 do alvo, 4 têm precursor físico que o detector pega dentro da janela (07/04, 11/04, 29/04 e 09/12: mancal, com a vibração à frente); 1 é selagem com
precursor de 1,6 h pego pelo p (27/02); 1 é mancal com sinal fraco e longo pego só pelo episódio de 8 dias (17/03); os 2 de óleo não têm precursor nas variáveis medidas (o
04/11 a base atual perde; o crédito do 26/02/2026 é coincidência com um episódio de transiente). O 29/04 não é trip de proteção: é parada controlada com o mancal a 124 °C.
O registro de 2024 aponta a parada errada (16/01 06:10 em vez de 17/01 07:56 local). Ele só alimenta a referência do vb, que apaga [trip − 7 d, trip + 2 d]; com
VB_HORAS_BASE = 400 h, o erro só toca a inferência do início de 2024, fora do período avaliado e da produção. A correção do `registro_trips.csv` fica para decisão.

Executor: `experiments/sota/anatomia_trips.py`.  Local: PYTHONPATH=src:experiments .venv/bin/python -m sota.anatomia_trips
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp anatomia_trips --tags trips-anatomia --enqueue --queue default
"""
