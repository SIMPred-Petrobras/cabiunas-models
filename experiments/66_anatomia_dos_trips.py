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

Executor: `experiments/sota/anatomia_trips.py`.  Local: PYTHONPATH=src:experiments .venv/bin/python -m sota.anatomia_trips
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp anatomia_trips --tags trips-anatomia --enqueue --queue default
"""
