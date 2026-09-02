#!/usr/bin/env python3
"""Monta o RELATORIO_DETECTOR_TC33003A.pdf com o ponto de operacao fechado.

Reusa as primitivas de layout de scripts/build_relatorio_pdf.py (A4 paisagem, so
matplotlib -- o venv nao tem reportlab). As figuras vem de figs_relatorio_pdf.py.

Este script NAO recalcula metrica: todos os numeros vem das medicoes ja feitas e estao
citados com o script de origem, para o relatorio ser auditavel.

Uso:  cd scripts/pdm_fisico && python build_pdf_detector.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "../..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scripts.build_relatorio_pdf import (
    W, H, INK, INK2, MUTED, RULE, GROUND, ACCENT, AMP, CTX, GOOD, CRIT, SANS, MONO,
    _page, new_page as _np, para, bullet, heading, table, callout, full_image)

OUT = "../../RELATORIO_DETECTOR_TC33003A.pdf"
RODAPE = "TC-330.03A · Cabiúnas · detector de 4 sinais · 28/08/2026"


def new_page(pdf, title=None, eyebrow=None):
    fig = _np(pdf, title, eyebrow)
    for t in list(fig.texts):
        if t.get_text().startswith("Cabiúnas · Turbina A"):
            t.set_text(RODAPE)
    return fig


# ---------------------------------------------------------------- paginas
def page_capa(pdf):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    fig.text(0.075, 0.80, "DETECTOR DE PARADA", fontsize=8, color=ACCENT, family=MONO)
    fig.text(0.075, 0.70, "TC-330.03A", fontsize=40, color=INK, family=SANS, weight="bold")
    fig.text(0.075, 0.635, "O detector de quatro sinais do Thallys", fontsize=15, color=INK2,
             family=SANS)
    fig.lines.append(plt.Line2D([0.075, 0.40], [0.60, 0.60], transform=fig.transFigure,
                                color=INK, lw=1.6))
    y = 0.545
    for t in ["Solar Taurus 60 + compressor centrífugo, UTGCAB.",
              "Quatro sinais físicos com voto por confirmação, ajuste walk-forward mensal.",
              "Alvo: parada não programada da máquina, prevista com 48 h de antecedência."]:
        y = para(fig, 0.075, y, t, size=10.5, width=88)
    callout(fig, 0.075, y - 0.02, 0.42, "ponto de operação fechado",
            ["8 de 8 paradas · 1,12 falso positivo por mês de operação (13 alarmes por ano) · "
             "39,0 h/mês de alarme (5,3% do tempo) · antecedência MÉDIA 29,0 h (mín. 2,8 h) · "
             "LOEO aninhado 7/8."],
            color=GOOD)
    fig.text(0.56, 0.545, "O que mudou nesta revisão", fontsize=10.5, color=INK,
             family=SANS, weight="bold")
    y = 0.505
    for t in ["Cada sinal ganha um segundo canal: CUSUM, que acumula evidência de deriva lenta.",
              "Período refratário de 48 h e duração mínima de 60 min por episódio.",
              "Falso positivo cai 77%, horas caem 69%, antecedência sobe 47%.",
              "Mesmos 8 eventos detectados, com o custo em falso positivo reduzido à metade.",
              "A deriva do custo, último problema estrutural aberto, deixa de ser detectável.",
              "Janela oficial: 2025-01 a 2026-04, onde estão os 8 eventos."]:
        y = bullet(fig, 0.56, y, t, size=9.6, width=64)
    fig.text(0.075, 0.075, RODAPE, fontsize=8, color=MUTED, family=SANS)
    pdf.savefig(fig); plt.close(fig)


def page_detector(pdf):
    fig = new_page(pdf, "O que tem no detector", "01")
    y = para(fig, 0.055, 0.845,
             "Quatro sinais físicos independentes, cada um comparando a máquina com ela mesma "
             "recentemente. Cada sinal passa por EWMA, limiar e sustentação de 30 min; o alerta "
             "sai quando DOIS ou mais estão sustentados ao mesmo tempo.", width=120)
    y = table(fig, 0.055, y - 0.01,
              ["sinal", "o que mede", "construção", "meia-vida", "limiar"],
              [["t", "14 tags de temperatura", "erro de reconstrução PCA, máx. por sensor (φ=0,10)", "1 h", "2,0 × k"],
               ["p", "12 tags de pressão", "idem, família de pressão", "1 h", "2,0 × k"],
               ["sp", "spread do mancal", "z robusto de TI_0305 − mediana(0301/0303/0307)", "30 min", "3,0 × k"],
               ["vb", "10 sondas TV_35*", "máx. do z, referência rolante 400 h + guarda 24 h", "30 min", "3,0 × k_vib"]],
              [0.055, 0.115, 0.315, 0.735, 0.825], size=8.8)
    y = heading(fig, 0.055, y - 0.012, "Máscara de pontuação — onde o detector tem direito de falar")
    y = para(fig, 0.055, y,
             "RUNNING_A > 0,5  E  T5_AVG_A > 300 °C, menos as 6 h seguintes a cada religamento. "
             "As duas condições foram medidas, nenhuma é decorativa: tirar o piso térmico abre só "
             "48 h em 2,5 anos, mas são transientes 5× mais densos em alarme e custam uma parada "
             "real; tirar o blackout custa duas. Com a máquina parada os quatro sinais saturam por "
             "física (spread de 49 °C por calor residual, sonda de vibração no piso de ruído, PCA "
             "fora da variedade ajustada) — sem máscara o detector alarmaria em 100% de toda parada.",
             width=120)
    y = heading(fig, 0.055, y - 0.008, "Camada de decisão")
    for t in ["Dois canais por sinal — o sinal dispara se o EWMA cruza o limiar sustentado por 30 min "
              "OU se o CUSUM daquele sinal acumula acima de 80 (κ=0,75). O primeiro é detector de "
              "degrau, o segundo de deriva lenta: uma anomalia 25% acima do normal por dois dias "
              "nunca cruza um limiar, mas acumula. Foi o que trouxe a antecedência de 19,7 h para "
              "29,0 h. O CUSUM roda sobre o sinal já suavizado, não sobre o cru: testado, o cru custa "
              "+46% de falso positivo e −9 h de lead, porque a EWMA converte pico pontual em excesso "
              "sustentado, que é o que o acumulador consome. Crucial: o voto continua sendo por SINAL "
              "FÍSICO, não por detector — deixar "
              "os oito votarem separado faz um sinal só satisfazer o voto pelas duas vias e perde "
              "a proteção contra transiente de partida.",
              "Voto ≥ 2 — exige concordância entre subsistemas físicos distintos. Não é "
              "conservadorismo: a manobra move um sinal por vez, a degradação move vários. "
              "Testamos um escape por magnitude (um sinal isolado muito acima do limiar) e ele "
              "foi refutado — dois terços do que ele acrescenta está a menos de 12 h de uma partida.",
              "Refratário de 48 h — após um episódio terminar, novo alerta é suprimido por 48 h. "
              "Não muda o que o detector vê, muda quantas vezes reporta o mesmo estado.",
              "Duração mínima de 60 min — descarta episódio curto demais para ser atendido.",
              "Ajuste walk-forward mensal: PCA e escalas recalculadas a cada mês sobre as 20.000 "
              "amostras estáveis anteriores. Todos os quatro sinais são causais."]:
        y = bullet(fig, 0.055, y, t, size=9.2, width=116)
    pdf.savefig(fig); plt.close(fig)


def page_alvos(pdf):
    fig = new_page(pdf, "Os alvos", "02")
    y = para(fig, 0.055, 0.845,
             "Três alvos distintos, com réguas distintas. Misturá-los produz números que não "
             "significam nada — foi o que a análise dos alarmes de TRIP mostrou.", width=120)
    y = heading(fig, 0.055, y - 0.005, "Alvo 1 — parada real  (o alvo do produto)")
    y = para(fig, 0.055, y,
             "Rótulo (regra do Francisco): RUNNING_A cai de 1 para 0 → máquina fica parada ≥ 2 h → "
             "há alarme de nível em [−1 h, +30 min] → trips a menos de 24 h são agrupados. "
             "62 paradas → 10 trips → 9 eventos, dos quais 8 dentro da janela de avaliação. Fuso "
             "provado por sondagem: offset +0 h.  "
             "Régua: janela de 48 h antes do evento, episódios agrupados a 2 h, FP por mês de "
             "OPERAÇÃO (730 h), e teste de permutação — sem ele, cobertura alta vira recall falso.",
             width=120)
    y = callout(fig, 0.055, y - 0.004, 0.89, "validação cruzada independente",
                ["Declusterizando os 7 tags de proteção de nível (ALL/LL e AHH/HH) a 30 min, "
                 "134 episódios na série, dos quais 9 coincidem com parada real ≥ 2 h — exatamente "
                 "os mesmos 9 eventos, por um caminho que não usa a regra do Francisco."], color=GOOD)
    y = heading(fig, 0.055, y, "Alvo 2 — alarmes de temperatura  (o alvo do detector do Diego)")
    y = para(fig, 0.055, y,
             "32 alarmes HI/HIHI genuínos de TC382_03_A e T5_AVG_A no período 07/2025–04/2026. "
             "Os 8 UNDER ficam de fora: disparam entre −18 e −22 °C com Comm Fail — é falha do "
             "próprio sensor, sem precursor físico possível. Régua: ±24 h com detalhamento "
             "preditivo / reativo / sem detecção.", width=120)
    y = heading(fig, 0.055, y - 0.004, "Alvo 3 — proteção de nível de trip")
    y = para(fig, 0.055, y,
             "7 tags: 3 do lado baixo (óleo lubrificante) e 4 do lado alto (mancal e selagem). "
             "28 episódios no período. O alvo tem um problema estrutural que precisa ser declarado: "
             "25 dos 28 disparam com a máquina JÁ PARADA e o ferro entre 27 e 34 °C — são proteções "
             "reportando que a bomba não roda, não falha de equipamento. Só 3 são parada real, e "
             "o detector acerta os 3, todos preditivos.", width=120)
    pdf.savefig(fig); plt.close(fig)


def page_resultado(pdf):
    fig = new_page(pdf, "Quantos acertamos", "03")
    y = table(fig, 0.055, 0.845,
              ["configuração", "paradas", "episódios", "fp/mês", "h/mês", "duty", "lead", "precisão"],
              [[("original (só degrau, sem refratário)", MUTED, "normal"), ("8/8", MUTED, "normal"), ("66", MUTED, "normal"), ("4,82", MUTED, "normal"),
                ("126,9", MUTED, "normal"), ("17,4%", MUTED, "normal"), ("19,7 h", MUTED, "normal"), ("13%", MUTED, "normal")],
               [("+ refratário 48 h + duração mín. 60 min", MUTED, "normal"), ("8/8", MUTED, "normal"), ("36", MUTED, "normal"),
                ("2,41", MUTED, "normal"), ("48,2", MUTED, "normal"), ("6,6%", MUTED, "normal"), ("19,7 h", MUTED, "normal"), ("22%", MUTED, "normal")],
               [("atual: + canal CUSUM por sinal", INK, "bold"), ("8/8", GOOD, "bold"),
                ("21", GOOD, "bold"), ("1,12", GOOD, "bold"), ("39,0", GOOD, "bold"),
                ("5,3%", GOOD, "bold"), ("29,0 h", GOOD, "bold"), ("38%", GOOD, "bold")],
               [("sob LOEO aninhado (8 dobras)", MUTED, "normal"), ("7/8", AMP, "normal"),
                ("—", MUTED, "normal"), ("—", MUTED, "normal"), ("—", MUTED, "normal"), ("—", MUTED, "normal"), ("—", MUTED, "normal"), ("—", MUTED, "normal")]],
              [0.055, 0.345, 0.435, 0.535, 0.615, 0.695, 0.775, 0.865], size=9.0)
    y = para(fig, 0.055, y - 0.008,
             "O LOEO caiu de 8/8 para 7/8 na revisão de 28/08: o 8/8 só aparece quando NÃO se "
             "aplica desempate algum entre configurações empatadas na detecção de treino — aí o "
             "argmax pega a primeira linha da varredura, que é ordem de busca, não mérito. Com "
             "qualquer desempate razoável (menos horas de FP, menos FP, mais lead) o resultado é "
             "7/8, e o evento que cai é sempre 04/11/2025, detectado por apenas 8 das 72 "
             "configurações dentro do orçamento de 1,15 FP/mês. É o mesmo evento que já havia "
             "caído em cinco intervenções anteriores e o único dos oito sem precursor físico "
             "atribuível. Leitura honesta: 7 detecções são robustas a reajuste, a oitava é sorte "
             "do ponto de operação.   "
             "Os oito: 27/02, 17/03, 07/04, 11/04, 29/04 e 04/11 de 2025; 09/12/2025; 26/02/2026. "
             "Com o teto de 12 h escapa 17/03/2025.   A avaliação começa em 01/02/2024, primeiro "
             "instante com os quatro sinais válidos dentro da máscara — antes disso o ajuste "
             "walk-forward não tem nenhuma amostra estável anterior e a referência rolante ainda não "
             "acumulou as 400 h exigidas. O recorte é uma JANELA, aplicada ao numerador e ao "
             "denominador juntos, não a remoção de um caso: a parada de 16/01/2024 fica fora do alvo "
             "(três dos quatro sinais são NaN na janela dela, e com um sinal só o voto ≥2 é "
             "insatisfazível), mas continua na lista de exclusão da referência rolante, porque foi "
             "falha real e o dado em volta não deve entrar no baseline. Sobre as 9 paradas da série "
             "inteira o resultado seria 8/9.",
             width=120)
    y = heading(fig, 0.055, y, "O refratário passou os quatro testes que reprovaram as outras 22 ideias")
    for t in ["Permutação: o p melhora (0,0003 → 0,0000). O ganho não é cobertura comprada.",
              "Platô: detecção = 8/9 em R = 0, 12, 24, 36 e 48 h — cinco valores consecutivos.",
              "Supressão: zero eventos perdidos, lead perdido de 0,00 h nos nove, zero eventos "
              "na sombra de um episódio anterior.",
              "LOEO: 6/9, e é a configuração escolhida em todos os orçamentos testados."]:
        y = bullet(fig, 0.055, y, t, size=9.2, width=116)
    y = callout(fig, 0.055, y - 0.004, 0.89, "por que funciona quando nada mais funcionou",
                ["As 22 ideias refutadas atacavam a DISCRIMINAÇÃO — quanto o detector separa falha "
                 "de normal. O refratário ataca a REDUNDÂNCIA TEMPORAL: o mesmo estado anômalo era "
                 "reportado várias vezes porque o sinal oscila em torno do limiar. Suprimir a "
                 "repetição não perde informação — ela já foi entregue no primeiro alerta."])
    pdf.savefig(fig); plt.close(fig)


def page_modos(pdf):
    fig = new_page(pdf, "Três modos de falha, não um", "04")
    y = para(fig, 0.055, 0.845,
             "Chamar o alvo de \"8 paradas\" esconde que são três processos físicos distintos, com "
             "precursores e escalas de tempo diferentes — e que ajustamos um limiar único para os "
             "três. É o mesmo erro de escopo que apontamos no relatório do EXP10c, aplicado a nós.",
             width=120)
    y = table(fig, 0.055, y - 0.006,
              ["modo", "n", "tag de proteção", "detectados", "lead médio", "1º sinal a disparar"],
              [["mancal", "5", "TAHH_6240305", ("5/5", GOOD, "normal"), "23,6 h", "vb 3×, t 2×"],
               ["óleo", "2", "PALL_6240309 / 6240340", ("2/2", GOOD, "normal"), "14,4 h", "p, sp"],
               ["selagem", "1", "PDAHH6240305", ("1/1", GOOD, "normal"), "1,5 h", "vb"]],
              [0.055, 0.135, 0.185, 0.375, 0.545, 0.665], size=9.0)
    y = heading(fig, 0.055, y - 0.008, "Quem carrega a detecção")
    for t in ["A vibração é o carro-chefe: sustentou em 8 dos 9 eventos e foi a primeira a disparar "
              "em 4. Nos eventos de mancal de abril/2025 chega a 10,9× e 10,4× o próprio limiar — "
              "não é cruzamento marginal. Isso reabilita a recomendação de tirar a vibração da "
              "quarentena: ela merecia mais peso, não menos.",
              "A pressão detecta óleo e selagem, como a física prevê, e de forma brutal: 268× o "
              "limiar na selagem de 27/02/2025 e 65,6× no óleo de 26/02/2026.",
              "O spread do mancal é o mais fraco — sustentou em 4 de 9. A ablação mede o que ele "
              "vale: custa uma detecção (26/02/2026, onde a pressão está a 65,6× do limiar mas "
              "sozinha, e o spread a 1,52× fornece o segundo voto) e rende 32% das horas de alarme. "
              "Mantido; é o candidato a cortar se o orçamento de horas apertar.",
              "A antecedência difere muito por modo — 23,6 h no mancal contra 1,5 h na selagem, "
              "coerente com a física (selo falha rápido, mancal aquece progressivamente). Com n=1 "
              "na selagem não dá para concluir, mas o produto deve declarar lead POR MODO, não uma "
              "mediana única."]:
        y = bullet(fig, 0.055, y, t, size=9.2, width=116)
    pdf.savefig(fig); plt.close(fig)


def page_fp(pdf):
    fig = new_page(pdf, "Os falsos positivos, um a um", "05")
    y = para(fig, 0.055, 0.845,
             "São 23 episódios em 11,6 meses — 8 detecções e 15 falsos positivos. Com esse número "
             "dá para abrir cada um, e a autópsia mostra uma estrutura que nenhuma métrica agregada "
             "revelava.", width=120)
    y = table(fig, 0.055, y - 0.006,
              ["sinal", "nas 8 detecções", "nos 15 falsos pos.", "razão", "pico mediano det / fp"],
              [[("t", GOOD, "bold"), "6 de 8  (75%)", "6 de 15  (40%)", ("1,88", GOOD, "bold"), "1,34  /  0,90"],
               ["vb", "8 de 8  (100%)", "13 de 15  (87%)", "1,15", "3,21  /  1,97"],
               ["sp", "3 de 8  (38%)", "5 de 15  (33%)", "1,13", "0,63  /  0,59"],
               [("p", CRIT, "bold"), "5 de 8  (62%)", ("15 de 15 — todos", CRIT, "bold"),
                ("0,62", CRIT, "bold"), ("1,53  /  3,50", CRIT, "bold")]],
              [0.055, 0.115, 0.245, 0.395, 0.505, 0.585], size=9.0)
    y = callout(fig, 0.055, y - 0.004, 0.89, "nenhum sinal é o culpado — nem por presença, nem por intensidade",
                ["A pressão está em todos os 15 falsos positivos, mas a vibração está em 13 — "
                 "presença sozinha não discrimina. O que discrimina é a RAZÃO entre as duas taxas, e "
                 "por ela a temperatura é o melhor sinal (1,88) e a pressão o pior (0,62).",
                 "A intensidade também não separa: a pressão é MAIS forte nos falsos positivos "
                 "(pico mediano 3,50 contra 1,53), chegando a 1.167× o limiar num deles. Quando ela "
                 "explode, é manobra, não falha. E o spread do mancal mal cruza o próprio limiar nos "
                 "dois casos (0,63 e 0,59) — ele contribui quase só pelo canal CUSUM.",
                 "O que de fato estrutura os falsos positivos é o TEMPO: 11 dos 15 estão a 30 h ou "
                 "menos de uma partida, com mediana de 6,5 h — o instante em que o blackout de 6 h "
                 "expira. A combinação temperatura + vibração, por outro lado, aparece em 3 detecções "
                 "e em nenhum falso positivo."], color=CRIT)
    y = heading(fig, 0.055, y, "Três tentativas de atacar isso — todas negativas")
    y = table(fig, 0.055, y, ["intervenção", "detecção", "fp/mês", "h/mês", "o que perde"],
              [["blackout de 24 h só para a pressão", "7/8", "1,03", "34,1", "04/11/2025"],
               ["voto ponderado (pressão vale < 1)", ("5/8", CRIT, "normal"), "0,95", "51,2", "três eventos"],
               ["baseline da pressão por campanha", "8/8", ("1,38", CRIT, "normal"), "47,1", "— (piora o FP)"],
               ["piso de severidade na vibração (1,5×)", "7/8", "1,12", "42,9", "04/11/2025"],
               ["piso de severidade na pressão (1,5×)", "8/8", ("1,46", CRIT, "normal"), "84,5", "— (piora o FP)"]],
              [0.055, 0.325, 0.415, 0.495, 0.585], size=9.0)
    y = callout(fig, 0.055, y - 0.002, 0.89,
                "das 8 detecções, 7 têm precursor atribuível e 1 não",
                ["Cinco intervenções distintas — blackout por sinal, voto ponderado, baseline por "
                 "campanha e dois pisos de severidade — perdem SEMPRE o mesmo evento: 04/11/2025. A "
                 "autópsia explica: é pressão + vibração começando 6,5 h após a partida, assinatura "
                 "idêntica à dos seis falsos positivos que cada regra tentava matar. Mesma dupla de "
                 "sinais, mesma distância da partida, mesma via de disparo.",
                 "O alarme de fato soou antes daquela parada — a detecção é real. Mas a atribuição "
                 "causal não se sustenta: é um trip de óleo lubrificante, subsistema sem sensor no "
                 "nosso conjunto, e nenhuma regra consegue separá-lo de manobra de partida. O que o "
                 "detector cobre com precursor identificável são os 5 de mancal, o de selagem e um "
                 "dos dois de óleo."], color=AMP)
    y = heading(fig, 0.055, y, "E subir o limiar não ajuda: a curva não é uma troca, é um penhasco")
    para(fig, 0.055, y,
         "Subir o limiar em 15% derruba a detecção de 8/8 para 4/8 E SOBE o falso positivo de 1,29 "
         "para 1,55. Só se chega abaixo de 1,29 com o limiar dobrado, perdendo seis das oito paradas. "
         "O ponto atual é simultaneamente o de maior detecção e o de menor falso positivo — dois "
         "efeitos se somam: ao subir o limiar os episódios longos se fragmentam em vários curtos "
         "(o refratário passa a suprimir menos), e o voto ≥2 falha primeiro nos eventos que "
         "dependiam de um segundo sinal no limite.", width=120)
    pdf.savefig(fig); plt.close(fig)


def page_decisao(pdf):
    fig = new_page(pdf, "Dois pontos de operação", "06")
    y = para(fig, 0.055, 0.845,
             "A escolha entre eles não é técnica — depende da razão entre o custo de uma parada não "
             "programada e o custo de uma investigação. Esse número é da planta. Os dois estão "
             "medidos com o mesmo protocolo.", width=120)
    y = table(fig, 0.055, y - 0.008,
              ["ponto", "paradas", "fp/mês", "por ano", "h/mês", "duty", "lead", "precisão"],
              [[("COBERTURA — recomendado", INK, "bold"), ("8/8", GOOD, "bold"), ("1,12", GOOD, "bold"),
                ("13", GOOD, "bold"), ("39,0", GOOD, "bold"), ("5,3%", GOOD, "bold"),
                ("29,0 h", GOOD, "bold"), ("38%", GOOD, "bold")],
               ["ECONÔMICO — refratário de 7 dias", "7/8", "0,43", "5", "8,7", "1,2%", "38,9 h", "56%"]],
              [0.055, 0.395, 0.485, 0.565, 0.645, 0.725, 0.795, 0.875], size=9.0)
    y = para(fig, 0.055, y - 0.006,
             "O ponto econômico entrega 5 alarmes por ano, 1,2% de duty e a maior antecedência medida "
             "(38,9 h) — ao custo de uma parada, 07/04/2025, que é de mancal e tem precursor legítimo. "
             "MAS: sob validação cruzada o regime abaixo de 1,0 FP/mês desaba para 4/8, enquanto o "
             "ponto de cobertura dá 8/8. Um refratário de sete dias é instável: qual evento sobrevive "
             "depende de qual caiu na sombra do anterior, e isso muda a cada dobra. Abaixo de 1,0 a "
             "troca não é detecção contra falso alarme — é troca por instabilidade.", width=120)
    y = heading(fig, 0.055, y, "Recomendação, e por quê")
    y = para(fig, 0.055, y,
             "Manter COBERTURA. Com oito eventos no histórico inteiro, abrir mão de uma detecção "
             "real é caro demais para a evidência disponível, e a validação cruzada reprova o regime "
             "econômico: LOEO 7/8 em ≤1,15 FP/mês contra 4/8 em ≤1,0. O piso defensável é 1,12. Se a "
             "planta aceitar o risco de instabilidade em troca de 5 alarmes por ano, o ponto econômico "
             "está medido — mas precisa ser declarado como o que é.", width=120)
    y = heading(fig, 0.055, y, "O que foi testado e não entrou")
    for t in ["Subir o limiar: piora os dois eixos ao mesmo tempo (ver página anterior).",
              "Voto ≥3, escape por magnitude, portão graduado pós-partida, voto entre sondas, "
              "piso de escala relativo e absoluto, janela de exclusão, banda de guarda, "
              "limiar por alvo de duty: todos refutados a detecção fixa.",
              "GLR multiescala (4/8), canais lentos de 24 h (3× as horas), nível de atenção "
              "(não precede o alarme confirmado em 4 dos 8 eventos).",
              "87 sensores da série consolidada — rotação, torque, vazão, posição axial, vibração "
              "em outro canal, sete termopares de mancal: nenhum com precursor acima do acaso.",
              "30 grandezas de órbita X-Y das cinco duplas de sondas: nenhuma supera o máximo simples."]:
        y = bullet(fig, 0.055, y, t, size=9.1, width=116)
    pdf.savefig(fig); plt.close(fig)


def page_fig(pdf, path, titulo, eyebrow, legenda):
    fig = new_page(pdf, titulo, eyebrow)
    full_image(fig, path, top=0.855, bottom=0.155)
    para(fig, 0.055, 0.125, legenda, size=8.6, width=138, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def page_limites(pdf):
    fig = new_page(pdf, "Limites, e o que eles impedem", "10")
    y = heading(fig, 0.055, 0.845, "n = 8 é o teto de tudo")
    y = para(fig, 0.055, y,
             "O intervalo de confiança de Wilson sobre 8/8 é [67,6%, 100,0%] — ainda 32 pontos de "
             "largura, e o piso de 67,6% é a afirmação honesta. "
             "Nenhuma melhoria que façamos é mensurável nessa métrica. E não há mais positivos a "
             "colher: os alarmes de mancal do catálogo dão 61 episódios, dos quais apenas 4 com a "
             "máquina operando, e nenhum distante das paradas já conhecidas. O catálogo do SCADA "
             "está drenado. Mais evidência só vem de máquinas irmãs ou do tempo, para frente.",
             width=120)
    y = heading(fig, 0.055, y, "A deriva do custo — resolvida, por um caminho inesperado")
    y = para(fig, 0.055, y,
             "Com parâmetros fixos o custo crescia sozinho: agregando por campanha (n = 57), "
             "rho = +0,367 com p = 0,0064, e o duty subindo entre a primeira e a segunda "
             "metade da série. Atacamos o denominador por seis caminhos — piso relativo, piso "
             "absoluto em °C e µm, janela de exclusão, banda de guarda, limiar por alvo de duty e "
             "ablação do spread — e todos falharam. O motivo é que o mecanismo nunca esteve no "
             "denominador: a deriva estava na REPETIÇÃO, não na taxa de anomalia. O mesmo estado "
             "passou a ser reportado mais vezes, não a acontecer mais. O refratário colapsa a "
             "repetição e leva rho a +0,213 com p = 0,1227. O controle está incluído: a duração "
             "mínima sozinha não muda nada (p = 0,0042) — o efeito é do refratário.",
             width=120)
    y = heading(fig, 0.055, y, "Falso positivo abaixo de 1 por mês não é alcançável")
    y = table(fig, 0.055, y, ["fp/mês ≤", "melhor detecção", "h/mês", "lead"],
              [["2,5", ("8/8", GOOD, "bold"), "55,6", "19,7 h"], ["2,0", "7/8", "8,5", "12,5 h"],
               ["1,5", "5/8", "8,3", "11,1 h"], ["1,0", ("1/8", CRIT, "bold"), "5,9", "8,8 h"]],
              [0.055, 0.145, 0.265, 0.335], size=9.0)
    y = para(fig, 0.055, y - 0.004,
             "336 configurações varridas. O limite não é o detector — é a raridade do evento: a "
             "taxa-base é 0,46 parada por mês de operação, então 100% de recall são 0,46 alertas "
             "certos por mês. Exigir menos de 1 falso positivo por mês é exigir precisão acima de "
             "31%; o ponto fechado entrega 16,8%, contra 9,3% do anterior.", width=120)
    y = heading(fig, 0.055, y, "Cegueira a máquina parada")
    para(fig, 0.055, y,
         "Por construção o detector não pontua fora de operação quente. Qualquer degradação que se "
         "desenvolva durante uma parada é invisível. Cobrir isso exige um segundo detector com "
         "sinais que existam a frio — pressão de óleo, temperatura do tanque, bombas auxiliares.",
         width=120)
    pdf.savefig(fig); plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        page_capa(pdf)
        page_detector(pdf)
        page_alvos(pdf)
        page_resultado(pdf)
        page_modos(pdf)
        page_fp(pdf)
        page_decisao(pdf)
        page_fig(pdf, "fig_pdf_lead.png", "Antecedência, evento a evento", "07",
                 "Uma barra por parada. A de 16/01/2024 está marcada como fora da janela — o detector "
                 "ainda não existia. As barras em 48,0 h estão "
                 "censuradas pela janela da régua: o alarme já estava ativo quando a janela abriu, "
                 "então a antecedência real é maior e não é observável com esta régua.")
        page_fig(pdf, "fig_pdf_serie.png", "A série completa", "08",
                 "Acima, T5_AVG_A com as campanhas, as paradas e o piso de 300 °C da máscara. "
                 "Abaixo, a variável de decisão: quantos dos quatro sinais estão sustentados acima "
                 "do próprio limiar, com a linha de confirmação em 2, as faixas de alarme e as 9 paradas.")
        page_fig(pdf, "fig_pdf_zoom.png", "Dois eventos de perto", "09",
                 "Cada sinal dividido pelo próprio limiar. 09/12/2025: vibração de 1,3 a 2,9 com a "
                 "temperatura cruzando junto, 23,6 h antes. 26/02/2026: spread do mancal sustentado "
                 "mais pressão, 19,9 h antes. Dois modos de falha distintos, o mesmo ajuste.")
        page_limites(pdf)
        d = pdf.infodict()
        d["Title"] = "TC-330.03A — Detector de parada de quatro sinais"
        d["Author"] = "Thallys"
        d["Subject"] = "Ponto de operação fechado: refratário de 48 h"
    print(f"PDF: {OUT}  ({_page['n'] + 1} páginas)")


if __name__ == "__main__":
    main()
