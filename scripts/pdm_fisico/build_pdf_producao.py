#!/usr/bin/env python3
"""Monta RELATORIO_PRODUCAO_TC33003A.pdf -- como chegamos ao 8/8 a 0,517 FP/mes.

Relatorio de engenharia do ponto de operacao de producao: a regua, a arquitetura,
cada decisao com a medicao que a sustenta, o que foi refutado e os limites conhecidos.

Reusa as primitivas de layout de scripts/build_relatorio_pdf.py (A4 paisagem, so
matplotlib -- o venv nao tem LaTeX, pandoc nem reportlab).

Este script NAO recalcula metrica. Todo numero vem de medicao ja feita nesta linha de
trabalho e esta citado com o script de origem, para o relatorio ser auditavel.

Uso:  cd scripts/pdm_fisico && python build_pdf_producao.py
"""
from __future__ import annotations
import os, sys, subprocess
sys.path.insert(0, "../..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scripts.build_relatorio_pdf import (
    W, H, INK, INK2, MUTED, RULE, GROUND, ACCENT, AMP, CTX, GOOD, CRIT, SANS, MONO,
    new_page as _np, para, bullet, heading, table, callout, full_image)

OUT = "../../RELATORIO_PRODUCAO_TC33003A.pdf"
FIG = "fig_nosso_estilo_francisco.png"


def git(*args, default="?"):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd="../..", timeout=10).stdout.strip() or default
    except Exception:
        return default


BRANCH = git("rev-parse", "--abbrev-ref", "HEAD")
COMMIT = git("rev-parse", "--short", "HEAD")
COMMIT_DT = git("log", "-1", "--format=%ad", "--date=short")
RODAPE = (f"TC-330.03A · Cabiúnas · detector físico de 4 sinais · "
          f"branch {BRANCH} @ {COMMIT} · 01/09/2026")


def new_page(pdf, title=None, eyebrow=None):
    fig = _np(pdf, title, eyebrow)
    for t in list(fig.texts):
        if t.get_text().startswith("Cabiúnas · Turbina A"):
            t.set_text(RODAPE)
    return fig


def fecha(pdf, fig):
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ================================================================== 1. capa
def page_capa(pdf):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.075, 0.83, "RELATÓRIO DE ENGENHARIA", fontsize=8, color=ACCENT, family=MONO)
    fig.text(0.075, 0.745, "Detector físico de falhas", fontsize=30, color=INK,
             family=SANS, weight="bold")
    fig.text(0.075, 0.675, "TC-330.03A — como chegamos ao 8/8", fontsize=30,
             color=ACCENT, family=SANS, weight="bold")
    fig.lines.append(plt.Line2D([0.075, 0.50], [0.635, 0.635],
                                transform=fig.transFigure, color=INK, lw=2.0))
    y = para(fig, 0.075, 0.585,
             "Antecipação de trips por quatro sinais físicos com voto cruzado, "
             "PCA walk-forward mensal e canal duplo degrau/CUSUM. Este documento "
             "registra a régua, cada decisão de projeto com a medição que a "
             "sustenta, o que foi testado e refutado, e os limites conhecidos "
             "do resultado.", size=11, width=86, leading=0.028)

    caixa = [("8/8", "eventos-alvo detectados", GOOD),
             ("0,517", "falsos positivos / mês", ACCENT),
             ("7,15 h", "por mês em alarme falso", ACCENT),
             ("29,0 h", "antecedência média", CTX)]
    x = 0.075
    for val, rot, cor in caixa:
        fig.patches.append(plt.Rectangle((x, 0.30), 0.195, 0.135,
                                         transform=fig.transFigure, facecolor=GROUND,
                                         edgecolor=RULE, lw=0.8))
        fig.patches.append(plt.Rectangle((x, 0.30), 0.0035, 0.135,
                                         transform=fig.transFigure, facecolor=cor,
                                         edgecolor="none"))
        fig.text(x + 0.016, 0.375, val, fontsize=21, color=INK, family=SANS, weight="bold")
        fig.text(x + 0.016, 0.335, rot, fontsize=8.5, color=MUTED, family=SANS)
        x += 0.215

    fig.text(0.075, 0.20, "JANELA DE AVALIAÇÃO", fontsize=7.2, color=MUTED,
             family=MONO, weight="bold")
    fig.text(0.075, 0.168, "01/01/2025 a 30/04/2026 · 11,6 meses de operação",
             fontsize=10, color=INK2, family=SANS)
    fig.text(0.40, 0.20, "CÓDIGO", fontsize=7.2, color=MUTED, family=MONO, weight="bold")
    fig.text(0.40, 0.168, f"branch {BRANCH} @ {COMMIT} ({COMMIT_DT})",
             fontsize=10, color=INK2, family=MONO)
    fig.text(0.40, 0.138, "scripts/pdm_fisico/publica_clearml.py",
             fontsize=9, color=MUTED, family=MONO)
    fig.text(0.075, 0.075, "Projeto Cabiúnas · UTGCAB / Petrobras · 1 de setembro de 2026",
             fontsize=8.5, color=MUTED, family=SANS)
    fecha(pdf, fig)


# ======================================================= 2. sumário executivo
def page_sumario(pdf):
    fig = new_page(pdf, "Sumário executivo", "01")
    y = 0.83
    y = para(fig, 0.055, y,
             "O detector antecipa os 8 trips catalogados do TC-330.03A na janela de "
             "avaliação, com 0,517 falso positivo por mês e 7,15 horas mensais em alarme "
             "falso. A antecedência média é de 29,0 horas e a mínima de 2,8 horas. Todos "
             "os 8 são antecipação genuína: a janela de avaliação é [t−48h, t], "
             "estritamente anterior ao evento, e nenhum acerto é contado depois do trip.")
    y -= 0.006
    y = heading(fig, 0.055, y, "As três decisões que produziram o resultado")
    y = bullet(fig, 0.055, y,
               "Quatro sinais físicos com voto cruzado, em vez de um modelo sobre os 39 "
               "sensores. Cada sinal isolado é fraco — o spread de mancal sozinho detecta "
               "1 de 8, a pressão sozinha 2 de 8. A força está na exigência de dois sinais "
               "simultâneos, que é também o que protege de transiente de partida.")
    y = bullet(fig, 0.055, y,
               "Reajuste mensal do PCA (walk-forward, só com dados anteriores ao mês). "
               "Congelar o modelo custa 2 detecções e multiplica as horas de alarme falso "
               "por 15 — de 7,15 para 108,19 h/mês.")
    y = bullet(fig, 0.055, y,
               "Canal duplo por sinal: degrau sustentado por 30 min OU CUSUM acumulado. "
               "O CUSUM foi a adição de maior ganho isolado do projeto — reduziu falso "
               "positivo em 67% e aumentou a antecedência em 54%.")
    y -= 0.010
    y = heading(fig, 0.055, y, "O que este relatório também registra")
    y = para(fig, 0.055, y,
             "Um ponto de operação só é defensável se as alternativas foram medidas. As "
             "seções 7 e 8 listam quinze linhas de investigação testadas e refutadas — "
             "incluindo seis tentativas independentes de separar a classe de falso "
             "positivo que nasce na borda do blackout de partida, todas malsucedidas pela "
             "mesma razão física. A seção 9 registra os limites: com 8 eventos, o "
             "intervalo de confiança do custo é [0,53; 1,81], e o gargalo do projeto "
             "passou a ser o rótulo, não o modelo.")
    y -= 0.004
    callout(fig, 0.055, y, 0.89, "leitura rápida para quem vai decidir",
            ["O detector está no ponto de recall máximo: 8 de 8, ao custo de pouco mais "
             "de meio alarme falso por mês. Existe um ponto alternativo mapeado e "
             "reversível por uma constante (blackout de 12h) que entrega 5 de 8 a 0,178 "
             "FP/mês e 1,00 h/mês, caso o orçamento de alarme falso passe a valer mais "
             "que a cobertura. A escolha atual privilegia recall porque neste ativo uma "
             "parada não antecipada custa mais que um alarme a confirmar."],
            color=GOOD)
    fecha(pdf, fig)


# ============================================================ 3. o alvo
def page_alvo(pdf):
    fig = new_page(pdf, "A régua: alvo, fuso e critério de custo", "02")
    y = 0.83
    y = heading(fig, 0.055, y, "O alvo é derivado por regra, não por curadoria")
    y = para(fig, 0.055, y,
             "Os 8 eventos vêm de uma regra reprodutível (scripts/pdm_fisico/verdade.py), "
             "aplicada ao histórico completo: (1) transição RUNNING_A de 1 para 0; "
             "(2) parada durou 2 h ou mais; (3) havia alarme de nível na janela de −1 h a "
             "+30 min da queda; (4) trips a menos de 24 h contam como um evento. Isso "
             "elimina o julgamento manual do denominador.")
    y -= 0.004
    rows = [
        ("1", "27/02/2025 08:38", "4,4 h", "Pr.Dif.Mt.Alta Vaz. Selo Prim.", "selagem"),
        ("2", "17/03/2025 18:16", "6,2 h", "Temp.Mt.Alta Manc.Rad.LNA CP", "mancal"),
        ("3", "07/04/2025 21:18", "18,2 h", "Temp.Mt.Alta Manc.Rad.LNA CP", "mancal"),
        ("4", "11/04/2025 17:02", "41,4 h", "Temp.Mt.Alta Manc.Rad.LNA CP", "mancal"),
        ("5", "29/04/2025 03:04", "564,2 h", "Temp.Mt.Alta Manc.Rad.LNA CP", "mancal"),
        ("6", "04/11/2025 06:22", "144,1 h", "TRIP-Pr.Mt.Bx. Óleo Lub.", "óleo"),
        ("7", "09/12/2025 08:36", "7,8 h", "Temp.Mt.Alta Manc.Rad.LNA CP", "mancal"),
        ("8", "26/02/2026 15:34", "5,7 h", "TRIP-Pr.Mt.Bx. Óleo Lub.", "óleo"),
    ]
    y = table(fig, 0.055, y, ["#", "evento (utc)", "parada", "alarme de nível", "mecanismo"],
              rows, [0.018, 0.135, 0.075, 0.245, 0.09], size=8.6)
    y -= 0.004
    y = para(fig, 0.055, y,
             "Um 9º evento (16/01/2024) existe no catálogo bruto mas antecede T0 e está "
             "fora da janela de avaliação. Três mecanismos físicos distintos aparecem: "
             "temperatura de mancal radial, pressão de óleo lubrificante e vazamento de "
             "selo primário.", size=9.2)
    y -= 0.008

    yb = y
    heading(fig, 0.055, yb, "O fuso foi provado, não assumido")
    para(fig, 0.055, yb - 0.032,
         "Os dados vêm do PI Web API, que retorna UTC — documentado no notebook de "
         "extração. O offset do log de alarmes foi determinado por busca de −3 h a +3 h, "
         "escolhendo o que melhor alinha alarme de nível com a queda real da máquina. "
         "O pico é inequívoco em +0 h: 10 paradas coincidentes, contra 6 no segundo "
         "melhor. Grade e alarme estão no mesmo relógio.",
         size=9.2, width=57)

    heading(fig, 0.52, yb, "Duas réguas de custo, não uma")
    para(fig, 0.52, yb - 0.032,
         "FP/mês mede frequência de episódios falsos. h/mês mede horas acumuladas em "
         "alarme falso e pesa a duração — reflete melhor o desgaste de confiança do "
         "operador, e expõe episódios longos que a contagem esconde. As duas são "
         "reportadas sempre juntas; otimizar só a primeira já nos levou a pontos com "
         "poucos episódios de mais de 100 h cada.",
         size=9.2, width=57)
    fecha(pdf, fig)


# ====================================================== 4. classificação regra C
def page_regrac(pdf):
    fig = new_page(pdf, "Como um episódio é classificado", "03")
    y = 0.83
    y = para(fig, 0.055, y,
             "A regra de classificação foi acordada com a equipe do detector paralelo "
             "(Francisco/Lara) para que os números das duas linhas de trabalho sejam "
             "comparáveis. Pontos de alarme separados por menos de 2 h formam um único "
             "episódio; cada episódio recebe um de três rótulos.")
    y -= 0.006
    rows = [
        (("DETECÇÃO", GOOD, "bold"), "o episódio cai nas 48 h anteriores a um dos 8 eventos-alvo", "8 episódios"),
        (("NEUTRO", AMP, "bold"), "não pegou falha, mas houve parada real ≥2 h em [início, fim+48 h]", "6 episódios"),
        (("FALSO POSITIVO", CRIT, "bold"), "nenhum dos dois", "6 episódios"),
    ]
    y = table(fig, 0.055, y, ["rótulo", "critério", "no ponto atual"],
              rows, [0.155, 0.545, 0.13], size=9.0)
    y -= 0.006
    y = para(fig, 0.055, y,
             "O balde NEUTRO existe porque a máquina parou de verdade logo depois do "
             "alarme: chamar isso de erro do detector seria injusto, e chamar de acerto "
             "seria inflar o resultado. A caixa do meio é a leitura honesta. Das 6 "
             "reclassificações, 3 têm hiato praticamente nulo entre o fim do alerta e a "
             "parada (0,03 h, a própria resolução da grade) — nesses o alarme estava "
             "ativo até o instante da parada, e o rótulo é robusto a qualquer escolha "
             "de janela.")
    y -= 0.008
    y = heading(fig, 0.055, y, "Sensibilidade da janela de 48 h")
    rows2 = [("2 h", "9", "0,775"), ("6 h", "9", "0,775"), ("12 h", "9", "0,775"),
             ("24 h", "8", "0,689"), (("48 h (adotada)", INK, "bold"), ("6", INK, "bold"),
                                      ("0,517", INK, "bold"))]
    y = table(fig, 0.055, y, ["janela", "falsos positivos", "fp/mês"], rows2,
              [0.13, 0.14, 0.10], size=9.0)
    y -= 0.004
    callout(fig, 0.055, y, 0.89, "por que 48 h e não menos",
            ["A janela foi escolhida pela equipe paralela e adotada por nós sem "
             "renegociar depois de ver o resultado — a ordem importa para a regra não "
             "virar ajuste. Mesmo na versão mais apertada possível (2 h), o resultado "
             "é 9 falsos positivos e 0,775 FP/mês, que continua dentro do orçamento "
             "de projeto de 1,15 FP/mês."])
    fecha(pdf, fig)


# ========================================================= 5. arquitetura
def page_arquitetura(pdf):
    fig = new_page(pdf, "Arquitetura: quatro sinais físicos", "04")
    y = 0.83
    y = para(fig, 0.055, y,
             "A decisão de projeto mais consequente foi não treinar um modelo genérico "
             "sobre os 39 sensores. Em vez disso, quatro sinais derivados, cada um "
             "ancorado num mecanismo de falha conhecido da máquina. Todos os limiares "
             "são adimensionais — múltiplos do p99 por sensor e de MADs do próprio "
             "baseline mensal, nada em °C ou bar — então o detector porta para outro "
             "compressor com os mesmos multiplicadores, sem re-derivar unidade.")
    y -= 0.006
    rows = [
        (("t", ACCENT, "bold"), "erro de reconstrução de PCA", "14 tags de temperatura (TI, TC382, T5)", "degradação térmica de mancal"),
        (("p", ACCENT, "bold"), "erro de reconstrução de PCA", "12 tags de pressão (PI, PDI, PDIT)", "óleo lubrificante, selagem"),
        (("sp", ACCENT, "bold"), "z robusto do spread", "|TI_0305 − mediana(0301, 0303, 0307)|", "divergência entre mancais irmãos"),
        (("vb", ACCENT, "bold"), "máximo do z robusto", "10 sondas de vibração (TV_351..355 X/Y)", "vibração mecânica"),
    ]
    y = table(fig, 0.055, y, ["sinal", "construção", "entrada", "mecanismo coberto"],
              rows, [0.045, 0.20, 0.29, 0.25], size=8.7)
    y -= 0.008
    y = heading(fig, 0.055, y, "Por que quatro sinais em voto, e não um modelo único")
    y = para(fig, 0.055, y,
             "Medido, não presumido. Cada sinal isolado como obrigatório do voto entrega "
             "pouco; e a arquitetura de dois detectores especializados por subsistema — "
             "que funciona no detector paralelo baseado em OCSVM/iforest — piora aqui, "
             "porque os nossos quatro sinais já são agregados derivados que atravessam "
             "subsistemas. Separá-los de novo remove os votos cruzados, que é onde metade "
             "da detecção vive.")
    y -= 0.004
    rows2 = [
        (("voto ≥2 dos 4, exigindo sp ou vb  (produção)", INK, "bold"),
         ("8/8", GOOD, "bold"), ("0,517", INK, "bold"), ("7,15", INK, "bold")),
        ("união de {t, sp, vb} com {p, vb} — dois especializados", "6/8", "0,517", "39,99"),
        ("só o de mancal/térmico {t, sp, vb}", "5/8", "0,344", "6,66"),
        ("só o de óleo/pressão {p, vb}", "2/8", "0,689", "47,98"),
    ]
    y = table(fig, 0.055, y, ["configuração", "detecção", "fp/mês", "h/mês"],
              rows2, [0.46, 0.11, 0.10, 0.10], size=9.0)
    fecha(pdf, fig)


# ====================================================== 6. cadeia de decisão
def page_cadeia(pdf):
    fig = new_page(pdf, "A cadeia de decisão, passo a passo", "05")
    y = 0.83
    passos = [
        ("1", "GRADE 2 MIN",
         "Mediana por janela de 2 min sobre o histórico de 30 s. Mediana e não média: "
         "robusta a spike isolado, dispensa filtro de Hampel."),
        ("2", "MÁSCARA OPERACIONAL",
         "estável = RUNNING_A > 0,5 E T5_AVG_A > 300 °C, menos 6 h de blackout após cada "
         "religamento. O blackout é o filtro pesado; tirar o corte de T5 abre 48 h em "
         "2,5 anos e custa uma parada."),
        ("3", "WALK-FORWARD MENSAL",
         "A cada mês, PCA de t e p e normalização de sp reajustados nas últimas 20.000 "
         "amostras estáveis (666,7 h), usando só dados anteriores ao mês — sem vazamento. "
         "RobustScaler, PCA com 95% de variância explicada."),
        ("4", "EWMA POR SINAL",
         "Meia-vida de 1 h para t e p, 30 min para sp e vb — casada com a escala de tempo "
         "física de cada mecanismo: temperatura e pressão evoluem mais devagar que vibração."),
        ("5", "CANAL DUPLO: DEGRAU OU CUSUM",
         "Degrau: sinal acima do limiar por 15 amostras consecutivas (30 min), pega "
         "excursão franca. CUSUM: soma acumulada com κ = 0,75 e h = 80, resetada a cada "
         "partida, pega deriva lenta que nunca cruza o limiar sozinha."),
        ("6", "VOTO ≥2, EXIGINDO MANCAL",
         "alarme = (nº de sinais ativos ≥ 2) E (sp OU vb ativo). O voto ≥2 é o que protege "
         "de transiente: um sinal sozinho dispara em partida, dois simultâneos não."),
        ("7", "REFRATÁRIO 48 H + DURAÇÃO ≥120 MIN",
         "O refratário resolveu a deriva de custo, que era repetição e não taxa (p de "
         "0,006 para 0,134). Varridos em conjunto: 720 pontos de pós-processamento."),
    ]
    for n, tit, txt in passos:
        fig.text(0.055, y, n, fontsize=13, color=ACCENT, family=MONO, weight="bold")
        fig.text(0.080, y, tit, fontsize=8.4, color=INK, family=MONO, weight="bold")
        y = para(fig, 0.080, y - 0.026, txt, size=9.0, width=112, leading=0.0205)
        y -= 0.002
    fecha(pdf, fig)


# ========================================================= 7. constantes
def page_constantes(pdf):
    fig = new_page(pdf, "Ponto de operação: as constantes", "06")
    y = 0.83
    y = para(fig, 0.055, y,
             "Todas as constantes vivem em scripts/pdm_fisico/publica_clearml.py e são "
             "reproduzidas bit a bit pela função reproduz(). Nenhuma foi escolhida por "
             "conveniência: cada uma tem varredura registrada nesta linha de trabalho.")
    y -= 0.008
    yb = y
    rows_a = [
        ("GRID", "2 min", "resolução da grade"),
        ("BLACKOUT", "6 h", "silêncio após religamento"),
        ("SUSTAIN", "15 amostras", "30 min acima do limiar (degrau)"),
        ("T0", "2025-01-01 UTC", "início da janela de avaliação"),
        ("FIT_POINTS", "20.000", "666,7 h estáveis por ajuste de PCA"),
        ("n_components", "0,95", "variância explicada retida"),
    ]
    table(fig, 0.055, yb, ["constante", "valor", "papel"], rows_a,
          [0.135, 0.115, 0.19], size=8.8)

    rows_b = [
        ("BASE t, p", "2,0", "múltiplo do p99 do baseline"),
        ("BASE sp, vb", "3,0", "múltiplo de MAD"),
        ("K base (t, p, sp)", "1,7", "multiplicador adimensional"),
        ("K vibração", "2,2", "multiplicador adimensional"),
        ("κ, h (CUSUM)", "0,75 · 80", "folga e limiar do acumulador"),
        ("REFRAT · DUR_MIN", "48 h · 120 min", "pós-processamento"),
    ]
    y = table(fig, 0.52, yb, ["constante", "valor", "papel"], rows_b,
              [0.145, 0.115, 0.20], size=8.8)
    y -= 0.012
    y = heading(fig, 0.055, y, "O limiar não é alavanca de custo")
    y = para(fig, 0.055, y,
             "Varrer o multiplicador de base de 1,5 a 5,0 leva a detecção de 8/8 a 0/8 — "
             "e o custo fica praticamente parado. Com 0/8 detecções ainda há cerca de "
             "1 falso positivo por mês. Isso encerra a intuição de que basta \"subir o "
             "limiar\" para comprar custo: o que se compra é perda de cobertura, não "
             "redução de alarme falso. A mesma conclusão apareceu no varrimento conjunto "
             "de refratário × duração × escalada: nenhum dos 720 pontos mantém 8/8 abaixo "
             "de 1,033 FP/mês na régua bruta. O piso de custo é duro.")
    y -= 0.004
    callout(fig, 0.055, y, 0.89, "voto mínimo — não há folga nesse eixo",
            ["Varredura exaustiva das 30 combinações de sinal obrigatório × voto mínimo. "
             "O ponto atual (sp|vb, voto ≥2) é o argmax: 8/8 a 0,517 FP/mês e 7,15 h/mês. "
             "A segunda melhor configuração que também atinge 8/8 é p|vb, a 0,689 FP/mês "
             "e 13,66 h/mês — quase o dobro das horas. Subir o voto mínimo para 3 "
             "colapsa a detecção de 8/8 para 1/8."], color=CTX)
    fecha(pdf, fig)


# ================================================= 8. walk-forward
def page_walkforward(pdf):
    fig = new_page(pdf, "Cadência de retreino: por que mensal", "07")
    y = 0.83
    y = para(fig, 0.055, y,
             "A cadência de reajuste do PCA foi medida contra todas as alternativas, com "
             "controle de corretude: a implementação mensal reproduz o ponto publicado "
             "bit a bit antes de qualquer comparação. Em todas as linhas, o sinal vb fica "
             "intocado — ele usa referência rolante de 400 h com passo de 6 h, mecanismo "
             "diferente do ajuste mensal.")
    y -= 0.006
    rows = [
        (("mensal  (produção)", INK, "bold"), ("27", INK, "bold"), ("8/8", GOOD, "bold"),
         ("0,517", INK, "bold"), ("7,15", INK, "bold"), ("29,0 h", INK, "bold"), "—"),
        ("trimestral", "10", "7/8", "0,431", "4,48", "23,4 h", "17/03/2025"),
        ("semestral", "5", "6/8", "0,431", "16,94", "36,0 h", "09/12 · 17/03"),
        ("anual", "3", "7/8", "0,775", "73,45", "37,8 h", "17/03/2025"),
        (("congelado", CRIT, "normal"), "1", ("6/8", CRIT, "normal"), "0,861",
         ("108,19", CRIT, "bold"), "36,0 h", "17/03 · 26/02"),
    ]
    y = table(fig, 0.055, y,
              ["cadência", "ajustes", "detecção", "fp/mês", "h/mês", "lead", "eventos perdidos"],
              rows, [0.135, 0.075, 0.085, 0.075, 0.085, 0.075, 0.16], size=8.8)
    y -= 0.008
    y = heading(fig, 0.055, y, "O mecanismo — e por que o detector paralelo mede o oposto")
    y = para(fig, 0.055, y,
             "Congelar o modelo custa 2 detecções e multiplica as horas de alarme falso "
             "por 15. A razão é mecânica: t e p são resíduos de reconstrução ancorados "
             "num baseline. Um PCA ajustado em baseline velho descola conforme o ponto de "
             "operação anda — campanha, carga, ambiente — e o resíduo cresce por motivo "
             "que não é saúde da máquina. O reajuste mensal reancora.")
    y = para(fig, 0.055, y,
             "A equipe do detector paralelo mediu +53% de falso positivo com retreino "
             "mensal e por isso não usa. Os dois resultados são compatíveis: as features "
             "de lá são estatísticas multiescala (média, desvio, tendência, curtose, "
             "assimetria, crest), já invariantes a deriva lenta, então retreinar injeta "
             "principalmente ruído de amostragem.")
    y -= 0.004
    callout(fig, 0.055, y, 0.89, "regra generalizável, extraída da divergência",
            ["Retreine se o seu sinal for resíduo de reconstrução ancorado num baseline. "
             "Não retreine se o sinal já for invariante à deriva do ponto de operação. "
             "A pergunta certa não é \"retreinar ou não\", é qual a natureza do sinal."],
            color=GOOD)
    fecha(pdf, fig)


# ========================================================= 9. resultado
def page_resultado(pdf):
    fig = new_page(pdf, "Resultado de produção", "08")
    y = 0.83
    yb = y
    rows_m = [
        (("detecção", INK, "bold"), ("8 / 8   (100%)", GOOD, "bold")),
        ("falsos positivos por mês", ("0,517", INK, "bold")),
        ("horas por mês em alarme falso", ("7,15", INK, "bold")),
        ("antecedência média", "29,0 h"),
        ("antecedência mínima", "2,8 h"),
        ("episódios totais", "20  (8 TP · 6 FP · 6 neutros)"),
        ("janela de operação", "11,6 meses"),
        ("validação leave-one-event-out", "7/8"),
    ]
    table(fig, 0.055, yb, ["métrica", "valor"], rows_m, [0.26, 0.19], size=9.0)

    rows_l = [
        ("27/02/2025", "143,9 h"), ("17/03/2025", "194,9 h"),
        ("07/04/2025", "5,0 h"), ("11/04/2025", "2,8 h"),
        ("29/04/2025", "51,8 h"), ("04/11/2025", "8,8 h"),
        ("09/12/2025", "23,6 h"), ("26/02/2026", "670,0 h"),
    ]
    y = table(fig, 0.55, yb, ["evento-alvo", "antecedência"], rows_l,
              [0.155, 0.13], size=9.0)
    y -= 0.010
    y = para(fig, 0.055, y,
             "A antecedência média de 29,0 h é média, não mediana — 4 dos 8 eventos estão "
             "censurados no teto de 48 h da janela de avaliação, e o mínimo é 2,8 h. "
             "Reportar como \"29 h de aviso típico\" seria incorreto: o número honesto é "
             "que em todos os 8 houve aviso antes do evento, com o pior caso em 2,8 h.")
    y -= 0.006
    y = heading(fig, 0.055, y, "Anatomia do custo")
    y = para(fig, 0.055, y,
             "Os 6 falsos positivos somam 83,0 h em 16 meses — média de 13,8 h por "
             "episódio, faixa de 3,8 h a 32,8 h. Os 6 neutros somam 365,8 h, e são "
             "neutros porque cada um é seguido de parada real da máquina dentro da "
             "janela: os dois episódios de mais de 130 h que dominavam a régua bruta "
             "estão nesse grupo. A validação leave-one-event-out devolve 7/8 sob "
             "qualquer regra de desempate — o 8/8 no LOEO era artefato de desempate, e "
             "o evento que cai é sempre 04/11/2025, que tem apenas 15 h de operação "
             "contínua antes do trip.")
    fecha(pdf, fig)


# ========================================================= 10. figura
def page_figura(pdf):
    if not os.path.exists(FIG):
        return
    fig = new_page(pdf, "A série completa, classificada", "09")
    full_image(fig, FIG, top=0.865, bottom=0.185)
    y = para(fig, 0.055, 0.135,
             "Mediana diária de T5_AVG_A. Faixas cinzas = máquina parada. Linhas "
             "tracejadas laranja = os 8 eventos-alvo. Pontos vermelhos = episódio que "
             "antecipou falha; pretos = falso positivo; laranja = neutro, isto é, parada "
             "real da máquina dentro da janela de 48 h após o fim do alerta.",
             size=8.6, color=MUTED, width=128, leading=0.020)
    fig.text(0.055, y, "Origem: scripts/pdm_fisico/plota_estilo_francisco.py",
             fontsize=8, color=MUTED, family=MONO)
    fecha(pdf, fig)


# ================================================= 11. refutados I
def page_refutados(pdf):
    fig = new_page(pdf, "A borda do blackout: seis tentativas, seis refutações", "10")
    y = 0.83
    y = para(fig, 0.055, y,
             "Oito dos doze episódios não-detecção nascem em dist_partida = 6,4667 h "
             "cravado, o mesmo valor até a quarta casa decimal. Não é coincidência: é "
             "(180 + 14) × 2 min, ou seja, as 6 h de blackout mais os 14 passos que faltam "
             "para fechar o SUSTAIN de 30 min. São alarmes que disparam no primeiro "
             "instante em que a máscara permite. A EWMA é calculada sobre o sinal cru e "
             "atravessa o blackout carregando o pico do religamento.")
    y -= 0.006
    rows = [
        ("gate de não-decaimento (1.152 pontos, 2 graus de rigor)", "custa exatamente o 04/11/2025, em toda a grade"),
        ("ordenação por severidade e LOEO", "o episódio mais grave por CUSUM é falso positivo"),
        ("duração da parada antes do religamento", "TP inteiramente contido na faixa dos FP (0,03 a 190,6 h)"),
        ("rampa de T5 pós-partida (slope e rugosidade)", "empate exato em 3,79 °C/min entre um FP e uma detecção"),
        ("assinatura de sinais", "2 dos 3 TP na borda têm a mesma combinação (p+vb) que 4 FP"),
        ("blackout mais longo (9, 12, 18, 24 h)", "24 h custa metade das detecções e piora h/mês em 5x"),
    ]
    y = table(fig, 0.055, y, ["tentativa", "por que falhou"], rows,
              [0.40, 0.49], size=8.7)
    y -= 0.008
    y = heading(fig, 0.055, y, "A causa raiz é física, não algorítmica")
    y = para(fig, 0.055, y,
             "A cauda do transiente de religamento e a falha de religamento têm a mesma "
             "geometria nos quatro sinais. Não há regra sobre derivada que separe as "
             "duas. Isso também explica por que o LOEO sempre derruba o 04/11/2025: a "
             "detecção daquele evento vive na borda do blackout, e a evidência que a "
             "sustenta é indistinguível da que produz os falsos positivos.")
    y -= 0.004
    callout(fig, 0.055, y, 0.89, "convergência independente que reforça o diagnóstico",
            ["A equipe do detector paralelo, com arquitetura completamente diferente "
             "(OCSVM e isolation forest sobre features multiescala, com portões causais), "
             "chegou ao mesmo muro no mesmo lugar: cerca de 48% do resíduo residual deles "
             "são pontos que ocorrem exatamente uma amostra antes de um portão engatar. "
             "Ambos os times testaram um filtro de duração genérico e ambos viram a "
             "detecção desabar, pela mesma razão — um fragmento curto de precursor real "
             "truncado e um fragmento curto de ruído genuíno são indistinguíveis por "
             "duração. Duas arquiteturas independentes batendo no mesmo limite é "
             "evidência de que ele é estrutural, não de modelagem."], color=AMP)
    fecha(pdf, fig)


# ================================================= 12. refutados II + controle
def page_controle(pdf):
    fig = new_page(pdf, "Contexto de alarme como filtro: refutado por taxa de base", "11")
    y = 0.83
    y = para(fig, 0.055, y,
             "Uma hipótese recorrente é usar o catálogo de alarmes da planta para "
             "reclassificar falso positivo como \"sinal real de outro alarme\". Testado "
             "com o controle que a afirmação exige: qual fração de instantes aleatórios "
             "dentro da máscara de operação também seria \"explicada\" pelo mesmo "
             "critério? Com 47 tags e 3,43 alarmes por dia, o resultado é decisivo.")
    y -= 0.006
    rows = [
        ("± 1 h", "0%", "0%", "6%", "—"),
        (("± 2 h", INK, "bold"), ("33%", INK, "bold"), "12%", ("10%", INK, "bold"),
         ("3,42x — informativo", GOOD, "bold")),
        ("± 4 h", "33%", "38%", "18%", "1,85x"),
        ("± 6 h", "50%", "50%", "26%", "1,94x"),
        ("± 12 h", "83%", "100%", "47%", "1,78x"),
        (("± 24 h", CRIT, "normal"), "83%", ("100%", CRIT, "normal"),
         ("70%", CRIT, "bold"), ("1,20x — sem enriquecimento", CRIT, "normal")),
    ]
    y = table(fig, 0.055, y,
              ["janela", "nossos fp", "nossos tp", "aleatório", "enriquecimento"],
              rows, [0.09, 0.11, 0.11, 0.11, 0.24], size=9.0)
    y -= 0.008
    y = para(fig, 0.055, y,
             "A ±24 h, 70% de instantes puramente aleatórios são \"explicados\" — e a "
             "±12 h os acertos verdadeiros são explicados em 100% contra 83% dos falsos "
             "positivos, ou seja, o critério não distingue acerto de erro. Só a ±2 h há "
             "enriquecimento real. Qualquer reclassificação por proximidade de alarme "
             "precisa reportar contra essa linha de base, não contra zero.")
    y -= 0.008
    y = heading(fig, 0.055, y, "Outras linhas testadas e refutadas")
    y = para(fig, 0.055, y,
             "Voto entre sondas de vibração (sobreajuste — o resultado vive num único "
             "k_vib e o LOEO cai a 3/9) · quinto sinal a partir do stack do EXP7 · piso "
             "relativo de escala · potência espectral por STFT na banda de 20 min a 4 h "
             "(percentil médio 52, nível de acaso) · divergência generalizada de "
             "temperatura para vibração (0 de 9 eventos cruza o limiar já validado) · "
             "veto de sensor congelado (não aplicável: apenas 0,3% do tempo mascarado, "
             "e 0,0% de sobreposição com os episódios) · modelos especializados por "
             "subsistema · autocalibração de limiar por percentil do baseline.")
    fecha(pdf, fig)


# ========================================================= 13. limites
def page_limites(pdf):
    fig = new_page(pdf, "Limites conhecidos do resultado", "12")
    y = 0.83
    itens = [
        ("A RÉGUA ACABOU ANTES DO DETECTOR",
         "Com 8 eventos, o intervalo de confiança de Poisson do custo é [0,53; 1,81]. "
         "Provar uma melhora de 1,03 para 0,75 FP/mês exigiria 92 meses de observação. "
         "O gargalo do projeto passou a ser o rótulo, não o modelo — e nenhuma diferença "
         "de configuração dentro desse intervalo deve ser tratada como real."),
        ("RUÍDO DE RETREINO LIMITA A LEITURA DE GANHOS",
         "Duas execuções de configuração idêntica diferem em 20,7 pontos percentuais de "
         "recall por ruído de retreino. Dominância de Pareto entre configurações não "
         "prova efeito; qualquer ganho abaixo desse piso é indistinguível de sorte."),
        ("AMBIGUIDADE IRREDUTÍVEL NA PARTIDA",
         "A informação que falta — classificação do religamento entre partida comandada "
         "e recuperação de trip, vinda do SOE ou do log do operador — não existe na grade "
         "atual. Enquanto não existir, os episódios nascidos em até ~7 h de um "
         "religamento devem ir para uma fila de \"confirmar com operação\", não ser "
         "suprimidos. Isso preserva a cobertura de 8/8 e reduz em 67% os alarmes de "
         "prioridade plena."),
        ("UM MISS CONHECIDO, FORA DO ALVO OFICIAL",
         "A parada de 24/11/2025 (43 h), precedida em 1 minuto de anunciação de baixa "
         "pressão no header de óleo lubrificante, não é detectada. Não entra no alvo hoje "
         "porque PAL_6240339 não é tag de intertravamento — mas é o mesmo mecanismo "
         "físico dos eventos 6 e 8, um estágio antes. É a investigação técnica mais "
         "promissora em aberto."),
    ]
    for tit, txt in itens:
        fig.text(0.055, y, tit, fontsize=8.4, color=CRIT, family=MONO, weight="bold")
        y = para(fig, 0.055, y - 0.028, txt, size=9.2, width=116, leading=0.021)
        y -= 0.008
    fecha(pdf, fig)


# ================================================= 14. alternativo + repro
def page_fecho(pdf):
    fig = new_page(pdf, "Ponto alternativo e reprodutibilidade", "13")
    y = 0.83
    y = heading(fig, 0.055, y, "Se o orçamento de alarme falso passar a valer mais que a cobertura")
    y = para(fig, 0.055, y,
             "Existe um ponto medido e reversível por uma única constante: blackout de "
             "12 h em vez de 6 h, com os mesmos limiares. Ele entrega 5/8 a 0,178 FP/mês "
             "e 1,00 h/mês. No mesmo nível de cobertura dos detectores univariados de "
             "referência (5/8), isso é de 2,4 a 3,3 vezes melhor em FP/mês e de 2 a 10 "
             "vezes melhor em horas.")
    y -= 0.004
    rows = [
        (("blackout 6 h  (produção)", INK, "bold"), ("8/8", GOOD, "bold"),
         ("0,517", INK, "bold"), ("7,15", INK, "bold"), ("29,0 h", INK, "bold")),
        ("blackout 9 h", "5/8", "0,613", "7,86", "25,2 h"),
        (("blackout 12 h  (alternativo)", CTX, "bold"), ("5/8", CTX, "bold"),
         ("0,178", CTX, "bold"), ("1,00", CTX, "bold"), ("34,1 h", CTX, "bold")),
        ("blackout 18 h", "5/8", "0,368", "34,28", "34,1 h"),
        ("blackout 24 h", "4/8", "0,380", "34,80", "41,9 h"),
    ]
    y = table(fig, 0.055, y, ["configuração", "detecção", "fp/mês", "h/mês", "lead"],
              rows, [0.28, 0.10, 0.10, 0.10, 0.10], size=9.0)
    y -= 0.006
    y = para(fig, 0.055, y,
             "A escolha de produção continua sendo o 8/8, porque neste ativo uma parada "
             "não antecipada custa mais que um alarme a confirmar. Note que blackout "
             "longo interage mal com a regra de classificação: empurrar o início do "
             "episódio para frente faz a parada real que justificava o rótulo neutro cair "
             "antes do novo início, e o episódio vira falso positivo cheio — por isso "
             "18 h e 24 h pioram h/mês em vez de melhorar.", size=9.2)
    y -= 0.010
    y = heading(fig, 0.055, y, "Onde está o modelo")
    rows2 = [
        ("branch", (f"{BRANCH}", ACCENT, "bold")),
        ("commit", f"{COMMIT}  ({COMMIT_DT})"),
        ("ponto de operação", "scripts/pdm_fisico/publica_clearml.py  →  reproduz()"),
        ("classificação e figura", "scripts/pdm_fisico/plota_estilo_francisco.py"),
        ("derivação do alvo", "scripts/pdm_fisico/verdade.py  →  falhas.csv"),
        ("varredura de pós-processamento", "scripts/pdm_fisico/pos_processamento.py"),
        ("ponto alternativo de 12 h", "scripts/pdm_fisico/plota_silencio12h.py"),
    ]
    y = table(fig, 0.055, y, ["item", "caminho"], rows2, [0.24, 0.55], size=8.8)
    fecha(pdf, fig)


def main():
    with PdfPages(OUT) as pdf:
        page_capa(pdf)
        page_sumario(pdf)
        page_alvo(pdf)
        page_regrac(pdf)
        page_arquitetura(pdf)
        page_cadeia(pdf)
        page_constantes(pdf)
        page_walkforward(pdf)
        page_resultado(pdf)
        page_figura(pdf)
        page_refutados(pdf)
        page_controle(pdf)
        page_limites(pdf)
        page_fecho(pdf)
        d = pdf.infodict()
        d["Title"] = "Detector físico TC-330.03A — como chegamos ao 8/8"
        d["Author"] = "Projeto Cabiúnas"
        d["Subject"] = f"Relatório de engenharia do ponto de produção · branch {BRANCH}"
    print(f"-> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
