#!/usr/bin/env python3
"""
build_relatorio_pdf.py
Monta o relatório de sessão em PDF (A4 paisagem) para circular com o time.

Conteúdo: o que foi feito até 10/08/2026 no TC382_03_A — resultado da Frente B
(treino estendido para 2024), a série temporal com alarmes DETECTADOS vs
REGISTRADOS no DCS, a investigação de falso positivo desta sessão e as decisões
pendentes.

Sem dependência externa: só matplotlib (o venv não tem reportlab/weasyprint).
As figuras são as já geradas em eval_predictive_out/ — este script não recalcula
métrica nenhuma, só compõe. Os números vêm dos CSVs da auditoria.

Uso:
    PYTHONPATH=. python scripts/build_relatorio_pdf.py
"""
from __future__ import annotations

import os
import textwrap
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = "relatorio_anexos/RELATORIO_CABIUNAS_2026_08_10.pdf"
FIG_FRENTEB = "eval_predictive_out/fig_frenteB_TC382_03_A_serie.png"
FIG_ZOOM = "eval_predictive_out/fig_oos2025_TC382_03_A_zoom.png"
FIG_FP_AGO = "eval_predictive_out/fig_fp_agosto2025_TC382_03_A.png"

W, H = 11.69, 8.27          # A4 paisagem
INK, INK2, MUTED = "#131a20", "#3d4a55", "#6b7885"
RULE, GROUND = "#dde2e6", "#f4f6f7"
ACCENT, AMP, CTX = "#0f6e78", "#b8792a", "#2b6ca3"
GOOD, CRIT = "#2e7d4f", "#b03a2e"

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

_page = {"n": 0}


# ---------------------------------------------------------------------------
# primitivas de layout
# ---------------------------------------------------------------------------

def new_page(pdf, title: str | None = None, eyebrow: str | None = None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    _page["n"] += 1
    if eyebrow:
        fig.text(0.055, 0.945, "  ".join(eyebrow.upper()) if len(eyebrow) < 4 else eyebrow.upper(),
                 fontsize=7.5, color=ACCENT, family=MONO)
    if title:
        fig.text(0.055, 0.905, title, fontsize=16, color=INK, family=SANS, weight="bold")
        fig.lines.append(plt.Line2D([0.055, 0.945], [0.888, 0.888],
                                    transform=fig.transFigure, color=INK, lw=1.4))
    fig.text(0.945, 0.038, f"{_page['n']}", fontsize=8, color=MUTED,
             family=MONO, ha="right")
    fig.text(0.055, 0.038, "Cabiúnas · Turbina A · detecção de anomalias · 10/08/2026",
             fontsize=7.5, color=MUTED, family=SANS)
    fig.lines.append(plt.Line2D([0.055, 0.945], [0.058, 0.058],
                                transform=fig.transFigure, color=RULE, lw=0.8))
    return fig


def para(fig, x, y, text, size=9.5, color=INK2, width=118, leading=0.0225,
         weight="normal", family=SANS):
    """Escreve parágrafo com quebra manual (matplotlib não quebra sozinho)."""
    for line in textwrap.wrap(text, width=width):
        fig.text(x, y, line, fontsize=size, color=color, family=family, weight=weight)
        y -= leading
    return y - leading * 0.35


def bullet(fig, x, y, text, size=9.3, width=112, leading=0.0225):
    """Item de lista com recuo de continuação (linhas seguintes alinham ao texto)."""
    lines = textwrap.wrap(text, width=width)
    for i, line in enumerate(lines):
        if i == 0:
            fig.text(x, y, "•", fontsize=size, color=MUTED, family=SANS)
        fig.text(x + 0.012, y, line, fontsize=size, color=INK2, family=SANS)
        y -= leading
    return y - leading * 0.25


def heading(fig, x, y, text, size=10.5):
    fig.text(x, y, text, fontsize=size, color=INK, family=SANS, weight="bold")
    return y - 0.032


def table(fig, x, y, cols, rows, widths, size=9.0, header_color=MUTED):
    """Tabela simples: cols = títulos, rows = listas de (texto, cor) ou texto."""
    xs, acc = [], x
    for w in widths:
        xs.append(acc)
        acc += w
    for cx, c in zip(xs, cols):
        fig.text(cx, y, c.upper(), fontsize=7.2, color=header_color, family=SANS,
                 weight="bold")
    y -= 0.008
    fig.lines.append(plt.Line2D([x, acc - 0.012], [y, y], transform=fig.transFigure,
                                color=RULE, lw=1.0))
    y -= 0.024
    for r in rows:
        for cx, cell in zip(xs, r):
            txt, col, wt = (cell if isinstance(cell, tuple) else (cell, INK2, "normal"))
            fig.text(cx, y, txt, fontsize=size, color=col, family=SANS, weight=wt)
        y -= 0.0245
    return y - 0.008


def callout(fig, x, y, w, label, lines, color=ACCENT, size=9.2):
    """Caixa com barra lateral colorida."""
    n = sum(len(textwrap.wrap(t, width=int(w * 118 / 0.89))) for t in lines)
    # `para` acrescenta leading*0.35 no fim de CADA parágrafo — sem contabilizar
    # isso a última linha vaza para fora da caixa
    h = 0.032 + n * 0.0215 + len(lines) * 0.0215 * 0.35
    fig.patches.append(plt.Rectangle((x, y - h + 0.018), w, h, transform=fig.transFigure,
                                     facecolor=GROUND, edgecolor=RULE, lw=0.8, zorder=0))
    fig.patches.append(plt.Rectangle((x, y - h + 0.018), 0.0035, h,
                                     transform=fig.transFigure, facecolor=color,
                                     edgecolor="none", zorder=1))
    yy = y
    fig.text(x + 0.014, yy, label.upper(), fontsize=7.2, color=color, family=MONO,
             weight="bold")
    yy -= 0.026
    for t in lines:
        yy = para(fig, x + 0.014, yy, t, size=size,
                  width=int(w * 118 / 0.89), leading=0.0215)
    return y - h - 0.012


def full_image(fig, path, top=0.86, bottom=0.115):
    ax = fig.add_axes([0.045, bottom, 0.91, top - bottom])
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# páginas
# ---------------------------------------------------------------------------

def page_capa(pdf):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.075, 0.80, "CABIÚNAS · TURBINA A", fontsize=9, color=ACCENT,
             family=MONO, weight="bold")
    fig.text(0.075, 0.705, "Detecção de anomalias", fontsize=30, color=INK,
             family=SANS, weight="bold")
    fig.text(0.075, 0.645, "Estado do modelo e decisões pendentes", fontsize=17,
             color=INK2, family=SANS)
    fig.lines.append(plt.Line2D([0.075, 0.40], [0.60, 0.60], transform=fig.transFigure,
                                color=ACCENT, lw=2.5))

    y = para(fig, 0.075, 0.545,
             "O treino estendido para 2024 resolveu o TC382_03_A: 86,2% de recall na janela completa "
             "contra 51,7% do controle, com 17 de 17 incidentes detectados fora da amostra. A caça ao "
             "falso positivo fechou duas famílias de solução e abriu uma terceira.",
             size=10.5, width=92, leading=0.026)
    para(fig, 0.075, y - 0.01,
         "O que falta agora não é modelagem — é escolher um ponto de operação.",
         size=10.5, width=92, weight="bold", color=INK)

    # painel de números
    tiles = [("17/17", "incidentes detectados\nfora da amostra (2025)"),
             ("0,103", "falsos alarmes por dia\num a cada 10 dias"),
             ("−38%", "de falso positivo possível\ncustando 6,9 pp de recall"),
             ("nada", "em produção — bundles\natuais são de junho/v9")]
    x0, wt = 0.075, 0.21
    for i, (v, k) in enumerate(tiles):
        x = x0 + i * wt
        fig.patches.append(plt.Rectangle((x, 0.20), wt - 0.018, 0.145,
                                         transform=fig.transFigure, facecolor=GROUND,
                                         edgecolor=RULE, lw=0.8))
        fig.text(x + 0.016, 0.288, v, fontsize=23, color=INK, family=SANS, weight="bold")
        for j, line in enumerate(k.split("\n")):
            fig.text(x + 0.016, 0.252 - j * 0.021, line, fontsize=8, color=MUTED,
                     family=SANS)

    fig.text(0.075, 0.12, f"Relatório de sessão · {date.today().strftime('%d/%m/%Y')} · commit 6465d42",
             fontsize=8.5, color=MUTED, family=MONO)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_estado(pdf):
    fig = new_page(pdf, "Onde o modelo está", "Resultado · Frente B")
    y = 0.845
    y = para(fig, 0.055, y,
             "O melhor candidato é POR SENSOR, não um modelo só. O treino estendido para junho/2024 "
             "promoveu no TC382_03_A e piorou no T5 — cada um fica com sua receita.")
    y -= 0.012
    y = table(fig, 0.055, y,
              ["Sensor", "Receita vencedora", "FULL", "OOS 2025", "Backcast 2024", "FA/dia"],
              [[("TC382_03_A", INK, "bold"), "b2024 — treino jun/24→jul/25",
                ("86,2%", GOOD, "bold"), ("17/17", GOOD, "bold"), ("21,4%", CRIT, "bold"), "0,103"],
               ["TC382_03_A", "controle — treino jan→jul/25", "51,7%", "17/17",
                ("21,4%", CRIT, "normal"), "0,172"],
               [("T5_AVG_A", INK, "bold"), "controle / v10", ("100%", GOOD, "bold"),
                "1/1  (!)", "83,3%", "0,082"],
               ["T5_AVG_A", "b2024", ("81,8%", CRIT, "normal"), "1/1  (!)",
                ("50,0%", CRIT, "normal"), "0,151"]],
              widths=[0.115, 0.245, 0.085, 0.095, 0.125, 0.08])

    y = para(fig, 0.055, y - 0.008,
             "Protocolo honesto: incidentes HI/HIHI com máquina ligada, horizonte de 8 h, recall_raw "
             "(sem crédito da cauda do sticky), duty pós-sticky ≤ 0,25. O OOS do T5 tem UM incidente — "
             "aquele 100% não sustenta conclusão nenhuma, está na tabela por completude.",
             size=8.6, color=MUTED)

    y = callout(fig, 0.055, y - 0.02, 0.89, "O que o ganho do b2024 NÃO é",
                ["O backcast de 2024 ficou idêntico nos dois braços (21,4%). Ou seja: o modelo não "
                 "aprendeu o regime de 2024. O ganho de 34,5 pp veio do threshold global se "
                 "reposicionar quando o histórico de treino mudou — é um ganho real e reprodutível, "
                 "mas pela razão errada, e por isso não transfere para outros sensores.",
                 "Ruído de semente neste problema é de ±27 pp. Por isso toda comparação usa um braço "
                 "de controle re-treinado, nunca o número histórico."])

    para(fig, 0.055, y - 0.005,
         "A figura da próxima página mostra os dois braços lado a lado, com os alarmes registrados no "
         "DCS numa pista própria e os alertas do modelo classificados em detecção, perda e falso positivo.",
         size=9.0)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_frenteb(pdf):
    fig = new_page(pdf, "Alarmes detectados × alarmes registrados no DCS",
                   "Frente B · TC382_03_A · jun/2024 → abr/2026")
    full_image(fig, FIG_FRENTEB, top=0.875, bottom=0.145)
    y = 0.135
    para(fig, 0.055, y,
         "Painel 1: temperatura bruta. Pista 'alarme DCS': os 70 eventos HI/HIHI reais (onset→OK) — a "
         "verdade de campo. Painéis 3 e 4: o health index de cada braço sob UM único threshold. Verde = "
         "incidente detectado (cruzamento bruto em até 8 h antes); vermelho = incidente perdido; laranja = "
         "episódio de alerta sem nenhum alarme por perto, o falso positivo. Cinza = equipamento desligado.",
         size=8.4, color=MUTED, width=138, leading=0.019)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_zoom(pdf):
    fig = new_page(pdf, "Antecedência: o alerta acende antes do alarme",
                   "Incidentes fora da amostra · TC382_03_A")
    full_image(fig, FIG_ZOOM, top=0.855, bottom=0.20)
    y = 0.175
    y = para(fig, 0.055, y,
             "A figura anterior comprime 23 meses e esconde o que mais importa operacionalmente: quanto "
             "tempo antes o alerta acende. Aqui, três incidentes fora da amostra em detalhe. A linha "
             "vermelha é o alarme HI/HIHI do DCS; a faixa amarela é o alerta do modelo já ativo.",
             size=9.0, width=138, leading=0.0205)
    para(fig, 0.055, y - 0.004,
         "Dois dos três dão 8 h de antecedência — o horizonte máximo que a métrica credita. O de "
         "20/10/2025 dá 0,3 h: é detecção, mas praticamente simultânea ao alarme. A mediana de "
         "antecedência na janela completa é de 7,4 h.",
         size=9.0, width=138, leading=0.0205, color=INK, weight="bold")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_fp(pdf):
    fig = new_page(pdf, "Falso positivo: três famílias testadas",
                   "Investigação · sessão de 10/08/2026")
    y = 0.845
    y = para(fig, 0.055, y,
             "Um falso positivo aqui é UM EPISÓDIO de alerta sem nenhum alarme do DCS em [início, fim + 8h]. "
             "São 71 episódios em 687 dias. Testei três famílias, todas como pós-processamento — nenhuma "
             "exige retreinar.")

    y = heading(fig, 0.055, y - 0.012, "1. Amplitude — encerrada")
    y = para(fig, 0.055, y,
             "Histerese e confirmação k-de-N trocam falso positivo por detecção na razão de 1 para 1. A "
             "causa foi medida: o pico do score no falso positivo é tão alto quanto no evento real "
             "(AUC 0,44, ligeiramente MAIOR no FP). Subir a barra não seleciona nada. O falso positivo não "
             "é tremor marginal no limiar — é excursão de amplitude legítima que não se sustenta.")

    y = heading(fig, 0.055, y - 0.010, "2. Contexto operacional — 6 a 8 vezes melhor de troca")
    y = para(fig, 0.055, y,
             "Os falsos positivos acontecem MANOBRANDO; os eventos reais acontecem com a máquina estável e "
             "quente. Na janela do disparo, a rampa de carga mediana é de 172 °C/h nos FPs contra 21 °C/h "
             "nos eventos reais. Um portão que suprime o disparo durante manobra custa ZERO de "
             "antecedência — a rampa já é conhecida no instante do alerta.")

    y = table(fig, 0.055, y - 0.012,
              ["Regra", "FP", "Recall", "FA/dia", "FP removidos por incidente perdido"],
              [["histerese", "66", "44/58", "0,096", ("0,8", CRIT, "bold")],
               ["k-de-N (6 h, 90%)", "50", "32/58", "0,073", ("1,2", CRIT, "bold")],
               ["rampa < 80 °C/h + T5 ≥ 600 °C", "44", "46/58", "0,064", ("6,8", GOOD, "bold")],
               ["só nível T5 ≥ 640 °C", "33", "45/58", "0,048", ("7,6", GOOD, "bold")],
               [("ponto atual (sem portão)", INK, "bold"), ("71", INK, "bold"),
                ("50/58", INK, "bold"), ("0,103", INK, "bold"), "—"]],
              widths=[0.235, 0.06, 0.075, 0.08, 0.30])

    callout(fig, 0.055, y - 0.012, 0.89, "Por que eu não promovi isso sozinho",
            ["Fixei o critério ANTES de rodar: FA cair 30% perdendo no máximo 5 pp de recall E manter "
             "17/17 no OOS. Nenhum ponto passou — o melhor perde por um único incidente na régua de "
             "recall e cai para 16/17 no OOS. Não afrouxei o critério depois de ver o resultado.",
             "Risco físico: as medianas de rampa diferem 8×, mas as caudas se sobrepõem. 7 dos 58 "
             "incidentes reais têm rampa acima de 40 °C/h (máximo 222). Falha durante manobra existe, "
             "e o portão a apagaria."],
            color=CRIT)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_fronteira(pdf):
    fig = new_page(pdf, "A fronteira: quanto de recall custa cada falso alarme evitado",
                   "TC382_03_A · janela FULL · 58 incidentes / 687 dias")
    ax = fig.add_axes([0.075, 0.30, 0.52, 0.53])
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.grid(True, color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)

    amp = [(0.099, 84.5), (0.096, 75.9), (0.084, 60.3), (0.073, 55.2)]
    ctx = [(0.067, 81.0), (0.064, 79.3), (0.048, 77.6), (0.047, 75.9), (0.045, 74.1)]
    ax.plot(*zip(*amp), "-o", color=AMP, lw=1.8, ms=7, mec="white", mew=1.5,
            label="regras de amplitude (histerese, k-de-N)", zorder=3)
    ax.plot(*zip(*ctx), "-o", color=CTX, lw=1.8, ms=7, mec="white", mew=1.5,
            label="portão de contexto (rampa + nível de carga)", zorder=3)
    ax.plot([0.103], [86.2], "o", color=INK, ms=11, mec="white", mew=2,
            label="ponto de operação atual", zorder=4)
    ax.annotate("hoje\n86,2% @ 0,103", xy=(0.103, 86.2), xytext=(-10, 11),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5,
                color=INK, family=SANS, weight="bold")
    ax.annotate("79,3% @ 0,064", xy=(0.064, 79.3), xytext=(4, -14),
                textcoords="offset points", fontsize=8, color=CTX, family=SANS)
    ax.annotate("74,1% @ 0,045", xy=(0.045, 74.1), xytext=(2, -14),
                textcoords="offset points", fontsize=8, color=CTX, family=SANS)
    ax.annotate("", xy=(0.042, 88.5), xytext=(0.060, 88.5),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.text(0.0615, 88.2, "melhor", fontsize=8.5, color=MUTED, family=SANS)
    ax.set_xlabel("falso alarme por dia  →  pior", fontsize=9, color=INK2, family=SANS)
    ax.set_ylabel("recall (incidentes detectados)", fontsize=9, color=INK2, family=SANS)
    ax.set_ylim(50, 92)
    ax.set_xlim(0.035, 0.112)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK2)

    x = 0.635
    y = heading(fig, x, 0.815, "O que a curva diz")
    y = para(fig, x, y, "A curva de contexto DOMINA a de amplitude em toda a faixa: para a mesma "
             "perda de recall, remove muito mais falso positivo.", width=52, size=9.2)
    y = heading(fig, x, y - 0.012, "3. Graduação de confiança")
    y = para(fig, x, y, "O alerta acende no mesmo instante de hoje; nada é suprimido nem atrasado. "
             "Seis horas depois o nível é revisto pela SUSTENTAÇÃO do sinal.", width=52, size=9.2)
    y = table(fig, x, y - 0.008,
              ["Revisão", "FP rebaixados", "Reais mantidos"],
              [["3 h  conservadora", "26/71  (37%)", ("25/29  (86%)", GOOD, "bold")],
               [("6 h  recomendada", INK, "bold"), ("46/71  (65%)", GOOD, "bold"),
                ("24/29  (83%)", GOOD, "bold")],
               ["12 h  agressiva", "55/71  (77%)", ("20/29  (69%)", CRIT, "bold")]],
              widths=[0.115, 0.105, 0.10], size=8.6)
    para(fig, x, y - 0.004,
         "Um evento real rebaixado NÃO é um incidente perdido: o alerta existe e está visível, só num "
         "nível menor. É por isso que a graduação vale onde o portão não vale.",
         width=52, size=8.6, color=MUTED)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_decisoes(pdf):
    fig = new_page(pdf, "As duas decisões e o que está pendente", "Encaminhamento")
    y = 0.845

    for lab, q, d in [
        ("Decisão de operação · time", "Qual ponto de operação o campo quer?",
         "Manter 86,2% de recall com um falso alarme a cada 10 dias, ou aceitar 79,3% com um a cada "
         "16 dias? É escolha de negócio, não técnica: depende de quanto custa uma ida a campo em falso "
         "contra o valor de pegar mais um evento. Recomendo decidir antes de gerar o bundle, porque o "
         "ponto fica gravado nele."),
        ("Decisão de escopo · time", "Qual candidato vira a referência oficial?",
         "Existem três — v9, v10 e agora o b2024 — e nenhum está deployado. O README aponta v9; a "
         "documentação interna preferia v10; o b2024 é o melhor no TC382_03_A e o pior no T5. Isso trava "
         "a geração do bundle há semanas. Recomendação: b2024 no TC382_03_A e a receita de controle no "
         "T5, assumindo de vez que a configuração é por sensor."),
    ]:
        fig.text(0.055, y, lab.upper(), fontsize=7.2, color=ACCENT, family=MONO, weight="bold")
        y -= 0.026
        fig.text(0.055, y, q, fontsize=11, color=INK, family=SANS, weight="bold")
        y -= 0.030
        y = para(fig, 0.055, y, d, size=9.3) - 0.016

    y = heading(fig, 0.055, y - 0.008, "Pendências prontas para executar")
    for t in ["Gerar o bundle de inferência do TC382_03_A/b2024 — hoje ele só existe como task do "
              "ClearML, não como artefato de produção.",
              "Implementar a graduação de confiança no bundle (pós-processamento, sem retreino)."]:
        y = bullet(fig, 0.068, y, t)

    y = heading(fig, 0.055, y - 0.008, "Pendências que dependem de dado da Petrobras")
    for t in ["Re-extração do NGP_A. O export record_2025 está truncado no primeiro dia de cada mês "
              "(~1000 registros por tag). Só janeiro e maio são utilizáveis em 2025, o que impede "
              "auditar agosto com o árbitro físico de 'equipamento ligado'.",
              "Séries dos sensores novos nos anos anteriores (já solicitadas). Sem elas, o TI_0305 fica "
              "como piloto de 4 incidentes."]:
        y = bullet(fig, 0.068, y, t)

    callout(fig, 0.055, y - 0.014, 0.89, "Ressalva de método que vale repetir",
            ["'Falso positivo' significa 'o DCS não alarmou', não 'não estava acontecendo nada'. Se o "
             "modelo pegar uma degradação real que nunca cruzou o setpoint de HI, ela nos penaliza. Foi "
             "só olhando curva a curva que confirmamos que os FPs de agosto/2025 eram rampa de descarga.",
             "Reprodutível a partir de sweep_onset_rules_offline.py, sweep_load_gate_offline.py e "
             "eval_confidence_grading_offline.py — os três reproduzem o ponto de operação da auditoria "
             "antes de comparar e abortam se não bater."])
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def main() -> None:
    for f in (FIG_FRENTEB, FIG_ZOOM):
        if not os.path.exists(f):
            raise SystemExit(f"figura ausente: {f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        page_capa(pdf)
        page_estado(pdf)
        page_frenteb(pdf)
        page_zoom(pdf)
        page_fp(pdf)
        page_fronteira(pdf)
        page_decisoes(pdf)
        d = pdf.infodict()
        d["Title"] = "Cabiúnas — Detecção de anomalias: estado e decisões pendentes"
        d["Author"] = "Equipe de modelagem"
        d["Subject"] = "Relatório de sessão 10/08/2026 — Frente B, falso positivo, encaminhamento"
    print(f"PDF: {OUT}  ({_page['n'] + 1} páginas)")


if __name__ == "__main__":
    main()
