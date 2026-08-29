#!/usr/bin/env python3
"""Monta RELATORIO_EXECUCAO_WORKER_TC33003A.pdf -- o detector rodado do zero num
worker do ClearML, nao mais so na minha maquina.

Reusa as primitivas de scripts/build_relatorio_pdf.py. As tres figuras (linha do
tempo, lead por evento, teste de nulo) sao as mesmas geradas para o artifact HTML
publicado em 29/08/2026, ja salvas em fig_resultado_*.png.

Uso:  cd scripts/pdm_fisico && python build_pdf_worker.py
"""
from __future__ import annotations
import sys
sys.path.insert(0, "../..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scripts.build_relatorio_pdf import (
    W, H, INK, INK2, MUTED, RULE, GROUND, ACCENT, AMP, CTX, GOOD, CRIT, SANS, MONO,
    _page, new_page as _np, para, bullet, heading, table, callout, full_image)

OUT = "../../RELATORIO_EXECUCAO_WORKER_TC33003A.pdf"
RODAPE = "TC-330.03A · Cabiúnas · execução remota no ClearML · 29/08/2026"


def new_page(pdf, title=None, eyebrow=None):
    fig = _np(pdf, title, eyebrow)
    for t in list(fig.texts):
        if t.get_text().startswith("Cabiúnas · Turbina A"):
            t.set_text(RODAPE)
    return fig


def scorebox(fig, x, y, w, h, val, lbl, color=INK):
    fig.patches.append(plt.Rectangle((x, y - h), w, h, transform=fig.transFigure,
                                     facecolor="white", edgecolor=RULE, lw=0.8, zorder=0))
    fig.text(x + w/2, y - h*0.42, val, fontsize=15.5, color=color, family=SANS,
             weight="bold", ha="center", va="center")
    fig.text(x + w/2, y - h*0.80, lbl, fontsize=7.0, color=MUTED, family=SANS,
             ha="center", va="center", wrap=True)


# ---------------------------------------------------------------- capa
def page_capa(pdf):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    fig.text(0.075, 0.83, "EXECUÇÃO REMOTA · CLEARML", fontsize=8, color=ACCENT, family=MONO)
    fig.text(0.075, 0.735, "O detector, rodado", fontsize=32, color=INK, family=SANS, weight="bold")
    fig.text(0.075, 0.675, "do zero num worker", fontsize=32, color=INK, family=SANS, weight="bold")
    fig.text(0.075, 0.615, "TC-330.03A · Cabiúnas", fontsize=14, color=INK2, family=SANS)
    fig.lines.append(plt.Line2D([0.075, 0.40], [0.575, 0.575], transform=fig.transFigure,
                                color=INK, lw=1.6))
    y = 0.525
    for t in ["Não é mais um número que saiu da minha máquina.",
              "O modelo foi reajustado e pontuado inteiramente dentro de um worker do "
              "ClearML, a partir só do dataset publicado.",
              "O resultado reproduziu, na casa decimal, o calculado localmente."]:
        y = para(fig, 0.075, y, t, size=11, width=52, color=INK2)

    # placar em caixas
    px = 0.075; py = 0.375; pw = (0.85 - 3*0.014) / 4
    stats = [("8/8", "paradas\nantecipadas", ACCENT),
             ("1,12", "FP por mês\nde operação", INK),
             ("39,0 h", "em alarme\nfalso por mês", INK),
             ("29,0 h", "antecedência\nmédia", INK)]
    for i, (v, l, c) in enumerate(stats):
        scorebox(fig, px + i*(pw+0.014), py, pw, 0.10, v, l, c)
    stats2 = [("p = 0,0000", "teste de nulo\n20.000 permutações", ACCENT),
              ("7/8", "LOEO aninhado\n(perde 04/11/2025)", AMP),
              ("0,95", "FP/mês na régua\ndo PCA walk-forward", INK),
              ("21", "episódios em\n11,6 meses", INK)]
    for i, (v, l, c) in enumerate(stats2):
        scorebox(fig, px + i*(pw+0.014), py - 0.115, pw, 0.10, v, l, c)

    fig.text(0.075, 0.145, "worker  clearml-worker-1", fontsize=8.3, color=MUTED, family=MONO)
    fig.text(0.075, 0.122, "projeto  TesteMLCab", fontsize=8.3, color=MUTED, family=MONO)
    fig.text(0.075, 0.099, "régua    janela 48 h · gap 2 h · FP por mês de operação (730 h)",
             fontsize=8.3, color=MUTED, family=MONO)
    fig.text(0.075, 0.06, RODAPE, fontsize=8, color=MUTED, family=SANS)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- execucao
def page_execucao(pdf):
    fig = new_page(pdf, "Como a execução aconteceu", "01")
    y = para(fig, 0.055, 0.845,
             "Até esta rodada, todo número publicado tinha saído da minha máquina — "
             "reproduzível, mas dependente do meu disco. Esta execução muda isso: as "
             "entradas foram publicadas como Dataset do ClearML e o script que ajusta o "
             "PCA mês a mês, monta os quatro sinais e aplica a régua foi enfileirado para "
             "rodar dentro de um worker, sem tocar em nada do meu ambiente local.",
             width=120)
    rows = [
        ("01", "Dataset publicado", "grade de 2 min, cache de sinais, alvo de paradas e os CSVs "
         "dos estudos de comparação — 95 MiB, versionado no ClearML."),
        ("02", "Script autocontido enfileirado", "mesma convenção do detector PCA walk-forward: "
         "um único arquivo, sem imports locais, para que qualquer worker o execute."),
        ("03", "Duas falhas de ambiente, corrigidas no caminho", "a imagem docker padrão do "
         "agente (nvidia/cuda…ubuntu20.04) tem Python 3.8 e não encontra as versões de numpy/"
         "pandas fixadas na minha máquina — trocada pela imagem TensorFlow que o resto do "
         "repositório já usa. Depois, o container acusou pytz ausente ao ler os timestamps "
         "com fuso do parquet — adicionado aos requisitos."),
        ("04", "Reajuste completo dentro do worker", "PCA walk-forward mensal, os quatro "
         "sinais, EWMA, canal duplo degrau/CUSUM, voto ≥ 2, refratário e duração mínima — "
         "tudo recalculado do dataset, não copiado de um cache."),
        ("05", "Resultado idêntico ao calculado localmente", "na casa decimal: 8/8 · 1,12 "
         "FP/mês · 39,0 h/mês · lead 29,03 h. A garantia que a execução remota acrescenta é "
         "que o resultado não depende do meu ambiente."),
    ]
    y -= 0.008
    for n, t, s in rows:
        fig.text(0.055, y, n, fontsize=9, color=ACCENT, family=MONO, weight="bold")
        fig.text(0.088, y, t, fontsize=9.6, color=INK, family=SANS, weight="bold")
        y -= 0.026
        y = para(fig, 0.088, y, s, size=8.8, width=112, leading=0.0205)
        y -= 0.006
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- figuras
def page_figura(pdf, path, eyebrow, titulo, legenda):
    """full_image() usa a caixa inteira sem respeitar a proporcao da imagem --
    sobra vazio nas figuras baixas e largas. Aqui a imagem e encaixada por
    proporcao real (contida, centralizada), como um object-fit: contain."""
    import matplotlib.image as mpimg
    fig = new_page(pdf, titulo, eyebrow)
    top, bottom, left, right = 0.855, 0.20, 0.055, 0.945
    box_w_in, box_h_in = (right - left) * W, (top - bottom) * H
    im = mpimg.imread(path)
    img_h, img_w = im.shape[0], im.shape[1]
    img_ratio = img_w / img_h
    box_ratio = box_w_in / box_h_in
    if img_ratio > box_ratio:
        w_in = box_w_in; h_in = w_in / img_ratio
    else:
        h_in = box_h_in; w_in = h_in * img_ratio
    w_fr, h_fr = w_in / W, h_in / H
    x0 = left + ((right - left) - w_fr) / 2
    y0 = top - h_fr                       # alinhado ao topo, nao centralizado --
    ax = fig.add_axes([x0, y0, w_fr, h_fr])  # figuras largas nao deixam vazio embaixo
    ax.imshow(im); ax.axis("off")
    para(fig, 0.055, y0 - 0.045, legenda, size=8.8, width=124)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- ressalvas
def page_ressalvas(pdf):
    fig = new_page(pdf, "As três ressalvas que acompanham o placar", "05")
    y = 0.845
    y = callout(fig, 0.055, y, 0.89, "LOEO aninhado é 7/8, não 8/8", [
        "Sob qualquer regra de desempate razoável entre configurações empatadas, uma parada "
        "não sobrevive à remoção do próprio evento da busca: 04/11/2025, detectada por apenas "
        "8 das 72 configurações dentro do orçamento de falso positivo. É o mesmo evento que "
        "já havia escapado de cinco intervenções anteriores e o único dos oito sem precursor "
        "físico atribuível."], color=AMP)
    y -= 0.012
    y = callout(fig, 0.055, y, 0.89, "29,0 h é média, não mediana", [
        "Quatro dos oito leads estão censurados exatamente em 48,0 h pela borda da janela de "
        "detecção. O número que define o que dá para fazer com o aviso é o mínimo real: 2,8 h "
        "(11/04/2025)."], color=CTX)
    y -= 0.012
    y = callout(fig, 0.055, y, 0.89, "Um episódio concentra um terço do custo", [
        "As 153,7 h de falso positivo de janeiro de 2025 — quando a máquina foi para carga "
        "alta e o PCA, ajustado em dado de nov–dez/2024, não conhecia aquele regime — "
        "respondem por 33,9% de todas as horas de alarme falso do período."], color=CRIT)
    y -= 0.02
    heading(fig, 0.055, y, "Onde reproduzir")
    y -= 0.032
    y = para(fig, 0.055, y,
             "Tarefa publicada em detector-fisico::TC33003A_4sinais_v1 (complementada pela "
             "execução remota …_v2_worker), projeto TesteMLCab, ao lado das reproduções "
             "pca-walkforward::. Reprodução local: scripts/pdm_fisico/roda_clearml.py "
             "--remote — o script é autocontido, sem dependência de nenhum outro arquivo do "
             "repositório.", size=9.0, width=118)
    pdf.savefig(fig); plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        page_capa(pdf)
        page_execucao(pdf)
        page_figura(pdf, "fig_resultado_serie.png", "02",
                    "A linha do tempo",
                    "21 episódios de alerta em 11,6 meses de operação elegível. Oito antecipam "
                    "uma parada catalogada (dentro da janela de 48 h que a termina); os treze "
                    "restantes são falso positivo. Dois deles — 08/01/2025 e 31/03/2025 — são "
                    "seguidos por uma parada real em menos de 48 h e se reclassificam como "
                    "acerto na régua do detector PCA walk-forward.")
        page_figura(pdf, "fig_resultado_lead.png", "03",
                    "A antecedência, evento a evento",
                    "A média de 29,0 h esconde uma característica importante: metade dos "
                    "eventos bate exatamente no teto da janela de 48 h porque o alerta já "
                    "estava ativo quando ela abriu — o lead real é maior, e não sabemos "
                    "quanto. O mínimo garantido, sem essa censura, é 2,8 h (11/04/2025).")
        page_figura(pdf, "fig_resultado_nulo.png", "04",
                    "Contra o acaso",
                    "Um detector com a mesma cobertura de alarme (24,7% das janelas de 48 h "
                    "antes de cada parada), mas disparando em instantes sorteados, acerta em "
                    "média 1,97 das 8 paradas. Em 20.000 sorteios, nenhum chegou a 8. O "
                    "detector publicado não é sorte.")
        page_ressalvas(pdf)
    print("gerado:", OUT)


if __name__ == "__main__":
    main()
