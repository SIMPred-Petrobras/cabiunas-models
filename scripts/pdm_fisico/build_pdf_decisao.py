#!/usr/bin/env python3
"""Gera o PDF do relatorio de decisao a partir do MESMO HTML publicado como artifact.

POR QUE ASSIM, E NAO COM matplotlib/PdfPages. Os outros relatorios do projeto sao
desenhados celula a celula em PdfPages porque nasceram de figuras. Este nasceu como
pagina: refaze-lo em matplotlib duplicaria o conteudo em duas fontes que divergem na
primeira correcao. Renderizando o proprio HTML pelo Chrome headless, PDF e pagina
compartilham a unica fonte -- corrigiu um numero, corrige nos dois.

O QUE MUDA NO MODO IMPRESSAO. Tema claro forcado (o PDF nao tem tema do leitor),
margens A4, controle de quebra de pagina (nota/cartao/figura/tabela nao partem ao
meio), tooltip escondido e o SVG reescalado para caber na largura util.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
# A fonte vive NO REPOSITORIO, ao lado deste script: o mesmo arquivo publicado como
# artifact. Uma unica fonte para os dois formatos -- corrigiu um numero, corrige nos
# dois. `HTML_DECISAO` permite apontar para outro arquivo sem editar o script.
FONTE = os.environ.get("HTML_DECISAO", os.path.join(AQUI, "relatorio_decisao.html"))
SAIDA = os.path.join(AQUI, "..", "..", "RELATORIO_DECISAO_DETECTOR.pdf")

PRINT_CSS = """
<style>
@page { size: A4; margin: 15mm 13mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { padding: 0 !important; background: #ffffff !important; font-size: 10.4pt; line-height: 1.55; }
.wrap { max-width: 100% !important; }
.prose { max-width: 62ch; }

header { padding: 0 0 18px !important; margin-bottom: 26px !important; }
h1 { font-size: 30pt !important; }
.standfirst { font-size: 11.6pt; }
.eyebrow { font-size: 8pt; margin-bottom: 14px; }

/* nada de orfao: cabecalho de secao nunca fica sozinho no fim da pagina */
section { break-inside: auto; margin-bottom: 30px !important; }
.shead { break-after: avoid; break-inside: avoid; }
h2, h3 { break-after: avoid; break-inside: avoid; }
h2 { font-size: 17pt !important; }
h3 { font-size: 12.5pt !important; margin-top: 20px !important; }
p, li { orphans: 3; widows: 3; }

.verdict, .note, .op, figure, .tw { break-inside: avoid; }
.verdict { margin-bottom: 30px !important; padding: 20px 24px !important; }
.stat .k { font-size: 21pt !important; }
.stat .l { font-size: 7.4pt !important; }
.ops { break-inside: avoid; }
.op .big { font-size: 18pt !important; }

table { font-size: 9pt; min-width: 0 !important; }
th, td { padding: 6px 10px !important; }
thead th { font-size: 7.6pt !important; }
td.n, th.n { font-size: 8.6pt !important; }
.tw { overflow: visible !important; }
.cap { font-size: 8.4pt !important; }

/* no papel nada rola: o que era scroll-x vira quebra de linha */
.cfg { white-space: normal !important; overflow: visible !important; font-size: 8.6pt !important; line-height: 1.7; }

figure { padding: 14px 12px 8px !important; }
.svg-scroll { overflow: visible !important; }
svg { min-width: 0 !important; width: 100% !important; height: auto; }
.chart-ttl { font-size: 11.5pt; }
/* a legenda inteira numa linha so -- quebrada, ela empurra o grafico pra baixo */
.chart-hd { flex-wrap: nowrap !important; align-items: center; }
.legend { font-size: 8.6pt; flex-wrap: nowrap !important; gap: 0 14px !important; }
.legend span { white-space: nowrap; }

/* o SVG e reescalado para ~77% na largura util da A4; compensa no tamanho do texto */
.tick     { font-size: 13px !important; }
.axlab    { font-size: 15px !important; }
.dlab     { font-size: 14px !important; }
.bandlab  { font-size: 13px !important; }

.note p { font-size: 9.8pt; }
.note .lbl { font-size: 7.6pt !important; }

#tip { display: none !important; }
.mark { cursor: default; }

footer { break-inside: avoid; margin-top: 34px !important; }
footer p { font-size: 8.2pt !important; }
</style>
"""


def main() -> int:
    if not os.path.exists(FONTE):
        print(f"erro: HTML de origem nao encontrado em {FONTE}", file=sys.stderr)
        return 1
    with open(FONTE, encoding="utf-8") as fh:
        corpo = fh.read()

    # O artifact e publicado sem <html>/<head>/<body> -- o host embrulha na hora.
    # Para o Chrome precisamos do documento completo: separa <title>/<link>/<style>
    # para o head e manda o resto (marcacao + <script>) para o body, com o tema
    # claro cravado no elemento raiz.
    # cada padrao fecha no seu proprio terminador -- um `.*?>` generico casaria no `>`
    # da PROPRIA tag de abertura e deixaria o CSS vazar para o corpo como texto.
    HEAD = re.compile(
        r"<style\b[^>]*>.*?</style>|<title\b[^>]*>.*?</title>|<link\b[^>]*>",
        re.S | re.I,
    )
    head_bits = HEAD.findall(corpo)
    body = HEAD.sub("", corpo)
    doc = (
        '<!doctype html><html lang="pt-BR" data-theme="light"><head>'
        '<meta charset="utf-8">'
        + "".join(head_bits) + PRINT_CSS
        + "</head><body>" + body + "</body></html>"
    )

    with tempfile.TemporaryDirectory() as tmp:
        entrada = os.path.join(tmp, "relatorio.html")
        with open(entrada, "w", encoding="utf-8") as fh:
            fh.write(doc)

        saida = os.path.abspath(SAIDA)
        cmd = [
            "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",          # deixa fonte + JS do grafico terminarem
            f"--print-to-pdf={saida}",
            f"--user-data-dir={os.path.join(tmp, 'perfil')}",
            "file://" + entrada,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not os.path.exists(saida):
            print("erro do Chrome:", r.stderr[-2000:], file=sys.stderr)
            return 1

    print(f"-> {saida}  ({os.path.getsize(saida)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
