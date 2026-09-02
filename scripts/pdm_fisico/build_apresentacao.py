#!/usr/bin/env python3
"""Monta o deck da apresentacao: embute as figuras no HTML e gera o PDF.

POR QUE UM BUILDER. O deck e versionavel (`apresentacao.html`, so texto), as figuras
nao (sao artefato, e a regra do projeto e commitar codigo). O builder junta os dois
na hora: le os marcadores {{FIG_*}} e troca por data URI. Assim o HTML no git
continua legivel e diffavel, e o arquivo publicado sai autocontido.

Uso:
    python build_apresentacao.py           # gera o HTML montado
    python build_apresentacao.py --pdf     # gera tambem o PDF (uma pagina por slide)
"""
from __future__ import annotations
import argparse
import base64
import os
import re
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
FONTE = os.path.join(AQUI, "apresentacao.html")
MONTADO = os.path.join(AQUI, "apresentacao_montada.html")
PDF = os.path.join(AQUI, "..", "..", "APRESENTACAO_TC33003A.pdf")

FIGS = {
    "FIG_ALVO": ("fig_alvo.png", "Três famílias de sinal do TC-330.03A ao longo de 16 meses, "
                                 "com os oito trips catalogados marcados"),
    "FIG_RESULTADO": ("fig_nosso_estilo_francisco.png",
                      "Série de temperatura de exaustão com os episódios do detector "
                      "classificados em antecipou falha, antes de parada e falso positivo"),
    "FIG_FRONTEIRA": ("fig_fronteira.png",
                      "Fronteira custo por detecção do detector de 4 sinais comparada aos "
                      "pontos publicados por Francisco e Lara"),
}

# no papel o slide vira pagina; a barra de navegacao e a sombra do palco somem
PRINT_CSS = """
<style>
@page { size: A4 landscape; margin: 0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { background: #ffffff !important; }
#deck { display: block !important; padding: 0 !important; gap: 0 !important; }
#bar { display: none !important; }
.slide {
  width: 100% !important; max-width: none !important; aspect-ratio: auto !important;
  height: 100vh !important; border: 0 !important; border-radius: 0 !important;
  break-after: page; break-inside: avoid; padding: 34px 44px 30px !important;
}
.slide:last-child { break-after: auto; }
</style>
"""


def monta() -> str:
    with open(FONTE, encoding="utf-8") as fh:
        html = fh.read()
    faltando = []
    for chave, (arquivo, alt) in FIGS.items():
        caminho = os.path.join(AQUI, arquivo)
        if not os.path.exists(caminho):
            faltando.append(arquivo)
            continue
        with open(caminho, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        tag = f'<img src="data:image/png;base64,{b64}" alt="{alt}">'
        html = html.replace("{{" + chave + "}}", tag)
        print(f"  {arquivo:34s} {len(b64)/1024:7.0f} KB em base64")
    if faltando:
        print("\nerro: figuras ausentes -> " + ", ".join(faltando), file=sys.stderr)
        print("gere com: python plota_alvo.py · plota_estilo_francisco.py · plota_fronteira.py",
              file=sys.stderr)
        raise SystemExit(1)
    restantes = re.findall(r"\{\{[A-Z_]+\}\}", html)
    if restantes:
        print(f"erro: marcador sem figura -> {restantes}", file=sys.stderr)
        raise SystemExit(1)
    return html


def gera_pdf(html: str) -> None:
    # o deck e publicado sem <html>/<head>/<body>; para o Chrome precisa do documento
    # completo. Cada padrao fecha no proprio terminador -- um `.*?>` generico casaria
    # no `>` da tag de abertura e deixaria o CSS vazar para o corpo como texto.
    HEAD = re.compile(r"<style\b[^>]*>.*?</style>|<title\b[^>]*>.*?</title>|<link\b[^>]*>",
                      re.S | re.I)
    cabeca = HEAD.findall(html)
    corpo = HEAD.sub("", html)
    doc = ('<!doctype html><html lang="pt-BR" data-theme="light"><head>'
           '<meta charset="utf-8">' + "".join(cabeca) + PRINT_CSS
           + "</head><body>" + corpo + "</body></html>")

    with tempfile.TemporaryDirectory() as tmp:
        entrada = os.path.join(tmp, "deck.html")
        with open(entrada, "w", encoding="utf-8") as fh:
            fh.write(doc)
        saida = os.path.abspath(PDF)
        r = subprocess.run([
            "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer", "--virtual-time-budget=25000",
            f"--print-to-pdf={saida}", f"--user-data-dir={os.path.join(tmp, 'perfil')}",
            "file://" + entrada,
        ], capture_output=True, text=True, timeout=240)
        if r.returncode != 0 or not os.path.exists(saida):
            print("erro do Chrome:", r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(1)
    print(f"-> {saida}  ({os.path.getsize(saida)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="gera tambem o PDF do deck")
    args = ap.parse_args()

    print("figuras embutidas:")
    html = monta()
    with open(MONTADO, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\n-> {MONTADO}  ({os.path.getsize(MONTADO)/1024/1024:.1f} MB)")
    if args.pdf:
        gera_pdf(html)
