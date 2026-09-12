#!/usr/bin/env python3
"""DETECTOR_V2.md -> HTML -> PDF (Chrome headless).

Mesma abordagem de `build_pdf_decisao.py`: uma fonte so, o PDF e derivado. O
conversor cobre o subconjunto de Markdown usado no documento -- titulos, tabelas
com pipe, negrito, codigo, citacao, regua e listas. Nao usa biblioteca externa
porque o venv nao tem `markdown` e o subconjunto e pequeno.
"""
from __future__ import annotations
import html as H
import re, subprocess, sys, os

AQUI = os.path.dirname(os.path.abspath(__file__))
FONTE = os.path.join(AQUI, "DETECTOR_V2.md")
SAIDA = os.path.join(AQUI, "..", "..", "DETECTOR_V2.pdf")

CSS = """
@page { size: A4; margin: 17mm 15mm 15mm 15mm; }
* { box-sizing: border-box; }
body { font: 10.2pt/1.5 "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
       color: #16202a; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 2mm; color: #0b2d4d; letter-spacing: -.3px; }
h1 + p { color: #5a6b7c; font-size: 9.4pt; margin: 0 0 7mm;
         border-bottom: 2px solid #0b2d4d; padding-bottom: 3mm; }
h2 { font-size: 13pt; margin: 8mm 0 2.5mm; color: #0b2d4d;
     border-bottom: 1px solid #c9d6e2; padding-bottom: 1.2mm;
     page-break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 2mm; color: #1d4e77; page-break-after: avoid; }
p { margin: 0 0 2.6mm; }
ol, ul { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt;
       background: #eef3f8; padding: .4mm 1.1mm; border-radius: 2px; color: #0b3d62; }
pre { background: #f6f9fc; border: 1px solid #d7e3ee; border-left: 3px solid #0b2d4d;
      padding: 2.6mm 3mm; font-family: "DejaVu Sans Mono", monospace;
      font-size: 8.5pt; line-height: 1.45; overflow-x: auto;
      page-break-inside: avoid; margin: 0 0 3.5mm; }
pre code { background: none; padding: 0; color: inherit; }
blockquote { margin: 3mm 0; padding: 2.4mm 4mm; background: #fff8e6;
             border-left: 3px solid #d99b0a; font-size: 10pt; page-break-inside: avoid; }
blockquote p { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 0 0 4mm;
        font-size: 9.1pt; page-break-inside: avoid; }
th { background: #0b2d4d; color: #fff; text-align: left; padding: 1.7mm 2.2mm;
     font-weight: 600; font-size: 8.9pt; }
td { padding: 1.5mm 2.2mm; border-bottom: 1px solid #dde6ee; vertical-align: top; }
tr:nth-child(even) td { background: #f7fafd; }
td:first-child { white-space: nowrap; }
hr { border: 0; border-top: 1px solid #c9d6e2; margin: 6mm 0; }
strong { color: #0b2d4d; }
/* negrito dentro do cabecalho herda o branco -- sem isto o texto fica
   azul-escuro sobre azul-escuro e SOME (pego na inspecao visual do PDF) */
th strong, th code, th em { color: inherit; background: none; }
"""

def inline(t: str) -> str:
    t = H.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    return t

def converte(md: str) -> str:
    out, i, linhas = [], 0, md.split("\n")
    while i < len(linhas):
        L = linhas[i]
        if L.startswith("```"):                       # bloco de codigo
            i += 1; buf = []
            while i < len(linhas) and not linhas[i].startswith("```"):
                buf.append(H.escape(linhas[i])); i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>"); i += 1; continue
        if re.match(r"^\|.*\|\s*$", L) and i+1 < len(linhas) \
           and re.match(r"^\|[\s:|-]+\|\s*$", linhas[i+1]):
            cels = lambda s: [c.strip() for c in s.strip().strip("|").split("|")]
            cab = cels(L); i += 2; corpo = []
            while i < len(linhas) and re.match(r"^\|.*\|\s*$", linhas[i]):
                corpo.append(cels(linhas[i])); i += 1
            t = ["<table><thead><tr>"] + [f"<th>{inline(c)}</th>" for c in cab] \
                + ["</tr></thead><tbody>"]
            for r in corpo:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("".join(t) + "</tbody></table>"); continue
        if L.startswith("> "):
            buf = []
            while i < len(linhas) and linhas[i].startswith("> "):
                buf.append(linhas[i][2:]); i += 1
            out.append("<blockquote><p>" + inline(" ".join(buf)) + "</p></blockquote>"); continue
        m = re.match(r"^(#{1,4})\s+(.*)$", L)
        if m:
            n = len(m.group(1)); out.append(f"<h{n}>{inline(m.group(2))}</h{n}>"); i += 1; continue
        if re.match(r"^---+\s*$", L):
            out.append("<hr>"); i += 1; continue
        if re.match(r"^\s*[-*]\s+", L) or re.match(r"^\s*\d+\.\s+", L):
            ord_ = bool(re.match(r"^\s*\d+\.", L)); tag = "ol" if ord_ else "ul"
            itens = []
            while i < len(linhas) and (re.match(r"^\s*[-*]\s+", linhas[i])
                                       or re.match(r"^\s*\d+\.\s+", linhas[i])):
                itens.append(inline(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", linhas[i]))); i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{x}</li>" for x in itens) + f"</{tag}>")
            continue
        if L.strip():
            buf = []
            while i < len(linhas) and linhas[i].strip() and not re.match(
                    r"^(#{1,4}\s|\||>|```|---+\s*$|\s*[-*]\s|\s*\d+\.\s)", linhas[i]):
                buf.append(linhas[i]); i += 1
            out.append("<p>" + inline(" ".join(buf)) + "</p>"); continue
        i += 1
    return "\n".join(out)


def main() -> int:
    md = open(FONTE, encoding="utf-8").read()
    corpo = converte(md)
    doc = (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
           f"<title>Detector v2 — TC-330.03A</title><style>{CSS}</style></head>"
           f"<body>{corpo}</body></html>")
    tmp = os.path.join(AQUI, "_detector_v2.html")
    open(tmp, "w", encoding="utf-8").write(doc)
    r = subprocess.run([
        "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", f"--print-to-pdf={os.path.abspath(SAIDA)}",
        "--virtual-time-budget=6000", f"file://{tmp}"],
        capture_output=True, text=True, timeout=180)
    if not os.path.exists(SAIDA):
        print(r.stderr[-900:]); return 1
    print(f"-> {os.path.abspath(SAIDA)}  ({os.path.getsize(SAIDA)/1024:.0f} KB)")
    os.remove(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
