#!/usr/bin/env python3
"""HTML -> PDF via Chrome headless, com folha de impressao propria."""
import os, subprocess, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
FONTE = sys.argv[1]
SAIDA = sys.argv[2]

IMPRESSAO = """
@page { size: A4; margin: 15mm 14mm 16mm; }
/* PDF e papel: fixa o tema claro, venha o leitor de onde vier */
:root, :root[data-theme="dark"] {
  --ground:#FFFFFF; --surface:#FFFFFF; --surface-2:#F0F4F8; --sunken:#F4F7FA;
  --ink:#16202E; --ink-2:#3C4E63; --muted:#64768C;
  --hairline:#D2DCE6; --hairline-2:#BCCAD8;
  --accent:#0E5A78; --accent-soft:#E4F0F5; --accent-line:#8FBDD0;
  --ok:#2C6B50; --ok-soft:#E4F0E9;
  --warn:#8F5D0C; --warn-soft:#F7EEDD;
  --crit:#A03728; --crit-soft:#F8E7E3;
}
* { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
body { font-size: 10.2pt; line-height: 1.55; background:#FFFFFF; }
.wrap { max-width: none; padding-inline: 0; padding-block: 0; }
.hd { max-width: none; padding-inline: 0; padding-block: 0 18px; }
header { margin-bottom: 26px; }
.col { max-width: none; }
h1 { font-size: 22pt; } h2 { font-size: 14pt; } h3 { font-size: 11pt; }
/* A secao PODE quebrar: mante-la inteira deixava paginas 3/4 vazias.
   Quem nao pode quebrar sao os blocos -- tabela, placar, checklist, codigo. */
section { break-inside: auto; margin-bottom: 20px; }
.shead, h2, h3 { break-after: avoid; }
.placar, .tw, .check, .aviso, .nota, pre, footer { break-inside: avoid; }
.tw + h3, .check + h3, pre + h3 { margin-top: 24px; }
p + .tw, p + pre, p + .check { margin-top: 2px; }
thead { display: table-header-group; }
tr, .item { break-inside: avoid; }
.placar { grid-template-columns: repeat(4, 1fr); }
.placar .v { font-size: 1.4rem; }
.placar .k { font-size: .6rem; }
"""

doc = ("<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
       "<meta name='viewport' content='width=device-width,initial-scale=1'>"
       "</head><body>" + open(FONTE, encoding="utf-8").read()
       + f"<style>{IMPRESSAO}</style></body></html>")
tmp = os.path.join(AQUI, "_print.html")
open(tmp, "w", encoding="utf-8").write(doc)
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                "--no-pdf-header-footer", f"--print-to-pdf={os.path.abspath(SAIDA)}",
                "--virtual-time-budget=12000", f"file://{tmp}"],
               capture_output=True, text=True, timeout=240)
if os.path.exists(SAIDA):
    print(f"-> {SAIDA}  ({os.path.getsize(SAIDA)/1024:.0f} KB)")
    os.remove(tmp)
else:
    print("falhou"); sys.exit(1)
