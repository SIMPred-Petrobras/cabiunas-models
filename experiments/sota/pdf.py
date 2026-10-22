#!/usr/bin/env python3
"""Markdown do relatório -> PDF (Chrome headless, A4, folha de impressão própria). O PDF fica fora do git, como o relatório.

Uso:  .venv/bin/python experiments/sota/pdf.py [reports/estado_da_arte_comparativo.md]
"""
import subprocess, sys
from pathlib import Path
import markdown

RAIZ = Path(__file__).resolve().parents[2]
FONTE = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "reports" / "estado_da_arte_comparativo.md"
SAIDA = FONTE.with_suffix(".pdf")

CSS = """
@page { size: A4; margin: 14mm 12mm 15mm; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; box-sizing: border-box; }
body { font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif; font-size: 9.6pt; line-height: 1.5; color: #16202E; background: #FFFFFF; margin: 0; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 6px; color: #0E3A52; }
h2 { font-size: 12.5pt; margin: 20px 0 6px; padding-bottom: 3px; border-bottom: 1px solid #D2DCE6; color: #0E3A52; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 14px 0 4px; color: #0E3A52; break-after: avoid; }
p, li { margin: 4px 0; }
ul, ol { padding-left: 18px; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt; background: #F0F4F8; padding: 0 3px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 7.4pt; line-height: 1.3; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #F0F4F8; color: #16202E; text-align: left; font-weight: 600; }
th, td { border: 1px solid #D2DCE6; padding: 3px 4px; vertical-align: top; word-break: normal; overflow-wrap: normal; hyphens: manual; }
tbody tr:nth-child(even) td { background: #FAFBFC; }
img { max-width: 100%; display: block; margin: 8px auto 12px; break-inside: avoid; }
strong { color: #0B1A28; }
"""


def main():
    corpo = markdown.markdown(FONTE.read_text(encoding="utf-8"), extensions=["tables", "sane_lists"])
    html = FONTE.with_name("_print.html")
    html.write_text(f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>{FONTE.stem}</title><style>{CSS}</style></head><body>{corpo}</body></html>", encoding="utf-8")
    r = subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer", f"--print-to-pdf={SAIDA}",
                        "--virtual-time-budget=10000", html.resolve().as_uri()], capture_output=True, text=True, timeout=240)
    if not SAIDA.exists():
        print("falhou:", r.stderr[-800:]); sys.exit(1)
    html.unlink(); print(f"-> {SAIDA} ({SAIDA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
