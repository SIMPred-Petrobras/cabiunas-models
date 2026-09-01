"""Gera os formatos de entrega da documentação do detector a partir de um fonte único.

Fonte: docs/detector_tc33003a.html (documento HTML completo, autocontido).
Saídas:
    reports/DOCUMENTACAO_DETECTOR_TC33003A.html   cópia de entrega
    reports/DOCUMENTACAO_DETECTOR_TC33003A.pdf    via google-chrome headless
    reports/DOCUMENTACAO_DETECTOR_TC33003A.docx   via libreoffice
    <tmp>/artifact_body.html                      corpo sem <head>, para publicar

Uso:  python docs/build_doc.py [--pdf] [--docx] [--artifact]
"""
import pathlib, re, shutil, subprocess, sys, tempfile

RAIZ = pathlib.Path(__file__).resolve().parents[1]
FONTE = RAIZ / "docs" / "detector_tc33003a.html"
SAIDA = RAIZ / "reports"
BASE = "DOCUMENTACAO_DETECTOR_TC33003A"


def corpo(html: str) -> str:
    """Só o miolo: o publicador de artifact injeta o próprio <head>."""
    m = re.search(r"<title>.*?</style>\s*</head>\s*<body>\s*(.*)\s*</body>", html, re.S)
    if not m:
        raise SystemExit("não achei o corpo do documento no fonte")
    cabeca = re.search(r"(<title>.*?</style>)", html, re.S).group(1)
    return cabeca + "\n\n" + m.group(1) + "\n"


def para_docx(html: str) -> str:
    """Word e LibreOffice não importam grid: o placar vira tabela de verdade."""
    def tabela(m):
        celulas = re.findall(r"<div><b>(.*?)</b><span>(.*?)</span></div>", m.group(1), re.S)
        linhas = "".join('<tr><td class="v"><b>%s</b></td><td>%s</td></tr>' % c for c in celulas)
        return '<div class="scroller"><table><tbody>%s</tbody></table></div>' % linhas
    return re.sub(r'<div class="placar">(.*?)</div>\s*</div>',
                  lambda m: tabela(m) + "</div>", html, flags=re.S)


def main() -> None:
    html = FONTE.read_text(encoding="utf-8")
    SAIDA.mkdir(exist_ok=True)
    entrega = SAIDA / f"{BASE}.html"
    shutil.copyfile(FONTE, entrega)
    print(f"html   {entrega} ({len(html):,} bytes)")

    if "--pdf" in sys.argv or len(sys.argv) == 1:
        pdf = SAIDA / f"{BASE}.pdf"
        pdf.unlink(missing_ok=True)
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--virtual-time-budget=10000", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", str(entrega)],
                       check=True, capture_output=True)
        print(f"pdf    {pdf} ({pdf.stat().st_size:,} bytes)")

    if "--docx" in sys.argv or len(sys.argv) == 1:
        tmp = pathlib.Path(tempfile.mkdtemp()) / f"{BASE}.html"
        tmp.write_text(para_docx(html), encoding="utf-8")
        docx = SAIDA / f"{BASE}.docx"
        docx.unlink(missing_ok=True)
        subprocess.run(["soffice", "--headless", "--convert-to", "docx:MS Word 2007 XML",
                        "--outdir", str(SAIDA), str(tmp)], check=True, capture_output=True)
        print(f"docx   {docx} ({docx.stat().st_size:,} bytes)")

    if "--artifact" in sys.argv or len(sys.argv) == 1:
        alvo = pathlib.Path(tempfile.gettempdir()) / "artifact_body.html"
        alvo.write_text(corpo(html), encoding="utf-8")
        print(f"corpo  {alvo} (para publicar como artifact)")


if __name__ == "__main__":
    main()
