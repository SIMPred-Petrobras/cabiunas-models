# Como refazer o PDF da integração

`INTEGRACAO_DASHBOARD.md` é a fonte de verdade; este HTML é a versão diagramada
dela, e o PDF sai do HTML. Mudou o markdown, mude o HTML aqui também.

```bash
python3 build_pdf.py integracao.html ../../../../INTEGRACAO_DASHBOARD_TC33003A.pdf
```

Precisa do `google-chrome` no PATH (headless). Sem rede, as fontes IBM Plex caem
para a fallback do sistema e o layout continua correto.

A folha de impressão deixa a **seção** quebrar entre páginas e protege só os
blocos (tabela, placar, checklist, código). Proteger a seção inteira, como faz o
`build_pdf.py` dos outros relatórios, deixava páginas 3/4 vazias: 11 páginas
contra as 9 de agora.

Confira o resultado olhando, não só pelo número de páginas:

```bash
pdftoppm -png -r 72 ../../../../INTEGRACAO_DASHBOARD_TC33003A.pdf pg
```
