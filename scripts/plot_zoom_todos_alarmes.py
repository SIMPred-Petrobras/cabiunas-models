#!/usr/bin/env python3
"""
plot_zoom_todos_alarmes.py
Um recorte por incidente HI/HIHI do TC382_03_A, com 7 dias de contexto ANTES do alarme —
o mesmo formato do recorte de 10–18/01/2025, aplicado a todos.

Sai em PDF de uma página por alarme (dá para folhear) mais um CSV de resumo com, por
incidente: se cada braço detectou, a antecedência REAL (sem o teto de 8 h da métrica),
quantos episódios de falso positivo caem no recorte e quanto tempo a máquina ficou ligada.

⚠️ Ponto de operação GLOBAL, calculado uma vez sobre toda a janela de avaliação e reusado
em todos os recortes. Nenhuma página é recalibrada — do contrário cada figura mostraria um
desempenho que o sistema não tem.

O CSV é o que responde perguntas agregadas ("em quantos o AE chegou antes?"); o PDF é para
inspecionar mecanismo. Nenhum dos dois muda o veredito estatístico: a diferença entre os
braços é de 1 incidente em 58.

Uso:
    PYTHONPATH=. python scripts/plot_zoom_todos_alarmes.py
    PYTHONPATH=. python scripts/plot_zoom_todos_alarmes.py --dias_antes 7 --dias_depois 2
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zj = _load("plot_zoom_janela")

OUT_PDF = "relatorio_anexos/recortes_alarmes_TC382_03_A.pdf"
OUT_CSV = "eval_predictive_out/recortes_alarmes_TC382_03_A.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias_antes", type=float, default=7.0)
    ap.add_argument("--dias_depois", type=float, default=2.0)
    ap.add_argument("--png_dir", default=None, help="também salva PNG por incidente")
    args = ap.parse_args()

    ctx = zj.prepare()
    inc = ctx["inc_glob"]
    print(f"{len(inc)} incidentes · {args.dias_antes:.0f} d antes + {args.dias_depois:.0f} d depois")

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)

    linhas = []
    fps_unicos = {"ae": set(), "temp": set()}   # janelas se sobrepõem: deduplicar
    with PdfPages(OUT_PDF) as pdf:
        for i, t in enumerate(inc, 1):
            A = t - pd.Timedelta(days=args.dias_antes)
            B = t + pd.Timedelta(days=args.dias_depois)
            fig, axes = plt.subplots(4, 1, figsize=(13.5, 9.8), sharex=True,
                                     gridspec_kw={"height_ratios": [1.0, 0.14, 1, 1]})
            fig.patch.set_facecolor("white")
            tit = (f"Incidente {i}/{len(inc)} — alarme em {t.strftime('%d/%m/%Y %H:%M')} UTC"
                   f"   ·   {zj.SENSOR}   ·   ponto de operação GLOBAL, não recalibrado")
            st = zj.draw_zoom(axes, A, B, ctx, tit, alvo=t)
            zj.legenda(axes[-1])
            fig.tight_layout()
            pdf.savefig(fig, facecolor="white")
            if args.png_dir:
                fig.savefig(os.path.join(args.png_dir,
                                         f"{i:02d}_{t.strftime('%Y%m%d_%H%M')}.png"),
                            dpi=130, facecolor="white")
            plt.close(fig)

            for k in ("ae", "temp"):
                fps_unicos[k].update(st[f"{k}_fps"])

            linhas.append(dict(
                n=i, alarme=t, inicio=A, fim=B, frac_on=st["frac_on"], t_max=st["t_max"],
                n_inc_na_janela=st["n_inc"],
                ae_detectou=st["ae_detectou"], ae_lead_h=st["ae_lead_h"], ae_fp=st["ae_n_fp"],
                temp_detectou=st["temp_detectou"], temp_lead_h=st["temp_lead_h"],
                temp_fp=st["temp_n_fp"]))
            print(f"  [{i:>2}/{len(inc)}] {t.strftime('%Y-%m-%d %H:%M')}  "
                  f"AE={'sim' if st['ae_detectou'] else 'NÃO':<3} "
                  f"lead={st['ae_lead_h'] if st['ae_lead_h'] is None else round(st['ae_lead_h'],1)}  |  "
                  f"limiar={'sim' if st['temp_detectou'] else 'NÃO':<3} "
                  f"lead={st['temp_lead_h'] if st['temp_lead_h'] is None else round(st['temp_lead_h'],1)}",
                  flush=True)

    df = pd.DataFrame(linhas)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nPDF: {OUT_PDF}\nCSV: {OUT_CSV}")

    d = df
    print("\n=== RESUMO ===")
    print(f"  incidentes                         {len(d)}")
    print(f"  detectados pelo AE                 {int(d.ae_detectou.sum())}")
    print(f"  detectados pelo limiar             {int(d.temp_detectou.sum())}")
    print(f"  só o AE pegou                      {int((d.ae_detectou & ~d.temp_detectou).sum())}")
    print(f"  só o limiar pegou                  {int((~d.ae_detectou & d.temp_detectou).sum())}")
    print(f"  nenhum dos dois                    {int((~d.ae_detectou & ~d.temp_detectou).sum())}")
    amb = d[d.ae_detectou & d.temp_detectou].dropna(subset=["ae_lead_h", "temp_lead_h"])
    if len(amb):
        print(f"\n  nos {len(amb)} que AMBOS pegaram, antecedência real mediana:")
        print(f"    AE      {amb.ae_lead_h.median():5.1f} h")
        print(f"    limiar  {amb.temp_lead_h.median():5.1f} h")
        print(f"    limiar avisou ANTES em {int((amb.temp_lead_h > amb.ae_lead_h).sum())}/{len(amb)}")
    print(f"\n  episódios FP DISTINTOS que caem nos recortes (deduplicados):")
    print(f"    AE      {len(fps_unicos['ae'])}")
    print(f"    limiar  {len(fps_unicos['temp'])}")
    print(f"  (somar as páginas daria {int(d.ae_fp.sum())} e {int(d.temp_fp.sum())} — "
          f"as janelas de 9 dias se sobrepõem e recontariam o mesmo episódio)")
    print(f"\n  Total de episódios FP na janela inteira (auditoria):  "
          f"AE {len(ctx['arms']['ae']['fps'])}  ·  limiar {len(ctx['arms']['temp']['fps'])}")


if __name__ == "__main__":
    main()
