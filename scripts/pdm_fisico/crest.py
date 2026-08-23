#!/usr/bin/env python3
"""O ganho do crest factor e informacao nova ou so sensibilidade a mais?

textura.py deu o primeiro 9/9 da investigacao: base (4 sinais) faz 8/9 a 106,7 h de
alarme/mes; base + crest factor da vibracao como 5o sinal faz 9/9 a 108,0 h/mes -- +1
evento por +1,2% de horas. Isso e barato demais para aceitar sem controle.

Ja caimos duas vezes na armadilha de Pareto nesta investigacao: comparar dois
detectores no limiar deles quando a escala do escore mudou confunde ganho de
discriminacao com dessensibilizacao. Aqui o controle e direto -- se dessensibilizar o
detector de 4 sinais ate gastar as MESMAS 108 h/mes tambem der 9/9, o crest nao
acrescenta nada. Varre-se k_base e k_vib para baixo (mais sensivel) exatamente para
isso.

Depois, LOEO: escolher o parametro nos outros 8 eventos e testar no que ficou de fora.
E o protocolo que falta no EXP10b/10c do Diego (la ramp_max e o limiar de volatilidade
foram escolhidos para preservar os preditivos do proprio OOS) -- nao faria sentido
cobrar isso dele e nao aplicar aqui.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from textura import textura, alerta_n

K_TX = [1.0, 1.4, 1.7, 2.1, 2.5, 3.0, 4.0]


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    stable = df["stable"].astype(bool)
    mask = mascara_pontuacao(df)

    print("montando 'out' e crest ...", flush=True)
    out = roda(BRACO, df, falhas)
    CR = textura(g, stable, falhas, 1, "crest")
    base = alerta_2k(out, mask, K_BASE, K_VIB)

    def m(al):
        x = A.avalia(al, falhas, mask)
        x.update(A.permuta(al, mask, x["det"], len(falhas)))
        return x

    b = m(base)
    print(f"\nbase (4 sinais): {b['det']}/9  {b['episodios']} episodios  "
          f"{b['h_fp_mes']:.1f} h/mes  lead={b['lead_med']:.1f}h  p={b['p']:.4f}")
    print(f"  detectados: {','.join(b['detectados'])}\n")

    # ---- CONTROLE: dessensibilizar os 4 sinais ate gastar as mesmas horas
    print("=== CONTROLE: 4 sinais, so mexendo na sensibilidade (sem crest) ===")
    print(f"{'k_base':>7} {'k_vib':>6} {'det':>4} {'eps':>5} {'h/mes':>7} {'lead':>6}  quais")
    ctrl = []
    for kb in [1.0, 1.2, 1.4, 1.5, 1.6, 1.7]:
        for kv in [1.2, 1.6, 2.2, 3.0]:
            x = m(alerta_2k(out, mask, kb, kv))
            ctrl.append(dict(k_base=kb, k_vib=kv, det=x["det"], eps=x["episodios"],
                             h=x["h_fp_mes"], lead=x["lead_med"], quais=",".join(x["detectados"])))
    ct = pd.DataFrame(ctrl).sort_values("h")
    for _, r in ct.iterrows():
        marca = "  <-- 9/9" if r.det == 9 else ""
        print(f"{r.k_base:7.1f} {r.k_vib:6.1f} {r.det:4d} {r.eps:5d} {r.h:7.1f} {r.lead:6.1f}{marca}")
    perto = ct[(ct.h >= 100) & (ct.h <= 125)]
    print(f"\ndetector de 4 sinais na faixa de 100-125 h/mes: deteccao maxima = "
          f"{perto.det.max() if len(perto) else 'n/a'}/9  "
          f"(o crest faz 9/9 a 108,0 h/mes)")

    # ---- crest: varredura fina + qual evento entra
    print("\n=== 5o SINAL = crest factor da vibracao ===")
    print(f"{'k_tx':>6} {'det':>4} {'eps':>5} {'h/mes':>7} {'lead':>6} {'p':>8}  ganho vs base")
    linhas = []
    for k in K_TX:
        al = alerta_n(out, mask, [(CR, "30min", 3.0)], K_BASE, K_VIB, k)
        x = m(al)
        novos = sorted(set(x["detectados"]) - set(b["detectados"]))
        perdidos = sorted(set(b["detectados"]) - set(x["detectados"]))
        print(f"{k:6.1f} {x['det']:4d} {x['episodios']:5d} {x['h_fp_mes']:7.1f} "
              f"{x['lead_med']:6.1f} {x['p']:8.5f}  +[{','.join(novos)}] -[{','.join(perdidos)}]")
        linhas.append(dict(k_tx=k, det=x["det"], eps=x["episodios"], h_mes=x["h_fp_mes"],
                           lead=x["lead_med"], p=x["p"], novos=",".join(novos),
                           perdidos=",".join(perdidos), quais=",".join(x["detectados"])))
    pd.DataFrame(linhas).to_csv("crest.csv", index=False)
    ct.to_csv("crest_controle.csv", index=False)

    # ---- LOEO: parametro escolhido nos outros 8
    print("\n=== LOEO: k_tx escolhido nos outros 8 eventos, testado no que ficou de fora ===")
    print("   (criterio: maior deteccao entre os 8, empate -> menor h/mes)")
    print(f"{'evento fora':>12} {'k_tx*':>6} {'8 restantes':>12} {'h/mes':>7} | {'detectado?':>11} {'lead':>6}")
    res = []
    for ev in falhas:
        outros = falhas[falhas != ev]
        melhor = None
        for k in K_TX:
            al = alerta_n(out, mask, [(CR, "30min", 3.0)], K_BASE, K_VIB, k)
            x = A.avalia(al, outros, mask)
            chave = (x["det"], -x["h_fp_mes"])
            if melhor is None or chave > melhor[0]:
                melhor = (chave, k, x, al)
        (_, _), k, x8, al = melhor
        xe = A.avalia(al, pd.Series([ev]), mask)
        pega = xe["det"] == 1
        print(f"{ev.strftime('%Y-%m-%d'):>12} {k:6.1f} {x8['det']:>8}/8   {x8['h_fp_mes']:7.1f} | "
              f"{'SIM' if pega else 'nao':>11} {xe['lead_med'] if pega else float('nan'):6.1f}")
        res.append(dict(evento=ev, k_tx=k, det8=x8["det"], h_mes=x8["h_fp_mes"],
                        detectado=pega, lead=xe["lead_med"] if pega else np.nan))
    r = pd.DataFrame(res); r.to_csv("crest_loeo.csv", index=False)
    print(f"\nLOEO com crest: {int(r.detectado.sum())}/9")


main()
