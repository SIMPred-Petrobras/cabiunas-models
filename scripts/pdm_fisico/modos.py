#!/usr/bin/env python3
"""Decompoe o alvo por MODO DE FALHA e testa se 2024-01-16 e artefato de partida a frio.

Motivo. Criticamos a Secao 18 do EXP10c por misturar subsistemas com fisicas diferentes
num denominador so. Fazemos o mesmo: chamamos de "9 paradas" um conjunto que e

    6 de MANCAL   (TAHH_6240305, temperatura muito alta do mancal radial)
    1 de SELAGEM  (PDAHH6240305, pressao diferencial alta no selo primario)
    2 de OLEO     (PALL_6240309/6240340, pressao muito baixa do oleo lubrificante)

e ajustamos UM k global para os tres. Precursor, escala de tempo e sensor relevante sao
diferentes em cada um.

PARTE 1 -- atribuicao. Para cada evento, qual sinal sustentou primeiro dentro das 48 h,
com quanto de antecedencia, e a que fracao do proprio limiar cada sinal chegou. Se os
eventos de mancal sao carregados por sp+vb e os de oleo por p, a afirmacao defensavel
deixa de ser "preve parada" e passa a ser especifica por modo -- mais estreita em escopo,
muito mais forte em evidencia.

PARTE 2 -- partida a frio. 2024-01-16 09:10 e o unico evento que nunca detectamos, em
nenhuma configuracao testada. A serie comeca em 2024-01-01 03:00, quinze dias antes. Mas:
  - o ajuste walk-forward usa `fit = stable & (idx < inicio_do_mes)`; para janeiro/2024
    isso e VAZIO, entao t, p e sp podem nem existir no mes;
  - a referencia rolante da vibracao exige 400 h de operacao quente + 24 h de guarda,
    ou seja ~18 dias de operacao antes de produzir o primeiro valor.
Se o detector estava desligado por falta de historico, o 8/9 nos penaliza por limitacao
de DADO, nao de modelo, e o recall sobre eventos com historico adequado e 8/8.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from portoes import K_BASE, K_VIB
from auto_reset import trunca
import reduz_fp as RF

JAN = pd.Timedelta(hours=48)
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}


def modo(desc):
    d = str(desc)
    if "Manc" in d: return "mancal"
    if "Selo" in d or "Vaz." in d: return "selagem"
    if "Óleo" in d or "Oleo" in d: return "oleo"
    return "?"


def main():
    df = canonico(); idx = df.index
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])
    fal["evento"] = fal["evento"].dt.tz_convert("UTC")
    fal["modo"] = fal["alarmes"].map(modo)
    falhas = fal["evento"]
    mask = mascara_pontuacao(df)
    out = roda(BRACO := "max+vib_rol", df, falhas)
    T = {"t": DET.THR_FAM*K_BASE, "p": DET.THR_FAM*K_BASE,
         "sp": DET.THR_SPREAD*K_BASE, "vb": 3.0*K_VIB}
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    S = {c: DET._sustained(E[c], T[c]) for c in E}
    n = sum(S[c].astype(int) for c in E)
    al = RF.dur_min(RF.refratario((n >= 2) & mask, 48), 60)

    print("=" * 100)
    print("PARTE 2 -- 2024-01-16 e artefato de partida a frio?")
    print("=" * 100)
    t0 = falhas.iloc[0]
    print(f"serie comeca em {idx[0]:%Y-%m-%d %H:%M}; evento em {t0:%Y-%m-%d %H:%M} "
          f"({(t0-idx[0]).days} dias depois)\n")
    print(f"{'evento':>17} {'h de mascara':>13} {'t':>10} {'p':>10} {'sp':>10} {'vb':>10}")
    print(f"{'':>17} {'em +-48h':>13} " + " ".join(f"{'pts validos':>10}" for _ in range(4)))
    for _, r in fal.iterrows():
        t = r["evento"]; a, b = t - JAN, t
        hm = mask.loc[a:b].sum() * 2 / 60
        val = [int(out[c].loc[a:b].notna().sum()) for c in ["t", "p", "sp", "vb"]]
        marca = "   <-- nunca detectado" if t == t0 else ""
        print(f"{t:%d/%m/%Y %H:%M} {hm:11.1f} h " + " ".join(f"{v:10d}" for v in val) + marca)
    tot = int(JAN / pd.Timedelta("2min"))
    print(f"\n(uma janela de 48 h tem {tot} pontos de 2 min no total)")
    est = df["stable"].astype(bool)
    print(f"horas de operacao quente ANTES de 2024-01-16: {est.loc[:t0].sum()*2/60:.0f} h "
          f"(a referencia rolante exige 400 h + 24 h de guarda)")
    print(f"amostras estaveis antes de 2024-01-01 (fit do walk-forward): "
          f"{int(est.loc[:idx[0]].sum())} (o fit exige {DET.FIT_POINTS//4} no minimo)")

    print("\n" + "=" * 100)
    print("PARTE 1 -- atribuicao por sinal e por modo de falha")
    print("=" * 100)
    print(f"{'evento':>17} {'modo':>9} {'det':>5} {'lead':>7} {'1o sinal':>9} | "
          f"{'pico/limiar de cada sinal':>34} | sinais que sustentaram")
    print(f"{'':>17} {'':>9} {'':>5} {'':>7} {'':>9} | "
          f"{'t':>8} {'p':>8} {'sp':>8} {'vb':>8} |")
    linhas = []
    for _, r in fal.iterrows():
        t = r["evento"]; a = t - JAN
        w = al.loc[a:t]; on = w[w.fillna(False)]
        det = bool(len(on)); lead = (t - on.index[0]).total_seconds()/3600 if det else np.nan
        picos, quais, prim, tprim = {}, [], "-", None
        for c in ["t", "p", "sp", "vb"]:
            s = E[c].loc[a:t].dropna()
            picos[c] = (s.max()/T[c]) if len(s) else np.nan
            ss = S[c].loc[a:t]; ss = ss[ss.fillna(False)]
            if len(ss):
                quais.append(c)
                if tprim is None or ss.index[0] < tprim: tprim, prim = ss.index[0], c
        print(f"{t:%d/%m/%Y %H:%M} {r['modo']:>9} {'SIM' if det else 'nao':>5} "
              f"{(f'{lead:.1f} h' if det else '-'):>7} {prim:>9} | "
              + " ".join(f"{picos[c]:8.2f}" if np.isfinite(picos[c]) else f"{'-':>8}"
                         for c in ["t","p","sp","vb"])
              + f" | {','.join(quais) if quais else '-'}")
        linhas.append(dict(evento=t, modo=r["modo"], det=det, lead=lead, primeiro=prim,
                           **{f"pico_{c}": picos[c] for c in picos},
                           sustentaram=",".join(quais)))
    D = pd.DataFrame(linhas); D.to_csv("modos.csv", index=False)

    print("\n" + "-" * 100)
    print("por modo de falha:")
    for m, g in D.groupby("modo"):
        gd = g[g.det]
        prim = g.loc[g.det, "primeiro"].value_counts().to_dict()
        print(f"  {m:>9}: {int(g.det.sum())}/{len(g)} detectados   "
              f"lead mediano {gd.lead.median() if len(gd) else float('nan'):5.1f} h   "
              f"primeiro sinal a disparar: {prim}")
    print("\n  fracao dos eventos em que cada sinal sustentou:")
    for c in ["t", "p", "sp", "vb"]:
        k = D.sustentaram.str.contains(c + ",|" + c + "$", regex=True).sum()
        print(f"     {c:>3}: {k}/9")


if __name__ == "__main__":
    main()
