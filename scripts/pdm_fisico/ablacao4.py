#!/usr/bin/env python3
"""Desacopla o botao unico de sensibilidade do braco 'max+vib_rol'.

ablacao3 escolheu um k UNICO que multiplica os quatro limiares (temp, pressao,
spread do mancal, vibracao). Isso e um artefato: os quatro sinais tem
distribuicoes de erro diferentes, entao um k que acerta o orcamento de FP dos
tres sinais "antigos" pode estar sub- ou super-sensibilizando a vibracao ao
mesmo tempo -- e vice-versa. O salto de 5/6 (k=1.7, FP treino 4.90) para 3/6
(k=2.2, FP treino 3.20) no grid 1-D e exatamente essa colisao.

Aqui o botao vira dois: k_base (temp+pressao+spread, os tres sinais do
detector original) e k_vib (so a vibracao). O braco so existe com os dois
juntos ('max+vib_rol'), entao o grid e 2-D. Selecao igual a ablacao3: dentre
os pontos com FP de TREINO <= orcamento (o nativo do detector base, 3.40/mes),
o de maior deteccao de treino; empate quebra pelo menor FP. Nunca se olha o
teste para escolher o ponto.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE

BRACO = "max+vib_rol"
KS = [0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.5, 7.5, 10.0]
ALVO = 3.400823   # FP/mes do detector base no k nativo (ablacao2.csv), fixado a priori


def alerta_2k(out, mask, k_base, k_vib):
    idx = out.index
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    n = (DET._sustained(ew("t", "1h"), DET.THR_FAM * k_base).astype(int)
         + DET._sustained(ew("p", "1h"), DET.THR_FAM * k_base).astype(int)
         + DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * k_base).astype(int)
         + DET._sustained(ew("vb", "30min"), 3.0 * k_vib).astype(int))
    return (n >= 2) & mask


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    tr = pd.Series(idx < CORTE, index=idx); te = ~tr
    ev_tr, ev_te = falhas[falhas < CORTE], falhas[falhas >= CORTE]
    mask = mascara_pontuacao(df)

    print("montando 'out' (walk-forward mensal, uma vez) ...", flush=True)
    out = roda(BRACO, df, falhas)

    linhas = []
    for k_base in KS:
        for k_vib in KS:
            al = alerta_2k(out, mask, k_base, k_vib)
            d = {"k_base": k_base, "k_vib": k_vib}
            for tag, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te)]:
                am, qm = al[m], (mask & m)[m]
                x = A.avalia(am, ev, qm); x.update(A.permuta(am, qm, x["det"], len(ev)))
                d.update({f"{tag}_n": x["det"], f"{tag}_det": f"{x['det']}/{x['n_ev']}",
                          f"{tag}_lead": x["lead_med"], f"{tag}_fp": x["fp_mes"],
                          f"{tag}_h": x["h_fp_mes"], f"{tag}_p": x["p"],
                          f"{tag}_quais": ",".join(x["detectados"])})
            linhas.append(d)
        print(f"k_base={k_base:<5} varrido", flush=True)

    t = pd.DataFrame(linhas)
    t.to_csv("ablacao4.csv", index=False)

    print(f"\n=== ponto escolhido NO TREINO: FP treino <= {ALVO:.2f}/mes, maior deteccao "
          "(empate -> menor FP)")
    s = t[t.tr_fp <= ALVO].copy()
    if s.empty:
        print("nenhum ponto do grid respeita o orcamento")
        return
    r = s.sort_values(["tr_n", "tr_fp"], ascending=[False, True]).iloc[0]
    print(f"k_base={r.k_base}  k_vib={r.k_vib}")
    print(f"  treino: {r.tr_det}  FP={r.tr_fp:.2f}/mes  {r.tr_h:.1f}h/mes  p={r.tr_p:.3f}  "
          f"[{r.tr_quais}]")
    print(f"  TESTE : {r.te_det}  FP={r.te_fp:.2f}/mes  {r.te_h:.1f}h/mes  p={r.te_p:.3f}  "
          f"[{r.te_quais}]")

    print("\n--- para contexto: o mesmo corte, mas so entre os pontos com k_base==k_vib "
          "(replica o botao unico da ablacao3) ---")
    s1 = t[(t.k_base == t.k_vib) & (t.tr_fp <= ALVO)].copy()
    if not s1.empty:
        r1 = s1.sort_values(["tr_n", "tr_fp"], ascending=[False, True]).iloc[0]
        print(f"k={r1.k_base}  treino {r1.tr_det} FP={r1.tr_fp:.2f}  |  teste {r1.te_det} "
              f"FP={r1.te_fp:.2f} p={r1.te_p:.3f}")


main()
