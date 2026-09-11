#!/usr/bin/env python3
"""LOEO com o alarme travado removido -- fecha a composicao das duas objecoes.

O time ja conhece o 7/8 do LOEO (derruba 04/11/2025, fragilidade a escolha de
configuracao). O achado de 04/09/2026 e outro: a deteccao de 26/02/2026 e um
alarme travado. Sao eventos diferentes -- entao as objecoes compoem, e o numero
honesto seria 6/8. Mas o LOEO reescolhe a config em cada fold, e isso nao se
deduz: tem de ser medido com a busca REFEITA sob o corte.

Mesma logica de publica_clearml.py::loeo_aninhado, mesmo orcamento, mesmas
quatro regras de desempate.
"""
import numpy as np, pandas as pd, os

ORC_FP = 1.15
DIAS = ["2025-02-27", "2025-03-17", "2025-04-07", "2025-04-11",
        "2025-04-29", "2025-11-04", "2025-12-09", "2026-02-26"]
REGRAS = {"sem_desempate": (0.0, 0.0, 0.0), "menos_horas_fp": (1.0, 0.0, 0.0),
          "menos_fp": (0.0, 10.0, 0.0), "mais_lead": (0.0, 0.0, 0.01)}


def loeo(csv, orc=ORC_FP):
    df = pd.read_csv(csv)
    df["set"] = df["quais"].fillna("").apply(lambda s: {x for x in s.split(",") if x})
    cand = df[df["fp"] <= orc].reset_index(drop=True)
    if not len(cand):
        return None, None, 0
    lead = np.nan_to_num(cand["lead"].to_numpy(), nan=0.0)
    out = {}
    for nome, (wh, wf, wl) in REGRAS.items():
        ok, perdidos = 0, []
        for d in DIAS:
            tr = set(DIAS) - {d}
            n = cand["set"].apply(lambda s: len(s & tr)).to_numpy()
            sc = n * 1000.0 - cand["hm"].to_numpy() * wh - cand["fp"].to_numpy() * wf + lead * wl
            L = cand.iloc[int(np.argmax(sc))]
            if d in L["set"]:
                ok += 1
            else:
                perdidos.append(d)
        out[nome] = (ok, perdidos)
    frag = {d: float(cand["set"].apply(lambda s: d in s).mean()) for d in DIAS}
    return out, frag, len(cand)


for rot, csv in [("SEM corte (o publicado)", "busca_conjunta.csv"),
                 ("COM corte de 5% (sem travado)", "busca_conjunta_semtravado.csv")]:
    if not os.path.exists(csv):
        print(f"{rot}: {csv} ausente"); continue
    res, frag, n = loeo(csv)
    print(f"\n{rot}   [{n} configs dentro do orcamento de {ORC_FP} FP/mes]")
    print("=" * 78)
    for nome, (ok, perd) in res.items():
        print(f"  {nome:<16} {ok}/8   perdidos: {', '.join(perd) if perd else '--'}")
    print("  fragilidade por evento (fracao das configs no orcamento que detectam):")
    for d, f in sorted(frag.items(), key=lambda kv: kv[1]):
        print(f"     {d}  {f:5.0%}" + ("   <<< mais fragil" if f == min(frag.values()) else ""))
