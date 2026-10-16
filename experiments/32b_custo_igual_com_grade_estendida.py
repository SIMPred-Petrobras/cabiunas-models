#!/usr/bin/env python3
"""EXP 32b — O MESMO TESTE DO EXP 32 COM A GRADE DE ESCALA ESTENDIDA E A CLASSIFICAÇÃO SÓ DAS CÉLULAS A CUSTO IGUAL.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR. (Correção do EXP 32: lá o custo igual só foi alcançado em 3 de 40 células e a classificação impressa ignorou isso.)

O QUE MUDA. Os mesmos escores em cache (data_local/escores_fiel8, desde 01/02/2025; fiel, pca3, ridge, gbm, ocsvm, iforest, knn, lof, gmm, ecod; canais t e p; n = 20.000 e 19.000) e a mesma decisão de produção com os sinais de 8 eventos; a escala s dos dois limiares do canal trocado passa a uma grade FINA e LONGA: 0,5 a 3,0 em passos de 0,1 e {3,5; 4; 5; 6; 8} (a decisão não é monótona em s: o refratário, a escalada e a duração criam degraus, e por isso uma grade fina ajuda a encontrar um s que iguale o custo).
CUSTO IGUAL (definição fixada antes): as horas de alarme fora das janelas de trip dentro de +-10% de 407 h (366 a 448 h) e, entre os s que cumprem, o mais perto de 407 h (o critério é o CUSTO, nunca a detecção). Reporta-se também o FP e se o FP fica a <= +2 do da referência da mesma variante.
CLASSIFICAÇÃO (só para células que alcançam o custo igual nas DUAS variantes de n; as outras são 'SEM CUSTO IGUAL', sem rótulo): GANHO = detecção >= a da referência da mesma variante + 1 em ambas; EQUIVALENTE = a +-1 em ambas; PIOR = o resto. Piso de ruído: a referência varia +-2 eventos entre n = 20.000 e 19.000 (EXP 32).
EXPECTATIVAS: >= 4 dos 9 escores alternativos alcançam o custo igual nas duas variantes em t ~50% e em p ~40%; entre as células a custo igual, nenhuma é GANHO ~90%; EQUIVALENTE ~40%; e a diferença de FP: as alternativas a custo igual têm >= 2 FP a mais que a referência em >= metade das células ~60% (o custo em horas igual não garante o custo em notificações igual).
INTERPRETAÇÃO PREVISTA. Se nenhuma alternativa chega ao custo igual com detecção melhor, a resposta à pergunta 'o PCA é simples demais?' é NÃO neste conjunto, com o custo de notificações agora medido de forma justa. Se alguma alcança o custo e ganha, segue o placebo e o 13b.
LIMITES. 8 eventos; só dois tamanhos de treino; a grade de s mexe só nos dois níveis do canal trocado; uma célula que iguala o custo por um degrau do refratário pode estar igualando por acaso (a procura de s numa grade fina é uma família de comparações); nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/32b_custo_igual_com_grade_estendida.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E32 = _carrega("exp32", "32_escores_alternativos_a_custo_igual.py"); E20 = E32.E20
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG

ROOT = E32.ROOT; INICIO8 = E32.INICIO8; ESC = E32.ESC; NS = E32.NS; ALVO = 407.0
GRADE = tuple(round(x, 1) for x in np.arange(0.5, 3.01, 0.1)) + (3.5, 4.0, 5.0, 6.0, 8.0)


def mede(a):
    return E32.mede(a)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items()}
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    E32.GL.update(g=g, est=est, fim=vig_c.index[-1], tags=tags, proc=proc, det=det, reg=reg, vig=vig_c, idx_c=al_c.index)
    pedidos = [(nm, c, n, s) for c in tags for nm in ESC for n in NS for s in GRADE]
    print(f"{len(pedidos)} execuções da decisão (grade de {len(GRADE)} valores de s)", flush=True)
    with Pool(4) as p:
        R = dict(p.map(mede, pedidos, chunksize=4))
    tab = {}
    print("\nA CUSTO IGUAL (hours fora em 366-448 h; s mais perto de 407 h): detecção | FP | h fora | s", flush=True)
    for c in tags:
        for nm in ESC:
            lin = []
            for n in NS:
                ok = [s for s in GRADE if abs(R[(nm, c, n, s)]["fora"] - ALVO) <= 0.1 * ALVO]
                if ok:
                    s = min(ok, key=lambda s: abs(R[(nm, c, n, s)]["fora"] - ALVO)); r = R[(nm, c, n, s)]; tab[(c, nm, n)] = (r["det"], r["fp"], r["fora"], s); lin.append(f"n={n}: {r['det']}/8, FP {r['fp']}, {r['fora']:.0f} h, s={s}")
                else:
                    tab[(c, nm, n)] = None; mn = min(R[(nm, c, n, s)]["fora"] for s in GRADE); lin.append(f"n={n}: sem custo igual (melhor {mn:.0f} h)")
            print(f"   {c} <- {nm:7s} | " + " | ".join(lin), flush=True)
    print("\nCLASSIFICAÇÃO (só células a custo igual nas duas variantes de n)", flush=True)
    for c in tags:
        ref = [tab[(c, "fiel", n)] for n in NS]
        if any(x is None for x in ref):
            print(f"   [{c}] a referência fiel não alcança o custo igual nas duas variantes: {ref}", flush=True); continue
        print(f"   [{c}] referência fiel: detecção {[x[0] for x in ref]}, FP {[x[1] for x in ref]}", flush=True); nm_ok = 0
        for nm in ESC[1:]:
            cel = [tab[(c, nm, n)] for n in NS]
            if any(x is None for x in cel):
                print(f"      {nm:7s}: SEM CUSTO IGUAL em {sum(x is None for x in cel)} das 2 variantes", flush=True); continue
            nm_ok += 1; d = [x[0] for x in cel]; dfp = [x[1] - y[1] for x, y in zip(cel, ref)]
            cl = "GANHO" if all(x[0] >= y[0] + 1 for x, y in zip(cel, ref)) else "EQUIVALENTE" if all(abs(x[0] - y[0]) <= 1 for x, y in zip(cel, ref)) else "PIOR"
            print(f"      {nm:7s}: detecção {d} (ref {[x[0] for x in ref]}), FP {[x[1] for x in cel]} (diferença para a ref {dfp}) -> {cl}", flush=True)
        print(f"      alternativas que alcançam o custo igual nas duas variantes: {nm_ok} de {len(ESC) - 1}", flush=True)


if __name__ == "__main__":
    main()
