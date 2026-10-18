#!/usr/bin/env python3
"""EXP 49b — VERIFICAÇÃO DO CONTROLE DO EXP 49 (PÓS-HOC, DESCRITIVA), a pedido do usuário ("certeza? confira novamente os resultados").

O EXP 49 concluiu que o PCA unido a ele mesmo (n = 20.000 com 19.000) explica o ganho nominal das fusões com outros modelos, com base em UM par de n. Aqui o controle é refeito para os 6 pares entre n = 18.000, 19.000, 20.000 e 21.000 (o de 18.000 calculado no harness; o PCA sozinho a 1.457 h por n), pelas duas fusões: F2 (máximo dos escores) e F1 (união dos alarmes, agora com busca fina do orçamento comum em 36 valores, escolhendo o que mais se aproxima de 1.457 h).
Uso:  PYTHONPATH=src .venv/bin/python experiments/49b_verificacao_do_controle.py
"""
import sys, itertools, importlib.util, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("exp49", "experiments/49_complementaridade_de_modelos.py"); m = importlib.util.module_from_spec(spec); sys.modules["exp49"] = m; spec.loader.exec_module(m)
from nv import io as nvio
from nv.avalia import carrega_controle
from nv.causal import carrega_registro
from nv.walk import walk_forward
E46 = m.E46; E20 = m.E46.E20
reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]
EV = {"real": [t for t in reg if t <= fim]}
for dd in m.SHIFTS: EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
ini = m.INICIO8 - pd.Timedelta(days=10); m.GL.update(vig=vig_c, fim=fim, ini=ini, EV=EV, EW={})
g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = m.E46._ci()
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()): modelos = ci.carregar_modelos(m.E46.PKG / "modelos")
tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items() if c in ("t", "p")}
NSV = (18000, 19000, 20000, 21000)
# escores PCA para os 4 n (18000 calculado agora com o mesmo harness; os demais vêm do escores_sub34, e confiro um contra o harness)
for c in ("t", "p"):
    for n in NSV:
        f = m.ROOT / "escores_sub34" / f"{c}_A_{n}.parquet"
        if f.exists(): s = pd.read_parquet(f)["s"].loc[ini:fim]
        else:
            s = walk_forward(lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente), g, tags[c], est, m.INICIO8, fim, n_treino=n, semente=0).loc[ini:fim]
        m.GL["EW"][(c, "PCA", n)] = m.ewma_vigiado(s, vig_c.loc[s.index], "1.0h")
def uniao_fina(e1, e2, alvo=1457.0):
    bs = np.geomspace(60, 1457, 36); best = None
    for b in bs:
        a = m.alarme_orcamento(e1, b) | m.alarme_orcamento(e2, b); h = m.horas(a)
        if best is None or abs(h - alvo) < abs(best[1] - alvo): best = (a, h, b)
    return best
print("PCA SOZINHO por n (det/FP/horas):")
solo = {}
for c in ("t", "p"):
    for n in NSV:
        al = m.alarme_orcamento(m.GL["EW"][(c, "PCA", n)], 1457.0); r = m.mede(al, EV); solo[(c, n)] = sum(r["real"][0]); print(f"  [{c}] n={n}: {sum(r['real'][0])}/8 FP {r['real'][1]} {m.horas(al):.0f} h")
print("\nCONTROLE PCA+PCA com todos os pares de n:")
for c in ("t", "p"):
    for n1, n2 in itertools.combinations(NSV, 2):
        e1, e2 = m.GL["EW"][(c, "PCA", n1)], m.GL["EW"][(c, "PCA", n2)]
        a2 = m.alarme_orcamento(m.maximo_escores(e1, e2), 1457.0); r2 = m.mede(a2, EV); h2 = m.horas(a2)
        a1, h1, b1 = uniao_fina(e1, e2); r1 = m.mede(a1, EV)
        base = (solo[(c, n1)] + solo[(c, n2)]) / 2
        pl2 = float(np.median([sum(v[0]) / len(v[0]) for k, v in r2.items() if k != "real" and len(v[0])]) * 8)
        print(f"  [{c}] ({n1},{n2}) PCA sozinho médio {base:.1f} | F2 máx: {sum(r2['real'][0])}/8 FP {r2['real'][1]} {h2:.0f} h (placebo {pl2:.1f}) | F1 união fina: {sum(r1['real'][0])}/8 FP {r1['real'][1]} {h1:.0f} h")
