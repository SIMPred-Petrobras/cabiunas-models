#!/usr/bin/env python3
"""EXP 25b — O CONSENSO DE VIZINHANÇA DOMINA A FRONTEIRA DAS CONFIGURAÇÕES INDIVIDUAIS?

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR (depois de ver o resultado do EXP 25, e por isso o desenho é pós-hoc quanto à PERGUNTA, mas os critérios são escritos antes desta rodada).

POR QUE. No EXP 25 o consenso (maioria de 60 configurações em +-10% do ponto publicado) traça uma curva suave: q = 0,4: 6/7, 13 FP, 594 h | 0,5: 5/7, 10 FP, 412 h | 0,6: 5/7, 8 FP, 237 h | 0,7: 5/7, 7 FP, 234 h | 0,8: 5/7, 6 FP, 227 h (publicado: 6/7, 9 FP, 407 h). A 5/7 o consenso tem FP 6-8 e ~230 h: perto do que o usuário pediu (FP baixo com 1 a 2 detecções a menos). Mas isso só é um RESULTADO se o consenso fizer melhor do que o que uma configuração individual da mesma vizinhança consegue na mesma detecção: senão é só um ponto da mesma fronteira de custo.
O QUE SE FAZ. Dois conjuntos independentes de 60 configurações (os 8 k x fatores uniformes em [0,9; 1,1]; sementes 20261016 e 20261017). O CONSENSO vem do conjunto 1; as INDIVIDUAIS de comparação, do conjunto 2 (para a comparação não ser a seleção dentro da mesma amostra). Para cada q em {0,4; 0,5; 0,6; 0,7; 0,8} e a detecção d do consenso: entre as configurações do conjunto 2 com detecção >= d, o MENOR FP e as MENORES horas fora.
CRITÉRIOS (fixados antes desta rodada). O consenso em q DOMINA se, com detecção d, tem FP <= o menor FP das individuais com detecção >= d E horas fora <= 0,9 x as menores horas fora delas. Para uma amostra mais justa, também se mede a posição do consenso: o percentil do FP e das horas fora do consenso entre as individuais com detecção >= d (0% = melhor que todas). SUPERA A FRONTEIRA = o consenso domina em >= 2 valores de q vizinhos.
EXPECTATIVAS: o consenso domina em algum q ~40%; SUPERA A FRONTEIRA (>= 2 q vizinhos) ~25%; o percentil de FP do consenso entre as individuais com detecção >= d é <= 25% em >= 3 dos 5 q ~45%.
INTERPRETAÇÃO PREVISTA. Se supera, o consenso é um achado de operação (a votação entre vizinhos tira FP idiossincráticos), candidato a hipótese prospectiva na sombra com o q escolhido por critério de custo, não por detecção. Se não supera, é um ponto da fronteira já conhecida: nenhuma melhoria.
LIMITES. 7 eventos; os 5 q e a escolha dos sorteios são uma família de comparações; o consenso e as individuais variam em torno do MESMO centro (o ponto publicado, que já é o topo da vizinhança: o EXP 25 B mostrou que em centros aleatórios o consenso não ajuda); nada se adota sem sombra.

RESULTADO (15/10/2026) -- O CONSENSO DOMINA A FRONTEIRA DAS CONFIGURAÇÕES INDIVIDUAIS EM q = 0,6, 0,7 E 0,8.
  Conjunto 2 (60 individuais independentes do consenso, semente 20261017): 7 de 60 BONS; detecção por contagem [4/7: 10, 5/7: 23, 6/7: 24, 7/7: 3]; FP mediano 12. Das 50 individuais com detecção >= 5, o MENOR FP é 9 e as MENORES horas fora são 384 h; nenhuma delas tem FP < 9 ou menos de 384 h.
  Consenso (do conjunto 1): q = 0,4: 6/7, FP 13, 594 h (56% das individuais com detecção >= 6 têm FP menor: não domina) | q = 0,5: 5/7, FP 10, 412 h (8% têm FP menor, 14% horas menores: não domina por pouco) | q = 0,6: 5/7, FP 8, 237 h (0% das individuais têm FP ou horas menores: DOMINA) | q = 0,7: 5/7, FP 7, 234 h (DOMINA) | q = 0,8: 5/7, FP 6, 227 h (DOMINA).
  CRITÉRIOS: DOMINA em 3 de 5 valores de q; SUPERA A FRONTEIRA (>= 2 q vizinhos): SIM.
LEITURA. A 5/7 o consenso entrega FP 6 a 8 e ~230 h fora; nenhuma configuração individual de uma amostra independente da mesma vizinhança com detecção >= 5 faz FP < 9 nem horas < 384: a maioria entre vizinhas tira FP que as individuais carregam (os FP são idiossincráticos de cada configuração) e é um ponto de custo melhor que o da fronteira individual. Ressalvas: o centro é o ponto publicado (o EXP 25 B mostrou que em centros aleatórios não ajuda), 5 valores de q, os mesmos 7 eventos.
PLACAR: ACERTEI: o consenso domina em algum q (~40%); SUPERA A FRONTEIRA em >= 2 q vizinhos (~25%: sim, três); o percentil de FP do consenso <= 25% em >= 3 dos 5 q (~45%: em quatro, 8% e 0%).

Uso:  PYTHONPATH=src .venv/bin/python experiments/25b_consenso_contra_a_fronteira.py
"""
from __future__ import annotations
import importlib.util, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E25 = _carrega("exp25", "25_consenso_de_vizinhanca.py"); E6 = E25.E6; E5 = E25.E5
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG

INICIO = E25.INICIO; GL = E25.GL; CHAVES = E25.CHAVES


def tarefa(a):
    return a[0], E25.alarme(a[1])


def main():
    import contextlib, io
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]; al_c, vig_c = carrega_controle("atual"); ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg, vig=vig_c.reindex(proc.index).fillna(False)); idx = proc.index
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True)
    conjuntos = []
    for sem in (20261016, 20261017):
        rng = np.random.default_rng(sem); conjuntos.append([((sem, i), {k: float(rng.uniform(0.9, 1.1)) for k in CHAVES}) for i in range(60)])
    with Pool(4) as p:
        R = dict(p.map(tarefa, conjuntos[0] + conjuntos[1], chunksize=2))
    M = np.mean([R[(20261016, i)] for i in range(60)], axis=0); indiv = [E25.mede(R[(20261017, i)], idx) for i in range(60)]
    print(f"\nINDIVIDUAIS (conjunto 2): BONS {sum(E25.BOM(r) for r in indiv)} de 60 | detecção {np.bincount([r.det for r in indiv], minlength=8).tolist()} (0..7) | FP mediano {np.median([r.fp for r in indiv]):.0f}", flush=True)
    dominios = []
    for q in (0.4, 0.5, 0.6, 0.7, 0.8):
        r = E25.mede(M >= q, idx); pool_ = [x for x in indiv if x.det >= r.det]
        if not pool_:
            print(f"   q = {q}: consenso {r.det}/7 | FP {r.fp} | fora {r.horas_alarme_fora:.0f} h | nenhuma individual com detecção >= {r.det}", flush=True); continue
        mfp = min(x.fp for x in pool_); mfo = min(x.horas_alarme_fora for x in pool_); pfp = 100 * np.mean([x.fp < r.fp for x in pool_]); pfo = 100 * np.mean([x.horas_alarme_fora < r.horas_alarme_fora for x in pool_])
        dom = r.fp <= mfp and r.horas_alarme_fora <= 0.9 * mfo; dominios.append(dom)
        print(f"   q = {q}: consenso {r.det}/7 | FP {r.fp} | fora {r.horas_alarme_fora:.0f} h || individuais com detecção >= {r.det} ({len(pool_)}): menor FP {mfp}, menores horas fora {mfo:.0f} | posição do consenso: {pfp:.0f}% das individuais têm FP menor, {pfo:.0f}% têm horas fora menores | DOMINA: {dom}", flush=True)
    print(f"\nconsenso DOMINA em {sum(dominios)} de {len(dominios)} valores de q; SUPERA A FRONTEIRA (>= 2 q vizinhos): {any(a and b for a, b in zip(dominios, dominios[1:]))}", flush=True)


if __name__ == "__main__":
    main()
