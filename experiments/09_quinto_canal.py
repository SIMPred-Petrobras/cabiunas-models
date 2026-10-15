#!/usr/bin/env python3
"""EXP 09 — O AUTOENCODER COMO QUINTO CANAL da decisão do v2 (acrescenta informação em vez de substituir).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. Os EXP 06, 07 e 08 testaram sinais novos como SUBSTITUTOS de t ou p e nenhum dominou o controle. Um sinal que não vence um canal do v2 ainda pode ACRESCENTAR evidência independente. A decisão de produção aceita uma lista de canais (`det["sinais"]`): aqui entra um quinto canal `ae`, com a base e os k dos níveis A e B do canal t, o sinal levado à escala do t por mapeamento de quantis
(então tem o mesmo ciclo de trabalho do t em operação normal), e se roda `ci.prever` como caixa-preta.
VOTO: com 5 canais o voto "3 de 4" não é o mesmo. Duas regras, fixadas antes: (V1) as do bundle, nível A >= 3 e nível B >= 2 (mais permissivas: 3 de 5); (V2) nível A >= 4 e nível B >= 2 (mais estritas, o equivalente de "3 de 4" escalado para 5). O portão sp|vb do nível B não muda.
CANDIDATOS (escores em cache, sem retreino): AE D0_20k de temperatura (10 sementes), AE D0_20k de pressão (10 sementes), AE D2 de temperatura (5 sementes). 4 + 1 canais; 2 regras de voto; 25 sementes x 2 = 50 execuções.
CRITÉRIOS (iguais aos do EXP 07/08), mediana das sementes: DIGNO = detecção >= 6/7, FP <= 9, horas fora das janelas <= 407 h; PROMISSOR = detecção >= 5/7, FP <= 12, horas fora <= 800 h.
EXPECTATIVAS REGISTRADAS: com o voto V1 o quinto canal aumenta o FP em >= 3 contra o controle (~70%) e mantém a detecção >= 6/7 (~60%); com V2 o FP fica <= 10 (~50%) e a detecção cai a <= 5/7 (~50%); algum DIGNO ~8%, algum PROMISSOR ~25%.
LIMITES. 7 eventos; o quinto canal herda os limiares do t sem recalibração; o voto V2 é uma escolha minha de equivalência; mapa de quantis global no período.

Uso:  PYTHONPATH=src .venv/bin/python experiments/09_quinto_canal.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py"); E7 = _carrega("exp07", "07_ae_condicional.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, mapeia_quantis
from nv.politica import ewma_vigiado

INICIO = E4.INICIO; GL = {}
VOTOS = {"V1": (3, 2), "V2": (4, 2)}


def tarefa_avalia(a):
    voto, cand, sem = a
    proc, det, vigS, reg = GL["proc"], GL["det"], GL["vigS"], GL["reg"]
    d2 = copy.deepcopy(det); d2["sinais"] = list(det["sinais"]) + ["ae"]
    for k in ("base", "k_nivel_a", "k_nivel_b"):
        d2[k]["ae"] = det[k]["t"]
    d2["voto_nivel_a"], d2["voto_nivel_b"] = VOTOS[voto]
    p2 = proc.copy(); p2["ae"] = mapeia_quantis(GL["sinais"][(cand, sem)], proc["t"], vigS)
    r = E6.mede(decide_v2(p2, d2), reg)
    return a, dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h)


def tarefa_ae(a):
    return E7.tarefa(a)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {"temperatura": modelos[-1]["normalizacao"]["temperatura"]["cols"], "pressao": modelos[-1]["normalizacao"]["pressao"]["cols"]}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    sinais = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = sinais[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    E7._G.update(g=g, est=est, tags=tags, fim=fim)
    cands = {"AE D0_20k temperatura": ("D0_20k", "temperatura", range(10)), "AE D0_20k pressao": ("D0_20k", "pressao", range(10)), "AE D2 temperatura": ("D2", "temperatura", range(5))}
    pedidos = [(c, f, s) for c, f, ss in cands.values() for s in ss]
    with Pool(3) as p:
        S = dict(p.map(tarefa_ae, pedidos, chunksize=1))
    GL.update(proc=proc, det=det, vigS=vigS, reg=reg, sinais={})
    for (cfg, fam, sem), s in S.items():
        GL["sinais"][((cfg, fam), sem)] = ewma_vigiado(s, vig_g, "1h").reindex(proc.index)
    tarefas = [(v, (c, f), s) for v in VOTOS for c, f, ss in cands.values() for s in ss]
    print(f"CONTROLE: {E6.linha(ctrl)}\n{len(tarefas)} execuções do plug-in de 5 canais", flush=True)
    with Pool(4) as p:
        R = p.map(tarefa_avalia, tarefas, chunksize=1)
    print("\nQUINTO CANAL (mediana e Q1-Q3 das sementes)")
    grupos = {}
    for (v, cand, sem), m in R:
        grupos.setdefault((v, cand), []).append(m)
    for (v, cand), ms in grupos.items():
        d = [m["det"] for m in ms]; f = [m["fp"] for m in ms]; o = [m["fora"] for m in ms]; md, mf, mo = np.median(d), np.median(f), np.median(o)
        ver = "DIGNO" if md >= 6 and mf <= 9 and mo <= 407 else "PROMISSOR" if md >= 5 and mf <= 12 and mo <= 800 else "-"
        print(f"   voto {v} {VOTOS[v]} + {cand[0]} {cand[1]:11s} ({len(ms)} sementes): detecção {md:.0f}/7 (faixa {min(d)}-{max(d)}) | FP {mf:.0f} ({np.quantile(f, .25):.0f}-{np.quantile(f, .75):.0f}) | horas fora {mo:.0f} ({np.quantile(o, .25):.0f}-{np.quantile(o, .75):.0f}) | lead {np.nanmedian([m['lead'] for m in ms]):.1f} h -> {ver}")


if __name__ == "__main__":
    main()
