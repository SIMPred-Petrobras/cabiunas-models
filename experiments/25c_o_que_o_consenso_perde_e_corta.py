#!/usr/bin/env python3
"""EXP 25c — O QUE O CONSENSO PERDE E O QUE CORTA (por evento e por episódio), e a ESPECIFICAÇÃO da hipótese prospectiva.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

CONTEXTO. EXP 25: o consenso (maioria de 60 configurações em +-10% dos k do ponto publicado) em q = 0,6 a 0,8 dá 5/7 com FP 8 a 6 e 237 a 227 h fora (publicado: 6/7, 9 FP, 407 h); EXP 25b: em um conjunto independente de 60 individuais, nenhuma com detecção >= 5 tem FP < 9 nem horas fora < 384, logo o consenso DOMINA a fronteira das individuais em 3 de 5 q vizinhos. Antes de tratar como achado: QUAL evento o consenso perde contra o publicado, os FP que sobram são um SUBCONJUNTO dos FP do publicado ou episódios novos, e quanto isso vale em FP por mês de máquina vigiada (a unidade do usuário: 'abaixo de 0,5 ou próximo, com 1 ou 2 detecções a menos').
O QUE SE FAZ. O conjunto 1 do EXP 25 (60 configurações, semente 20261016), consenso em q = 0,5; 0,6; 0,7; 0,8. Por q: (i) detecção, início, banda e lead POR EVENTO, comparados aos do publicado (qual evento muda); (ii) episódios do consenso contra os do publicado: quantos são FP, quantos intersectam um FP do publicado (subconjunto) e quantos são novos; quais FP do publicado somem; (iii) FP por mês de máquina vigiada (FP por 730 h vigiadas) e horas fora por mês.
CRITÉRIOS (fixados antes). O CORTE É GENUÍNO se: (a) o consenso perde UM só evento contra o publicado (e é um evento que o publicado detecta com lead curto, < 12 h); (b) >= 80% dos episódios de FP do consenso intersectam FP do publicado (o consenso REMOVE FP em vez de criar outros); (c) o FP/mês em q = 0,7 fica <= 0,75 (o publicado dá 0,87). É ARTEFATO se o consenso perde 2 eventos ou cria FP que o publicado não tem.
EXPECTATIVAS: o evento perdido é o de 11/04/2025 (lead 4 h) ou o de 29/04/2025 (episódio de 2,2 h) ~75%; >= 80% dos FP do consenso são subconjunto dos FP do publicado ~75%; o FP/mês de q = 0,7 fica <= 0,75 ~65%; o corte é genuíno ~50%.
HIPÓTESE PROSPECTIVA (escrita antes de qualquer dado novo; a sombra não recebe mudança de configuração, então a regra é aplicada OFFLINE sobre os mesmos sinais que a sombra grava, quando o dado novo chegar): H-CONSENSO = com os k_nivel_a e k_nivel_b do detector.json vigente (t 1,10, p 1,20, sp 0,90, vb 2,00; 1,70, 1,70, 1,70, 2,20), 60 configurações cujos oito k são multiplicados por fatores independentes uniformes em [0,9; 1,1] (numpy default_rng(20261016), sorteados na ordem: para cada configuração, os oito k na ordem (a,t), (a,p), (a,sp), (a,vb), (b,t), (b,p), (b,sp), (b,vb)); alarme de consenso = fração de configurações em alarme >= 0,7. Predição prospectiva: em dado novo, o consenso tem FP/mês menor que o do detector publicado e detecção igual ou 1 evento a menos. Refuta-se se o FP/mês do consenso ficar >= o do publicado, ou se a detecção cair >= 2 eventos abaixo.
LIMITES. 7 eventos; q é escolhido num intervalo (0,6 a 0,8) onde o resultado é estável, mas ainda é uma família de comparações; o consenso é em torno do ponto publicado, que é o topo de sua vizinhança (o EXP 25 B mostrou que em centros aleatórios o consenso não ajuda); nada se adota sem a prova prospectiva.

Uso:  PYTHONPATH=src .venv/bin/python experiments/25c_o_que_o_consenso_perde_e_corta.py
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
E25 = _carrega("exp25", "25_consenso_de_vizinhanca.py"); E6 = E25.E6; E5 = E25.E5
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import sinais_causais, _ci, PKG

INICIO = E25.INICIO; GL = E25.GL; CHAVES = E25.CHAVES; JAN = pd.Timedelta(hours=48)


def tarefa(a):
    return a[0], E25.alarme(a[1])


def eps(arr, idx):
    return episodios_de_serie(pd.Series(arr, index=idx).loc[INICIO:])


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]; al_c, vig_c = carrega_controle("atual"); ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg, vig=vig_c.reindex(proc.index).fillna(False)); idx = proc.index
    rng = np.random.default_rng(20261016); conj = [(i, {k: float(rng.uniform(0.9, 1.1)) for k in CHAVES}) for i in range(60)]
    with Pool(4) as p:
        R = dict(p.map(tarefa, conj, chunksize=2))
    M = np.mean([R[i] for i in range(60)], axis=0); pub = E25.alarme({}); Ep = eps(pub, idx); Rp = E25.mede(pub, idx)
    fp_de = lambda E: [(a, b) for a, b in E if not any(a <= t and b >= t - JAN for t in reg)]
    FPp = fp_de(Ep); hv = float(GL["vig"].loc[INICIO:].sum()) * 2 / 60 / 730
    lin = lambda r: f"{r.det}/7 | FP {r.fp} | fora {r.horas_alarme_fora:.0f} h | FP/mês {r.fp / hv:.3f} | fora/mês {r.horas_alarme_fora / hv:.0f} h"
    print(f"PUBLICADO (reproduzido): {lin(Rp)} | meses de máquina vigiada {hv:.1f}", flush=True)
    print("  por evento (início/banda/lead): " + "; ".join(f"{k[:10]} {int(x['inicio'])}{int(x['banda'])} {x['lead_h'] if x['lead_h'] is None else round(x['lead_h'], 1)}" for k, x in Rp.por_evento.items()), flush=True)
    for q in (0.5, 0.6, 0.7, 0.8):
        cons = M >= q; r = E25.mede(cons, idx); Ec = eps(cons, idx); FPc = fp_de(Ec)
        subc = [x for x in FPc if any(x[0] <= b and x[1] >= a for a, b in FPp)]; novos_fp = [x for x in FPc if x not in subc]; somem = [x for x in FPp if not any(x[0] <= b and x[1] >= a for a, b in Ec)]
        perdidos = [k[:10] for k, x in Rp.por_evento.items() if x["det"] and not r.por_evento[k]["det"]]; ganhos = [k[:10] for k, x in Rp.por_evento.items() if not x["det"] and r.por_evento[k]["det"]]
        print(f"\nq = {q}: {lin(r)}", flush=True)
        print(f"   eventos perdidos contra o publicado: {perdidos} (lead no publicado: {[round(Rp.por_evento[k]['lead_h'], 1) for k in Rp.por_evento if k[:10] in perdidos]}) | ganhos: {ganhos}", flush=True)
        print(f"   FP do consenso {len(FPc)}: subconjunto dos FP do publicado {len(subc)}, NOVOS {len(novos_fp)} | FP do publicado que somem: {len(somem)} de {len(FPp)}: {[f'{a:%Y-%m-%d} ({(b - a).total_seconds() / 3600:.0f} h)' for a, b in somem]}", flush=True)
        print("   por evento (início/banda/lead): " + "; ".join(f"{k[:10]} {int(x['inicio'])}{int(x['banda'])} {x['lead_h'] if x['lead_h'] is None else round(x['lead_h'], 1)}" for k, x in r.por_evento.items()), flush=True)


if __name__ == "__main__":
    main()
