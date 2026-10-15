#!/usr/bin/env python3
"""EXP 05b — O MESMO CONFRONTO COM O ORÇAMENTO EM EPISÓDIOS FALSOS (o que o operador sente), não em horas.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE EXISTE. Os EXP 04 e 05 igualaram as HORAS de alarme fora das janelas (407 h). O controle gasta essas horas em 9 episódios; a política simples, em 10 a 47. Igualar horas favorece a política simples. O custo que o operador sente é o NÚMERO de alarmes falsos (FP/mês). Aqui o orçamento é: no máximo 9 FP (o do controle).

O QUE SE FAZ. Todos os escores (os 6 baselines do EXP 04 em n = 20.000 e semente 0, e os 8 escores dos sinais do v2 do EXP 05), a mesma política simples, o limiar = o MENOR de uma grade logarítmica de 80 valores (do quantil 50% ao máximo do escore) com FP <= 9 (escolha por FP apenas, sem olhar a detecção). Mesmas métricas e mesmo harness. Controle: 6/7 a 9 FP.

EXPECTATIVAS REGISTRADAS: a 9 FP episódios nenhum escore simples passa de 3 de 7 (~75%); o melhor dos sinais do v2 (ord2 ou canal_t) fica em 2 ou 3 de 7; os baselines em 0 a 2; a diferença para o controle (6/7) a 9 episódios é de >= 3 eventos (~70%).
LEITURA: se a lacuna a 9 episódios é >= 3 eventos, a CONSOLIDAÇÃO DO ALARME (CUSUM com memória, níveis, portão, escalada) vale pelo menos 3 detecções à mesma contagem de falsos alarmes: é onde um candidato novo (modelo de decisão) tem mais a ganhar.

LIMITES. 7 eventos; FP não monótono no limiar (por isso a grade e o menor limiar viável); mesmo piso de ruído dos EXP 04 e 05 (detecção +-1 evento).

Uso:  PYTHONPATH=src .venv/bin/python experiments/05b_orcamento_em_episodios.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, cobertura_ao_acaso
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.politica import ewma_vigiado, episodios, alarme_de

INICIO = E4.INICIO; ORC_FP = 9


def a_9_episodios(e, vig, reg, pre_suave):
    ini = INICIO - pd.Timedelta(days=10); e = e.loc[ini:]; v = vig.loc[ini:]
    s = e.dropna()
    grade = np.exp(np.linspace(np.log(max(float(np.nanquantile(s, 0.5)), 1e-9)), np.log(float(s.max())), 80))
    for thr in grade:                                                     # do mais sensível ao menos: o primeiro com FP <= 9
        al = alarme_de(episodios(e, thr), e.index); r = avalia(al, v, reg, inicio_periodo=INICIO)
        if r.fp <= ORC_FP:
            return r, al, thr
    return r, al, thr


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); tags = [c for c in g.columns if c not in E4.SEM_TAGS]
    al_c, vig_c = carrega_controle("atual"); reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    vig_g = vig_c.reindex(g.index).fillna(False)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE)
    with contextlib.redirect_stdout(io.StringIO()):
        det = _ci().carregar_modelos(PKG / "modelos")[-1]["detector"]
    vigS = S["mask"].astype(bool)
    print(f"CONTROLE: detecção {ctrl.det}/{ctrl.n_eventos}, FP {ctrl.fp}, lead {ctrl.lead_medio_h:.1f} h, {ctrl.horas_alarme:.0f} h de alarme\n")
    linhas = []
    fim = vig_c.index[-1]
    for n in E4.FABRICAS:
        s = E4.escore(n, g, tags, est, fim); e = ewma_vigiado(s, vig_g)
        r, al, thr = a_9_episodios(e, vig_g, reg, False)
        linhas.append(dict(familia="baseline", escore=n, limiar=round(thr, 3), det=f"{r.det}/{r.n_eventos}", inicio=r.inicio, lead_h=round(r.lead_medio_h, 1), fp=r.fp, horas=round(r.horas_alarme)))
    for n, e in E5.pontua(S, det).items():
        r, al, thr = a_9_episodios(e, vigS, reg, True)
        linhas.append(dict(familia="sinais v2", escore=n, limiar=round(thr, 3), det=f"{r.det}/{r.n_eventos}", inicio=r.inicio, lead_h=round(r.lead_medio_h, 1), fp=r.fp, horas=round(r.horas_alarme)))
    T = pd.DataFrame(linhas); print("A NO MÁXIMO 9 FP EPISÓDIOS (o do controle)"); print(T.to_string(index=False))
    melhor = max(int(x.split('/')[0]) for x in T.det)
    print(f"\nmelhor escore simples: {melhor}/{ctrl.n_eventos} contra o controle {ctrl.det}/{ctrl.n_eventos} a {ctrl.fp} FP -> lacuna de {ctrl.det - melhor} evento(s)")


if __name__ == "__main__":
    main()
