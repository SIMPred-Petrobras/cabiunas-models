#!/usr/bin/env python3
"""EXP 05b — O MESMO CONFRONTO COM O ORÇAMENTO EM EPISÓDIOS FALSOS (o que o operador sente), não em horas.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE EXISTE. Os EXP 04 e 05 igualaram as HORAS de alarme fora das janelas (407 h). O controle gasta essas horas em 9 episódios; a política simples, em 10 a 47. Igualar horas favorece a política simples. O custo que o operador sente é o NÚMERO de alarmes falsos (FP/mês). Aqui o orçamento é: no máximo 9 FP (o do controle).

O QUE SE FAZ. Todos os escores (os 6 baselines do EXP 04 em n = 20.000 e semente 0, e os 8 escores dos sinais do v2 do EXP 05), a mesma política simples, o limiar = o MENOR de uma grade logarítmica de 80 valores (do quantil 50% ao máximo do escore) com FP <= 9 (escolha por FP apenas, sem olhar a detecção). Mesmas métricas e mesmo harness. Controle: 6/7 a 9 FP.

EXPECTATIVAS REGISTRADAS: a 9 FP episódios nenhum escore simples passa de 3 de 7 (~75%); o melhor dos sinais do v2 (ord2 ou canal_t) fica em 2 ou 3 de 7; os baselines em 0 a 2; a diferença para o controle (6/7) a 9 episódios é de >= 3 eventos (~70%).
LEITURA: se a lacuna a 9 episódios é >= 3 eventos, a CONSOLIDAÇÃO DO ALARME (CUSUM com memória, níveis, portão, escalada) vale pelo menos 3 detecções à mesma contagem de falsos alarmes: é onde um candidato novo (modelo de decisão) tem mais a ganhar.

LIMITES. 7 eventos; FP não monótono no limiar (por isso a grade e o menor limiar viável); mesmo piso de ruído dos EXP 04 e 05 (detecção +-1 evento).

RESULTADO (15/10/2026). CONTROLE: 6/7 a 9 FP, lead 32,7 h, 1.304 h de alarme. A NO MÁXIMO 9 FP EPISÓDIOS (limiar = o menor da grade com FP <= 9; detecção | lead h | horas de alarme):
  baselines: rz_max 1/7 | 7,4 | 44; pca_spe 1/7 | 20,3 | 249; pca_t2 1/7 | 7,4 | 81; maha_mcd 1/7 | 9,3 | 404; iforest 1/7 | 23,5 | 97; knn 0/7 | 149.
  sinais do v2 (política simples): canal_t 0/7 | 119; canal_p 1/7 | 20,3 | 63; canal_sp 2/7 | 31,8 | 196; canal_vb 1/7 | 48,0 | 372; max4 1/7 | 177; ord2 0/7 | 141; ord3 0/7 | 100; ord2_spvb 0/7 | 103.
  MELHOR ESCORE SIMPLES: 2/7 (canal_sp) contra 6/7 do controle: LACUNA DE 4 EVENTOS à mesma contagem de falsos alarmes. A política simples gasta de 44 a 404 h de alarme (o controle, 1.304 h): o controle COBRE os eventos com episódios LONGOS (memória do CUSUM), enquanto a política simples a 9 episódios só acende por pouco tempo.
LEITURA. Há DOIS custos e eles dão respostas diferentes: (a) a custo igual em HORAS de alarme fora das janelas, os sinais do v2 sob a política simples ficam a 1 evento do controle (5/7 contra 6/7); (b) a custo igual em EPISÓDIOS falsos (o que o operador sente como notificações), a lacuna é de 4 eventos (2/7 contra 6/7). A consolidação do alarme em episódios longos (CUSUM com memória, níveis A/B, portão, escalada)
  é, portanto, o que compra a detecção por notificação; o preço é a CARGA ATIVA em horas (o que a pesquisa atacou com a gestão de alarme R1, que reduz a carga ativa sem mudar a notificação). Todo candidato deve ser reportado NOS DOIS custos (episódios e horas).
PLACAR: ACERTEI: nenhum escore simples passa de 3/7 (o máximo é 2); a lacuna para o controle é >= 3 eventos (4); os baselines em 0 a 2 (0 a 1). ERREI: previ o melhor dos sinais do v2 (ord2 ou canal_t) em 2 ou 3 de 7: ambos deram 0 (o melhor é o canal_sp, 2/7).

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
