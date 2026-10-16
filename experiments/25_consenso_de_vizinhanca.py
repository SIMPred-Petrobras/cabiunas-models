#!/usr/bin/env python3
"""EXP 25 — CONSENSO DE VIZINHANÇA (bagging dos parâmetros de decisão do v2): a maioria de configurações vizinhas é mais robusta que um ponto?

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 24 mostrou que o ponto publicado (k de nível A e B) é o topo de sua vizinhança: perturbando os 8 k em +-10% de uma vez, só 10% dos sorteios mantêm o desempenho (mediana 6/7, 12 FP, 552 h; FP de 9 a 18 entre sorteios). Os FP variam MUITO entre configurações vizinhas: se são idiossincráticos de cada configuração e os precursores reais disparam em quase todas, o alarme por MAIORIA (consenso) de configurações vizinhas tira os primeiros e mantém os segundos. Não é informação nova: é um ponto de operação sem o otimismo do pico.
O QUE SE FAZ (caminho exato de produção; sinais causais do EXP 05; `ci.prever` como caixa-preta):
  (A) CONSENSO NO PONTO PUBLICADO: 60 configurações com os 8 k multiplicados por fatores independentes uniformes em [0,9; 1,1] (semente fixa 20261016); o alarme de consenso em cada instante é verdadeiro se a fração de configurações em alarme >= q, com q em {0,2; 0,3; 0,4; 0,5; 0,6; 0,7; 0,8}. Cada configuração já aplica o refratário e a duração mínima do v2; o consenso opera sobre as séries de alarme resultantes (os episódios do consenso são as corridas contíguas do consenso).
  (B) ROBUSTEZ DO CONSENSO: 12 CENTROS aleatórios (os 8 k do ponto publicado x fatores em [0,9; 1,1]); para cada centro, o detector de um ponto (o centro) e o consenso (q = 0,4 e 0,5; 30 configurações em +-10% em torno do centro). Mede-se a fração de centros BONS (>= 6/7, FP <= 11, <= 550 h) com o detector de um ponto e com o consenso: o EXP 24 deu 10% BONS para um ponto.
CRITÉRIOS (fixados antes). (A) GANHO sobre o ponto publicado (6/7, 9 FP, 407 h) = detecção >= 6/7 com FP <= 7 e horas fora <= 326 h para algum q (cuidado: 7 valores de q); MANTÉM = detecção >= 6/7, FP <= 10 e horas fora <= 500 h para algum q. (B) ROBUSTECE = a fração de centros BONS com o consenso (em q = 0,4 ou 0,5) >= 60% e >= 3x a do detector de um ponto.
EXPECTATIVAS: (A) algum q MANTÉM ~50%; GANHO ~12% (o consenso tende a FP menor, mas a detecção cai quando poucas configurações acendem juntas); a detecção do consenso cai a <= 5/7 em q >= 0,7 ~70%; (B) ROBUSTECE ~35%; o detector de um ponto dos 12 centros tem <= 25% BONS ~70% (como no EXP 24).
INTERPRETAÇÃO PREVISTA. Se o consenso ROBUSTECE, ele é uma hipótese prospectiva para o sombra.py (produção intocada): o ponto de operação esperado é o do consenso, que é menos sensível ao envelhecimento. Se não, a variação entre configurações é sistemática (os mesmos FP em todas) e a fragilidade vem do sinal, não da calibração.
LIMITES. 7 eventos; o consenso custa 30 a 60 execuções da decisão (barato: os sinais são calculados uma vez); o consenso pode fragmentar episódios (conta mais FP): reportado; o q de 7 valores é uma família de comparações; nada se adota sem sombra.

RESULTADO (15/10/2026) -- (A) O CONSENSO TRAÇA UMA CURVA SUAVE E NÃO MANTÉM 6/7; (B) NÃO ROBUSTECE EM CENTROS ALEATÓRIOS; a pergunta que sobra (25b) é se a curva domina a fronteira das individuais.
  (A) Consenso de 60 configurações em +-10% do ponto publicado (individuais: 7 de 60 BONS; detecção mediana 6/7; FP mediano 12; horas fora mediana 465): q = 0,2: 7/7 | FP 21 | 1.000 h | q = 0,3: 7/7 | 19 | 916 h | q = 0,4: 6/7 | 13 | 594 h | q = 0,5: 5/7 | 10 | 412 h | q = 0,6: 5/7 | 8 | 237 h | q = 0,7: 5/7 | 7 | 234 h | q = 0,8: 5/7 | 6 | 227 h. Nenhum q GANHA (>= 6/7 com FP <= 7 e <= 326 h) nem MANTÉM (>= 6/7, FP <= 10, <= 500 h): com detecção 6/7 só em q = 0,4 e a 13 FP.
  (B) 12 centros aleatórios (+-10% do publicado): o detector de UM ponto (o centro) tem 1 de 12 BONS (detecção mediana 5,0; FP 11; 563 h); o consenso em q = 0,4: 0 de 12 BONS (6,0; FP 15; 622 h) e em q = 0,5: 2 de 12 (5,5; FP 12; 460 h). ROBUSTECE: não.
  LEITURA. Em centros deslocados o consenso é tão ruim quanto o ponto: não conserta calibração envelhecida (a fragilidade é do sinal e do pico, não de ruído de calibração que a média tira). Em torno do ponto publicado (que já é o topo da sua vizinhança) a curva em q é monótona e plana de 0,6 a 0,8 (5/7 com FP 8 a 6). Ver 25b e 25c.
PLACAR: ERREI: algum q MANTÉM (~50%: nenhum); ROBUSTECE (~35%: não). ACERTEI: o consenso cai a <= 5/7 em q >= 0,7 (~70%); o detector de um ponto tem <= 25% BONS nos 12 centros (~70%: 1 de 12); nenhum GANHO (~88%).

Uso:  PYTHONPATH=src .venv/bin/python experiments/25_consenso_de_vizinhanca.py
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
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2

INICIO = E4.INICIO; GL = {}; CANAIS = ("t", "p", "sp", "vb"); CHAVES = [(nv, c) for nv in ("a", "b") for c in CANAIS]
BOM = lambda r: r.det >= 6 and r.fp <= 11 and r.horas_alarme_fora <= 550


def alarme(mults):
    det = copy.deepcopy(GL["det"])
    for (nv, c), f in mults.items():
        det["k_nivel_" + nv][c] = det["k_nivel_" + nv][c] * f
    return decide_v2(GL["proc"], det)["is_anomaly"].astype(bool).to_numpy()


def tarefa(a):
    return a[0], alarme(a[1])


def mede(arr, idx):
    return avalia(pd.Series(arr, index=idx), GL["vig"], GL["reg"], inicio_periodo=INICIO)


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]; al_c, vig_c = carrega_controle("atual"); ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg, vig=vig_c.reindex(proc.index).fillna(False)); idx = proc.index
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True)
    rng = np.random.default_rng(20261016); amp = 0.10
    draw = lambda base: {k: base.get(k, 1.0) * float(rng.uniform(1 - amp, 1 + amp)) for k in CHAVES}
    A = [(("A", i), draw({})) for i in range(60)]
    centros = [{k: float(rng.uniform(1 - amp, 1 + amp)) for k in CHAVES} for _ in range(12)]
    B = [(("Cc", j), c) for j, c in enumerate(centros)] + [(("B", j, i), draw(c)) for j, c in enumerate(centros) for i in range(30)]
    with Pool(4) as p:
        R = dict(p.map(tarefa, A + B, chunksize=2))
    print("\n(A) CONSENSO NO PONTO PUBLICADO (60 configurações em +-10%)", flush=True)
    M = np.mean([R[("A", i)] for i in range(60)], axis=0); indiv = [mede(R[("A", i)], idx) for i in range(60)]
    print(f"   referência: configurações individuais: BONS {sum(BOM(r) for r in indiv)} de 60 | detecção mediana {np.median([r.det for r in indiv]):.0f}/7 | FP mediano {np.median([r.fp for r in indiv]):.0f} | h fora mediana {np.median([r.horas_alarme_fora for r in indiv]):.0f}", flush=True)
    ganho, mantem = [], []
    for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        r = mede(M >= q, idx); print(f"   consenso q = {q:.1f}: {r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | lead {r.lead_medio_h:.1f} h", flush=True)
        if r.det >= 6 and r.fp <= 7 and r.horas_alarme_fora <= 326:
            ganho.append(q)
        if r.det >= 6 and r.fp <= 10 and r.horas_alarme_fora <= 500:
            mantem.append(q)
    print(f"   GANHO sobre o publicado em q = {ganho if ganho else 'nenhum'} | MANTÉM em q = {mantem if mantem else 'nenhum'}", flush=True)
    print("\n(B) ROBUSTEZ: 12 centros aleatórios (+-10% do publicado); um ponto contra consenso (30 configurações em +-10% do centro)", flush=True)
    um = [mede(R[("Cc", j)], idx) for j in range(12)]; print(f"   detector de UM ponto (o centro): BONS {sum(BOM(r) for r in um)} de 12 | detecção mediana {np.median([r.det for r in um]):.1f} | FP mediano {np.median([r.fp for r in um]):.1f} | h fora mediana {np.median([r.horas_alarme_fora for r in um]):.0f}", flush=True)
    for q in (0.4, 0.5):
        cs = [mede(np.mean([R[("B", j, i)] for i in range(30)], axis=0) >= q, idx) for j in range(12)]
        print(f"   CONSENSO q = {q}: BONS {sum(BOM(r) for r in cs)} de 12 | detecção mediana {np.median([r.det for r in cs]):.1f} (mín {min(r.det for r in cs)}) | FP mediano {np.median([r.fp for r in cs]):.1f} (máx {max(r.fp for r in cs)}) | h fora mediana {np.median([r.horas_alarme_fora for r in cs]):.0f}", flush=True)


if __name__ == "__main__":
    main()
