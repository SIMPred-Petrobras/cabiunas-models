#!/usr/bin/env python3
"""EXP 24 — VIZINHANÇA DOS k DE NÍVEL A E B NO CAMINHO EXATO DE PRODUÇÃO: o ponto publicado é topo de pico ou platô?

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

CONTEXTO. O EXP 21/22 mostrou que o limiar de variância do PCA em t é um pico (0,95: 6/7, 9 FP; vizinhos 5/7, 10-15 FP) e que o p é um platô. O EXP 13 varreu os botões da política (refratário, duração, CUSUM, escalada, votos) um por vez; os limiares por canal dos níveis A e B (`k_nivel_a`: t 1,10, p 1,20, sp 0,90, vb 2,00; `k_nivel_b`: t 1,70, p 1,70, sp 1,70, vb 2,20) não foram varridos no harness exato com piso de ruído. A memória 'ponto-de-deploy-margem-a-borda' registra que k_p já esteve a um passo de um precipício e foi movido para ter margem de 2,00x; aqui se mede a margem de TODOS os k, juntos e um a um, na base causal de 7 eventos.
DESENHO. Os sinais causais do EXP 05 (os de produção) e `ci.prever` como caixa-preta (validado: o controle é reproduzido idêntico).
  (A) UM POR VEZ: cada um dos 8 k (4 canais x níveis A e B) multiplicado por {0,7; 0,8; 0,9; 1; 1,1; 1,2; 1,3} com os demais no publicado: 56 execuções.
  (B) CONJUNTA ALEATÓRIA: os 8 k multiplicados, de uma vez, por fatores independentes uniformes em [0,9; 1,1] (100 sorteios) e em [0,8; 1,2] (100 sorteios), semente fixa 20261015: a distribuição do desempenho numa vizinhança de volume, que é o que importa quando a calibração envelhece.
CRITÉRIOS (fixados antes). Por execução: BOM = detecção >= 6/7, FP <= 11 e horas fora <= 550 (o ponto publicado com uma folga de +2 FP e +35% de horas); RUIM = detecção <= 4/7 ou FP >= 15. ROBUSTO LOCAL = >= 60% dos sorteios de +-10% são BONS. MARGEM de um k = o maior passo (multiplicador) em cada direção em que o desempenho continua BOM.
EXPECTATIVAS: o ponto publicado é BOM (obviamente); >= 60% dos sorteios de +-10% BONS ~45%; >= 60% de +-20% BONS ~15%; a mediana do sorteio +-10% tem detecção 6/7 ~50% e FP <= 11 ~60%; o k de maior sensibilidade (menor margem) é o vb (nível A) ~35%, o t (nível A) ~30%; algum movimento de um k dá GANHO (detecção >= 6/7, FP <= 7, <= 326 h) ~15% (com 56 comparações e o piso de ruído, é o que se espera por acaso; só valeria se os dois vizinhos imediatos também forem BONS).
INTERPRETAÇÃO PREVISTA. Se o robusto local for baixo, o ponto publicado se comunica como 'o centro de uma vizinhança com X% de BONS', e o desempenho esperado da calibração envelhecendo é o da mediana dos sorteios, não o do ponto. Se for alto, a fragilidade do t é só do limiar de variância.
LIMITES. 7 eventos; os 8 k variam independentes (não há correlação estrutural entre eles); nenhuma calibração se adota sem sombra; os sinais t, p, sp, vb são os do v2 (o t e o p com o EWMA de produção).

Uso:  PYTHONPATH=src .venv/bin/python experiments/24_vizinhanca_dos_k_nivel.py
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

INICIO = E4.INICIO; GL = {}; CANAIS = ("t", "p", "sp", "vb"); MULT = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
BOM = lambda r: r["det"] >= 6 and r["fp"] <= 11 and r["fora"] <= 550
RUIM = lambda r: r["det"] <= 4 or r["fp"] >= 15


def roda(mults):
    """mults: dict {('a'|'b', canal): fator}."""
    det = copy.deepcopy(GL["det"])
    for (nv, c), f in mults.items():
        det["k_nivel_" + nv][c] = det["k_nivel_" + nv][c] * f
    r = E6.mede(decide_v2(GL["proc"], det), GL["reg"]); return dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, inicio=r.inicio)


def tarefa(a):
    return a[0], roda(a[1])


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]; al_c, vig_c = carrega_controle("atual"); ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg)
    print(f"CONTROLE: {E6.linha(ctrl)}\nk_nivel_a {det['k_nivel_a']} | k_nivel_b {det['k_nivel_b']}", flush=True)
    chaves = [(nv, c) for nv in ("a", "b") for c in CANAIS]; pedidos = [((nv, c, m), {(nv, c): m}) for (nv, c) in chaves for m in MULT]
    rng = np.random.default_rng(20261015); sorteios = []
    for amp in (0.10, 0.20):
        for i in range(100):
            sorteios.append(((f"s{int(amp * 100)}", i), {k: float(rng.uniform(1 - amp, 1 + amp)) for k in chaves}))
    with Pool(4) as p:
        R = dict(p.map(tarefa, pedidos + sorteios, chunksize=2))
    print("\n(A) UM POR VEZ (detecção | FP | h fora) — * = publicado; BOM: ✓, RUIM: ✗", flush=True)
    margem = {}
    for (nv, c) in chaves:
        linha = []; ok = {}
        for m in MULT:
            r = R[(nv, c, m)]; ok[m] = BOM(r); linha.append(f"{m:.1f}{'*' if m == 1.0 else ' '}:{r['det']}/{r['fp']}/{r['fora']:.0f}{'✓' if BOM(r) else ('✗' if RUIM(r) else '·')}")
        down = max([m for m in MULT if m <= 1.0 and all(ok[x] for x in MULT if m <= x <= 1.0)], default=1.0); up = max([m for m in MULT if m >= 1.0 and all(ok[x] for x in MULT if 1.0 <= x <= m)], default=1.0)
        margem[(nv, c)] = (down, up); print(f"   k_nivel_{nv}[{c:2s}] {det['k_nivel_' + nv][c]:.2f}: " + "  ".join(linha) + f"  | margem {down:.1f} a {up:.1f}", flush=True)
    ganhos = [(k, m) for (k, m) in [((nv, c), m) for nv, c in chaves for m in MULT if m != 1.0] if R[(k[0], k[1], m)]["det"] >= 6 and R[(k[0], k[1], m)]["fp"] <= 7 and R[(k[0], k[1], m)]["fora"] <= 326]
    print(f"   células com GANHO (>= 6/7, FP <= 7, <= 326 h): {ganhos if ganhos else 'nenhuma'}", flush=True)
    estreitos = sorted(margem.items(), key=lambda kv: (kv[1][1] - kv[1][0])); print("   margens mais estreitas:", [(f"{nv}.{c}", f"{d:.1f}-{u:.1f}") for (nv, c), (d, u) in estreitos[:4]], flush=True)
    print("\n(B) SORTEIOS CONJUNTOS", flush=True)
    for tag, rot in (("s10", "±10%"), ("s20", "±20%")):
        rs = [R[(tag, i)] for i in range(100)]; d = [r["det"] for r in rs]; f = [r["fp"] for r in rs]; o = [r["fora"] for r in rs]; bons = sum(BOM(r) for r in rs); ruins = sum(RUIM(r) for r in rs)
        print(f"   {rot}: BONS {bons}% | RUINS {ruins}% | detecção mediana {np.median(d):.0f}/7 (Q1-Q3 {np.quantile(d, .25):.0f}-{np.quantile(d, .75):.0f}; mín {min(d)}, máx {max(d)}) | FP mediano {np.median(f):.0f} (Q1-Q3 {np.quantile(f, .25):.0f}-{np.quantile(f, .75):.0f}; mín {min(f)}, máx {max(f)}) | horas fora mediana {np.median(o):.0f} (Q1-Q3 {np.quantile(o, .25):.0f}-{np.quantile(o, .75):.0f})", flush=True)
        print(f"        distribuição da detecção: " + ", ".join(f"{k}/7: {sum(1 for x in d if x == k)}" for k in sorted(set(d))), flush=True)


if __name__ == "__main__":
    main()
