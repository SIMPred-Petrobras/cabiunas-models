#!/usr/bin/env python3
"""EXP 22 — ENSEMBLE SOBRE O LIMIAR DE VARIÂNCIA DO PCA: a média de vários 0,9x mantém o desempenho do ponto publicado com menos fragilidade?

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

CONTEXTO (EXP 21). Pelo caminho exato do v2 (que o harness reproduz com correlação 1,000000), o limiar de variância do PCA em 0,95 é um PICO em t: 0,93 dá 5/7, 14 FP, 880 h; 0,97 dá 5/7, 10 FP, 453 h; 0,99 dá 5/7, 15 FP; 0,90 dá 3/7. O ponto publicado (6/7, 9 FP, 407 h) é o melhor da vizinhança. Pergunta: calcular o canal como a MÉDIA (e a mediana) dos escores normalizados de vários limiares troca o pico por um platô, sem perder o ponto de operação?
DESENHO. Os escores brutos fiéis (RobustScaler, últimos 20.000 estáveis, escore em unidades do próprio recon_p99) com limiar 0,90, 0,93, 0,95, 0,97 e 0,99, por canal; combinados ANTES do EWMA (o EWMA é linear: a média dos EWMAs é o EWMA da média). Composições fixadas agora:
   C1 = {0,93, 0,95, 0,97};  C2 = {0,90, 0,95, 0,99};  C3 = {0,93, 0,95, 0,97, 0,99}.
Combinadores: média (principal) e mediana (robustez). EWMA do v2 sobre o sinal cru, sem mapa, direto no canal, `ci.prever` como caixa-preta. Avaliações: só t, só p, e t e p JUNTOS (o uso de produção). Referências: 0,95 sozinho (controle: 6/7, 9 FP, 407 h) e os vizinhos sozinhos do EXP 21.
CRITÉRIOS (fixados antes). ESTABILIZA = entre as três composições (média) a detecção varia <= 1 evento e o FP <= 2 (os vizinhos individuais em t variam 3-6 eventos e 9-15 FP). MANTÉM = cada composição com detecção >= 6/7, FP <= 11 e horas fora <= 500 em t (e em p: >= 6/7, FP <= 11, <= 500 h). GANHO = detecção >= 6/7, FP <= 7 e horas fora <= 0,8 x 407 = 326 h (menos 2 FP e menos 20% de horas que o 0,95 sozinho).
EXPECTATIVAS: ESTABILIZA em t ~60% (a média de vizinhos pior tende a um valor intermediário); MANTÉM em t ~25% (o pico não sobrevive à média com vizinhos que dão 5/7 e 14 FP); MANTÉM em p ~55% (p é plano até 0,93); MANTÉM com t e p juntos ~20%; GANHO ~4%; mediana igual à média em +-1 FP ~65%.
INTERPRETAÇÃO PREVISTA. Se a média cai para ~5/7 e 10-12 FP, o desempenho 'típico' do canal t sob variação do parâmetro é esse, e o ponto publicado deve ser comunicado como o topo de um pico. Se a média mantém 6/7 com <= 11 FP, o v2 tem um platô por ensemble e a fragilidade é só de parâmetro único (e o ensemble é candidato a hipótese prospectiva na sombra, sem mudar a produção).
LIMITES. 7 eventos; as composições foram escolhidas por mim, sem busca; o custo em produção seria múltiplos bundles por mês (3 a 4 PCAs por família, barato); nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/22_ensemble_de_limiares_de_variancia.py
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
E21 = _carrega("exp21", "21_ewma_mapa_e_robustez_do_k.py"); E6 = E21.E6; E5 = E21.E5
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, substitui

INICIO = E21.INICIO; VARS = (0.90, 0.93, 0.95, 0.97, 0.99)
COMPOSICOES = {"C1 {0,93; 0,95; 0,97}": (0.93, 0.95, 0.97), "C2 {0,90; 0,95; 0,99}": (0.90, 0.95, 0.99), "C3 {0,93; 0,95; 0,97; 0,99}": (0.93, 0.95, 0.97, 0.99)}


def main():
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E21.FAM.items()}; vib = list(det["tags_vibracao"]); todas = tags["t"] + tags["p"] + vib
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    est_b = ((g["RUNNING_A"] > 0.5).fillna(False) & (g["T5_AVG_A"] > 300).fillna(False)).to_numpy(bool); ini = pd.Timestamp("2025-03-01", tz="UTC")
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True)
    R = {c: {v: E21.walk2(g, tags[c], [x for x in todas if x not in tags[c]], est_b, lambda m, v=v: v, 20000, ini, fim) for v in VARS} for c in tags}

    def ewma(s, c):
        return s.ewm(halflife=pd.Timedelta(det["halflife"][c]), times=s.index).mean().where(s.notna())

    def comb(c, vs, how):
        M = pd.concat([R[c][v] for v in vs], axis=1); s = M.mean(axis=1, skipna=False) if how == "média" else M.median(axis=1, skipna=False); return s

    def mede(sinais):
        p2 = proc
        for c, s in sinais.items():
            p2 = substitui(p2, c, s.reindex(proc.index))
        return E6.mede(decide_v2(p2, det), reg)
    linha = lambda r: f"{r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | lead {r.lead_medio_h:.1f} h"
    print("\nVIZINHOS SOZINHOS (referência do EXP 21)", flush=True)
    for c in tags:
        for v in VARS:
            print(f"   {c} variância {v:.2f}: {linha(mede({c: ewma(R[c][v], c)}))}", flush=True)
    res = {}
    for how in ("média", "mediana"):
        print(f"\nENSEMBLES — combinador: {how}", flush=True)
        for nome, vs in COMPOSICOES.items():
            for modo, canais in (("só t", ["t"]), ("só p", ["p"]), ("t e p", ["t", "p"])):
                r = mede({c: ewma(comb(c, vs, how), c) for c in canais}); res[(how, nome, modo)] = r
                print(f"   {nome:30s} {modo:6s}: {linha(r)}", flush=True)
    for modo in ("só t", "só p", "t e p"):
        rs = [res[("média", n, modo)] for n in COMPOSICOES]; d = [r.det for r in rs]; f = [r.fp for r in rs]
        estab = max(d) - min(d) <= 1 and max(f) - min(f) <= 2; mant = all(r.det >= 6 and r.fp <= 11 and r.horas_alarme_fora <= 500 for r in rs); ganho = [n for n, r in zip(COMPOSICOES, rs) if r.det >= 6 and r.fp <= 7 and r.horas_alarme_fora <= 326]
        print(f"   [{modo}] ESTABILIZA: {estab} (detecção {min(d)}-{max(d)}, FP {min(f)}-{max(f)}) | MANTÉM: {mant} | GANHO: {ganho if ganho else 'nenhuma'}", flush=True)


if __name__ == "__main__":
    main()
