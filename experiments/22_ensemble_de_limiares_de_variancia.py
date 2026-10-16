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

RESULTADO (15/10/2026) -- EM t O PICO NÃO VIRA PLATÔ (NÃO ESTABILIZA, NÃO MANTÉM); EM p O ENSEMBLE ESTABILIZA E MANTÉM.
  VIZINHOS SOZINHOS (reconfere o EXP 21): t 0,90 3/7, 14 FP | 0,93 5/7, 14, 880 h | 0,95 6/7, 9, 407 h | 0,97 5/7, 10, 453 h | 0,99 5/7, 15, 512 h; p 0,90 6/7, 11, 508 h | 0,93 6/7, 9, 408 h | 0,95 6/7, 9, 407 h | 0,97 5/7, 10, 548 h | 0,99 5/7, 10, 542 h.
  ENSEMBLES, MÉDIA (detecção | FP | h fora). só t: C1 {0,93; 0,95; 0,97} 6/7 | 12 | 573 h; C2 {0,90; 0,95; 0,99} 3/7 | 12 | 707 h; C3 (4 limiares) 4/7 | 11 | 693 h. só p: C1 6/7 | 8 | 400 h; C2 6/7 | 9 | 455 h; C3 6/7 | 8 | 400 h. t e p juntos: C1 6/7 | 11 | 566 h; C2 4/7 | 12 | 755 h; C3 4/7 | 10 | 686 h.
  ENSEMBLES, MEDIANA: só t: C1 6/7 | 9 | 407 h (IDÊNTICO ao 0,95 sozinho: a mediana de três limiares escolhe o do meio); C2 5/7 | 10 | 688 h; C3 5/7 | 10 | 413 h (lead 38,4 h). só p: C1 6/7 | 9 | 407 h; C2 6/7 | 9 | 408 h; C3 6/7 | 8 | 400 h. t e p juntos: C1 6/7 | 9 | 407 h; C2 5/7 | 10 | 689 h; C3 5/7 | 9 | 406 h (início 4, 1.301 h totais).
  CRITÉRIOS: só t: ESTABILIZA falso (detecção 3-6, FP 11-12), MANTÉM falso, nenhum GANHO. só p: ESTABILIZA VERDADEIRO (detecção 6-6, FP 8-9), MANTÉM VERDADEIRO, nenhum GANHO (o FP de 8 está a 1 do limite de 7). t e p: ESTABILIZA falso (detecção 4-6), MANTÉM falso, nenhum GANHO.
LEITURA. (1) A MÉDIA de limiares NÃO recupera o ponto publicado em t: com vizinhos próximos (C1) mantém 6/7 mas custa +3 FP e +41% de horas (12 FP, 573 h); com o 0,90 ou o 0,99 na composição (C2, C3) a detecção cai a 3-4/7 (o 0,90 sozinho é 3/7 e envenena a média). A MEDIANA de 4 limiares (C3) dá 5/7 com 9-10 FP e 406-413 h fora: é o perfil dos vizinhos do pico (5/7, 10 FP), o desempenho TÍPICO sob variação do parâmetro. (2) O canal p é um PLATÔ: a média de limiares estabiliza em 6/7, 8-9 FP, 400-455 h, até um FP a menos que o 0,95 sozinho (8 contra 9, dentro do piso de ruído), mesmo com o 0,99 (5/7 sozinho) na composição. (3) Portanto o 6/7 com 9 FP do t publicado é o TOPO de um pico: a expectativa honesta do t sob perturbação do parâmetro é de 5/7 e 9 a 12 FP; o `p` do v2 é robusto. (4) A mediana de três limiares é inútil como teste (devolve o 0,95); a de quatro mostra o típico.
PLACAR: ACERTEI: MANTÉM em t (~25%: não manteve); MANTÉM em p (~55%: sim); MANTÉM com t e p juntos (~20%: não); nenhum GANHO (~96%). ERREI: ESTABILIZA em t (~60%: detecção 3-6); mediana igual à média em +-1 FP (~65%: a mediana difere muito, de 6/7 e 12 FP a 6/7 e 9 FP em C1 só t).

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
