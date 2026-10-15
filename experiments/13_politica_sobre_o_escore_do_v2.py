#!/usr/bin/env python3
"""EXP 13 — TRILHA B: A POLÍTICA DE DECISÃO SOBRE O ESCORE DO v2 (sensibilidade de cada botão, caixa-preta de produção, nos DOIS custos).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. O EXP 12 mostrou que aprender o escore com 6 grupos de evento não funciona. Resta a camada de DECISÃO: os botões da política do v2 (refratário, duração mínima, CUSUM, escalada, voto) estão em ponto de ótimo, ou há uma direção que reduz FP/horas sem perder detecção, no harness causal e com piso de ruído? Memórias antigas (piso de custo duro; limiar não é alavanca; gatilho de dois níveis) já varreram isso na base de 8 eventos com backtest otimista; aqui se repete na base CAUSAL de 7 eventos, com os sinais causais do EXP 05 e `ci.prever` como caixa-preta (validado no EXP 06: coincidência de 100%).
DESENHO (um botão por vez, os demais no valor publicado; NÃO é busca fatorial nem escolha de ponto de operação; relata-se a curva toda):
  refratario_h {24, 48, 72*, 120, 168} | duracao_min {60, 120*, 240} min | h_cusum {40, 80*, 160} | kappa {0,5, 0,75*, 1,0} | carga_reset {0, 0,25*, 0,5} | escalada_idade_h {48, 96*, 192} | escalada_abs {10, 20*, 40} | voto_nivel_a {2, 3*, 4} | voto_nivel_b {1, 2*, 3}   (* = publicado).
  E UMA ideia nova de política, fixada antes: REFRATÁRIO ADAPTATIVO = o refratário do episódio é 72 h MAIS a duração do episódio (um alarme longo, que já ocupou o operador por dias, silencia por mais tempo; pós-filtro sobre a decisão do v2).
MÉTRICAS: detecção em 7 eventos, FP em episódios, horas fora das janelas, horas totais; piso de ruído da régua (detecção +-0,5, FP +-1, carga +-20%).
CRITÉRIOS (fixados antes). GANHO = detecção >= 6/7 (a do controle) com FP <= 7 E horas fora <= 326 h (FP -2 e carga -20%: acima do piso nos dois custos). PERDA = detecção <= 5/7. TROCA = o resto (um custo melhora e o outro piora, ou fica dentro do piso). Para chamar um botão de "alavanca" exige-se GANHO em um valor E não perder (<= 5/7) no vizinho imediato, para não premiar uma borda.
EXPECTATIVAS: nenhuma célula é GANHO (~85%); refratários mais longos que 72 h cortam FP mas perdem detecção em >= 1 evento (~65%); refratário adaptativo é TROCA ou PERDA (~80%); duracao_min = 240 perde >= 1 detecção (~70%); voto_nivel_b = 1 ou voto_nivel_a = 2 sobe o FP em >= 3 (~75%).
LIMITES. 7 eventos; botões vizinhos do publicado foram o que a pesquisa v2 já varreu em outra régua; um botão por vez não vê interações; nenhum valor aqui se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/13_politica_sobre_o_escore_do_v2.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2
from nv.avalia import episodios_de_serie

INICIO = E4.INICIO
BOTOES = {"refratario_h": (24, 48, 72, 120, 168), "duracao_min": (60, 120, 240), "h_cusum": (40, 80, 160), "kappa": (0.5, 0.75, 1.0), "carga_reset": (0.0, 0.25, 0.5),
          "escalada_idade_h": (48, 96, 192), "escalada_abs": (10.0, 20.0, 40.0), "voto_nivel_a": (2, 3, 4), "voto_nivel_b": (1, 2, 3)}


def refratario_adaptativo(res, base_h=72):
    al = res["is_anomaly"].astype(bool); eps = episodios_de_serie(al); out = np.zeros(len(al), bool); bloq = None; idx = al.index
    for a, b in eps:
        if bloq is not None and a <= bloq:
            continue
        out[idx.searchsorted(a):idx.searchsorted(b, side="right")] = True; bloq = b + pd.Timedelta(hours=base_h) + (b - a)
    r = res.copy(); r["is_anomaly"] = pd.Series(out, index=idx); return r


def classe(r, ctrl_det):
    if r.det <= ctrl_det - 1:
        return "PERDA"
    return "GANHO" if r.det >= ctrl_det and r.fp <= 7 and r.horas_alarme_fora <= 326 else "TROCA"


def main():
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    al_c, vig_c = carrega_controle("atual"); reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True); cel = {}
    for botao, valores in BOTOES.items():
        print(f"\n{botao} (publicado: {det[botao]})", flush=True)
        for v in valores:
            d2 = copy.deepcopy(det); d2[botao] = v; r = E6.mede(decide_v2(proc, d2), reg); cel[(botao, v)] = r
            print(f"   {botao} = {v:<6}{'*' if v == det[botao] else ' '} | detecção {r.det}/7 | início {r.inicio} | FP {r.fp:>2} | h totais {r.horas_alarme:6.0f} | h fora {r.horas_alarme_fora:5.0f} | lead {r.lead_medio_h:5.1f} -> {classe(r, ctrl.det)}", flush=True)
    r = E6.mede(refratario_adaptativo(decide_v2(proc, det)), reg)
    print(f"\nREFRATÁRIO ADAPTATIVO (72 h + duração do episódio): detecção {r.det}/7 | início {r.inicio} | FP {r.fp} | h totais {r.horas_alarme:.0f} | h fora {r.horas_alarme_fora:.0f} | lead {r.lead_medio_h:.1f} -> {classe(r, ctrl.det)}")
    ganhos = [k for k, x in cel.items() if classe(x, ctrl.det) == "GANHO"]
    print(f"\ncélulas GANHO: {ganhos if ganhos else 'nenhuma'} de {len(cel)}")


if __name__ == "__main__":
    main()
