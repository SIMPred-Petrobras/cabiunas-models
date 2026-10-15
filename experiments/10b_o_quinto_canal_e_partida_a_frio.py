#!/usr/bin/env python3
"""EXP 10b — O GANHO DO QUINTO CANAL É UM DETECTOR DE PARTIDA A FRIO? Onde caem os FP extras e o evento de 04/11.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. Diagnóstico de mecanismo sobre o resultado do EXP 10.

A DÚVIDA. No EXP 10 o episódio do 04/11/2025 (único evento que o controle perde) começa em 03/11 21:32, o instante em que a vigilância abre depois da partida a frio, e os canais t, p e vb (os originais) já estão acesos ali; o quinto canal só prolonga o voto até passar da duração mínima. Depois de uma parada longa as TI de mancal e óleo estão FRIAS: um canal que mede o desvio
das TI contra o estado quente do mês acende em QUALQUER partida a frio. Se for isso, (i) o ganho no 04/11 é coincidência de regime e (ii) os FP a mais (12 a 14 contra 9) são outras partidas a frio.

O QUE SE FAZ. Para o controle, o AE D0_20k de temperatura (semente 0), o H7a e o H7b (voto V1 do EXP 10), na decisão de produção: os episódios (início, duração) classificados em verdadeiro (cobre uma janela [T - 48 h, T]) e falso; para cada episódio o tau no início (horas desde a última partida, ligada suave do EXP 02) e a duração da parada anterior à partida.
PARTIDA A FRIO = o episódio nasce em tau <= 12 h de uma partida precedida de parada >= 12 h. Reportam-se: nº de FP, quantos deles são de partida a frio, a fração das horas vigiadas que são "partida a frio" (a taxa de base), e o mesmo para o evento de 04/11.
CRITÉRIO (escrito antes): se pelo menos 50% dos FP A MAIS dos canais simples e do AE (contra o controle) caem em partida a frio, que ocupa < 15% das horas vigiadas, o quinto canal é, em boa parte, um detector de regime de partida a frio.
EXPECTATIVAS REGISTRADAS: (a) os FP a mais do AE, do H7a e do H7b são em >= 60% de partida a frio (~60%); (b) a partida a frio (tau <= 12 h, parada >= 12 h) ocupa 5% a 15% das horas vigiadas (~75%); (c) o controle tem <= 20% dos seus FP em partida a frio (~65%).
LIMITES. 7 eventos; 1 deles; a definição de partida a frio (12 h/12 h) é uma escolha fixada aqui; FP "a mais" é a diferença de contagem, não o pareamento episódio a episódio.

Uso:  PYTHONPATH=src .venv/bin/python experiments/10b_o_quinto_canal_e_partida_a_frio.py
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
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py"); E7 = _carrega("exp07", "07_ae_condicional.py"); E10 = _carrega("exp10", "10_canal_termico_simples.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import episodios_de_serie
from nv.contexto import ligada_suave
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.modelos import FABRICAS
from nv.plugin import decide_v2
from nv.politica import ewma_vigiado
from nv.walk import walk_forward

INICIO = E4.INICIO; JAN = pd.Timedelta(hours=48)


def info_partidas(g):
    lig = ligada_suave(g); n = len(lig); sub = np.flatnonzero(lig[1:] & ~lig[:-1]) + 1; off = []
    for i in sub:
        j = i - 1
        while j >= 0 and not lig[j]:
            j -= 1
        off.append((i - j - 1) / 30.0 if j >= 0 else np.nan)
    return g.index[sub], np.array(off)


def classifica(res, reg, ts_part, off_part):
    al = res["is_anomaly"].astype(bool); eps = [(a, b) for a, b in episodios_de_serie(al.loc[INICIO - pd.Timedelta(days=10):]) if b >= INICIO]; linhas = []
    for a, b in eps:
        verd = any(a <= t and b >= t - JAN for t in reg)
        k = np.searchsorted(ts_part, a, side="right") - 1
        tau = (a - ts_part[k]).total_seconds() / 3600 if k >= 0 else np.nan; parada = off_part[k] if k >= 0 else np.nan
        linhas.append(dict(inicio=a, horas=(b - a).total_seconds() / 3600, verdadeiro=verd, tau=tau, parada_h=parada, frio=bool(tau <= 12 and parada >= 12)))
    return pd.DataFrame(linhas)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {"temperatura": modelos[-1]["normalizacao"]["temperatura"]["cols"], "pressao": modelos[-1]["normalizacao"]["pressao"]["cols"]}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]
    sinais = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = sinais[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    E7._G.update(g=g, est=est, tags=tags, fim=fim); ts_part, off_part = info_partidas(g)
    ae0 = E7.tarefa(("D0_20k", "temperatura", 0))[1]
    E10.GL.update(proc=proc, det=det, vigS=vigS, reg=reg, sinais={("ae", 0): ewma_vigiado(ae0, vig_g, "1h").reindex(proc.index)})
    for nome, tg in E10.CONJ.items():
        s = walk_forward(FABRICAS["rz_max"], g, tg, est, pd.Timestamp("2025-03-01", tz="UTC"), fim); E10.GL["sinais"][(nome, 0)] = ewma_vigiado(s, vig_g, "1h").reindex(proc.index)
    # taxa de base: fração das horas vigiadas em partida a frio
    k = np.searchsorted(ts_part, proc.index, side="right") - 1; tau = np.where(k >= 0, (proc.index - ts_part[np.maximum(k, 0)]).total_seconds() / 3600, np.nan)
    frio_h = (tau <= 12) & (off_part[np.maximum(k, 0)] >= 12) & (k >= 0) & vigS.to_numpy(bool); base = float(frio_h.sum() / max(vigS.loc[INICIO:].sum(), 1))
    print(f"taxa de base: a partida a frio (tau <= 12 h, parada >= 12 h) ocupa {100 * float(frio_h[proc.index >= INICIO].sum() / vigS.loc[INICIO:].sum()):.1f}% das horas vigiadas do período", flush=True)
    execs = {"controle": decide_v2(proc, det), "AE D0_20k temperatura": decide_v2(E10.sinal5(("ae", 0), 0), E10.monta(det, "V1")), "H7a (7 TI)": decide_v2(E10.sinal5(("H7a", 0), 0), E10.monta(det, "V1")), "H7b (4 mancais)": decide_v2(E10.sinal5(("H7b", 0), 0), E10.monta(det, "V1"))}
    resumo = {}
    for nome, res in execs.items():
        C = classifica(res, reg, ts_part, off_part); fpD = C[~C.verdadeiro]; tp = C[C.verdadeiro]
        resumo[nome] = (len(fpD), int(fpD.frio.sum()))
        e1104 = [r for r in tp.itertuples() if r.inicio <= pd.Timestamp("2025-11-04 06:22", tz="UTC") and r.inicio >= pd.Timestamp("2025-11-03", tz="UTC")]
        print(f"\n{nome}: {len(C)} episódios | FP {len(fpD)} (em partida a frio: {int(fpD.frio.sum())}) | verdadeiros {len(tp)} (a frio: {int(tp.frio.sum())}) | episódio do 04/11: " + (f"início {e1104[0].inicio:%m-%d %H:%M}, tau {e1104[0].tau:.1f} h, parada anterior {e1104[0].parada_h:.0f} h" if e1104 else "nenhum"))
        if len(fpD):
            print("   FP: " + "; ".join(f"{r.inicio:%m-%d} (tau {r.tau:.0f} h, parada {r.parada_h:.0f} h{', FRIO' if r.frio else ''})" for r in fpD.itertuples()))
    c_fp, c_frio = resumo["controle"]
    print("\nFP A MAIS CONTRA O CONTROLE E QUANTOS SÃO DE PARTIDA A FRIO")
    for nome, (n, f) in resumo.items():
        if nome != "controle":
            print(f"   {nome}: {n - c_fp:+d} FP a mais; partidas a frio entre os FP: {f} (controle: {c_frio}) -> {f - c_frio:+d}")


if __name__ == "__main__":
    main()
