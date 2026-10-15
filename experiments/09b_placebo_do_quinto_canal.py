#!/usr/bin/env python3
"""EXP 09b — O 7/7 DO QUINTO CANAL É INFORMAÇÃO OU VOTO FROUXO? Controle placebo e atribuição por evento.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. O resultado do EXP 09 já é conhecido: AE D0_20k de temperatura como quinto canal com o voto V1 (3 de 5 no nível A, 2 no B) detecta 7/7 em TODAS as 10 sementes, FP 12 (11-14), 563 h fora das janelas (466-689), lead 29,3 h; o controle faz 6/7, 9 FP, 407 h.

A DÚVIDA. Com 5 canais, "3 de 5" é mais permissivo que "3 de 4": um canal EXTRA qualquer eleva a sensibilidade e o FP por contagem. Se um canal sem relação com os eventos também chega a 7/7, o ganho é do voto, não do sinal.

O QUE SE FAZ (mesma decisão, mesmo voto V1, mesmos 7 eventos).
  · PLACEBO A: o mesmo sinal do AE (D0_20k de temperatura, sementes 0 a 4) DESLOCADO NO TEMPO: os valores sobre os instantes vigiados são rodados circularmente em n amostras vigiadas (n = 30, 60, 90, 120, 150, 180 e 210 dias x 480 amostras/dia, ~ a taxa de operação), re-expandidos nos mesmos instantes e levados à escala do t por mapeamento de quantis. Preserva a distribuição, a autocorrelação e o ciclo de trabalho; destrói o alinhamento com os eventos. 5 sementes x 7 deslocamentos = 35 execuções.
  · PLACEBO B (outros escores como quinto canal, sem alterar o voto): rz_max, pca_spe, pca_t2, maha_mcd, iforest, knn (n = 20.000, semente 0, EXP 04), cada um levado à escala do t. 6 execuções. Diz se o ganho é do AE ou de "qualquer escore decente".
  · ATRIBUIÇÃO: quais eventos o AE detecta a mais que o controle (por evento, nas 10 sementes) e quais os placebos detectam.
CRITÉRIO (escrito antes): o 7/7 do AE conta como INFORMAÇÃO se os placebos A atingem 7/7 em no máximo 10% das 35 execuções (<= 3) E a mediana de detecção dos placebos A fica <= 6/7. Se os placebos A atingem 7/7 em > 20% das execuções, o ganho é do voto frouxo.
EXPECTATIVAS REGISTRADAS: (a) os placebos A atingem 7/7 em >= 20% das execuções (~50%); (b) pelo menos 2 dos 6 baselines do placebo B chegam a 7/7 (~40%); (c) o evento a mais é o 29/04/2025, o que quase nunca se detecta (~50%); (d) o FP médio dos placebos A fica em 12 +-2 (o do voto frouxo).
LIMITES. 7 eventos; o deslocamento circular de uma série de 13 meses mantém a estrutura de ocasiões raras; os placebos A reaproveitam as 5 primeiras sementes do AE; o voto V1 é um dos dois registrados no EXP 09.

Uso:  PYTHONPATH=src .venv/bin/python experiments/09b_placebo_do_quinto_canal.py
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
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py"); E7 = _carrega("exp07", "07_ae_condicional.py"); E9 = _carrega("exp09", "09_quinto_canal.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, mapeia_quantis
from nv.politica import ewma_vigiado

INICIO = E4.INICIO; GL = {}


def desloca_vigiado(e: pd.Series, vig: pd.Series, n: int) -> pd.Series:
    pos = np.flatnonzero(vig.reindex(e.index).fillna(False).to_numpy(bool)); v = e.to_numpy(float)[pos]
    out = np.full(len(e), np.nan); out[pos] = np.roll(v, n)
    return pd.Series(out, index=e.index)


def tarefa(a):
    rot, sinal_chave, deslocamento = a
    proc, det, vigS, reg = GL["proc"], GL["det"], GL["vigS"], GL["reg"]
    d2 = copy.deepcopy(det); d2["sinais"] = list(det["sinais"]) + ["ae"]
    for k in ("base", "k_nivel_a", "k_nivel_b"):
        d2[k]["ae"] = det[k]["t"]
    d2["voto_nivel_a"], d2["voto_nivel_b"] = 3, 2
    e = GL["sinais"][sinal_chave]
    if deslocamento:
        e = desloca_vigiado(e, vigS, int(deslocamento * 480))
    p2 = proc.copy(); p2["ae"] = mapeia_quantis(e, proc["t"], vigS)
    from nv.avalia import avalia as _av
    res = decide_v2(p2, d2)
    r = _av(res["is_anomaly"].astype(bool), res["vigiado"].astype(bool), reg, inicio_periodo=INICIO)
    return a, dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, lead=r.lead_medio_h, por_evento={k: v["det"] for k, v in r.por_evento.items()})


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
    with Pool(3) as p:
        S = dict(p.map(tarefa_ae, [("D0_20k", "temperatura", s) for s in range(10)], chunksize=1))
    GL.update(proc=proc, det=det, vigS=vigS, reg=reg, sinais={})
    for (cfg, fam, sem), s in S.items():
        GL["sinais"][("ae", sem)] = ewma_vigiado(s, vig_g, "1h").reindex(proc.index)
    for n in E4.FABRICAS:
        s = E4.escore(n, g, [c for c in g.columns if c not in E4.SEM_TAGS], est, fim)
        GL["sinais"][("base", n)] = ewma_vigiado(s, vig_g, "1h").reindex(proc.index)
    tarefas = [("AE", ("ae", s), 0) for s in range(10)] + [(f"placebo A d={d}", ("ae", s), d) for s in range(5) for d in (30, 60, 90, 120, 150, 180, 210)] + [(f"placebo B {n}", ("base", n), 0) for n in E4.FABRICAS]
    print(f"CONTROLE: {E6.linha(ctrl)}\n{len(tarefas)} execuções", flush=True)
    with Pool(4) as p:
        R = p.map(tarefa, tarefas, chunksize=1)
    ev_ctrl = {k: v["det"] for k, v in ctrl.por_evento.items()}
    def resumo(ms):
        d = [m["det"] for m in ms]; return f"detecção mediana {np.median(d):.0f}/7 (faixa {min(d)}-{max(d)}; 7/7 em {sum(x == 7 for x in d)} de {len(d)}) | FP {np.median([m['fp'] for m in ms]):.0f} ({min(m['fp'] for m in ms)}-{max(m['fp'] for m in ms)}) | horas fora {np.median([m['fora'] for m in ms]):.0f}"
    ae = [m for a, m in R if a[0] == "AE"]; pa = [m for a, m in R if a[0].startswith("placebo A")]
    print("\nAE (10 sementes):", resumo(ae)); print("PLACEBO A (35 deslocamentos x sementes):", resumo(pa))
    print("\nPLACEBO B (outros escores como quinto canal):")
    for a, m in R:
        if a[0].startswith("placebo B"):
            print(f"   {a[0]:22s}: {m['det']}/7 | FP {m['fp']} | fora {m['fora']:.0f} h | lead {m['lead']:.1f} h")
    print("\nATRIBUIÇÃO POR EVENTO (fração das execuções que detecta o evento)")
    chaves = list(ev_ctrl)
    for k in chaves:
        f = lambda ms: np.mean([m["por_evento"].get(k, False) for m in ms])
        print(f"   {str(k)[:10]}: controle {'sim' if ev_ctrl[k] else 'não'} | AE {f(ae):.2f} | placebo A {f(pa):.2f}")
    n7 = sum(m["det"] == 7 for m in pa); ok = n7 <= 3 and np.median([m["det"] for m in pa]) <= 6
    print(f"\nVEREDITO: placebos A com 7/7 em {n7} de {len(pa)} ({100 * n7 / len(pa):.0f}%); mediana {np.median([m['det'] for m in pa]):.0f}/7 -> " + ("o 7/7 do AE é INFORMAÇÃO (acima do placebo)" if ok else "voto frouxo não excluído" if n7 / len(pa) > 0.2 else "inconclusivo (entre 10% e 20%)"))


if __name__ == "__main__":
    main()
