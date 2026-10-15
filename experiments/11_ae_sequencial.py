#!/usr/bin/env python3
"""EXP 11 — AE SEQUENCIAL (convolução 1-D em janela de 30 min) COMO QUINTO CANAL da decisão do v2.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. O AE pontual de temperatura (EXP 09, D0_20k) como quinto canal dá 7/7, FP 12 (faixa 11-14), 563 h fora (466-689), em 10 sementes. A dinâmica dentro de 30 min (taxa de subida, oscilação) acrescenta ESPECIFICIDADE? Mesmo plug-in, mesmo voto V1 (3,2), mesma base/k do canal t, mesmo mapa de quantis; só muda o escore:
  janela de W=15 amostras (30 min) terminando no instante, `nv.modelos.tcn_torch` (Conv1d, bottleneck k=4, mesma corrupção/perda/escore do AE pontual), as últimas 20.000 janelas estáveis anteriores ao mês (o desenho D0_20k, 30 épocas, sem warm-start), 10 sementes, tags de temperatura do bundle.
  Só existe escore onde a janela inteira é finita (a vigilância do v2 já exclui as 6 h pós-partida).
REFERÊNCIA PARA COMPARAR: AE pontual D0_20k temperatura (recalculado do cache, mesmas 10 sementes) e o CONTROLE (6/7, 9 FP, 407 h fora).
CRITÉRIO (mediana das 10 sementes, V1): MELHOR = detecção >= 7/7 com FP <= 10 e horas fora <= 450 (abaixo da faixa do AE pontual, que é o que o usuário pede: menos FP); IGUAL = detecção >= 6/7 com FP 11 a 14 e horas fora <= 700; PIOR = o resto. Não se chama de ganho nada dentro do piso de ruído (detecção +-0,5, FP +-1, carga +-20%) e, se der MELHOR, seguem placebo e leitura do 04/11 antes de qualquer conclusão.
EXPECTATIVAS: MELHOR ~12%; IGUAL ~48%; PIOR ~40% (o EXP 07/08 mostrou que contexto e estrutura extra não ajudam, e o piso de ruído neural é maior). O 04/11 continua detectado em >= 8 das 10 sementes ~65%. O escore sequencial acende mais tempo seguido que o pontual (mais horas por episódio) ~55%.
LIMITES. 7 eventos; uma única arquitetura e janela (sem varredura: seria sobreajuste ao alvo); herda limiares do t sem recalibração; mapa de quantis global.

Uso:  PYTHONPATH=src .venv/bin/python experiments/11_ae_sequencial.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py"); E7 = _carrega("exp07", "07_ae_condicional.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.modelos.tcn_torch import walk_tcn
from nv.plugin import decide_v2, mapeia_quantis
from nv.politica import ewma_vigiado

INICIO = E4.INICIO; CACHE = Path(__file__).resolve().parents[1] / "data_local" / "escores_tcn"; GL = {}; SEMENTES = range(10)


def tarefa_tcn(sem):
    f = CACHE / f"w15_temperatura_{sem}.parquet"
    if f.exists():
        return sem, pd.read_parquet(f)["s"]
    import torch
    torch.set_num_threads(2); t0 = time.time()
    s = walk_tcn(GL["g"], GL["tags"], GL["est"], pd.Timestamp("2025-03-01", tz="UTC"), GL["fim"], semente=sem)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      TCN semente {sem}: {time.time() - t0:.0f} s", flush=True)
    return sem, s


def tarefa_avalia(a):
    nome, sem = a; proc, det, vigS, reg = GL["proc"], GL["det"], GL["vigS"], GL["reg"]
    d2 = copy.deepcopy(det); d2["sinais"] = list(det["sinais"]) + ["ae"]
    for k in ("base", "k_nivel_a", "k_nivel_b"):
        d2[k]["ae"] = det[k]["t"]
    d2["voto_nivel_a"], d2["voto_nivel_b"] = 3, 2
    p2 = proc.copy(); p2["ae"] = mapeia_quantis(GL["sinais"][(nome, sem)], proc["t"], vigS); res = decide_v2(p2, d2); r = E6.mede(res, reg)
    ev = [e for e in getattr(r, "eventos", [])] if hasattr(r, "eventos") else []
    return a, dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h, inicio=r.inicio)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = modelos[-1]["normalizacao"]["temperatura"]["cols"]
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    sinais = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = sinais[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    GL.update(g=g, est=est, tags=tags, fim=fim, proc=proc, det=det, vigS=vigS, reg=reg)
    E7._G.update(g=g, est=est, tags={"temperatura": tags, "pressao": modelos[-1]["normalizacao"]["pressao"]["cols"]}, fim=fim)
    print(f"CONTROLE: {E6.linha(ctrl)} | {len(tags)} tags de temperatura", flush=True)
    with Pool(4) as p:
        S = dict(p.map(tarefa_tcn, list(SEMENTES), chunksize=1))
        P = dict(((c, f, s), v) for (c, f, s), v in p.map(E7.tarefa, [("D0_20k", "temperatura", s) for s in SEMENTES], chunksize=1))
    GL["sinais"] = {}
    for s in SEMENTES:
        GL["sinais"][("TCN sequencial W15", s)] = ewma_vigiado(S[s], vig_g, "1h").reindex(proc.index)
        GL["sinais"][("AE pontual D0_20k", s)] = ewma_vigiado(P[("D0_20k", "temperatura", s)], vig_g, "1h").reindex(proc.index)
    tarefas = [(n, s) for n in ("TCN sequencial W15", "AE pontual D0_20k") for s in SEMENTES]
    with Pool(4) as p:
        R = p.map(tarefa_avalia, tarefas, chunksize=1)
    G = {}
    for (n, s), m in R:
        G.setdefault(n, []).append(m)
    print("\nQUINTO CANAL, voto V1 (mediana e faixa das 10 sementes)")
    for n, ms in G.items():
        d = [m["det"] for m in ms]; f = [m["fp"] for m in ms]; o = [m["fora"] for m in ms]; h = [m["horas"] for m in ms]; md, mf, mo = np.median(d), np.median(f), np.median(o)
        ver = "MELHOR" if md >= 7 and mf <= 10 and mo <= 450 else "IGUAL" if md >= 6 and 11 <= mf <= 14 and mo <= 700 else "PIOR"
        print(f"   {n:20s}: detecção {md:.0f}/7 ({min(d)}-{max(d)}) | FP {mf:.0f} ({min(f)}-{max(f)}) | horas fora {mo:.0f} ({min(o)}-{max(o)}) | horas totais {np.median(h):.0f} | horas por episódio (fora/FP) {mo / max(mf, 1):.0f} | lead {np.nanmedian([m['lead'] for m in ms]):.1f} h -> {ver if n.startswith('TCN') else '(referência)'}")
    print("   detecção por semente (TCN):", [m["det"] for m in G["TCN sequencial W15"]])


if __name__ == "__main__":
    main()
