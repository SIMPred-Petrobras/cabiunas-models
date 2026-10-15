#!/usr/bin/env python3
"""EXP 10 — H7: UM CANAL TÉRMICO SIMPLES E INTERPRETÁVEL recupera o evento de 04/11/2025 como quinto canal, sem rede neural?

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

DE ONDE VEM. EXP 09/09b: o AE de temperatura como quinto canal (voto 3 de 5) detecta o único evento que o controle perde (04/11/2025, a partida a frio) em 100% das sementes (placebo por deslocamento: 20%), com FP 12 e 563 h; mas 5 de 6 baselines genéricos também o recuperam (a custo maior): a informação está nas tags térmicas. O atlas (EXP 01/01b) mostrou que as TI de mancal/óleo
são as tags de MAIOR ESPECIFICIDADE (corrida terminal >= 12 h em apenas 0,9% das janelas normais) e que o 04/11 tem o TI_0305 a +6 a +8 sigma por 76 h antes da parada. Um canal dedicado a elas, simples, sem rede, é o candidato natural.

O CANAL. Escore = máximo, entre as tags do conjunto, de |z robusto| contra a referência ESTÁTICA do mês (a mesma do v2: as últimas 20.000 amostras estáveis anteriores ao dia 1, retreino mensal; `nv.walk`, o `rz_max` restrito ao conjunto), depois EWMA de 1 h só em vigiado, levado à escala do canal t por mapeamento de quantis, quinto canal da decisão de produção. Mesmos votos V1 (3, 2) e V2 (4, 2) do EXP 09.
CONJUNTOS (um tiro cada, sem varredura): H7a = TI_0301, 0303, 0305, 0307 (mancais), 0315, 0317 e TI_0325 (óleo): as 7 TI da família de temperatura; H7b = só os 4 mancais (TI_0301, 0303, 0305, 0307).
PLACEBO: cada conjunto deslocado no tempo (30, 60, 90, 120, 150, 180 e 210 dias, como no EXP 09b: sobre os instantes vigiados, preserva distribuição e autocorrelação, destrói o alinhamento com os eventos); V1.
MECANISMO: para o 04/11, o episódio que cobre a janela [T - 48 h, T]: o instante de início e quais canais estão acesos nas primeiras amostras do episódio (nível A e B), para o AE (semente 0) e para H7a e H7b.
CRITÉRIOS (escritos antes). "H7 CONFIRMADA" = detecção 7/7 com FP <= 12 e horas fora <= 600 h (o custo do AE) E, nos placebos, 7/7 em no máximo 1 dos 7 deslocamentos. "PARCIAL" = 7/7 com custo maior que o do AE ou placebo em 2 a 3 dos 7. "REFUTADA" = detecção <= 6/7.
EXPECTATIVAS REGISTRADAS: (E1) H7a com V1 detecta 7/7 com FP <= 13 e horas fora <= 700 (~35%); H7b (~30%); (E2) o placebo de H7a atinge 7/7 em <= 1 dos 7 deslocamentos (~55%); (E3) no início do episódio de 04/11, o quinto canal está aceso junto com t ou vb (um mecanismo térmico, não um voto isolado) (~60%); (E4) com V2, H7a mantém 6/7 ou mais com FP <= 10 (~20%).
LIMITES. 7 eventos, UM deles é o ganho; os limiares do t são herdados sem recalibração; a hipótese saiu de uma leitura do mesmo evento (04/11) no atlas: o canal é desenhado depois de ver o evento (pós-hoc em um sentido), então a confirmação real é o sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/10_canal_termico_simples.py
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
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py"); E7 = _carrega("exp07", "07_ae_condicional.py"); E9b = _carrega("exp09b", "09b_placebo_do_quinto_canal.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.modelos import FABRICAS
from nv.plugin import decide_v2, mapeia_quantis
from nv.politica import ewma_vigiado
from nv.walk import walk_forward

INICIO = E4.INICIO; GL = {}; P = "954005_624_"
CONJ = {"H7a": [P + x for x in ("TI_0301", "TI_0303", "TI_0305", "TI_0307", "TI_0315", "TI_0317", "TI_0325")], "H7b": [P + x for x in ("TI_0301", "TI_0303", "TI_0305", "TI_0307")]}
VOTOS = {"V1": (3, 2), "V2": (4, 2)}


def monta(det, voto):
    d2 = copy.deepcopy(det); d2["sinais"] = list(det["sinais"]) + ["ae"]
    for k in ("base", "k_nivel_a", "k_nivel_b"):
        d2[k]["ae"] = det[k]["t"]
    d2["voto_nivel_a"], d2["voto_nivel_b"] = VOTOS[voto]
    return d2


def sinal5(chave, deslocamento):
    e = GL["sinais"][chave]
    if deslocamento:
        e = E9b.desloca_vigiado(e, GL["vigS"], int(deslocamento * 480))
    p2 = GL["proc"].copy(); p2["ae"] = mapeia_quantis(e, GL["proc"]["t"], GL["vigS"]); return p2


def tarefa(a):
    rot, chave, voto, desl = a
    res = decide_v2(sinal5(chave, desl), monta(GL["det"], voto)); r = avalia(res["is_anomaly"].astype(bool), res["vigiado"].astype(bool), GL["reg"], inicio_periodo=INICIO)
    return a, dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, lead=r.lead_medio_h, ev={str(k)[:10]: v["det"] for k, v in r.por_evento.items()})


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
    ae0 = E7.tarefa(("D0_20k", "temperatura", 0))[1]
    GL.update(proc=proc, det=det, vigS=vigS, reg=reg, sinais={("ae", 0): ewma_vigiado(ae0, vig_g, "1h").reindex(proc.index)})
    for nome, tg in CONJ.items():
        s = walk_forward(FABRICAS["rz_max"], g, tg, est, pd.Timestamp("2025-03-01", tz="UTC"), fim)
        GL["sinais"][(nome, 0)] = ewma_vigiado(s, vig_g, "1h").reindex(proc.index)
    tarefas = [(f"{n} {v}", (n, 0), v, 0) for n in CONJ for v in VOTOS] + [(f"placebo {n} d={d}", (n, 0), "V1", d) for n in CONJ for d in (30, 60, 90, 120, 150, 180, 210)]
    print(f"CONTROLE: {E6.linha(ctrl)}\n{len(tarefas)} execuções", flush=True)
    with Pool(4) as p:
        R = p.map(tarefa, tarefas, chunksize=1)
    print("\nCANAL TÉRMICO SIMPLES COMO QUINTO CANAL (referência: AE D0_20k temperatura V1 = 7/7 | FP 12 | 563 h)")
    for a, m in R:
        if not a[0].startswith("placebo"):
            print(f"   {a[0]:10s}: {m['det']}/7 | FP {m['fp']} | fora {m['fora']:.0f} h | lead {m['lead']:.1f} h | 04/11 {'detectado' if m['ev'].get('2025-11-04') else 'perdido'}")
    print("\nPLACEBOS (deslocamentos de 30 a 210 d; V1)")
    for n in CONJ:
        ms = [m for a, m in R if a[0].startswith(f"placebo {n}")]; d = [m["det"] for m in ms]
        print(f"   {n}: detecção {d} (7/7 em {sum(x == 7 for x in d)} de {len(d)}) | FP mediano {np.median([m['fp'] for m in ms]):.0f} | horas fora mediana {np.median([m['fora'] for m in ms]):.0f} | 04/11 detectado em {sum(bool(m['ev'].get('2025-11-04')) for m in ms)} de {len(ms)}")
    # mecanismo: o episódio do 04/11
    T = pd.Timestamp("2025-11-04 06:22", tz="UTC"); print("\nMECANISMO: o episódio que cobre [T - 48 h, T] no 04/11/2025")
    for rot, chave in (("AE D0_20k temperatura (semente 0)", ("ae", 0)), ("H7a (7 TI)", ("H7a", 0)), ("H7b (4 mancais)", ("H7b", 0))):
        res = decide_v2(sinal5(chave, 0), monta(det, "V1")); al = res["is_anomaly"].astype(bool)
        epi = [(a_, b_) for a_, b_ in E9b.E4.avalia.__globals__["episodios_de_serie"](al.loc[T - pd.Timedelta(days=4):T + pd.Timedelta(hours=2)]) if a_ <= T and b_ >= T - pd.Timedelta(hours=48)]
        if not epi:
            print(f"   {rot}: nenhum episódio na janela"); continue
        a_, b_ = epi[0]; w = res.loc[a_:a_ + pd.Timedelta(minutes=20)]
        acesos = [c for c in ("t", "p", "sp", "vb", "ae") if bool(w[f"a_{c}"].any())]; acesosB = [c for c in ("t", "p", "sp", "vb", "ae") if bool(w[f"b_{c}"].any())]
        print(f"   {rot}: episódio {a_:%m-%d %H:%M} a {b_:%m-%d %H:%M} ({(b_ - a_).total_seconds() / 3600:.1f} h) | canais do nível A acesos no início: {acesos} | do nível B: {acesosB}")


if __name__ == "__main__":
    main()
