#!/usr/bin/env python3
"""EXP 08 — NORMALIZAÇÃO LOCAL DO ESCORE (LRN): tirar do canal os desvios lentos e persistentes do normal, nos canais do próprio v2 e nos do autoencoder.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

DE ONDE VEM. O EXP 07 mostrou que o autoencoder não passa porque as excursões dele em operação normal são mais PERSISTENTES que as dos canais do v2 e o CUSUM as acumula em episódios longos; e que a normalidade não é estacionária (frescura vale mais que volume). Os canais t e p do v2 usam uma referência ESTÁTICA no mês (o bundle); o `vb` já usa uma referência LOCAL e atrasada.
IDEIA: normalizar o escore de cada canal por uma referência local e atrasada: z(t) = max(0, (e(t) - mediana_ref) / sigma_ref), com a referência = as medianas horárias dos instantes vigiados em [t - 35 d, t - 7 d) (janela de 28 d, guarda de 7 d; `nv.lrn`), e levar o resultado de volta à escala do canal por mapeamento de quantis, como no plug-in. Remove deriva lenta sem retreinar nada e sem rótulos.
A pesquisa recusou cinco técnicas de drift sobre o NORMALIZADOR do bundle e a recentragem pós-manutenção; a LRN age no ESCORE e é causal, mas pode cheirar à mesma família: por isso tem uma regra de robustez.

BRAÇOS.
  A  os canais do PRÓPRIO v2 com LRN: t, p e t+p (os sinais causais do EXP 05; a decisão de produção como caixa-preta). Parâmetros padrão (janela 672 h, guarda 168 h) e quatro perturbações (janela 576 e 768 h; guarda 144 e 192 h) como controle de robustez.
  B  o AE D0_20k (as últimas 20.000 amostras, o melhor do EXP 07) como canal t e como canal p, com e sem LRN, em 10 SEMENTES (o piso de ruído neural é grande).
  C  o AE D2 (contexto de partida + nível lento) como canal t, com e sem LRN, em 5 sementes.
CRITÉRIOS (escritos antes).
  · MELHORIA DO CONTROLE (braço A): com os parâmetros padrão, detecção >= 6/7, FP <= 9 e horas fora das janelas <= 326 h (-20% do controle: 407 h) E, nas quatro perturbações, detecção >= 5/7, FP <= 10 e horas fora <= 407 h (robustez).
  · Braços B e C: DIGNO = mediana das sementes com detecção >= 6/7, FP <= 9 e horas fora <= 407 h; PROMISSOR = detecção >= 5/7, FP <= 12 e horas fora <= 800 h.
  · Os dois custos (FP em episódios; horas) e os 7 eventos, como sempre.

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  E1  a LRN aplicada aos canais do v2 reduz as horas fora em >= 20% em pelo menos um braço (~40%); se reduz, mantém detecção >= 6/7 (~50% condicional); MELHORIA DO CONTROLE completa (com a robustez): ~15%.
  E2  a LRN no AE D0_20k canal p reduz as horas fora em >= 30% contra a mesma série sem LRN (~60%) e mantém a detecção mediana >= 5/7 (~65%).
  E3  com 10 sementes o D0_20k p sem LRN reproduz o EXP 07: detecção mediana 6/7, FP 11 +-2, horas fora 700 +-250 (~70%).
  E4  algum DIGNO no braço B ou C: canal p ~25%, canal t ~10%.
  E5  a robustez a parâmetros (braço A): >= 3 das 4 perturbações mantêm a detecção >= 5/7 (~55%).

LIMITES. 7 eventos; a LRN atenua também precursores lentos (uma queda de detecção pode ser o preço da especificidade); o mapa de quantis é global no período; os parâmetros padrão (28 d, 7 d) vêm do princípio do vb e da guarda de 168 h da pesquisa e foram fixados antes de ver resultado, mas não são independentes dos 8 eventos (a guarda de 168 h foi achada neles).

RESULTADO (15/10/2026) -- A LRN PIORA: NENHUMA MELHORIA DO CONTROLE, NENHUM DIGNO. (A 1.ª execução parou num erro de serialização entre processos, antes de qualquer resultado.)
  BRAÇO A (canais do v2 com LRN; detecção de 7 | FP | horas fora das janelas), padrão (janela 672 h, guarda 168 h) e perturbações: canal t: 3/7 | 15 | 718 (e 3/7 em todas as perturbações, FP 15 a 16, horas fora de 328 a 984); canal p: 5/7 | 12 | 741 (perturbações 5 a 6/7, FP 11 a 13, 738 a 1.354 h); t+p: 4/7 | 13 | 961 (perturbações 3 a 5/7, FP 13 a 15).
        CONTROLE: 6/7 | 9 | 407. Nenhum braço cumpre o padrão (det >= 6, FP <= 9, <= 326 h) nem é robusto: MELHORIA DO CONTROLE = não em nenhum.
  BRAÇOS B e C (mediana de 10 e de 5 sementes; Q1-Q3): D0_20k canal t sem LRN 4/7 | FP 16 | 764 h (665-958) e COM LRN 3/7 | 17 | 1.224 h; D0_20k canal p sem LRN 6/7 (6-6; faixa 6-6) | FP 12 (11-14) | 1.075 h (695-1.341) e COM LRN 6/7 (faixa 5-6) | FP 14 | 1.253 h; D2 canal t sem LRN 5/7 | FP 20 | 1.748 h e COM LRN 4/7 | 18 | 1.872 h. NENHUM DIGNO, NENHUM PROMISSOR.
CORREÇÃO DO EXP 07: com 10 sementes o D0_20k no canal p tem horas fora MEDIANA de 1.075 h (Q1-Q3 695-1.341), não os 722 h das 3 primeiras sementes: o "PROMISSOR" do EXP 07 era SORTE DE SEMENTE (o limite de 800 h não se mantém). A detecção de 6/7 é estável (todas as 10 sementes) e o FP de 11 a 14; a especificidade fica ~17% melhor que o melhor baseline no p (1.290 h), longe do p do v2 (407 h).
LEITURA. (1) A normalização local, tanto nos canais do v2 quanto nos do AE, REDUZ a detecção ou mantém e AUMENTA as horas falsas: a referência móvel ABSORVE a deriva lenta que é também o precursor (as corridas terminais dos trips duram 3 a 7 dias, EXP 01b) e ainda amplifica o ruído onde a dispersão da referência é pequena. (2) Junto do EXP 07 (frescura > volume), a mensagem é consistente: a referência do v2, ESTÁTICA no mês e FRESCA (20.000 amostras, retreino mensal),
        é um ponto doce entre "velha demais" (histórico longo) e "adaptativa demais" (referência móvel); os dois lados pioram. (3) Seis sementes a mais mudaram a leitura do melhor caso: com redes neurais, 3 sementes enganam.
PLACAR: ERREI tudo que previ de otimista: E1 (a LRN reduz >= 20% as horas fora nos canais do v2: não; o canal t caiu a 3/7), E2 (a LRN no AE reduz >= 30% as horas: aumentou), E3 (o D0_20k p reproduz 700 +-250: 1.075), E4 (algum digno: nenhum). ACERTEI E5 só para o canal p do v2 (4 de 4 perturbações com detecção >= 5/7; no t, 0 de 4; no t+p, 1 de 4).

Uso:  PYTHONPATH=src .venv/bin/python experiments/08_normalizacao_local.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
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
from nv.lrn import lrn
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado

INICIO = E4.INICIO
GL = {}
PADRAO = (672, 168); PERTURBA = [(576, 168), (768, 168), (672, 144), (672, 192)]


def tarefa_ae(a):
    return E7.tarefa(a)                       # invólucro: uma função de módulo carregado por importlib não se serializa para os processos


def tarefa_avalia(a):
    """a = (braco, rotulo, sinal_chave, canais, parametros|None): devolve as métricas do plug-in."""
    braco, rot, chave, canais, par = a
    proc, det, vigS, reg = GL["proc"], GL["det"], GL["vigS"], GL["reg"]
    p2 = proc
    for c in canais:
        e = GL["sinais"][(chave, c)]
        z = lrn(e, vigS, *par) if par else e
        p2 = substitui(p2, c, mapeia_quantis(z, proc[c], vigS))
    r = E6.mede(decide_v2(p2, det), reg)
    return a, dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {"temperatura": modelos[-1]["normalizacao"]["temperatura"]["cols"], "pressao": modelos[-1]["normalizacao"]["pressao"]["cols"]}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    sinais = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = sinais[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    E7._G.update(g=g, est=est, tags=tags, fim=fim)
    # sementes que faltam (D0_20k: 0..9; D2 temperatura: 0..4)
    falta = [("D0_20k", f, s) for f in E7.FAMILIAS for s in range(10)] + [("D2", "temperatura", s) for s in range(5)] + [("D0_20k", f, s) for f in E7.FAMILIAS for s in range(10)]
    falta = list(dict.fromkeys(falta))
    print(f"CONTROLE: {E6.linha(ctrl)}\ntreinando/lendo {len(falta)} escores do AE (cache)", flush=True)
    with Pool(3) as p:
        S = dict(p.map(tarefa_ae, falta, chunksize=1))
    GL.update(proc=proc, det=det, vigS=vigS, reg=reg, sinais={})
    for (cfg, fam, sem), s in S.items():
        GL["sinais"][((cfg, sem), {"temperatura": "t", "pressao": "p"}[fam])] = ewma_vigiado(s, vig_g, "1h").reindex(proc.index)
    for c in ("t", "p"):
        GL["sinais"][(("v2", 0), c)] = proc[c]
    tarefas = []
    for canais, nome in ((("t",), "t"), (("p",), "p"), (("t", "p"), "t+p")):
        for par in [PADRAO] + PERTURBA:
            tarefas.append(("A", f"v2 {nome} LRN {par}", ("v2", 0), canais, par))
    for cfg, sems in (("D0_20k", range(10)), ("D2", range(5))):
        for fam, c in (("temperatura", "t"), ("pressao", "p")):
            if cfg == "D2" and c == "p":
                continue
            for sem in sems:
                for par in (None, PADRAO):
                    tarefas.append(("B" if cfg == "D0_20k" else "C", f"{cfg} {c} {'LRN' if par else 'sem'}", (cfg, sem), (c,), par))
    print(f"{len(tarefas)} execuções do plug-in", flush=True)
    with Pool(4) as p:
        R = p.map(tarefa_avalia, tarefas, chunksize=1)
    print(f"\nCONTROLE: {E6.linha(ctrl)}\n\nBRAÇO A — OS CANAIS DO PRÓPRIO v2 COM LRN (detecção de 7 | FP | horas fora | horas totais | lead h)")
    A = [(a, m) for a, m in R if a[0] == "A"]
    for rot in dict.fromkeys(a[1].split(" LRN")[0] for a, _ in A):
        for a, m in A:
            if a[1].startswith(rot + " LRN"):
                print(f"   {a[1]:34s}: {m['det']}/7 | {m['fp']:2d} | {m['fora']:5.0f} | {m['horas']:5.0f} | {m['lead']:.1f}")
        print()
    pad = {a[1]: m for a, m in A if a[4] == PADRAO}
    melhoria = []
    for rot in ("v2 t", "v2 p", "v2 t+p"):
        m0 = pad[f"{rot} LRN {PADRAO}"]; pert = [m for a, m in A if a[1].startswith(rot + " LRN") and a[4] != PADRAO]
        ok0 = m0["det"] >= 6 and m0["fp"] <= 9 and m0["fora"] <= 326; okp = all(m["det"] >= 5 and m["fp"] <= 10 and m["fora"] <= 407 for m in pert)
        melhoria.append((rot, ok0, okp)); print(f"   {rot}: padrão {'cumpre' if ok0 else 'não cumpre'} | perturbações {'robusto' if okp else 'não robusto'} -> {'MELHORIA DO CONTROLE' if ok0 and okp else 'não'}")
    print("\nBRAÇOS B e C — AE COM E SEM LRN (mediana e Q1-Q3 das sementes)")
    grupos = {}
    for a, m in R:
        if a[0] in ("B", "C"):
            grupos.setdefault(a[1], []).append(m)
    for rot, ms in grupos.items():
        d = [m["det"] for m in ms]; f = [m["fp"] for m in ms]; o = [m["fora"] for m in ms]
        md, mf, mo = np.median(d), np.median(f), np.median(o)
        v = "DIGNO" if md >= 6 and mf <= 9 and mo <= 407 else "PROMISSOR" if md >= 5 and mf <= 12 and mo <= 800 else "-"
        print(f"   {rot:16s} ({len(ms)} sementes): detecção mediana {md:.0f}/7 (Q1-Q3 {np.quantile(d, .25):.0f}-{np.quantile(d, .75):.0f}; faixa {min(d)}-{max(d)}) | FP {mf:.0f} ({np.quantile(f, .25):.0f}-{np.quantile(f, .75):.0f}) | horas fora {mo:.0f} ({np.quantile(o, .25):.0f}-{np.quantile(o, .75):.0f}) -> {v}")


if __name__ == "__main__":
    main()
