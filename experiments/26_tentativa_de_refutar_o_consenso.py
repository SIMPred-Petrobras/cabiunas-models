#!/usr/bin/env python3
"""EXP 26 — TENTATIVA DE REFUTAR O CONSENSO DE VIZINHANÇA (análise crítica).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. Cada teste é escrito para DERRUBAR a conclusão do EXP 25, não para confirmá-la.

SUSPEITAS (do que as memórias e os EXP 13b/25c já mostram):
  S1 O corte são DOIS EPISÓDIOS: no 25c, as horas fora caem de 407 para 234 h (-173 h) e os FP que somem são o de 15/10/2025 (135 h) e o de 09/01/2026 (34 h) = 169 h. É o mesmo regime (ago-out/2025) e o mesmo tipo de efeito que o EXP 13b chamou de artefato (um episódio de 112 h). Com ~14 meses e 2 episódios, pode ser indistinguível de acaso.
  S2 O consenso é DOMINADO por uma alavanca simples: o EXP 13 mostrou `escalada_idade_h = 192` com 6/7, 7 FP, 261 h (sem perder detecção), contra 5/7, 7 FP, 234 h do consenso; e uma ESCALA UNIFORME de todos os k pode traçar a mesma curva sem votação.
  S3 A AMPLITUDE (+-10%) e o N (60) foram escolhidos por mim: o efeito pode existir só aí.
  S4 UMA SEMENTE: o consenso pode ser sorte do sorteio 20261016.
O QUE SE FAZ (caminho exato de produção; os sinais do EXP 05; `ci.prever`; consenso q = 0,7 e q = 0,8 salvo se indicado):
  T1 (S1) decomposição por mês e por episódio: FP e horas do publicado e do consenso por mês de início de episódio; a fração das horas removidas que vêm de <= 2 episódios; bootstrap por blocos MENSAIS (2.000 reamostragens dos meses com reposição) da diferença publicado - consenso em FP por mês e em horas fora por mês: p unilateral = fração das reamostras com diferença <= 0.
  T2 (S2) ESCALA UNIFORME de todos os 8 k por s em {0,80; 0,85; ...; 1,50} (uma só configuração, sem votação): detecção, FP, horas fora; e o `escalada_idade_h = 192` do EXP 13 como comparação.
  T3 (S3) amplitude em {5%, 10%, 15%, 20%} (N = 60, uma semente cada) e N em {30, 60, 100} a +-10%.
  T4 (S4) três sementes novas (20261018, 20261019, 20261020), +-10%, N = 60.
CRITÉRIOS DE REFUTAÇÃO (fixados antes). T1: o corte NÃO se distingue de acaso se >= 90% das horas removidas vêm de <= 2 episódios E o p unilateral do bootstrap do FP/mês é >= 0,05. T2: o consenso é DOMINADO se alguma escala uniforme (ou o escalada_idade_h = 192) tem detecção >= 5, FP <= 7 e horas fora <= 280 (a 5/7 ou melhor). T3: ROBUSTO se em >= 3 das 4 amplitudes o consenso (q = 0,7) tem detecção >= 5, FP <= 8 e horas fora <= 320. T4: REPRODUTÍVEL se nas 3 sementes (q = 0,7) a detecção é >= 5 e FP <= 8 e horas fora <= 320.
EXPECTATIVAS (nesta ordem de probabilidade de a SUSPEITA se confirmar): T1 o corte não se distingue de acaso ~75% (as horas removidas são dois episódios; o p do bootstrap ~0,10); T2 alguma escala uniforme ou o E192 domina o consenso ~55% (o E192 já o domina: 6/7, 7 FP, 261 h); T3 robusto em >= 3 de 4 amplitudes ~35%; T4 reprodutível em 3 de 3 ~65%.
INTERPRETAÇÃO PREVISTA. Se T1 confirma a suspeita e T2 acha uma alavanca simples que domina, o consenso NÃO é um achado: é a mesma remoção de 2 FP do regime ago-out/2025 e fica registrado como tal, sem hipótese prospectiva. Se T1 e T2 não confirmam e T3/T4 passam, o consenso é uma melhoria de operação, mas ainda com a prova prospectiva pendente.
LIMITES. Mesmos 7 eventos; o bootstrap por blocos mensais com ~14 blocos tem pouco poder (o p mínimo é limitado); o EXP 13 é uma varredura de 29 células que já tinha o mesmo risco de seleção.

Uso:  PYTHONPATH=src .venv/bin/python experiments/26_tentativa_de_refutar_o_consenso.py
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
E25 = _carrega("exp25", "25_consenso_de_vizinhanca.py"); E6 = E25.E6; E5 = E25.E5
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import sinais_causais, _ci, PKG

INICIO = E25.INICIO; GL = E25.GL; CHAVES = E25.CHAVES; JAN = pd.Timedelta(hours=48)


def tarefa(a):
    return a[0], E25.alarme(a[1])


def sorteios(semente, amp, n):
    rng = np.random.default_rng(semente); return [{k: float(rng.uniform(1 - amp, 1 + amp)) for k in CHAVES} for _ in range(n)]


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= INICIO]; al_c, vig_c = carrega_controle("atual"); ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    GL.update(det=det, proc=proc, reg=reg, vig=vig_c.reindex(proc.index).fillna(False)); idx = proc.index
    hv = float(GL["vig"].loc[INICIO:].sum()) * 2 / 60 / 730
    pedidos = [(("pub",), {})] + [(("esc", s), {k: s for k in CHAVES}) for s in np.round(np.arange(0.80, 1.501, 0.05), 2)]
    jobs = {"A10": (20261016, 0.10, 100), "A05": (20261031, 0.05, 60), "A15": (20261032, 0.15, 60), "A20": (20261033, 0.20, 60), "S1": (20261018, 0.10, 60), "S2": (20261019, 0.10, 60), "S3": (20261020, 0.10, 60)}
    for nome, (sem, amp, n) in jobs.items():
        pedidos += [((nome, i), m) for i, m in enumerate(sorteios(sem, amp, n))]
    print(f"CONTROLE: {E6.linha(ctrl)} | {len(pedidos)} execuções", flush=True)
    with Pool(4) as p:
        R = dict(p.map(tarefa, pedidos, chunksize=2))
    pub = R[("pub",)]; Rp = E25.mede(pub, idx)
    fp_de = lambda arr: [(a, b) for a, b in episodios_de_serie(pd.Series(arr, index=idx).loc[INICIO:]) if not any(a <= t and b >= t - JAN for t in reg)]
    cons = lambda nome, n, q: np.mean([R[(nome, i)] for i in range(n)], axis=0) >= q
    print(f"PUBLICADO: {Rp.det}/7 | FP {Rp.fp} | fora {Rp.horas_alarme_fora:.0f} h", flush=True)

    print("\nT1 — DECOMPOSIÇÃO (consenso q = 0,7, 60 configurações do EXP 25)", flush=True)
    for q in (0.7, 0.8):
        c = cons("A10", 60, q); FPp, FPc = fp_de(pub), fp_de(c); somem = [x for x in FPp if not any(x[0] <= b and x[1] >= a for a, b in FPc)]
        h_rem = Rp.horas_alarme_fora - E25.mede(c, idx).horas_alarme_fora; h_som = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in somem); top2 = sum(sorted(((b - a).total_seconds() / 3600 + 2 / 60 for a, b in somem), reverse=True)[:2])
        meses = sorted({str(a.to_period("M")) for a, b in FPp + FPc} | {str(x.to_period("M")) for x in pd.date_range(INICIO, idx[-1], freq="MS", tz="UTC")}); mp = {m: 0 for m in meses}; mc = {m: 0 for m in meses}; hp = {m: 0.0 for m in meses}; hc = {m: 0.0 for m in meses}
        for a, b in FPp:
            mp[str(a.to_period("M"))] += 1; hp[str(a.to_period("M"))] += (b - a).total_seconds() / 3600
        for a, b in FPc:
            mc[str(a.to_period("M"))] += 1; hc[str(a.to_period("M"))] += (b - a).total_seconds() / 3600
        d_fp = np.array([mp[m] - mc[m] for m in meses]); d_h = np.array([hp[m] - hc[m] for m in meses]); rng = np.random.default_rng(7); B = 2000
        bs_fp = np.array([d_fp[rng.integers(0, len(meses), len(meses))].sum() for _ in range(B)]); bs_h = np.array([d_h[rng.integers(0, len(meses), len(meses))].sum() for _ in range(B)])
        print(f"   q = {q}: horas fora removidas {h_rem:.0f} h; FP que somem {len(somem)} ({h_som:.0f} h; os 2 maiores = {top2:.0f} h = {100 * top2 / max(h_rem, 1):.0f}% das horas removidas) | meses com diferença: {[m for m, d in zip(meses, d_fp) if d != 0]} | p unilateral (bootstrap mensal, diferença <= 0): FP {np.mean(bs_fp <= 0):.3f}, horas {np.mean(bs_h <= 0):.3f}", flush=True)

    print("\nT2 — ESCALA UNIFORME DE TODOS OS k (uma configuração, sem votação)", flush=True)
    dom = []
    for s in np.round(np.arange(0.80, 1.501, 0.05), 2):
        r = E25.mede(R[("esc", s)], idx); ok = r.det >= 5 and r.fp <= 7 and r.horas_alarme_fora <= 280; dom.append(ok)
        print(f"   s = {s:.2f}: {r.det}/7 | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h{'  <- domina o consenso' if ok else ''}", flush=True)
    print(f"   (comparação: escalada_idade_h = 192 do EXP 13: 6/7, 7 FP, 261 h; consenso q = 0,7: 5/7, 7 FP, 234 h) | alguma escala uniforme domina: {any(dom)}", flush=True)

    print("\nT3 — AMPLITUDE E N (consenso q = 0,7 e 0,8)", flush=True); robusto = 0
    for nome, rot, n in (("A05", "±5%", 60), ("A10", "±10%", 60), ("A15", "±15%", 60), ("A20", "±20%", 60), ("A10", "±10%, N = 30", 30), ("A10", "±10%, N = 100", 100)):
        for q in (0.7, 0.8):
            r = E25.mede(cons(nome, n, q), idx); ok = r.det >= 5 and r.fp <= 8 and r.horas_alarme_fora <= 320
            if q == 0.7 and n == 60:
                robusto += ok
            print(f"   {rot:15s} q = {q}: {r.det}/7 | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h{' ✓' if ok else ''}", flush=True)
    print(f"   amplitudes (q = 0,7) que passam: {robusto} de 4 (limite escrito: >= 3)", flush=True)

    print("\nT4 — SEMENTES NOVAS (±10%, N = 60)", flush=True); rep = 0
    for nome in ("S1", "S2", "S3"):
        for q in (0.7, 0.8):
            r = E25.mede(cons(nome, 60, q), idx); ok = r.det >= 5 and r.fp <= 8 and r.horas_alarme_fora <= 320
            if q == 0.7:
                rep += ok
            print(f"   semente {nome} q = {q}: {r.det}/7 | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h{' ✓' if ok else ''}", flush=True)
    print(f"   sementes reprodutíveis (q = 0,7): {rep} de 3", flush=True)


if __name__ == "__main__":
    main()
